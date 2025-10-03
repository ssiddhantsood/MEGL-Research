#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_braess.py  — continuation-backed predictor

This version avoids sweep_linear_stability and instead uses the same method
as your "find braess" code: continuation_descend_K to get a robust Kc quickly.
Then it locks a steady state at K≈Kc and computes dK/da (same math as before).

Notes:
- We work in the co-rotating frame (center omega) when finding Kc, matching utils.
- For locking at Kc we use your U.solve_locked (warm-started with a Laplacian seed).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
import scipy.linalg as la
from scipy.optimize import least_squares

import utils as U


# ---------- small helpers ----------
def edge_matrix(n: int, p: int, q: int, w_add: float = 1.0) -> np.ndarray:
    E = np.zeros((n, n), dtype=float)
    if p == q:
        raise ValueError("Edge must connect two distinct nodes.")
    E[p, q] = float(w_add)
    E[q, p] = float(w_add)
    return E


def F_theta(theta: np.ndarray, K: float, W: np.ndarray, omega: np.ndarray) -> np.ndarray:
    dth = theta[np.newaxis, :] - theta[:, np.newaxis]
    S = W * np.sin(dth)
    return omega + K * S.sum(axis=1)


def H_residual(theta: np.ndarray,
               v: np.ndarray,
               K: float,
               a: float,
               W0: np.ndarray,
               E: np.ndarray,
               omega: np.ndarray) -> np.ndarray:
    W = W0 + a * E
    F = F_theta(theta, K, W, omega)           # n
    J = U.kuramoto_jacobian(theta, W, K)      # n×n (symmetric)
    eig_block = J @ v                          # n
    norm_block = np.array([np.dot(v, v) - 1.0])
    gauge_block = np.array([theta[0]])
    return np.concatenate([F, eig_block, norm_block, gauge_block], axis=0)


def dF_dtheta(theta: np.ndarray, K: float, W: np.ndarray) -> np.ndarray:
    return U.kuramoto_jacobian(theta, W, K)

def dF_dK(theta: np.ndarray, W: np.ndarray) -> np.ndarray:
    dth = theta[np.newaxis, :] - theta[:, np.newaxis]
    S = W * np.sin(dth)
    return S.sum(axis=1)

def dF_da(theta: np.ndarray, K: float, E: np.ndarray) -> np.ndarray:
    dth = theta[np.newaxis, :] - theta[:, np.newaxis]
    S = E * np.sin(dth)
    return K * S.sum(axis=1)

def dJv_dtheta_FD(theta: np.ndarray, v: np.ndarray, W: np.ndarray, K: float, eps: float = 1e-6) -> np.ndarray:
    """Finite-diff columns for ∂(Jv)/∂θ (simple & robust)."""
    n = len(theta)
    J = U.kuramoto_jacobian(theta, W, K)
    base = J @ v
    cols = np.zeros((n, n), dtype=float)
    for k in range(n):
        th = theta.copy(); th[k] += eps
        Jk = U.kuramoto_jacobian(th, W, K)
        cols[:, k] = (Jk @ v - base) / eps
    return cols

def dJv_dv(theta: np.ndarray, W: np.ndarray, K: float) -> np.ndarray:
    return U.kuramoto_jacobian(theta, W, K)

def dJv_dK(theta: np.ndarray, W: np.ndarray, K: float) -> np.ndarray:
    J = U.kuramoto_jacobian(theta, W, K)
    if abs(K) < 1e-14:
        dK = 1e-6
        Jp = U.kuramoto_jacobian(theta, W, K + dK)
        return ((Jp - J) @ np.ones(len(theta))) / dK  # fallback; not usually used
    return (J @ np.ones(len(theta))) / K  # we only need the vector (J/K) * v, but we reshape later


def dJv_dK_vec(theta: np.ndarray, W: np.ndarray, K: float, v: np.ndarray) -> np.ndarray:
    J = U.kuramoto_jacobian(theta, W, K)
    if abs(K) < 1e-14:
        dK = 1e-6
        Jp = U.kuramoto_jacobian(theta, W, K + dK)
        return ((Jp - J) @ v) / dK
    return (J @ v) / K

def dJv_da(theta: np.ndarray, E: np.ndarray, K: float, v: np.ndarray) -> np.ndarray:
    Ja = U.kuramoto_jacobian(theta, E, K)
    return Ja @ v


# ---------- continuation-backed critical solve ----------
@dataclass
class CriticalSolution:
    theta: np.ndarray
    v: np.ndarray
    K: float
    a: float
    res_norm: float
    Kc_from_continuation: float

def _kc_via_continuation(W0: np.ndarray,
                         omega: np.ndarray,
                         K_start: float,
                         K_min: float,
                         K_step: float,
                         r_thresh: float) -> float:
    """Use the same robust Kc finder as your scan pipeline."""
    omega_c = U._center_omega(omega)
    Kc, Ks, Rs = U.continuation_descend_K(
        omega_c, W0, K_start=K_start, K_min=K_min, K_step=K_step,
        r_thresh=r_thresh, rng=None
    )
    if Kc is None:
        # fall back to argmax R if threshold never hit
        if len(Ks) == 0:
            raise RuntimeError("continuation_descend_K returned no points")
        Kc = float(Ks[np.argmax(Rs)])
    return float(Kc)

def _laplacian_seed(omega: np.ndarray, W: np.ndarray, K: float) -> np.ndarray:
    return U._laplacian_initial_guess(U._center_omega(omega), W, K)

def solve_critical_point_continuation(W0: np.ndarray,
                                      omega: np.ndarray,
                                      K_start: float,
                                      K_min: float,
                                      K_step: float,
                                      r_thresh: float,
                                      a_value: float = 0.0,
                                      K_guess_override: Optional[float] = None) -> CriticalSolution:
    """
    Get Kc via continuation (or override if provided), lock a fixed point at ~Kc,
    then refine (theta, v, K) by minimizing the augmented residual.
    """
    n = W0.shape[0]
    # 1) Kc via continuation (or use override)
    Kc = float(K_guess_override) if K_guess_override is not None else _kc_via_continuation(
        W0, omega, K_start, K_min, K_step, r_thresh
    )

    # 2) Lock at Kc
    omega_c = U._center_omega(omega)
    theta0 = _laplacian_seed(omega, W0, Kc)
    theta, ok, _ = U.solve_locked(omega_c, W0, Kc, theta0=theta0, retries=2)
    if not ok:
        # try a slightly larger K to get a good fixed point, then come back
        theta, ok, _ = U.solve_locked(omega_c, W0, Kc * 1.05, theta0=None, retries=3)
        if not ok:
            raise RuntimeError("Failed to lock a steady state near Kc.")

    theta -= theta[0]  # gauge fix for initial iterate

    # 3) Build initial eigenvector v from J(θ,K)
    J0 = U.kuramoto_jacobian(theta, W0, Kc)
    eigvals, eigvecs = la.eigh(J0)
    jmin = int(np.argmin(np.abs(eigvals)))
    v0 = eigvecs[:, jmin].copy()
    v0 /= la.norm(v0) + 1e-16

    # 4) Refine (θ, v, K) with augmented residual (least squares)
    x0 = np.concatenate([theta, v0, np.array([Kc])], axis=0)

    def resfun(x: np.ndarray) -> np.ndarray:
        th = x[:n]; vv = x[n:2*n]; KK = float(x[-1])
        return H_residual(th, vv, KK, a_value, W0, np.zeros_like(W0), omega_c)

    ls = least_squares(
        resfun, x0, method="lm",
        xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=800
    )
    x = ls.x
    theta_star = x[:n]
    v_star = x[n:2*n]; v_star /= la.norm(v_star) + 1e-16
    K_star = float(x[-1])

    res = resfun(x)
    return CriticalSolution(theta=theta_star, v=v_star, K=K_star,
                            a=a_value, res_norm=float(la.norm(res)),
                            Kc_from_continuation=float(Kc))


# ---------- sensitivity dK/da ----------
@dataclass
class SensitivityResult:
    dK_da: float
    details: Dict[str, Any]

def compute_dK_da_at_solution(sol: CriticalSolution,
                              W0: np.ndarray,
                              E: np.ndarray,
                              omega: np.ndarray) -> SensitivityResult:
    n = len(sol.theta)
    theta = sol.theta
    v = sol.v
    K = sol.K
    a = sol.a
    W = W0 + a * E
    omega_c = U._center_omega(omega)

    # Build H_x blocks
    A_F_th = dF_dtheta(theta, K, W)                 # n×n
    A_F_v  = np.zeros((n, n), dtype=float)          # n×n
    A_F_K  = dF_dK(theta, W).reshape(n, 1)          # n×1

    A_Jv_th = dJv_dtheta_FD(theta, v, W, K)         # n×n
    A_Jv_v  = U.kuramoto_jacobian(theta, W, K)      # n×n
    A_Jv_K  = dJv_dK_vec(theta, W, K, v).reshape(n, 1)

    A_norm_th = np.zeros((1, n), dtype=float)
    A_norm_v  = (2.0 * v).reshape(1, n)
    A_norm_K  = np.zeros((1, 1), dtype=float)

    e0 = np.zeros((1, n), dtype=float); e0[0, 0] = 1.0
    A_gauge_th = e0
    A_gauge_v  = np.zeros((1, n), dtype=float)
    A_gauge_K  = np.zeros((1, 1), dtype=float)

    Hx = np.block([
        [A_F_th,    A_F_v,    A_F_K],
        [A_Jv_th,   A_Jv_v,   A_Jv_K],
        [A_norm_th, A_norm_v, A_norm_K],
        [A_gauge_th,A_gauge_v,A_gauge_K],
    ])

    Ha_F  = dF_da(theta, K, E)
    Ha_Jv = dJv_da(theta, E, K, v)
    Ha = np.concatenate([Ha_F, Ha_Jv, np.array([0.0, 0.0])], axis=0)

    x_a, *_ = la.lstsq(Hx, -Ha)     # least-squares solve
    dK_da = float(x_a[-1])

    details = {
        "Hx_cond": float(np.linalg.cond(Hx)),
        "residual_norm": float(np.linalg.norm(Hx @ x_a + Ha)),
    }
    return SensitivityResult(dK_da=dK_da, details=details)


# ---------- public API ----------
@dataclass
class BraessPrediction:
    dK_da: float
    Kc: float
    theta: np.ndarray
    v: np.ndarray
    info: Dict[str, Any]

def predict_braess_add_edge(
    W0: np.ndarray,
    omega: np.ndarray,
    p: int,
    q: int,
    w_add: float = 1.0,
    r_thresh: float = 0.7,
    K_start: float = 5.0,
    K_min: float = 0.02,
    K_step: float = 0.05,
    K_guess: Optional[float] = None,   # if you want to inject CSV baseline_Kc, you can
) -> BraessPrediction:
    n = W0.shape[0]
    if not (0 <= p < n and 0 <= q < n):
        raise ValueError("p and q must be valid node indices.")
    E = edge_matrix(n, p, q, w_add=w_add)

    sol = solve_critical_point_continuation(
        W0=W0, omega=omega,
        K_start=K_start, K_min=K_min, K_step=K_step, r_thresh=r_thresh,
        a_value=0.0,
        K_guess_override=K_guess
    )

    sens = compute_dK_da_at_solution(sol, W0=W0, E=E, omega=omega)
    info = dict(
        augment_res_norm=sol.res_norm,
        Hx_cond=sens.details["Hx_cond"],
        lin_resid_norm=sens.details["residual_norm"],
        edge=(int(p), int(q)),
        w_add=float(w_add),
        Kc_from_continuation=float(sol.Kc_from_continuation),
    )
    return BraessPrediction(dK_da=sens.dK_da, Kc=sol.K, theta=sol.theta, v=sol.v, info=info)


# ---------- CLI demo (optional) ----------
if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Predict Braess via dK/da using continuation-backed Kc.")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--p_edge", type=float, default=0.3)
    ap.add_argument("--omega-std", type=float, default=1.0)
    ap.add_argument("--edge", type=str, default="0,1")
    ap.add_argument("--w-add", type=float, default=1.0)
    ap.add_argument("--K-start", type=float, default=5.0)
    ap.add_argument("--K-min", type=float, default=0.02)
    ap.add_argument("--K-step", type=float, default=0.05)
    ap.add_argument("--r-thresh", type=float, default=0.7)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    W0 = U.random_connected_graph(n=args.n, p=args.p_edge, rng=rng)
    omega = U.random_omega(args.n, std=args.omega_std, rng=rng)
    u_str, v_str = args.edge.split(","); u, v = int(u_str), int(v_str)

    pred = predict_braess_add_edge(
        W0=W0, omega=omega, p=u, q=v, w_add=args.w_add,
        K_start=args.K_start, K_min=args.K_min, K_step=args.K_step, r_thresh=args.r_thresh
    )

    print(json.dumps({
        "edge": pred.info["edge"],
        "w_add": pred.info["w_add"],
        "Kc_from_continuation": pred.info["Kc_from_continuation"],
        "Kc_refined": pred.Kc,
        "dK_da": pred.dK_da,
        "braess_predicted": bool(pred.dK_da > 0.0),
        "augment_res_norm": pred.info["augment_res_norm"],
        "Hx_condition_number": pred.info["Hx_cond"],
        "linear_system_residual": pred.info["lin_resid_norm"],
    }, indent=2))
