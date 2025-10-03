#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_braess.py — continuation-backed predictor (augmented system version)

We use the SAME Kc finder as your scan code:
  U.continuation_descend_K  (co-rotating frame)

Then we lock a steady state at K≈Kc with U.solve_locked (gauge θ0=0),
form the augmented system

    G(θ,v,K,a) = [F(θ,K,a); J(θ,K,a)v; vᵀv - 1; θ₀]

Differentiate wrt a:

    (∂G/∂x) x_a + ∂G/∂a = 0

Solve least-squares for x_a = [dθ/da, dv/da, dK/da],
then extract dK/da = x_a[-1].

All intermediate arrays are stored in `info` for traceability.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import numpy as np
import scipy.linalg as la
import utils as U


# ---------------------- JSON helper ----------------------
def _to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    return x


# ---------------------- dataclass ------------------------
@dataclass
class PredictResult:
    Kc: float
    dK_da: float
    info: Dict[str, Any]


# ---------------------- system builders ------------------
def F_theta(theta: np.ndarray, K: float, W: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Steady-state equation: F_i = ω_i + K * Σ_j W_ij sin(θ_j - θ_i)."""
    dth = theta[np.newaxis, :] - theta[:, np.newaxis]
    return omega + K * (W * np.sin(dth)).sum(axis=1)


def augmented_system(theta, v, K, a, W0, E, omega):
    """G(θ,v,K,a) = [F; Jv; v^T v - 1; θ0]."""
    n = len(theta)
    W = W0 + a * E
    F = F_theta(theta, K, W, omega)
    J = U.kuramoto_jacobian(theta, W, K)
    Jv = J @ v
    norm = np.array([v @ v - 1.0])
    gauge = np.array([theta[0]])
    return np.concatenate([F, Jv, norm, gauge])


def linearize_augmented(theta, v, K, a, W0, E, omega, eps=1e-6):
    """Finite-diff Jacobians: Hx=∂G/∂x, Ha=∂G/∂a."""
    n = len(theta)
    x = np.concatenate([theta, v, [K]])

    def G_of(x_vec, a_val):
        th = x_vec[:n]
        vv = x_vec[n:2*n]
        KK = x_vec[-1]
        return augmented_system(th, vv, KK, a_val, W0, E, omega)

    G0 = G_of(x, a)

    # ∂G/∂x by FD
    Hx = np.zeros((len(G0), len(x)))
    for j in range(len(x)):
        x_pert = x.copy()
        x_pert[j] += eps
        Hx[:, j] = (G_of(x_pert, a) - G0) / eps

    # ∂G/∂a
    Ha = (G_of(x, a + eps) - G0) / eps

    return Hx, Ha, G0


def solve_dK_da(theta, v, K, a, W0, E, omega):
    """Compute dK/da via augmented system linearization."""
    Hx, Ha, G0 = linearize_augmented(theta, v, K, a, W0, E, omega)
    x_a, *_ = la.lstsq(Hx, -Ha)
    return float(x_a[-1]), x_a, Hx, Ha, G0


# ---------------------- main predictor -------------------
def predict_braess_add_edge(
    W0: np.ndarray,
    omega: np.ndarray,
    p: int, q: int, w_add: float = 1.0,
    K_start: float = 5.0, K_min: float = 0.4, K_step: float = 0.01, r_thresh: float = 0.7,
    K_guess: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> PredictResult:
    """
    Predict dKc/da for adding one edge (p,q) with weight w_add,
    using the full augmented system method.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = W0.shape[0]
    assert 0 <= p < n and 0 <= q < n and p != q

    # 1) Find baseline Kc
    omega_c = U._center_omega(omega)
    Kc_cont, K_grid, R_grid = U.continuation_descend_K(
        omega_c, W0, K_start=K_start, K_min=K_min, K_step=K_step,
        r_thresh=r_thresh, rng=rng
    )
    if Kc_cont is None:
        Kc_cont = float(K_grid[np.argmax(R_grid)]) if len(K_grid) else float(K_start)
    Kc_used = float(K_guess) if (K_guess is not None and np.isfinite(K_guess)) else float(Kc_cont)

    # 2) Lock steady state
    theta_star, ok, _ = U.solve_locked(omega_c, W0, Kc_used, theta0=None, retries=3, rng=rng)
    if not ok:
        th0 = U._laplacian_initial_guess(omega_c, W0, Kc_used)
        theta_star, ok, _ = U.solve_locked(omega_c, W0, Kc_used, theta0=th0, retries=3, rng=rng)
    if not ok:
        raise RuntimeError("Failed to lock steady state.")

    # 3) Null eigenvector
    J = U.kuramoto_jacobian(theta_star, W0, Kc_used)
    evals, evecs = la.eigh(J)
    v = evecs[:, np.argmin(np.abs(evals))].copy()
    v /= (la.norm(v) + 1e-16)

    # 4) Build edge matrix E
    E = np.zeros_like(W0)
    E[p, q] = E[q, p] = w_add

    # 5) Augmented system solve
    dK_da, x_a, Hx, Ha, G0 = solve_dK_da(theta_star, v, Kc_used, 0.0, W0, E, omega_c)

    info = {
        "method": "augmented_system",
        "edge": (p, q),
        "w_add": w_add,
        "Kc_from_continuation": float(Kc_cont),
        "K_guess_used": float(K_guess) if K_guess is not None else np.nan,
        "theta_star": theta_star,
        "v_null": v,
        "eigvals_J": evals,
        "Hx": Hx,
        "Ha": Ha,
        "lin_solution": x_a,
        "augment_residual": G0,
        "Hx_cond_est": float(np.linalg.cond(Hx)),
    }

    return PredictResult(Kc=float(Kc_used), dK_da=float(dK_da), info=info)


# ---------------------- trace writer ---------------------
def write_predict_trace(pred_result: PredictResult, txt_path: str, json_path: Optional[str] = None):
    import numpy as np
    I = pred_result.info or {}
    with open(txt_path, "w") as f:
        f.write("=== Braess Predictor Trace (Augmented) ===\n")
        f.write(f"Kc (refined): {pred_result.Kc:.12f}\n")
        f.write(f"dK/da       : {pred_result.dK_da:+.6e}\n")
        f.write(f"Method      : {I.get('method','')}\n")
        f.write(f"Edge, w_add : {I.get('edge')}, {I.get('w_add')}\n")
        f.write(f"Kc_from_cont: {I.get('Kc_from_continuation','nan')}\n")

    if json_path is None:
        json_path = txt_path.replace(".txt", ".json")
    with open(json_path, "w") as jf:
        json.dump({
            "Kc": pred_result.Kc,
            "dK_da": pred_result.dK_da,
            "info": _to_jsonable(I),
        }, jf)


# ---------------------- tiny CLI demo --------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Predict dK/da via augmented system.")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--p_edge", type=float, default=0.3)
    ap.add_argument("--omega-std", type=float, default=1.0)
    ap.add_argument("--edge", type=str, default="0,1")
    ap.add_argument("--w-add", type=float, default=1.0)
    ap.add_argument("--K-start", type=float, default=5.0)
    ap.add_argument("--K-min", type=float, default=0.4)
    ap.add_argument("--K-step", type=float, default=0.01)
    ap.add_argument("--r-thresh", type=float, default=0.7)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    W0 = U.random_connected_graph(n=args.n, p=args.p_edge, rng=rng)
    omega = U.random_omega(args.n, std=args.omega_std, rng=rng)
    u, v = map(int, args.edge.split(","))

    pred = predict_braess_add_edge(
        W0=W0, omega=omega, p=u, q=v, w_add=args.w_add,
        K_start=args.K_start, K_min=args.K_min, K_step=args.K_step,
        r_thresh=args.r_thresh, rng=rng
    )

    write_predict_trace(pred, txt_path="predict_trace.txt")
    print(json.dumps({"edge": pred.info["edge"], "dK_da": pred.dK_da}, indent=2))
