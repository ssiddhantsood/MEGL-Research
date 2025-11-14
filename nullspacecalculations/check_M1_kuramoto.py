#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check M * 1 = 0 for Kuramoto-style Jacobian on symmetric (undirected) graphs.

Definitions:
  F_i(θ) = ω_i + K * sum_j A_ij * sin(θ_j - θ_i)
  J(θ) = ∂F/∂θ  (n×n Jacobian)
  M = D_θ ( J(θ) v )   (an n×n matrix acting on a direction in θ)
We verify that for undirected graphs (A symmetric, zero diagonal):
  (M * 1) = 0  for any v and θ.

We do two checks:
  1) Analytic:   S := sum_l ∂J/∂θ_l  (n×n).  Show S ≈ 0  ⇒ (M*1) = S v ≈ 0
  2) Numerical:  (J(θ+ε*1) v - J(θ) v)/ε ≈ 0
"""

import numpy as np

def kuramoto_J(theta, A, K=1.0):
    """Kuramoto Jacobian J(θ) for F_i = ω_i + K Σ_j A_ij sin(θ_j - θ_i)."""
    n = len(theta)
    J = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            J[i, j] = K * A[i, j] * np.cos(theta[j] - theta[i])
        # Row-sum zero (comes from derivative of phase-difference form)
        J[i, i] = -np.sum(J[i, :])
    return J

def dJ_dtheta_l(theta, A, l, K=1.0):
    """Exact partial derivative ∂J/∂θ_l as an n×n matrix."""
    n = len(theta)
    dJ = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # J_ij = K A_ij cos(θ_j - θ_i)
            # ∂J_ij/∂θ_l = -K A_ij sin(θ_j - θ_i) * (δ_{jl} - δ_{il})
            dJ[i, j] = -K * A[i, j] * np.sin(theta[j] - theta[i]) * (
                (1.0 if j == l else 0.0) - (1.0 if i == l else 0.0)
            )
    # Diagonals from row-sum zero: J_ii = -Σ_{j≠i} J_ij  ⇒  ∂J_ii = -Σ_{j≠i} ∂J_ij
    for i in range(n):
        dJ[i, i] = -np.sum(dJ[i, :]) + dJ[i, i]
    return dJ

def M_times_w_analytic(theta, A, v, w, K=1.0):
    """Compute (M * w) analytically: (sum_l w_l * ∂J/∂θ_l) v."""
    n = len(theta)
    acc = np.zeros((n, n), dtype=float)
    for l in range(n):
        acc += w[l] * dJ_dtheta_l(theta, A, l, K)
    return acc @ v

def M_times_w_fdiff(theta, A, v, w, eps=1e-7, K=1.0):
    """Finite-difference (M * w) ≈ (J(θ+ε w) v - J(θ) v) / ε."""
    return (kuramoto_J(theta + eps * w, A, K) @ v - kuramoto_J(theta, A, K) @ v) / eps

def random_symmetric_A(n, rng, low=0.1, high=1.0):
    """Random symmetric adjacency with zero diagonal."""
    M = rng.uniform(low, high, size=(n, n))
    A = 0.5 * (M + M.T)
    np.fill_diagonal(A, 0.0)
    return A

def check_case(n, seed):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, size=n)
    A = random_symmetric_A(n, rng)
    v = rng.normal(size=n)
    v /= np.linalg.norm(v) + 1e-15
    one = np.ones(n)

    # Analytic S = sum_l ∂J/∂θ_l
    S = sum(dJ_dtheta_l(theta, A, l) for l in range(n))
    S_fro = np.linalg.norm(S, ord='fro')

    # (M * 1): analytic and finite-difference
    M1_analytic = M_times_w_analytic(theta, A, v, one)
    M1_fdiff    = M_times_w_fdiff(theta, A, v, one)

    return {
        "n": n,
        "seed": seed,
        "S_fro_norm": float(S_fro),
        "M1_norm_analytic": float(np.linalg.norm(M1_analytic)),
        "M1_norm_fdiff": float(np.linalg.norm(M1_fdiff)),
    }

def main():
    print("Checking M * 1 = 0 for symmetric Kuramoto J(θ).")
    sizes = [3, 4, 5, 6]  # 3x3 and slightly beyond
    results = []
    for n in sizes:
        for seed in range(5):
            res = check_case(n, seed)
            results.append(res)
            print(res)

    # Simple pass/fail summary with loose tolerance (floating-point)
    tol = 1e-7
    ok = all(r["S_fro_norm"] < tol and
             r["M1_norm_analytic"] < tol and
             r["M1_norm_fdiff"] < tol
             for r in results)
    print("\nRESULT:", "PASS" if ok else "CHECK NUMBERS ABOVE (some > tol)")

if __name__ == "__main__":
    main()
