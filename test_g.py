#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_g.py

Sanity-check g.G and g.predict_braess on the Braess CSV data.

This script:
  - Loads a CSV of single-edge additions (base network + new edge).
  - Reconstructs the base adjacency matrix W from edges_json.
  - Reconstructs omega from omega_json.
  - For each row:
      * Solves for a phase-locked state theta_0 at K = baseline_Kc on the base graph.
      * Builds the Jacobian J(theta_0, K0) for the Kuramoto map F(θ) = ω + K s(θ).
      * Extracts a critical eigenvector v_0 orthogonal to the all-ones vector.
      * Calls g.predict_braess(theta_0, v_0, K0, edge_to_add, W, omega).
      * Compares sign(k'(0)) with the ground-truth is_braess flag.

Usage:
  python test_g.py --csv braess_runs.csv --max-rows 200 --kprime-tol 1e-4
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import root

import g  # your file containing G() and predict_braess()


def build_W_from_row(row):
    """
    Build the base adjacency matrix W (without the candidate new edge)
    from a CSV row that contains:
        - 'n'
        - 'edges_json' : [[u, v, w], ...] for the ORIGINAL graph only
    """
    n = int(row["n"])
    edges = json.loads(row["edges_json"])

    W = np.zeros((n, n), float)
    for u, v, w in edges:
        u = int(u)
        v = int(v)
        w = float(w)
        W[u, v] = w
        W[v, u] = w

    return W


def solve_theta_locked(W, omega, K, max_iter=50, tol=1e-10):
    """
    Solve for a phase-locked state theta on the base graph:

        F_i(θ) = ω_i + K Σ_j W_ij sin(θ_j - θ_i) = 0

    We fix θ_0 = 0 to remove the gauge freedom and solve for θ_1..θ_{n-1}.

    Returns:
      theta (n,) on success, or None on failure.
    """
    n = len(omega)

    def residual(x):
        theta = np.zeros(n, float)
        theta[1:] = x
        dth = theta[np.newaxis, :] - theta[:, np.newaxis]
        s_theta = (W * np.sin(dth)).sum(axis=1)
        F = omega + K * s_theta
        # Drop equation for i = 0 (gauge)
        return F[1:]

    x0 = np.zeros(n - 1, float)
    sol = root(residual, x0, tol=tol, options={"maxiter": max_iter})
    if not sol.success:
        return None
    theta = np.zeros(n, float)
    theta[1:] = sol.x
    return theta


def jacobian_kuramoto(W, theta, K):
    """
    Jacobian of F(θ) = ω + K Σ_j W_ij sin(θ_j - θ_i) with respect to θ.

    J_ij = K W_ij cos(θ_j - θ_i) for i != j
    J_ii = -K Σ_j W_ij cos(θ_j - θ_i)
    """
    n = len(theta)
    dth = theta[np.newaxis, :] - theta[:, np.newaxis]
    C = W * np.cos(dth)

    J = np.zeros((n, n), float)
    for i in range(n):
        for j in range(n):
            if i == j:
                J[i, i] = -K * C[i, :].sum()
            else:
                J[i, j] = K * C[i, j]
    return J


def extract_critical_eigenvector(J, tol_align=1e-3):
    """
    Extract a candidate critical eigenvector v such that:

      - v is nearly in the kernel of J (eigenvalue with smallest |λ|),
      - v is approximately orthogonal to the all-ones vector.

    We use eigen-decomposition (symmetric J) and:
      1) sort eigenvalues by |λ|,
      2) pick the first eigenvector not too aligned with 1.

    Returns:
      v (n,) normalized, with v^T 1 = 0 enforced, or None on failure.
    """
    vals, vecs = np.linalg.eigh(J)  # vecs[:, k] eigenvector for vals[k]
    n = J.shape[0]
    ones = np.ones(n) / np.sqrt(n)

    # sort by |λ|
    idx = np.argsort(np.abs(vals))
    for k in idx:
        v = vecs[:, k]
        alignment = abs(np.dot(v, ones))
        if alignment < tol_align:
            # Enforce exact orthogonality and normalization
            v = v - np.dot(v, ones) * ones
            norm = np.linalg.norm(v)
            if norm < 1e-12:
                continue
            v = v / norm
            return v

    # fallback: take the smallest eigenvalue vector and project
    k0 = idx[0]
    v = vecs[:, k0]
    v = v - np.dot(v, ones) * ones
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return None
    v = v / norm
    return v


def to_bool(x):
    """Robust conversion of CSV is_braess field to bool."""
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    raise ValueError(f"Cannot parse is_braess value: {x!r}")


def main():
    ap = argparse.ArgumentParser(description="Test g.predict_braess against Braess CSV data.")
    ap.add_argument("--csv", required=True, help="Path to Braess runs CSV.")
    ap.add_argument("--max-rows", type=int, default=1000, help="Maximum rows to test.")
    ap.add_argument("--kprime-tol", type=float, default=1e-4,
                    help="Minimum |k'(0)| to treat a prediction as decisive.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    total_rows = 0
    solved_theta = 0
    got_v = 0
    used_for_eval = 0
    correct = 0
    incorrect = 0
    skipped_small_kprime = 0

    for idx, row in df.iterrows():
        if total_rows >= args.max_rows:
            break
        total_rows += 1

        try:
            n = int(row["n"])
            omega_list = json.loads(row["omega_json"])
            assert len(omega_list) == n, f"omega length {len(omega_list)} != n {n} at row {idx}"
            omega = np.array(omega_list, float)

            W = build_W_from_row(row)
            assert W.shape == (n, n), f"W shape {W.shape} != (n, n) at row {idx}"

            edge_u = int(row["edge_u"])
            edge_v = int(row["edge_v"])
            edge_to_add = (edge_u, edge_v)

            # The candidate edge should NOT be in the base graph
            if W[edge_u, edge_v] != 0.0 or W[edge_v, edge_u] != 0.0:
                print(f"[WARN] Row {idx}: edge ({edge_u},{edge_v}) already in base graph; skipping.")
                continue

            K0 = float(row["baseline_Kc"])
            delta = float(row["delta"])
            is_braess_true = to_bool(row["is_braess"])

            # Optional sanity: delta>0 ↔ is_braess
            if (delta > 0) != is_braess_true:
                print(f"[WARN] Row {idx}: delta / is_braess mismatch: delta={delta}, is_braess={is_braess_true}")

            # Solve for theta_0 at K0
            theta0 = solve_theta_locked(W, omega, K0)
            if theta0 is None:
                print(f"[skip] Row {idx}: could not solve theta at K0={K0}")
                continue
            solved_theta += 1

            # Build Jacobian and get critical eigenvector v0
            J = jacobian_kuramoto(W, theta0, K0)
            v0 = extract_critical_eigenvector(J)
            if v0 is None:
                print(f"[skip] Row {idx}: could not extract critical eigenvector")
                continue
            got_v += 1

            # Use predictor
            k_prime = g.predict_braess(theta0, v0, K0, edge_to_add, W, omega)

            if abs(k_prime) < args.kprime_tol:
                skipped_small_kprime += 1
                continue

            used_for_eval += 1
            is_braess_pred = (k_prime > 0)

            if is_braess_pred == is_braess_true:
                correct += 1
            else:
                incorrect += 1

        except Exception as e:
            print(f"[error] Row {idx}: {e}")
            continue

    print("\n===== test_g summary =====")
    print(f"Rows seen:               {total_rows}")
    print(f"Solved theta:            {solved_theta}")
    print(f"Got eigenvector v:       {got_v}")
    print(f"Used for evaluation:     {used_for_eval}")
    print(f"  Correct predictions:   {correct}")
    print(f"  Incorrect predictions: {incorrect}")
    print(f"Skipped |k'(0)| small:   {skipped_small_kprime}")

    if used_for_eval > 0:
        acc = correct / used_for_eval
        print(f"\nPredictor accuracy (on decisive cases): {acc:.4f}")
    else:
        print("\nNo decisive k'(0) cases to evaluate.")


if __name__ == "__main__":
    main()
