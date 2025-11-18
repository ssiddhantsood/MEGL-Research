#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate derivative-based Braess predictor g on a networks CSV.

Reads a CSV produced by the Braess scanner with columns like:

  is_braess,label,baseline_Kc,new_Kc,delta,braess_tol,
  edge_u,edge_v,n,graph_type,timestamp,code_version,
  seed_root,seed_child,run_id,graph_params_json,
  K_start,K_min,K_step,onset_eps,onset_minrun,add_weight,
  omega_json,edges_json,pos_json

For each run_id:
  - Reconstruct baseline graph W and omega from JSON.
  - Compute baseline r(K) and first-onset Kc (using K_start/K_min/K_step/etc).
  - For each edge row in that run, compute:

      deriv_g  = finite-difference estimate of how r(K) near Kc
                 changes when we slightly increase weight on (u,v).

      pred_is_braess_g = (deriv_g > 0)

Outputs a new CSV with two new columns at the front:

  pred_is_braess_g, deriv_g, <all original columns...>

So you can compare pred_is_braess_g vs is_braess to see if g is a good predictor.
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import utils as U  # same utils the scanner uses


# --------------------------
# Helpers
# --------------------------

def parse_json_field(s: Any):
    """Robust JSON parser for fields like omega_json, edges_json, pos_json."""
    if s is None or s == "" or (isinstance(s, float) and np.isnan(s)):
        return None
    return json.loads(s)


def W_from_edges(n: int, edges: List[Tuple[int, int, float]]) -> np.ndarray:
    """Rebuild adjacency matrix from (u, v, w) list."""
    W = np.zeros((n, n), float)
    for u, v, w in edges:
        u_i, v_i = int(u), int(v)
        w_f = float(w)
        W[u_i, v_i] = w_f
        W[v_i, u_i] = w_f
    return W


def kc_first_onset(
    Ks: np.ndarray,
    Rs: np.ndarray,
    eps: float = 0.05,
    min_run: int = 3
) -> Optional[float]:
    """Same first-onset rule as the generator."""
    n = len(Rs)
    for i in range(n):
        if Rs[i] > eps:
            j_end = min(n, i + min_run)
            if j_end - i < min_run:
                return None
            if np.all(np.diff(Rs[i:j_end]) >= -1e-10):
                return float(Ks[i])
    return None


def continuation_curve(
    omega: np.ndarray,
    W: np.ndarray,
    K_start: float,
    K_min: float,
    K_step: float,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper around continuation_descend_K with centered omega."""
    omega_c = U._center_omega(omega)
    _Kc_unused, K, R = U.continuation_descend_K(
        omega_c,
        W,
        K_start=K_start,
        K_min=K_min,
        K_step=K_step,
        r_thresh=0.999,
        rng=rng,
    )
    return np.asarray(K, float), np.asarray(R, float)


def compute_deriv_g_for_edge(
    omega: np.ndarray,
    W_base: np.ndarray,
    u: int,
    v: int,
    *,
    add_weight: float,
    K_start: float,
    K_min: float,
    K_step: float,
    onset_eps: float,
    onset_minrun: int,
    eps_frac: float,
    Kb: np.ndarray,
    Rb: np.ndarray,
    Kc_base: Optional[float],
) -> Tuple[Optional[float], bool]:
    """
    Compute derivative-based predictor g for a single edge (u,v).

    Inputs:
      - omega, W_base: baseline system (without added edge).
      - add_weight: base add weight (we perturb by eps_frac * add_weight).
      - K_start, K_min, K_step, onset_eps, onset_minrun: K grid + onset rule.
      - eps_frac: fraction of add_weight used as small perturbation.
      - Kb, Rb, Kc_base: precomputed baseline curve and Kc.

    Returns:
      (deriv_g, pred_is_braess_g):

        deriv_g: float or None
        pred_is_braess_g: bool, True if deriv_g > 0, else False
    """
    if Kc_base is None:
        # No well-defined baseline Kc; we just say "no prediction".
        return None, False

    # Small weight perturbation
    eps_w = max(1e-6, float(eps_frac) * float(add_weight))

    # Build W_plus = base + eps_w on edge (u,v)
    Wp = W_base.copy()
    Wp[u, v] += eps_w
    Wp[v, u] += eps_w

    # Compute r(K) for perturbed system on same K grid
    rng_plus = np.random.default_rng(123)
    Kp, Rp = continuation_curve(
        omega,
        Wp,
        K_start=K_start,
        K_min=K_min,
        K_step=K_step,
        rng=rng_plus,
    )

    # Be robust to small length mismatches
    m = min(len(Kb), len(Kp))
    if m == 0:
        return None, False
    Kb_use = Kb[:m]
    Rb_use = Rb[:m]
    Rp_use = Rp[:m]

    # Find indices near Kc_base
    idx = int(np.argmin(np.abs(Kb_use - Kc_base)))
    j1 = idx
    j2 = min(m, idx + 5)
    if j2 <= j1:
        j2 = min(m, j1 + 1)

    if j2 <= j1:
        return None, False

    dR = (Rp_use[j1:j2] - Rb_use[j1:j2]) / eps_w
    deriv_g = float(np.mean(dR)) if dR.size > 0 else 0.0

    pred_is_braess_g = bool(deriv_g > 0.0)
    return deriv_g, pred_is_braess_g


# --------------------------
# Main evaluation logic
# --------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate derivative-based Braess predictor g on networks CSV."
    )
    ap.add_argument(
        "--in-csv",
        default="networks.csv",
        help="Input CSV (full network dataset). Default: networks.csv",
    )
    ap.add_argument(
        "--out-csv",
        default="networks_with_g.csv",
        help="Output CSV with g predictions. Default: networks_with_g.csv",
    )
    ap.add_argument(
        "--eps-frac",
        type=float,
        default=0.1,
        help="Fraction of add_weight used for derivative perturbation (default: 0.1).",
    )
    args = ap.parse_args()

    print(f"[g-eval] reading dataset from: {os.path.abspath(args.in_csv)}")
    df = pd.read_csv(args.in_csv)

    if "is_braess" in df.columns:
        df["is_braess"] = df["is_braess"].astype(bool)

    out_rows: List[Dict[str, Any]] = []
    total = 0
    total_braess_gt = 0
    total_pred_true = 0
    total_pred_correct = 0

    # Group by run_id so we can reuse baseline computations
    for run_id, group in df.groupby("run_id"):
        group = group.copy()
        row0 = group.iloc[0]

        n = int(row0["n"])
        edges = parse_json_field(row0["edges_json"]) or []
        omega_list = parse_json_field(row0["omega_json"])
        if omega_list is None:
            # If omega missing, skip this run
            print(f"[g-eval] WARNING: missing omega_json for run_id={run_id}, skipping.")
            continue
        omega = np.array(omega_list, float)

        K_start = float(row0["K_start"])
        K_min = float(row0["K_min"])
        K_step = float(row0["K_step"])
        onset_eps = float(row0["onset_eps"])
        onset_minrun = int(row0["onset_minrun"])
        add_weight = float(row0["add_weight"])

        # Rebuild baseline adjacency
        W_base = W_from_edges(n, edges)

        # Baseline continuation and Kc
        rng_base = np.random.default_rng(0)
        Kb, Rb = continuation_curve(
            omega,
            W_base,
            K_start=K_start,
            K_min=K_min,
            K_step=K_step,
            rng=rng_base,
        )
        Kc_base = kc_first_onset(Kb, Rb, eps=onset_eps, min_run=onset_minrun)

        print(
            f"[g-eval] run_id={run_id}: n={n}, edges={len(edges)}, "
            f"Kc_base={Kc_base}"
        )

        # Loop over each edge row in this run
        for _, row in group.iterrows():
            total += 1
            if bool(row.get("is_braess", False)):
                total_braess_gt += 1

            u = int(row["edge_u"])
            v = int(row["edge_v"])

            deriv_g, pred_is_braess_g = compute_deriv_g_for_edge(
                omega=omega,
                W_base=W_base,
                u=u,
                v=v,
                add_weight=add_weight,
                K_start=K_start,
                K_min=K_min,
                K_step=K_step,
                onset_eps=onset_eps,
                onset_minrun=onset_minrun,
                eps_frac=args.eps_frac,
                Kb=Kb,
                Rb=Rb,
                Kc_base=Kc_base,
            )

            if pred_is_braess_g:
                total_pred_true += 1

            # Compare to ground truth if available
            is_braess_gt = bool(row.get("is_braess", False))
            if (deriv_g is not None) and (pred_is_braess_g == is_braess_gt):
                total_pred_correct += 1

            # Build output row: new fields + original fields
            base_dict = row.to_dict()
            base_dict["pred_is_braess_g"] = bool(pred_is_braess_g)
            base_dict["deriv_g"] = deriv_g
            out_rows.append(base_dict)

    # Build output DataFrame with new columns at the front
    df_out = pd.DataFrame(out_rows)

    # New columns at the front, originals following (in original order)
    new_front = ["pred_is_braess_g", "deriv_g"]
    original_order = list(df.columns)
    ordered_cols = new_front + [c for c in original_order if c in df_out.columns]
    df_out = df_out[ordered_cols]

    # Save
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True) if os.path.dirname(args.out_csv) else None
    df_out.to_csv(args.out_csv, index=False)
    print(f"[g-eval] wrote g predictions to: {os.path.abspath(args.out_csv)}")

    # Simple summary
    if total > 0:
        acc = total_pred_correct / total
    else:
        acc = 0.0
    print("=== g predictor summary ===")
    print(f"total_edges: {total}")
    print(f"groundtruth_braess: {total_braess_gt}")
    print(f"predicted_braess_g: {total_pred_true}")
    print(f"correct_predictions_g: {total_pred_correct}")
    print(f"accuracy_g: {acc:.4f}")


if __name__ == "__main__":
    main()
