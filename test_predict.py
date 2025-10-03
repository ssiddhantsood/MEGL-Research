#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_predict_from_csv.py — run predictor against braess_runs.csv using continuation Kc

For each CSV row:
  - rebuild W and omega,
  - predict dK/da for (edge_u, edge_v),
  - compare sign with observed delta,
  - write a results CSV.

Run:
  python test_predict_from_csv.py --csv braess_runs.csv --max-rows 100 --out results_predict.csv --verbose
"""

from __future__ import annotations
import argparse, json, sys
import numpy as np
import pandas as pd

import utils as U
import predict_braess as PB


def W_from_edges(n: int, edges_json: str) -> np.ndarray:
    edges = json.loads(edges_json)
    W = np.zeros((n, n), dtype=float)
    for u, v, w in edges:
        u, v, w = int(u), int(v), float(w)
        W[u, v] = w
        W[v, u] = w
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results_predict.csv")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        print("No rows found.", file=sys.stderr); sys.exit(1)

    required = ["n","edge_u","edge_v","omega_json","edges_json",
                "K_start","K_min","K_step","r_thresh","add_weight",
                "delta","baseline_Kc","new_Kc","run_id","graph_type"]
    for c in required:
        if c not in df.columns:
            print(f"Missing column: {c}", file=sys.stderr); sys.exit(2)

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows)

    out_rows = []
    ok = 0; total = 0

    for idx, row in df.iterrows():
        try:
            n = int(row["n"])
            u = int(row["edge_u"]); v = int(row["edge_v"])
            if np.isnan(u) or np.isnan(v): continue

            W = W_from_edges(n, row["edges_json"])
            omega = np.asarray(json.loads(row["omega_json"]), dtype=float)
            if omega.shape[0] != n:
                if args.verbose: print(f"[warn] row {idx}: omega length != n; skip")
                continue

            K_start = float(row["K_start"])
            K_min   = float(row["K_min"])
            K_step  = float(row["K_step"])
            r_thresh = float(row["r_thresh"])
            add_weight = float(row["add_weight"])
            baseline_Kc = float(row["baseline_Kc"]) if not pd.isna(row["baseline_Kc"]) else None
            delta = float(row["delta"]) if not pd.isna(row["delta"]) else np.nan

            pred = PB.predict_braess_add_edge(
                W0=W, omega=omega, p=u, q=v, w_add=add_weight,
                K_start=K_start, K_min=K_min, K_step=K_step, r_thresh=r_thresh,
                K_guess=baseline_Kc  # harmless; makes it even faster
            )
            dK_da = float(pred.dK_da)
            sign_pred = np.sign(dK_da)
            sign_obs = np.sign(delta) if not np.isnan(delta) else np.nan
            match = (not np.isnan(sign_obs)) and (sign_pred == sign_obs)

            total += 1; ok += int(bool(match))

            if args.verbose:
                print(
                    f"[{idx}] run={row['run_id']} {row['graph_type']} n={n} "
                    f"edge=({u},{v}) ΔKc={delta:+.6f} dK/da={dK_da:+.3e} "
                    f"Kc(cont)={pred.info['Kc_from_continuation']:.6f} "
                    f"Kc(refined)={pred.Kc:.6f} match={match}"
                )

            out_rows.append({
                "row_index": int(idx),
                "run_id": row["run_id"],
                "graph_type": row["graph_type"],
                "n": n,
                "edge_u": u,
                "edge_v": v,
                "delta_obs": delta,
                "baseline_Kc_csv": float(row["baseline_Kc"]) if not pd.isna(row["baseline_Kc"]) else np.nan,
                "new_Kc_csv": float(row["new_Kc"]) if not pd.isna(row["new_Kc"]) else np.nan,
                "dK_da_pred": dK_da,
                "sign_match": bool(match),
                "Kc_from_continuation": pred.info["Kc_from_continuation"],
                "Kc_refined": pred.Kc,
                "augment_res_norm": pred.info.get("augment_res_norm", np.nan),
                "Hx_condition_number": pred.info.get("Hx_cond", np.nan),
                "linear_system_residual": pred.info.get("lin_resid_norm", np.nan),
            })

        except Exception as e:
            out_rows.append({
                "row_index": int(idx),
                "run_id": row.get("run_id", ""),
                "graph_type": row.get("graph_type", ""),
                "n": int(row["n"]) if not pd.isna(row["n"]) else np.nan,
                "edge_u": row.get("edge_u", np.nan),
                "edge_v": row.get("edge_v", np.nan),
                "error": str(e),
            })
            if args.verbose:
                print(f"[error] row {idx}: {e}", file=sys.stderr)

    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    if total > 0:
        print(f"\nSummary: {ok}/{total} sign matches  ({100.0*ok/max(total,1):.1f}%)")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
