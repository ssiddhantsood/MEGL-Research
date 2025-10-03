#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_predict.py — Run the Braess predictor over multiple CSVs, save per-row results,
compute ONE global confusion matrix across all rows, and (optionally) export artifacts.

Usage (everything, with artifacts):
  python3 test_predict.py \
    --csv braess_runs.csv non_braess_runs.csv \
    --out results_predict.csv \
    --verbose \
    --make-artifacts --artdir predict_replays --top-per-run 0

Outputs:
  - results_predict.csv                    (per-row predictions + diagnostics)
  - results_predict_confusion.csv          (ONE global confusion matrix + metrics)
  - predict_replays/<run_id>_.../          (if --make-artifacts)
      baseline/ (plots)
      add-u-v/  (plots + info.txt + trace.txt)
      summary.txt (run settings + GLOBAL confusion metrics)
"""

from __future__ import annotations
import argparse, json, os, sys
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

import utils as U
import predict_braess as PB


# ---------- small helpers ----------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def W_from_edges(n: int, edges_json: str) -> np.ndarray:
    edges = json.loads(edges_json)
    W = np.zeros((n, n), dtype=float)
    for u, v, w in edges:
        u, v, w = int(u), int(v), float(w)
        W[u, v] = w
        W[v, u] = w
    return W

def plot_overlay_no_zoom(K_base, R_base, K_new, R_new, out_path,
                         Kc_base: Optional[float], Kc_new: Optional[float],
                         K_min: float, K_start: float,
                         r_thresh: float,
                         labels=("baseline", "modified")):
    ensure_dir(os.path.dirname(out_path) or ".")
    plt.figure(figsize=(8, 6))
    plt.plot(K_base, R_base, "o-", linewidth=2, markersize=4, label=labels[0])
    if Kc_base is not None:
        plt.axvline(Kc_base, linestyle="--", alpha=0.6, label=f"{labels[0]} Kc ≈ {Kc_base:.3f}")
    plt.plot(K_new, R_new, "s--", linewidth=2, markersize=4, label=labels[1])
    if Kc_new is not None:
        plt.axvline(Kc_new, linestyle=":", alpha=0.7, label=f"{labels[1]} Kc ≈ {Kc_new:.3f}")
    plt.axhline(r_thresh, linestyle="--", alpha=0.35, label=f"threshold {r_thresh}")
    plt.xlabel("Coupling strength K")
    plt.ylabel("Order parameter r")
    plt.title("Overlay: baseline vs modified r(K)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(float(K_min), float(K_start))   # fixed domain (no zooming)
    plt.ylim(0.0, 1.0)                       # r ∈ [0,1]
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def compute_global_confusion(rows_df: pd.DataFrame):
    """Compute ONE confusion matrix across all rows (both CSVs combined)."""
    mask = (
        rows_df["delta_obs"].notna()
        & rows_df["dK_da_pred"].notna()
        & np.isfinite(rows_df["delta_obs"])
        & np.isfinite(rows_df["dK_da_pred"])
    )
    df = rows_df.loc[mask, ["delta_obs", "dK_da_pred"]].copy()
    if df.empty:
        print("\n[GLOBAL] No valid rows for confusion matrix.")
        return None

    # Positive class = Braess on add => delta_obs > 0
    df["y_true"] = (df["delta_obs"] > 0).astype(int)
    df["y_pred"] = (df["dK_da_pred"] > 0).astype(int)

    TP = int(((df["y_true"] == 1) & (df["y_pred"] == 1)).sum())
    TN = int(((df["y_true"] == 0) & (df["y_pred"] == 0)).sum())
    FP = int(((df["y_true"] == 0) & (df["y_pred"] == 1)).sum())
    FN = int(((df["y_true"] == 1) & (df["y_pred"] == 0)).sum())

    total = TP + TN + FP + FN
    acc  = (TP + TN) / total if total else float("nan")
    prec = TP / (TP + FP) if (TP + FP) else float("nan")
    rec  = TP / (TP + FN) if (TP + FN) else float("nan")
    f1   = (2 * prec * rec) / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0 else float("nan")

    print(f"\n[GLOBAL] Confusion matrix (Braess on add = positive):")
    print(f"           Pred 0   Pred 1")
    print(f"True 0 ->   {TN:5d}    {FP:5d}")
    print(f"True 1 ->   {FN:5d}    {TP:5d}")
    print(f"\nValid rows: {total}")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1       : {f1:.3f}")

    return {
        "TN": TN, "FP": FP, "FN": FN, "TP": TP,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "valid_rows": total
    }

def regenerate_curve(omega: np.ndarray, W: np.ndarray,
                     K_start: float, K_min: float, K_step: float, r_thresh: float,
                     rng: Optional[np.random.Generator]=None) -> Tuple[Optional[float], np.ndarray, np.ndarray]:
    omega_c = U._center_omega(omega)
    return U.continuation_descend_K(
        omega_c, W, K_start=K_start, K_min=K_min, K_step=K_step, r_thresh=r_thresh, rng=rng
    )

def draw_network(W: np.ndarray,
                 omega: np.ndarray,
                 path: str,
                 title: str,
                 layout: str = "spring",
                 highlight_edge: Optional[Tuple[int,int]] = None):
    U.draw_network_graph(W, omega, path=path, layout=layout, title=title, highlight_edge=highlight_edge)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True,
                    help="One or more CSV files (e.g., braess_runs.csv non_braess_runs.csv).")
    ap.add_argument("--out", default="results_predict.csv", help="Per-row predictions CSV output.")
    ap.add_argument("--max-rows", type=int, default=0, help="Trim concatenated dataset to first N rows.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--make-artifacts", action="store_true", help="Create plots and text traces like replay.")
    ap.add_argument("--artdir", default="predict_replays", help="Artifact output root (with --make-artifacts).")
    ap.add_argument("--top-per-run", type=int, default=0, help="Per-run, export top |ΔKc| edges (0 = all).")
    args = ap.parse_args()

    # Load data
    frames = []
    for path in args.csv:
        try:
            df = pd.read_csv(path)
            if df.empty:
                print(f"[warn] Empty CSV: {path}", file=sys.stderr)
                continue
            df["__source_csv__"] = path
            frames.append(df)
        except Exception as e:
            print(f"[error] reading {path}: {e}", file=sys.stderr)
    if not frames:
        print("No input rows found.", file=sys.stderr); sys.exit(1)

    all_df = pd.concat(frames, ignore_index=True)
    if args.max_rows and args.max_rows > 0:
        all_df = all_df.head(args.max_rows)

    # Required columns
    required = ["n","edge_u","edge_v","omega_json","edges_json",
                "K_start","K_min","K_step","r_thresh","add_weight",
                "delta","baseline_Kc","new_Kc","run_id","graph_type"]
    for c in required:
        if c not in all_df.columns:
            print(f"Missing column: {c}", file=sys.stderr); sys.exit(2)

    out_rows = []
    # For artifact generation: keep rows grouped per run
    per_run_rows: Dict[str, List[Dict[str, Any]]] = {}

    for idx, row in all_df.iterrows():
        try:
            n = int(row["n"])
            u = int(row["edge_u"]); v = int(row["edge_v"])
            if np.isnan(u) or np.isnan(v):
                continue

            W0 = W_from_edges(n, row["edges_json"])
            omega = np.asarray(json.loads(row["omega_json"]), dtype=float)
            if omega.shape[0] != n:
                if args.verbose:
                    print(f"[warn] row {idx}: omega length {omega.shape[0]} != n={n}; skip")
                continue

            K_start = float(row["K_start"])
            K_min   = float(row["K_min"])
            K_step  = float(row["K_step"])
            r_thresh = float(row["r_thresh"])
            add_weight = float(row["add_weight"])
            baseline_Kc = float(row["baseline_Kc"]) if not pd.isna(row["baseline_Kc"]) else None
            delta = float(row["delta"]) if not pd.isna(row["delta"]) else np.nan

            # Predict (pass K_guess to keep it snappy)
            pred = PB.predict_braess_add_edge(
                W0=W0, omega=omega, p=u, q=v, w_add=add_weight,
                K_start=K_start, K_min=K_min, K_step=K_step, r_thresh=r_thresh,
                K_guess=baseline_Kc
            )

            dK_da = float(pred.dK_da)
            sign_pred = np.sign(dK_da)
            sign_obs = np.sign(delta) if not np.isnan(delta) else np.nan
            match = (not np.isnan(sign_obs)) and (sign_pred == sign_obs)

            if args.verbose:
                print(
                    f"[{idx}] run={row['run_id']} {row['graph_type']} n={n} "
                    f"edge=({u},{v}) ΔKc={delta:+.6f} dK/da={dK_da:+.3e} "
                    f"Kc(cont)={pred.info.get('Kc_from_continuation', np.nan):.6f} "
                    f"Kc(refined)={pred.Kc:.6f} match={match} src={row['__source_csv__']}"
                )

            rec = {
                "row_index": int(idx),
                "source_csv": row["__source_csv__"],
                "run_id": row["run_id"],
                "graph_type": row["graph_type"],
                "n": n,
                "edge_u": u,
                "edge_v": v,
                "delta_obs": delta,
                "baseline_Kc_csv": float(row["baseline_Kc"]) if not pd.isna(row["baseline_Kc"]) else np.nan,
                "new_Kc_csv": float(row["new_Kc"]) if not pd.isna(row["new_Kc"]) else np.nan,
                "dK_da_pred": dK_da,                     # <- raw dK/da saved
                "sign_match": bool(match),
                # diagnostics
                "Kc_from_continuation": pred.info.get("Kc_from_continuation", np.nan),
                "Kc_refined": pred.Kc,
                "augment_res_norm": pred.info.get("augment_res_norm", np.nan),
                "Hx_condition_number": pred.info.get("Hx_cond", pred.info.get("Hx_cond_est", np.nan)),
                "linear_system_residual": pred.info.get("lin_resid_norm", pred.info.get("residual_norm", np.nan)),
            }
            out_rows.append(rec)

            # Keep for artifact gen
            per_run_rows.setdefault(row["run_id"], []).append({
                "meta": row.to_dict(),
                "n": n, "u": u, "v": v, "delta": delta,
                "W0": W0, "omega": omega,
                "K_start": K_start, "K_min": K_min, "K_step": K_step,
                "r_thresh": r_thresh, "add_weight": add_weight,
                "pred": pred,
            })

        except Exception as e:
            out_rows.append({
                "row_index": int(idx),
                "source_csv": row.get("__source_csv__", ""),
                "run_id": row.get("run_id", ""),
                "graph_type": row.get("graph_type", ""),
                "n": int(row["n"]) if not pd.isna(row["n"]) else np.nan,
                "edge_u": row.get("edge_u", np.nan),
                "edge_v": row.get("edge_v", np.nan),
                "error": str(e),
            })
            if args.verbose:
                print(f"[error] row {idx}: {e}", file=sys.stderr)

    # Save per-row predictions
    res_df = pd.DataFrame(out_rows)
    res_df.to_csv(args.out, index=False)
    print(f"\nSaved per-row predictions → {args.out}")

    # ONE global confusion (over all rows combined)
    global_cm = compute_global_confusion(res_df)
    if global_cm is not None:
        pd.DataFrame([global_cm]).to_csv(args.out.replace(".csv", "_confusion.csv"), index=False)
        print(f"Saved global confusion metrics → {args.out.replace('.csv','_confusion.csv')}")

    # Artifact generation per run (baseline + edges) and summary (GLOBAL metrics only)
    if args.make_artifacts:
        ensure_dir(args.artdir)
        for run_id, rows in per_run_rows.items():
            try:
                row0 = rows[0]["meta"]
                n = int(row0["n"])
                graph_type = str(row0["graph_type"])
                layout = "spring"

                W0 = rows[0]["W0"]
                omega = rows[0]["omega"]
                K_start = float(rows[0]["K_start"]); K_min = float(rows[0]["K_min"])
                K_step  = float(rows[0]["K_step"]);  r_thresh = float(rows[0]["r_thresh"])
                add_weight = float(rows[0]["add_weight"])

                net_dirname = f"{run_id}_{graph_type}_n{n}"
                net_dir = os.path.join(args.artdir, net_dirname)
                ensure_dir(net_dir)
                base_dir = os.path.join(net_dir, "baseline")
                ensure_dir(base_dir)

                # Baseline regeneration + plots
                base_Kc, Kb, Rb = regenerate_curve(omega, W0, K_start, K_min, K_step, r_thresh)
                U.plot_r_vs_K(Kb, Rb, out_path=os.path.join(base_dir, "r_vs_K_baseline.png"),
                              Kc=base_Kc, title="Baseline r(K)", r_threshold=r_thresh)
                draw_network(W0, omega, path=os.path.join(base_dir, "network_baseline.png"),
                             title="Baseline network", layout=layout)

                # Select edges by |delta|
                rows_sorted = [r for r in rows if np.isfinite(r["delta"])]
                rows_sorted.sort(key=lambda r: abs(float(r["delta"])), reverse=True)
                if args.top_per_run > 0:
                    rows_sorted = rows_sorted[:args.top_per_run]

                for r in rows_sorted:
                    u, v = int(r["u"]), int(r["v"])
                    delta = float(r["delta"])
                    pred = r["pred"]

                    # Modified W (single added edge)
                    Wm = W0.copy()
                    Wm[u, v] = add_weight; Wm[v, u] = add_weight

                    # Regenerate modified curve
                    new_Kc, Km, Rm = regenerate_curve(omega, Wm, K_start, K_min, K_step, r_thresh)

                    edge_dir = os.path.join(net_dir, f"add-{u}-{v}")
                    ensure_dir(edge_dir)

                    draw_network(Wm, omega, path=os.path.join(edge_dir, "network_add.png"),
                                 title=f"Network after add {(u,v)}", layout=layout, highlight_edge=(u, v))
                    U.plot_r_vs_K(Km, Rm, out_path=os.path.join(edge_dir, "r_vs_K_add.png"),
                                  Kc=new_Kc, title=f"add {(u,v)}: r(K)", r_threshold=r_thresh)
                    plot_overlay_no_zoom(
                        K_base=Kb, R_base=Rb, K_new=Km, R_new=Rm,
                        out_path=os.path.join(edge_dir, "r_vs_K_overlay_add.png"),
                        Kc_base=base_Kc, Kc_new=new_Kc,
                        K_min=K_min, K_start=K_start, r_thresh=r_thresh,
                        labels=("baseline", f"add {(u,v)}")
                    )

                    # Predictor trace
                    trace_path = os.path.join(edge_dir, "trace.txt")
                    try:
                        PB.write_predict_trace(pred, trace_path)  # helper in predict_braess.py
                    except Exception as e:
                        with open(trace_path, "w") as f:
                            f.write(f"[trace unavailable] {e}\n")

                    # Edge info
                    with open(os.path.join(edge_dir, "info.txt"), "w") as f:
                        f.write(f"edge: ({u},{v})\n")
                        f.write(f"delta_from_csv: {delta}\n")
                        f.write(f"baseline_Kc_csv: {row0.get('baseline_Kc', None)}\n")
                        f.write(f"new_Kc_csv: {row0.get('new_Kc', None)}\n")
                        f.write(f"baseline_Kc_recomputed: {base_Kc}\n")
                        f.write(f"new_Kc_recomputed: {new_Kc}\n")
                        f.write(f"dK/da_pred: {float(pred.dK_da)}\n")
                        f.write(f"Kc_from_continuation (pred): {pred.info.get('Kc_from_continuation', np.nan)}\n")
                        f.write(f"Kc_refined (pred): {pred.Kc}\n")
                        f.write(f"K grid: K_start={K_start}, K_min={K_min}, K_step={K_step}, r_thresh={r_thresh}, add_weight={add_weight}\n")

                # Per-run summary with ONLY GLOBAL confusion
                with open(os.path.join(net_dir, "summary.txt"), "w") as f:
                    f.write("=== Run Summary ===\n")
                    for k, v in {
                        "run_id": run_id, "graph_type": graph_type, "n": n,
                        "K_start": K_start, "K_min": K_min, "K_step": K_step,
                        "r_thresh": r_thresh, "add_weight": add_weight
                    }.items():
                        f.write(f"{k}: {v}\n")
                    f.write("\n--- Global confusion (all runs combined) ---\n")
                    if global_cm is None:
                        f.write("No valid rows for global confusion.\n")
                    else:
                        for k, v in global_cm.items():
                            f.write(f"{k}: {v}\n")

            except Exception as e:
                print(f"[warn] artifacts for run_id={run_id} failed: {e}")

if __name__ == "__main__":
    main()
