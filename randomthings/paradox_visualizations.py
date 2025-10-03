from __future__ import annotations
import os, json, argparse, csv
from typing import List, Tuple
import numpy as np

import utils as U  # make sure utils.py is in the same folder


def W_from_weighted_edges(n: int, edges: List[Tuple[int, int, float]]) -> np.ndarray:
    """Rebuild symmetric adjacency matrix from weighted edge list."""
    W = np.zeros((n, n), dtype=float)
    for (u, v, w) in edges:
        W[u, v] = float(w)
        W[v, u] = float(w)
    return W


def ensure_dir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p)


def recompute_curves_and_plot(
    omega: np.ndarray,
    W_base: np.ndarray,
    edge: Tuple[int, int],
    out_png_overlay: str,
    K_start: float = 5.0,
    K_min: float = 0.4,
    K_step: float = 0.01,
    r_thresh: float = 0.7,
):
    """Recompute baseline and modified r(K) curves and save an overlay plot."""
    omega_c = U._center_omega(omega)

    # Baseline curve
    base_Kc, base_Ks, base_Rs = U.continuation_descend_K(
        omega_c, W_base,
        K_start=K_start, K_min=K_min, K_step=K_step,
        r_thresh=r_thresh, rng=np.random.default_rng()
    )

    # Modified curve (add the edge with weight=1.0, only if absent)
    i, j = edge
    W_mod = W_base.copy()
    if i != j and W_mod[i, j] == 0.0:
        W_mod[i, j] = 1.0
        W_mod[j, i] = 1.0

    mod_Kc, mod_Ks, mod_Rs = U.continuation_descend_K(
        omega_c, W_mod,
        K_start=K_start, K_min=K_min, K_step=K_step,
        r_thresh=r_thresh, rng=np.random.default_rng()
    )

    # Overlay plot
    U.plot_r_vs_K_overlay(
        base_Ks, base_Rs,
        mod_Ks,  mod_Rs,
        out_path=out_png_overlay,
        Kc_base=base_Kc, Kc_new=mod_Kc,
        title=f"Overlay: baseline vs add {edge}",
        r_threshold=r_thresh,
        labels=("baseline", f"add {edge}")
    )


def coerce_int(s: str) -> int:
    """Handle ints serialized as '11' or '11.0' in CSV."""
    try:
        return int(s)
    except ValueError:
        return int(float(s))


def main():
    ap = argparse.ArgumentParser(description="Visualize Braess paradox edges from CSV (streaming, no pandas).")
    ap.add_argument("--csv", required=True, help="Path to CSV created by find_paradox_examples.py")
    ap.add_argument("--outdir", default="paradox_visualizations", help="Directory for plots")
    ap.add_argument("--limit", type=int, default=0, help="Max rows to process (0 = all)")
    ap.add_argument("--start", type=int, default=0, help="Row index to start from (0-based, in file order)")
    ap.add_argument("--K-start", type=float, default=5.0)
    ap.add_argument("--K-min", type=float, default=0.4)
    ap.add_argument("--K-step", type=float, default=0.01)
    ap.add_argument("--r-thresh", type=float, default=0.7)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    print(f"[info] streaming CSV: {args.csv}")

    processed = 0
    raw_idx = -1  # counts rows as they appear in the CSV (excluding header)

    with open(args.csv, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            raw_idx += 1

            # honor start offset
            if raw_idx < args.start:
                continue

            # stop when hitting limit
            if args.limit and processed >= args.limit:
                break

            # if the CSV happens to include 'is_braess', skip non-true rows
            if "is_braess" in row:
                val = str(row["is_braess"]).strip().lower()
                if val not in ("true", "1", "yes"):
                    continue

            try:
                # Parse required fields
                n = coerce_int(row["n"])
                u = coerce_int(row["edge_u"])
                v = coerce_int(row["edge_v"])

                # Rebuild inputs
                omega_list = json.loads(row["omega_json"])
                edges_list = json.loads(row["edges_json"])
                omega = np.asarray(omega_list, dtype=float)
                edges_typed = [(coerce_int(a), coerce_int(b), float(w)) for a, b, w in edges_list]
                W_base = W_from_weighted_edges(n, edges_typed)

                # Subfolder named by the row index in the CSV
                subdir = os.path.join(args.outdir, str(raw_idx))
                ensure_dir(subdir)
                out_png = os.path.join(subdir, f"edge_{u}_{v}_overlay.png")

                # Compute and save overlay
                recompute_curves_and_plot(
                    omega, W_base, (u, v), out_png,
                    K_start=args.K_start, K_min=args.K_min,
                    K_step=args.K_step, r_thresh=args.r_thresh
                )

                processed += 1
                print(f"[ok] row {raw_idx}: edge=({u},{v}) → {out_png}")

            except KeyboardInterrupt:
                print("\n[info] interrupted by user; exiting.")
                return
            except Exception as e:
                # keep going on bad rows
                print(f"[warn] row {raw_idx}: skipped due to error: {e}")

    if args.limit:
        print(f"[done] processed {processed} rows (start={args.start}, limit={args.limit}).")
    else:
        print(f"[done] processed {processed} rows (start={args.start}).")


if __name__ == "__main__":
    main()
