# main.py
# Runner for Kuramoto + Braess experiments with plotting.
from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd

# per your request, import everything from utils
from utils import *

def _slug(s: str) -> str:
    import re
    return re.sub(r'[^a-zA-Z0-9_.-]+', '-', s).strip('-').lower()

def run_once(args, seed=None, outdir=None, label="run"):
    rng = np.random.default_rng(seed)

    # 1) Build random connected graph + intrinsic frequencies
    W = random_connected_graph(
        args.n, p=args.p,
        weight_low=args.w_low, weight_high=args.w_high,
        rng=rng
    )
    omega = random_omega(args.n, mean=args.omega_mean, std=args.omega_std, rng=rng)

    # 2) Baseline continuation (descending K)
    Kc, K_grid, r = continuation_descend_K(
        omega, W,
        K_start=args.K_start, K_min=args.K_min, K_step=args.K_step,
        r_thresh=args.r_thresh, rng=rng
    )

    # ---- Linearization around fixed points along K_grid ----
    if K_grid is not None and K_grid.size > 0:
        # Get stability and eigenvalues (no Jacobians stored)
        stab = sweep_linear_stability(
            omega, W, K_values=K_grid, rng=rng,
            include_eigs=True
        )

        # Bifurcations (zero-crossings of max real part)
        bifs = detect_bifurcations_from_stability(stab["K"], stab["max_real"], tol=1e-6)

        # Plot + CSV summaries (write to outdir, not trial_out)
        if outdir:
            plot_linear_stability(
                stab["K"], stab["max_real"],
                out_path=os.path.join(outdir, "linear_stability.png"),
                bifurcations=bifs,
                title="Linear stability: max Re(λ(J)) vs K (baseline)"
            )
            save_stability_csv(
                stab["K"], stab["max_real"], stab["num_unstable"],
                out_path=os.path.join(outdir, "linear_stability.csv")
            )
            # Optional: store full eigenvalues (one long CSV)
            save_eigs_long_csv(stab, out_path=os.path.join(outdir, "eigs_long.csv"))

    # 3) Persist outputs (meta, curves, plots)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

        meta = {
            "label": label,
            "n": args.n, "p": args.p,
            "w_low": args.w_low, "w_high": args.w_high,
            "omega_mean": args.omega_mean, "omega_std": args.omega_std,
            "K_start": args.K_start, "K_min": args.K_min,
            "K_step": args.K_step, "r_thresh": args.r_thresh,
            "edge_mode": args.edge_mode, "remove_mode": args.remove_mode,
            "add_weight": args.add_weight,
            "seed": seed,
            "baseline_Kc": None if Kc is None else float(Kc),
        }
        with open(os.path.join(outdir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        if K_grid is not None and K_grid.size > 0:
            pd.DataFrame({"K": K_grid, "r": r}).to_csv(
                os.path.join(outdir, "r_vs_K.csv"), index=False
            )
            plot_r_vs_K(
                K_grid, r,
                out_path=os.path.join(outdir, "r_vs_K_baseline.png"),
                Kc=Kc,
                title=f"Baseline r(K)  n={args.n}, p={args.p}, omega_std={args.omega_std}",
                r_threshold=args.r_thresh
            )

        # network snapshot
        draw_network_graph(
            W, omega,
            path=os.path.join(outdir, "network_baseline.png"),
            layout="spring",
            title="Baseline network"
        )

    # 4) Edge scans (each saves per-edge r(K) plots internally if outdir is given)
    rows_remove, rows_add = [], []
    if args.edge_mode in ("remove", "both"):
        rows_remove = braess_scan_remove_edges(
            omega, W,
            K_start=args.K_start, K_min=args.K_min, K_step=args.K_step, r_thresh=args.r_thresh,
            mode=args.remove_mode, rng=rng, outdir=outdir, layout="spring"
        )
        if outdir and rows_remove:
            pd.DataFrame([{
                'edge': str(row['edge']),
                'baseline_Kc': row['baseline_Kc'],
                'new_Kc': row['new_Kc'],
                'delta': row['delta'],
                'is_braess': row['is_braess'],
            } for row in rows_remove]).to_csv(
                os.path.join(outdir, "edge_removals.csv"), index=False
            )

    if args.edge_mode in ("add", "both"):
        rows_add = braess_scan_add_edges(
            omega, W,
            add_weight=args.add_weight,
            K_start=args.K_start, K_min=args.K_min, K_step=args.K_step, r_thresh=args.r_thresh,
            rng=rng, outdir=outdir, layout="spring"
        )
        if outdir and rows_add:
            pd.DataFrame([{
                'edge': str(row['edge']),
                'baseline_Kc': row['baseline_Kc'],
                'new_Kc': row['new_Kc'],
                'delta': row['delta'],
                'is_braess': row['is_braess'],
            } for row in rows_add]).to_csv(
                os.path.join(outdir, "edge_additions.csv"), index=False
            )

    return {
        "W": W, "omega": omega,
        "Kc": Kc, "K_grid": K_grid, "r": r,
        "rows_remove": rows_remove, "rows_add": rows_add
    }


def load_runs_config(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "runs" in data:
        return data["runs"]
    if isinstance(data, list):
        return data
    raise ValueError("Config must be a list of runs or an object with key 'runs'.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to JSON with runs")
    ap.add_argument("--outdir", type=str, default=None, help="Base output directory")

    # Single-run defaults (also used as fallbacks in config runs)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--p", type=float, default=0.3)
    ap.add_argument("--w_low", type=float, default=1.0)
    ap.add_argument("--w_high", type=float, default=1.0)
    ap.add_argument("--omega_mean", type=float, default=0.0)
    ap.add_argument("--omega_std", type=float, default=1.0)

    ap.add_argument("--K_start", type=float, default=15.0)
    ap.add_argument("--K_min", type=float, default=0.05)
    ap.add_argument("--K_step", type=float, default=0.01)
    ap.add_argument("--r_thresh", type=float, default=0.7)

    ap.add_argument("--remove_mode", choices=["random", "betweenness"], default="random")
    ap.add_argument("--edge_mode", choices=["remove", "add", "both"], default="remove")
    ap.add_argument("--add_weight", type=float, default=1.0)

    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)

    args = ap.parse_args()

    base_out = args.outdir or create_output_folder()
    os.makedirs(base_out, exist_ok=True)

    if args.config:
        runs = load_runs_config(args.config)
        print(f"Loaded {len(runs)} run(s) from {args.config}")

        for idx, run in enumerate(runs):
            # Merge defaults with per-run overrides
            class R: pass
            R.__dict__ = args.__dict__.copy()
            for k, v in run.items():
                if k in R.__dict__:
                    R.__dict__[k] = v

            label = run.get("name", f"run_{idx:03d}")
            run_out = os.path.join(base_out, _slug(label))
            os.makedirs(run_out, exist_ok=True)

            trials = int(run.get("trials", R.trials))
            seed0 = run.get("seed", R.seed)

            for t in range(trials):
                seed = (seed0 + t) if seed0 is not None else None
                trial_out = os.path.join(run_out, f"trial_{t:03d}")
                os.makedirs(trial_out, exist_ok=True)
                res = run_once(R, seed=seed, outdir=trial_out, label=f"{label}_trial_{t}")
                print(f"[{label}][trial {t}] baseline Kc = {res['Kc']}")
    else:
        # Single-run mode (may include multiple trials)
        for t in range(args.trials):
            seed = (args.seed + t) if args.seed is not None else None
            trial_out = os.path.join(base_out, f"trial_{t:03d}")
            os.makedirs(trial_out, exist_ok=True)
            res = run_once(args, seed=seed, outdir=trial_out, label=f"default_trial_{t}")
            print(f"[single-run][trial {t}] baseline Kc = {res['Kc']}")

    print(f"\nAll done. Results in: {base_out}")

if __name__ == "__main__":
    main()
