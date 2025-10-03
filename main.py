from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

from utils import *

# ======= CONFIG ==========

CONFIG = dict(
    # Output
    OUTDIR_ROOT = "runs_simple",  # base folder; subfolder created per run

    # Randomness
    TRIALS = 1,
    SEED = None,                  # e.g., 1234; or None for nondeterministic

    # Choose generator: "erdos" or "rgn"
    GRAPH_TYPE = "rgn",

    # --- ER (random_connected_graph) params ---
    N = 12,
    ER_P = 0.30,
    W_LOW = 1.0,
    W_HIGH = 1.0,

    # --- RGN (random_geometric_graph_2d) params ---
    RGN_N = 30,                   # used if GRAPH_TYPE="rgn"
    RGN_RADIUS = 20.0,            # clamped to [1, 99] in utils
    RGN_BOX_SIZE = 100.0,
    RGN_W_LOW = 1.0,
    RGN_W_HIGH = 1.0,

    # Omega (intrinsic frequencies)
    OMEGA_MEAN = 0.0,
    OMEGA_STD = 1.0,

    # Continuation
    K_START = 5.0,
    K_MIN   = 0.10,
    K_STEP  = 0.01,
    R_THRESH = 0.7,

    # Edge scans
    EDGE_MODE = "remove",         # "remove", "add", or "both"
    REMOVE_MODE = "random",       # "random" or "betweenness"
    ADD_WEIGHT = 1.0,

    # Stability (choose fast top-k or full spectrum)
    FAST_EIGS = True,             # True = use sparse top-k (fast); False = full eigs
    EIG_TOPK  = 5,                # only used if FAST_EIGS=True
)

# ====== RUNNER ===========
def _slug(s: str) -> str:
    import re
    return re.sub(r'[^a-zA-Z0-9_.-]+', '-', s).strip('-').lower()

def run_once(cfg: dict, seed=None, outdir=None, label="run"):
    rng = np.random.default_rng(seed)

    # 1) Build graph + omega
    if cfg["GRAPH_TYPE"] == "rgn":
        W, pos = random_geometric_graph_2d(
            n=cfg["RGN_N"], radius=cfg["RGN_RADIUS"], box_size=cfg["RGN_BOX_SIZE"],
            weight_low=cfg["RGN_W_LOW"], weight_high=cfg["RGN_W_HIGH"],
            rng=rng, return_pos=True
        )
        n = W.shape[0]
    else:  # "erdos"
        W = random_connected_graph(
            cfg["N"], p=cfg["ER_P"],
            weight_low=cfg["W_LOW"], weight_high=cfg["W_HIGH"],
            rng=rng
        )
        n = W.shape[0]
        pos = None  # not used unless you want a geometric plot

    omega = random_omega(n, mean=cfg["OMEGA_MEAN"], std=cfg["OMEGA_STD"], rng=rng)

    # 2) Baseline continuation
    Kc, K_grid, r = continuation_descend_K(
        omega, W,
        K_start=cfg["K_START"], K_min=cfg["K_MIN"], K_step=cfg["K_STEP"],
        r_thresh=cfg["R_THRESH"], rng=rng
    )

    # 3) Stability sweep (fast or full)
    if K_grid is not None and K_grid.size > 0:
        if cfg["FAST_EIGS"]:
            stab = sweep_linear_stability_topk(
                omega, W, K_values=K_grid, top_k=cfg["EIG_TOPK"], rng=rng,
                include_topk=False, return_thetas=False
            )
        else:
            stab = sweep_linear_stability(
                omega, W, K_values=K_grid, rng=rng, include_eigs=True
            )
        bifs = detect_bifurcations_from_stability(stab["K"], stab["max_real"], tol=1e-6)
    else:
        stab = {"K": np.array([]), "max_real": np.array([])}
        bifs = []

    # 4) Save artifacts
    if outdir:
        os.makedirs(outdir, exist_ok=True)

        meta = {
            "label": label,
            "generator": cfg["GRAPH_TYPE"],
            "n": n,
            "omega_mean": cfg["OMEGA_MEAN"], "omega_std": cfg["OMEGA_STD"],
            "K_start": cfg["K_START"], "K_min": cfg["K_MIN"],
            "K_step": cfg["K_STEP"], "r_thresh": cfg["R_THRESH"],
            "edge_mode": cfg["EDGE_MODE"], "remove_mode": cfg["REMOVE_MODE"],
            "add_weight": cfg["ADD_WEIGHT"],
            "seed": seed,
            "baseline_Kc": None if Kc is None else float(Kc),
            "fast_eigs": bool(cfg["FAST_EIGS"]), "eig_topk": int(cfg["EIG_TOPK"]),
        }
        with open(os.path.join(outdir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        if K_grid is not None and K_grid.size > 0:
            pd.DataFrame({"K": K_grid, "r": r}).to_csv(os.path.join(outdir, "r_vs_K.csv"), index=False)
            plot_r_vs_K(
                K_grid, r,
                out_path=os.path.join(outdir, "r_vs_K_baseline.png"),
                Kc=Kc, title=f"Baseline r(K)  n={n}",
                r_threshold=cfg["R_THRESH"]
            )
            plot_linear_stability(
                stab["K"], stab["max_real"],
                out_path=os.path.join(outdir, "linear_stability.png"),
                bifurcations=bifs,
                title="Linear stability: max Re(λ(J)) vs K (baseline)"
            )
            # Save CSV for stability (num_unstable not computed in fast path)
            num_unstable = np.full_like(stab["max_real"], np.nan, dtype=float)
            save_stability_csv(stab["K"], stab["max_real"], num_unstable, out_path=os.path.join(outdir, "linear_stability.csv"))
            # Save full eigenvalues only if we computed them
            if not cfg["FAST_EIGS"]:
                save_eigs_long_csv(stab, out_path=os.path.join(outdir, "eigs_long.csv"))

        # network snapshot
        if cfg["GRAPH_TYPE"] == "rgn":
            draw_geometric_network_graph(
                W, omega, pos,
                path=os.path.join(outdir, "network_baseline.png"),
                title="Baseline geometric network"
            )
        else:
            draw_network_graph(
                W, omega,
                path=os.path.join(outdir, "network_baseline.png"),
                layout="spring",
                title="Baseline ER network"
            )

    # 5) Optional Braess scans (same APIs as before)
    rows_remove, rows_add = [], []
    if cfg["EDGE_MODE"] in ("remove", "both"):
        rows_remove = braess_scan_remove_edges(
            omega, W,
            K_start=cfg["K_START"], K_min=cfg["K_MIN"], K_step=cfg["K_STEP"], r_thresh=cfg["R_THRESH"],
            mode=cfg["REMOVE_MODE"], rng=rng, outdir=outdir, layout="spring"
        )
        if outdir and rows_remove:
            pd.DataFrame([{
                'edge': str(row['edge']),
                'baseline_Kc': row['baseline_Kc'],
                'new_Kc': row['new_Kc'],
                'delta': row['delta'],
                'is_braess': row['is_braess'],
            } for row in rows_remove]).to_csv(os.path.join(outdir, "edge_removals.csv"), index=False)

    if cfg["EDGE_MODE"] in ("add", "both"):
        rows_add = braess_scan_add_edges(
            omega, W,
            add_weight=cfg["ADD_WEIGHT"],
            K_start=cfg["K_START"], K_min=cfg["K_MIN"], K_step=cfg["K_STEP"], r_thresh=cfg["R_THRESH"],
            rng=rng, outdir=outdir, layout="spring"
        )
        if outdir and rows_add:
            pd.DataFrame([{
                'edge': str(row['edge']),
                'baseline_Kc': row['baseline_Kc'],
                'new_Kc': row['new_Kc'],
                'delta': row['delta'],
                'is_braess': row['is_braess'],
            } for row in rows_add]).to_csv(os.path.join(outdir, "edge_additions.csv"), index=False)

    return {
        "W": W, "omega": omega,
        "Kc": Kc, "K_grid": K_grid, "r": r,
        "rows_remove": rows_remove, "rows_add": rows_add
    }

def main():
    cfg = CONFIG
    base_out = create_output_folder(cfg["OUTDIR_ROOT"])
    trials = int(cfg["TRIALS"])
    seed0 = cfg["SEED"]

    for t in range(trials):
        seed = (seed0 + t) if seed0 is not None else None
        trial_out = os.path.join(base_out, f"trial_{t:03d}")
        os.makedirs(trial_out, exist_ok=True)
        res = run_once(cfg, seed=seed, outdir=trial_out, label=f"default_trial_{t}")
        print(f"[trial {t}] baseline Kc = {res['Kc']}")

    print(f"\nAll done. Results in: {base_out}")

if __name__ == "__main__":
    main()
