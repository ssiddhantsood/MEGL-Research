
from __future__ import annotations
import argparse, os, sys, json, time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import networkx as nx
import scipy

# Expect utils.py (your provided module) to be in PYTHONPATH / same dir
import utils as U

CODE_VERSION = "2025-09-18_a+b_logging_v2"


def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def weighted_edge_list_from_W(W: np.ndarray) -> List[Tuple[int, int, float]]:
    """Return upper-triangular weighted edge list (u, v, w)."""
    n = W.shape[0]
    out: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            w = float(W[i, j])
            if w != 0.0:
                out.append((i, j, w))
    return out


def W_from_weighted_edges(n: int, edges: List[Tuple[int, int, float]]) -> np.ndarray:
    """Construct symmetric W from (u, v, w) list."""
    W = np.zeros((n, n), dtype=float)
    for (u, v, w) in edges:
        W[u, v] = float(w)
        W[v, u] = float(w)
    return W


def pick_graph(
    rng: np.random.Generator,
    child_seed: int,
    n_min: int = 10,
    n_max: int = 20,
) -> Tuple[str, Dict[str, Any], np.ndarray, Optional[Dict[int, Tuple[float, float]]]]:
    """
    Randomly choose ER or RGN and build a connected weighted adjacency matrix W.
    Uses deterministic RNG seeded by child_seed and also passes child_seed to NX for ER.
    Returns (graph_type, gen_params, W, pos_or_none).
    """
    n = int(rng.integers(n_min, n_max + 1))
    graph_type = rng.choice(["er", "rgn"])

    if graph_type == "er":
        # Connection prob chosen from rng; ER uses child_seed for NX determinism
        p = float(rng.uniform(0.25, 0.5))
        W = U.random_connected_graph(
            n=n, p=p, weight_low=1.0, weight_high=1.0, rng=rng
        )
        # NOTE: U.random_connected_graph already calls nx.erdos_renyi_graph
        # with a seed derived from rng; since rng is seeded with child_seed,
        # this is reproducible with the same child_seed + code.
        params = {"n": n, "p": p, "weight_low": 1.0, "weight_high": 1.0}
        return "er", params, W, None

    else:
        radius = float(rng.uniform(18.0, 30.0))
        box_size = 100.0
        W, pos = U.random_geometric_graph_2d(
            n=n, radius=radius, box_size=box_size,
            weight_low=1.0, weight_high=1.0, rng=rng, return_pos=True
        )
        params = {"n": n, "radius": radius, "box_size": box_size, "weight_low": 1.0, "weight_high": 1.0}
        return "rgn", params, W, pos


def compute_kc_and_scan_additions(
    rng: np.random.Generator,
    omega: np.ndarray,
    W: np.ndarray,
    K_start: float = 5.0,
    K_min: float = 0.4,
    K_step: float = 0.01,
    r_thresh: float = 0.7,
    add_weight: float = 1.0,
) -> Tuple[Optional[float], list[dict]]:
    """
    Returns (baseline_Kc, rows). Rows have:
      edge, baseline_Kc, new_Kc, delta, is_braess

    Implements: for each candidate missing edge, add exactly that one edge
    on a fresh copy of the baseline (no accumulation), compute Kc, discard.
    """
    omega_c = U._center_omega(omega)
    base_Kc, _, _ = U.continuation_descend_K(
        omega_c, W,
        K_start=K_start, K_min=K_min, K_step=K_step, r_thresh=r_thresh, rng=rng
    )

    rows = U.braess_scan_single_add_edges(   # <--- use the new helper
        omega, W, add_weight=add_weight,
        K_start=K_start, K_min=K_min, K_step=K_step,
        r_thresh=r_thresh, rng=rng, outdir=None
    )
    return base_Kc, rows



def append_rows_to_csv(csv_path: str, rows: list[dict]) -> None:
    """
    Append (or create) the CSV with the provided list of dicts.
    Ensures consistent column ordering and writes header if the file does not exist.
    """
    ensure_parent_dir(csv_path)
    df = pd.DataFrame(rows)
    preferred_cols = [
        "code_version",
        "numpy_version", "networkx_version", "scipy_version",
        "timestamp", "run_id",
        "seed_root", "seed_child",
        "graph_type", "n",
        "er_p", "rgn_radius", "rgn_box_size",
        "weight_low", "weight_high",
        "omega_mean", "omega_std",
        "K_start", "K_min", "K_step", "r_thresh", "add_weight",
        "baseline_Kc", "edge_u", "edge_v", "new_Kc", "delta", "is_braess",
        "omega_json", "edges_json",
        "notes_json",
    ]
    for c in preferred_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[preferred_cols + [c for c in df.columns if c not in preferred_cols]]
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def main():
    ap = argparse.ArgumentParser(description="Scan random graphs for Braess paradox instances and log to CSV (seeds + full inputs).")
    ap.add_argument("--csv", default="braess_runs.csv", help="Path to the aggregate CSV output.")
    ap.add_argument("--num-runs", type=int, default=100, help="Number of random graphs to generate (Ctrl-C to stop early).")
    ap.add_argument("--omega-std", type=float, default=1.0, help="Std dev for omega ~ N(0, std^2).")
    ap.add_argument("--r-thresh", type=float, default=0.7, help="Order parameter threshold for defining Kc.")
    ap.add_argument("--K-start", type=float, default=5.0, help="Start K for continuation (descend).")
    ap.add_argument("--K-min", type=float, default=0.4, help="Minimum K to stop.")
    ap.add_argument("--K-step", type=float, default=0.01, help="Step size in K for continuation.")
    ap.add_argument("--add-weight", type=float, default=1.0, help="Weight for added edges during scan.")
    args = ap.parse_args()

    # (A) Deterministic integer seeds: log + use
    seed_root = int(np.random.default_rng().integers(0, 2**63 - 1))
    root_rng = np.random.default_rng(seed_root)

    print(f"[info] code={CODE_VERSION}")
    print(f"[info] numpy={np.__version__} networkx={nx.__version__} scipy={scipy.__version__}")
    print(f"[info] root seed: {seed_root}")
    print(f"[info] writing CSV to: {args.csv}")
    print(f"[info] starting runs: {args.num_runs} (Ctrl-C to stop early)")

    try:
        for run_idx in range(args.num_runs):
            child_seed = int(root_rng.integers(0, 2**63 - 1))
            rng = np.random.default_rng(child_seed)

            # Pick graph
            graph_type, gparams, W, pos = pick_graph(rng, child_seed)
            n = W.shape[0]
            print(f"[new graph] run {run_idx:04d} | type={graph_type} | n={n} | seed_child={child_seed}")

            # Draw omega from this run's RNG
            omega = U.random_omega(n, mean=0.0, std=args.omega_std, rng=rng)

            # (B) Collect exact inputs for perfect replay
            edges_list = weighted_edge_list_from_W(W)           # (u, v, w)
            omega_list = [float(x) for x in omega.tolist()]
            print(f"[scan] run {run_idx:04d} | type={graph_type} | n={n} | missing_edges={len(U.missing_edges(W))} | scanning additions…")

            # Compute baseline Kc and scan additions
            base_Kc, rows = compute_kc_and_scan_additions(
                rng, omega, W,
                K_start=args.K_start, K_min=args.K_min, K_step=args.K_step,
                r_thresh=args.r_thresh, add_weight=args.add_weight
            )
            print(f"[scan] run {run_idx:04d} | done")

            # Library versions
            v_numpy = np.__version__
            v_nx = nx.__version__
            v_scipy = scipy.__version__

            # Prepare CSV rows (one per Braess-positive edge)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            run_id = f"{int(time.time())}_{run_idx}"
            out_rows = []
            for r in rows:
                if not bool(r.get("is_braess", False)):
                    continue
                (u, v) = r.get("edge", (None, None))
                meta = {
                    "code_version": CODE_VERSION,
                    "numpy_version": v_numpy, "networkx_version": v_nx, "scipy_version": v_scipy,
                    "timestamp": ts,
                    "run_id": run_id,
                    "seed_root": int(seed_root),
                    "seed_child": int(child_seed),
                    "graph_type": graph_type,
                    "n": int(n),
                    "er_p": float(gparams["p"]) if graph_type == "er" else np.nan,
                    "rgn_radius": float(gparams["radius"]) if graph_type == "rgn" else np.nan,
                    "rgn_box_size": float(gparams["box_size"]) if graph_type == "rgn" else np.nan,
                    "weight_low": float(gparams["weight_low"]),
                    "weight_high": float(gparams["weight_high"]),
                    "omega_mean": 0.0,
                    "omega_std": float(args.omega_std),
                    "K_start": float(args.K_start),
                    "K_min": float(args.K_min),
                    "K_step": float(args.K_step),
                    "r_thresh": float(args.r_thresh),
                    "add_weight": float(args.add_weight),
                    "baseline_Kc": None if r.get("baseline_Kc") is None else float(r["baseline_Kc"]),
                    "edge_u": None if u is None else int(u),
                    "edge_v": None if v is None else int(v),
                    "new_Kc": None if r.get("new_Kc") is None else float(r["new_Kc"]),
                    "delta": None if r.get("delta") is None else float(r["delta"]),
                    "is_braess": bool(r.get("is_braess", False)),
                    # Full inputs for exact replay:
                    "omega_json": json.dumps(omega_list),
                    "edges_json": json.dumps(edges_list),
                    # Extra notes
                    "notes_json": json.dumps({
                        "graph_pos_saved": bool(pos is not None),
                        "generator": graph_type,
                    }),
                }
                out_rows.append(meta)

            if out_rows:
                append_rows_to_csv(args.csv, out_rows)
                print(f"[run {run_idx:04d}] n={n} type={graph_type} braess_edges={len(out_rows)} base_Kc={base_Kc:.4f}" if base_Kc is not None else
                      f"[run {run_idx:04d}] n={n} type={graph_type} braess_edges={len(out_rows)} base_Kc=None")
            else:
                print(f"[run {run_idx:04d}] n={n} type={graph_type} no Braess edges found" + (f", base_Kc={base_Kc:.4f}" if base_Kc is not None else ", base_Kc=None"))

    except KeyboardInterrupt:
        print("\n[info] interrupted by user; exiting gracefully.")

if __name__ == "__main__":
    main()
