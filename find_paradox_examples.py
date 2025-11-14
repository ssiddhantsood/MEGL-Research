
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Strict Braess scanner (single-edge additions only).

For each run:
  1) Sample a connected base graph W and natural frequencies omega.
  2) Compute baseline Kc using FIRST-ONSET of incoherence loss on r(K) while descending K.
  3) Loop over *missing* edges. For each edge (u,v):
       - Make a COPY of W, add only that one edge with weight=add_weight.
       - Recompute Kc with the *same* K grid and onset rule.
       - If new_Kc > base_Kc + braess_tol, record a CSV row (Braess-on-add).
  4) Append all Braess-positive rows to the CSV.

Important: We never accumulate added edges. Each candidate is tested in isolation.
This guarantees "add 1, remove it, add another" semantics without mutating the baseline.
"""

from __future__ import annotations
import argparse, os, json, time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd

import utils as U  # do not modify utils.py

CODE_VERSION = "2025-10-10_single_edge_only_v2"

def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def weighted_edge_list_from_W(W: np.ndarray) -> List[Tuple[int,int,float]]:
    n = W.shape[0]
    out = []
    for i in range(n):
        for j in range(i+1, n):
            w = float(W[i, j])
            if w != 0.0:
                out.append((i, j, w))
    return out

def kc_first_onset(Ks: np.ndarray, Rs: np.ndarray, eps: float = 0.05, min_run: int = 3) -> Optional[float]:
    """Return first K where r(K) crosses eps and stays nondecreasing for `min_run` points."""
    n = len(Rs)
    for i in range(n):
        if Rs[i] > eps:
            j_end = min(n, i + min_run)
            if j_end - i < min_run:
                return None
            if np.all(np.diff(Rs[i:j_end]) >= -1e-10):
                return float(Ks[i])
    return None

def continuation_curve(omega: np.ndarray, W: np.ndarray,
                       K_start: float, K_min: float, K_step: float,
                       rng: Optional[np.random.Generator]=None) -> Tuple[np.ndarray, np.ndarray]:
    omega_c = U._center_omega(omega)
    _Kc_unused, K, R = U.continuation_descend_K(
        omega_c, W, K_start=K_start, K_min=K_min, K_step=K_step, r_thresh=0.999, rng=rng
    )
    return np.asarray(K, float), np.asarray(R, float)

def compute_kc_and_scan_additions_strict(
    rng: np.random.Generator,
    omega: np.ndarray,
    W: np.ndarray,
    *,
    K_start: float,
    K_min: float,
    K_step: float,
    onset_eps: float,
    onset_minrun: int,
    braess_tol: float,
    add_weight: float,
) -> Tuple[Optional[float], List[dict]]:
    # Baseline
    Kb, Rb = continuation_curve(omega, W, K_start, K_min, K_step, rng=rng)
    base_Kc = kc_first_onset(Kb, Rb, eps=onset_eps, min_run=onset_minrun)

    rows: List[dict] = []
    missing = U.missing_edges(W)  # list of (u,v) u<v

    for (u, v) in missing:
        # single-edge test in isolation
        Wm = W.copy()
        Wm[u, v] = add_weight
        Wm[v, u] = add_weight

        Km, Rm = continuation_curve(omega, Wm, K_start, K_min, K_step, rng=rng)
        new_Kc = kc_first_onset(Km, Rm, eps=onset_eps, min_run=onset_minrun)

        # Strict Braess decision
        if base_Kc is None or new_Kc is None:
            continue
        delta = float(new_Kc - base_Kc)
        if delta <= float(braess_tol):
            continue

        rows.append({
            "edge_u": int(u), "edge_v": int(v),
            "baseline_Kc": float(base_Kc),
            "new_Kc": float(new_Kc),
            "delta": float(delta),
            "is_braess": True,
        })
    return base_Kc, rows

def append_rows_to_csv(csv_path: str, rows: List[dict]) -> None:
    if not rows:
        return
    ensure_parent_dir(csv_path)
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    df.to_csv(csv_path, mode="a", header=write_header, index=False)

def pick_graph(rng: np.random.Generator, child_seed: int, n_min: int, n_max: int):
    n = int(rng.integers(n_min, n_max + 1))
    kind = rng.choice(["er", "rgn"])
    if kind == "er":
        p = float(rng.uniform(0.25, 0.5))
        W = U.random_connected_graph(n=n, p=p, weight_low=1.0, weight_high=1.0, rng=rng)
        pos = None
        gparams = {"n": n, "p": p, "weight_low": 1.0, "weight_high": 1.0}
    else:
        radius = float(rng.uniform(18.0, 30.0))
        box = 100.0
        W, pos = U.random_geometric_graph_2d(n=n, radius=radius, box_size=box,
                                             weight_low=1.0, weight_high=1.0, rng=rng, return_pos=True)
        gparams = {"n": n, "radius": radius, "box_size": box, "weight_low": 1.0, "weight_high": 1.0}
    return kind, gparams, W, pos

def main():
    ap = argparse.ArgumentParser(description="Find Braess examples by single-edge additions only.")
    ap.add_argument("--csv", default="braess_runs.csv")
    ap.add_argument("--num-runs", type=int, default=50)
    ap.add_argument("--omega-std", type=float, default=1.0)
    ap.add_argument("--K-start", type=float, default=5.0)
    ap.add_argument("--K-min", type=float, default=0.4)
    ap.add_argument("--K-step", type=float, default=0.01)
    ap.add_argument("--onset-eps", type=float, default=0.05)
    ap.add_argument("--onset-minrun", type=int, default=3)
    ap.add_argument("--braess-tol", type=float, default=1e-3)
    ap.add_argument("--add-weight", type=float, default=1.0)
    ap.add_argument("--n-min", type=int, default=10)
    ap.add_argument("--n-max", type=int, default=20)
    args = ap.parse_args()

    seed_root = int(np.random.default_rng().integers(0, 2**63 - 1))
    root_rng = np.random.default_rng(seed_root)
    print(f"[info] code={CODE_VERSION}")
    print(f"[info] root seed: {seed_root}")
    print(f"[info] writing CSV to: {args.csv}")
    print(f"[info] starting runs: {args.num_runs}")

    all_rows: List[dict] = []
    try:
        for run_idx in range(args.num_runs):
            seed_child = int(root_rng.integers(0, 2**63 - 1))
            rng = np.random.default_rng(seed_child)
            graph_type, gparams, W, pos = pick_graph(rng, seed_child, args.n_min, args.n_max)
            n = W.shape[0]

            # omega
            omega = U.random_omega(n=n, mean=0.0, std=args.omega_std, rng=rng)

            # compute Braess edges (single-edge only)
            base_Kc, rows_core = compute_kc_and_scan_additions_strict(
                rng, omega, W,
                K_start=args.K_start, K_min=args.K_min, K_step=args.K_step,
                onset_eps=args.onset_eps, onset_minrun=args.onset_minrun,
                braess_tol=args.braess_tol, add_weight=args.add_weight,
            )

            # decorate and collect rows
            if rows_core:
                meta = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "code_version": CODE_VERSION,
                    "seed_root": seed_root,
                    "seed_child": seed_child,
                    "run_id": f"run_{seed_child}",
                    "graph_type": graph_type,
                    "graph_params_json": json.dumps(gparams),
                    "n": n,
                    "K_start": float(args.K_start),
                    "K_min": float(args.K_min),
                    "K_step": float(args.K_step),
                    "onset_eps": float(args.onset_eps),
                    "onset_minrun": int(args.onset_minrun),
                    "add_weight": float(args.add_weight),
                    "omega_json": json.dumps(list(map(float, omega.tolist()))),
                    "edges_json": json.dumps(weighted_edge_list_from_W(W)),
                    "pos_json": json.dumps({int(k): [float(v[0]), float(v[1])] for k, v in (pos or {}).items()}) if pos is not None else "",
                }
                for r in rows_core:
                    r.update(meta)
                all_rows.extend(rows_core)

            if rows_core:
                print(f"[run {run_idx:04d}] STRICT Braess edges: {len(rows_core)} | base_Kc={base_Kc}")
            else:
                print(f"[run {run_idx:04d}] no strict Braess edges | base_Kc={base_Kc}")

        # write once
        append_rows_to_csv(args.csv, all_rows)
        print(f"[done] appended {len(all_rows)} strict Braess rows to {args.csv}")

    except KeyboardInterrupt:
        print("\n[info] interrupted by user; exiting gracefully.")

if __name__ == "__main__":
    main()
