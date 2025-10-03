#!/usr/bin/env python3
"""
Replay Braess CSV: reconstruct exact instances and export plots/artifacts.

Usage:
  python replay_braess_from_csv.py --csv braess_runs.csv --outdir replays --top 10

Requires:
  - utils.py available as "import utils as U" (your provided module)
  - pandas, numpy, networkx, matplotlib

What it does per run_id (network):
  replays/<run_id>_<graph>_n<n>/
    summary.txt
    baseline/
      network_baseline.png
      r_vs_K_baseline.png
    add-<u>-<v>/ or remove-<u>-<v>/
      info.txt
      network_<change>.png
      r_vs_K_<change>.png
      r_vs_K_overlay_<change>.png
"""

from __future__ import annotations
import argparse, os, json
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils as U  # your module


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def W_from_edges(n: int, edges: List[List[float]]) -> np.ndarray:
    """edges is [[u,v,w], ...] possibly from json; return symmetric W."""
    W = np.zeros((n, n), dtype=float)
    for u, v, w in edges:
        u, v = int(u), int(v)
        w = float(w)
        W[u, v] = w
        W[v, u] = w
    return W


def parse_notes(notes_json: str) -> Dict[str, Any]:
    try:
        return json.loads(notes_json)
    except Exception:
        return {}


def draw_network(W: np.ndarray,
                 omega: np.ndarray,
                 path: str,
                 title: str,
                 layout: str,
                 highlight_edge: Optional[Tuple[int,int]]=None,
                 pos: Optional[Dict[int, Tuple[float,float]]]=None):
    """Uses exact geometric positions if provided; otherwise fall back to U.draw_network_graph layout param."""
    if pos is not None:
        U.draw_geometric_network_graph(
            W=W, omega=omega, pos=pos,
            path=path, title=title,
            highlight_edge=highlight_edge
        )
    else:
        U.draw_network_graph(
            W=W, omega=omega, path=path,
            layout=layout, title=title,
            highlight_edge=highlight_edge
        )


def plot_overlay_no_zoom(K_base, R_base, K_new, R_new, out_path,
                         Kc_base: Optional[float], Kc_new: Optional[float],
                         K_min: float, K_start: float,
                         r_thresh: float,
                         labels=("baseline", "modified")):
    """Overlay r(K) with *fixed axes* to avoid zoom/misleading views."""
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
    plt.xlim(float(K_min), float(K_start))   # fixed domain
    plt.ylim(0.0, 1.0)                       # r ∈ [0,1]
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def reconstruct_positions_if_available(notes: Dict[str,Any]) -> Optional[Dict[int, Tuple[float,float]]]:
    """
    If your CSV includes pos under notes_json (e.g., {"pos_json": [[x,y], ...]}),
    reconstruct an index→(x,y) dict. Otherwise return None.
    """
    pos_json = notes.get("pos_json", None)
    if pos_json is None:
        return None
    try:
        arr = np.asarray(pos_json, dtype=float)
        pos = {i: (float(arr[i,0]), float(arr[i,1])) for i in range(arr.shape[0])}
        return pos
    except Exception:
        return None


def regenerate_curve(omega: np.ndarray, W: np.ndarray,
                     K_start: float, K_min: float, K_step: float, r_thresh: float,
                     rng: Optional[np.random.Generator]=None) -> Tuple[Optional[float], np.ndarray, np.ndarray]:
    """Call your continuation with omega centered (matches pipeline)."""
    omega_c = U._center_omega(omega)
    return U.continuation_descend_K(
        omega_c, W, K_start=K_start, K_min=K_min, K_step=K_step, r_thresh=r_thresh, rng=rng
    )


def process_run(run_rows: pd.DataFrame, outdir: str, top_edges: int):
    """
    For a single run_id, rebuild baseline, select top |delta| paradox edges, and export artifacts.
    """
    # All rows in a run should share these fields; take the first
    row0 = run_rows.iloc[0]
    meta = {c: row0.get(c, None) for c in run_rows.columns}

    # Parse inputs
    omega = np.array(json.loads(row0["omega_json"]), dtype=float)
    edges = json.loads(row0["edges_json"])  # [[u,v,w],...]
    n = int(row0["n"])
    W0 = W_from_edges(n, edges)

    # K grid + settings (used for *exact* regeneration)
    K_start = float(row0["K_start"])
    K_min   = float(row0["K_min"])
    K_step  = float(row0["K_step"])
    r_thresh= float(row0["r_thresh"])
    add_weight = float(row0.get("add_weight", 1.0))

    graph_type = str(row0["graph_type"])
    run_id = str(row0["run_id"])
    layout = "spring" if graph_type == "er" else "spring"  # default
    notes = parse_notes(row0.get("notes_json", "{}"))
    pos = reconstruct_positions_if_available(notes) if graph_type == "rgn" else None

    # Output layout
    net_dirname = f"{run_id}_{graph_type}_n{n}"
    net_dir = os.path.join(outdir, net_dirname)
    ensure_dir(net_dir)
    base_dir = os.path.join(net_dir, "baseline")
    ensure_dir(base_dir)

    # Baseline regeneration
    base_Kc, Kb, Rb = regenerate_curve(omega, W0, K_start, K_min, K_step, r_thresh)
    # Baseline artifacts
    U.plot_r_vs_K(Kb, Rb, out_path=os.path.join(base_dir, "r_vs_K_baseline.png"),
                  Kc=base_Kc, title="Baseline r(K)", r_threshold=r_thresh)
    draw_network(W0, omega, path=os.path.join(base_dir, "network_baseline.png"),
                 title="Baseline network", layout=layout, pos=pos)

    # Write summary.txt
    with open(os.path.join(net_dir, "summary.txt"), "w") as f:
        keys = [
            "code_version","numpy_version","networkx_version","scipy_version","timestamp","run_id",
            "seed_root","seed_child","graph_type","n","er_p","rgn_radius","rgn_box_size",
            "weight_low","weight_high","omega_mean","omega_std","K_start","K_min","K_step",
            "r_thresh","add_weight","baseline_Kc"
        ]
        f.write("=== Run Summary ===\n")
        for k in keys:
            val = meta.get(k, None)
            # prefer regenerated baseline if present
            if k == "baseline_Kc" and base_Kc is not None:
                val = base_Kc
            f.write(f"{k}: {val}\n")
        # geometry note
        if graph_type == "rgn":
            saved = bool(notes.get("graph_pos_saved", False))
            f.write(f"rgn positions saved in CSV: {saved}\n")
            if pos is None and saved:
                f.write("NOTE: positions flag is True but pos_json not found; layout fallback used.\n")
            elif pos is None and not saved:
                f.write("NOTE: exact RGN positions not stored; layout fallback used.\n")

    # Filter paradox rows (is_braess==True) and pick top by |delta|
    br = run_rows.copy()
    br = br[br["is_braess"] == True].dropna(subset=["delta"])
    if br.empty:
        return  # nothing to do for this network

    # Sort by absolute jump magnitude descending
    br = br.reindex(br["delta"].abs().sort_values(ascending=False).index)
    if top_edges > 0:
        br = br.head(top_edges)

    # Convert baseline edges to a set for "add/remove" inference
    base_edge_set = {(int(u), int(v)) if int(u) < int(v) else (int(v), int(u)) for (u, v, _) in edges}

    # Process each chosen edge
    for _, row in br.iterrows():
        u, v = int(row["edge_u"]), int(row["edge_v"])
        e = (u, v) if u < v else (v, u)
        delta = float(row["delta"])

        # Infer change type:
        #  - If edge is in baseline, this row logically corresponds to a "remove"
        #  - Else it's an "add"
        if e in base_edge_set:
            change = "remove"
            # removal must keep graph connected to be valid; if not, skip
            W_mod = U.try_remove_edge_keep_connected(W0, e)
            if W_mod is None:
                # fall back: force removal but note disconnected (skip plots to avoid confusion)
                folder = os.path.join(net_dir, f"{change}-{e[0]}-{e[1]}")
                ensure_dir(folder)
                with open(os.path.join(folder, "info.txt"), "w") as f:
                    f.write(f"edge: {e}\nchange: {change}\nNOTE: removal disconnects the graph; skipped r(K).\n")
                continue
        else:
            change = "add"
            W_mod = W0.copy()
            W_mod[e[0], e[1]] = add_weight
            W_mod[e[1], e[0]] = add_weight

        # Regenerate modified curve
        new_Kc, Km, Rm = regenerate_curve(omega, W_mod, K_start, K_min, K_step, r_thresh)

        # Folder for this edge
        folder = os.path.join(net_dir, f"{change}-{e[0]}-{e[1]}")
        ensure_dir(folder)

        # Plots
        # 1) network (highlight edge if present in modified graph)
        edge_present = (W_mod[e[0], e[1]] != 0.0)
        highlight = e if edge_present else None
        draw_network(
            W_mod, omega,
            path=os.path.join(folder, f"network_{change}.png"),
            title=f"Network after {change} {e}",
            layout=layout, pos=pos, highlight_edge=highlight
        )

        # 2) modified only
        U.plot_r_vs_K(Km, Rm,
                      out_path=os.path.join(folder, f"r_vs_K_{change}.png"),
                      Kc=new_Kc, title=f"{change} {e}: r(K)", r_threshold=r_thresh)

        # 3) overlay with fixed axes
        plot_overlay_no_zoom(
            K_base=Kb, R_base=Rb, K_new=Km, R_new=Rm,
            out_path=os.path.join(folder, f"r_vs_K_overlay_{change}.png"),
            Kc_base=base_Kc, Kc_new=new_Kc,
            K_min=K_min, K_start=K_start,
            r_thresh=r_thresh,
            labels=("baseline", f"{change} {e}")
        )

        # Info file
        with open(os.path.join(folder, "info.txt"), "w") as f:
            f.write(f"edge: {e}\n")
            f.write(f"change: {change}\n")
            f.write(f"delta_from_csv: {delta}\n")
            f.write(f"baseline_Kc_csv: {row.get('baseline_Kc', None)}\n")
            f.write(f"new_Kc_csv: {row.get('new_Kc', None)}\n")
            f.write(f"baseline_Kc_recomputed: {base_Kc}\n")
            f.write(f"new_Kc_recomputed: {new_Kc}\n")
            f.write(f"K_start={K_start}, K_min={K_min}, K_step={K_step}, r_thresh={r_thresh}, add_weight={add_weight}\n")


def main():
    ap = argparse.ArgumentParser(description="Replay biggest Braess jumps from CSV and export artifacts.")
    ap.add_argument("--csv", required=True, help="Path to CSV (from your scan script).")
    ap.add_argument("--outdir", default="replays", help="Output directory root.")
    ap.add_argument("--top", type=int, default=10, help="Per-network: how many biggest |ΔKc| paradox edges to export (0 = all).")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    df = pd.read_csv(args.csv)

    # Keep only rows with a valid run_id and paradox flag
    df = df.dropna(subset=["run_id", "omega_json", "edges_json", "is_braess"])
    if df.empty:
        print("No usable rows in CSV.")
        return

    # Group by run (network)
    for run_id, run_rows in df.groupby("run_id"):
        # Only consider paradox rows; if none present in this group, skip
        if not (run_rows["is_braess"] == True).any():
            continue
        try:
            process_run(run_rows, args.outdir, top_edges=args.top)
        except Exception as e:
            print(f"[warn] run_id={run_id} failed: {e}")


if __name__ == "__main__":
    main()
