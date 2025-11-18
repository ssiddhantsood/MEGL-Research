
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Visualize Braess runs from networks_with_g.csv with consistent scaling + predictions + rich texts.

This version is adapted to work on the 'networks_with_g.csv',
with column/header changes compared to 'braess_runs.csv'.
The functionality is otherwise nearly identical.

Enhancements:
  • Annotate plots with Kc values.
  • Per-edge folder includes:
      - overlay.png, network_added.png (already)
      - edge_info.txt summarizing baseline/new Kc, delta, prediction, correctness.
      - run_meta.txt copied into each edge folder with graph generation details.
  • Per-run folder has:
      - summary.txt (run meta)
      - kc_values.csv (per-edge Kc table + prediction, correctness)
      - baseline images
  • Global summary at outdir/_predict_summary.txt with accuracy across all runs.
  • A simple predict_braess() using finite-difference derivative on r(K) wrt edge weight.
    By YOUR rule: positive derivative ⇒ predict paradox.

Notes:
  - We recompute curves using the same K grid and onset rule as the generator.
  - Axes are [K_min, K_start] and [0,1] across all plots.
"""

from __future__ import annotations
import argparse, os, json
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils as U  # do not modify utils.py

def ensure_dir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def W_from_edges(n: int, edges: List[Tuple[int,int,float]]) -> np.ndarray:
    W = np.zeros((n, n), float)
    for u, v, w in edges:
        W[int(u), int(v)] = float(w)
        W[int(v), int(u)] = float(w)
    return W

def parse_json_field(s: str):
    if s is None or s == "" or (isinstance(s, float) and np.isnan(s)):
        return None
    return json.loads(s)

def kc_first_onset(Ks: np.ndarray, Rs: np.ndarray, eps: float = 0.05, min_run: int = 3) -> Optional[float]:
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

def plot_r_vs_K(K: np.ndarray, R: np.ndarray, out_path: str,
                Kc: Optional[float], title: str, K_min: float, K_start: float) -> None:
    plt.figure()
    plt.plot(K, R)
    plt.title(title)
    plt.xlabel("K")
    plt.ylabel("r")
    plt.xlim(min(K_min, K_start), max(K_min, K_start))
    plt.ylim(0.0, 1.0)
    if Kc is not None:
        plt.axvline(Kc, linestyle="--")
        # annotate
        y_annot = 0.05
        plt.text(Kc, y_annot, f"Kc={Kc:.3f}", rotation=90, va="bottom", ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_overlay(K_base: np.ndarray, R_base: np.ndarray,
                 K_new: np.ndarray, R_new: np.ndarray,
                 out_path: str, Kc_base: Optional[float], Kc_new: Optional[float],
                 K_min: float, K_start: float, labels=("baseline","added")) -> None:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(K_base, R_base, label=labels[0])
    plt.plot(K_new, R_new, label=labels[1])

    plt.xlabel("K")
    plt.ylabel("r")
    plt.title("Overlay r(K)")
    plt.xlim(min(K_min, K_start), max(K_min, K_start))
    plt.ylim(0.0, 1.0)

    # Add vertical lines and shifted text labels to the right side
    if Kc_base is not None:
        plt.axvline(Kc_base, linestyle="--", color="blue", linewidth=1.2)
        plt.text(max(K_min, K_start)*0.99, 0.85, f"base Kc={Kc_base:.3f}",
                 color="blue", ha="right", va="center")

    if Kc_new is not None:
        plt.axvline(Kc_new, linestyle=":", color="orange", linewidth=1.2)
        plt.text(max(K_min, K_start)*0.99, 0.78, f"new Kc={Kc_new:.3f}",
                 color="orange", ha="right", va="center")

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def draw_network(W: np.ndarray, omega: np.ndarray, path: str, title: str,
                 layout: str, pos: Optional[Dict[int, List[float]]] = None,
                 highlight_edge: Optional[Tuple[int,int]] = None):
    import networkx as nx
    G = nx.Graph()
    n = W.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if W[i,j] != 0.0:
                G.add_edge(i, j, weight=W[i,j])

    if layout == "geo" and pos is not None:
        xy = {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}
    else:
        xy = nx.spring_layout(G, seed=42)

    plt.figure()
    nx.draw_networkx(G, pos=xy, with_labels=True, node_size=300, font_size=7)
    if highlight_edge is not None:
        u, v = highlight_edge
        nx.draw_networkx_edges(G, pos=xy, edgelist=[(u,v)], width=3)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def build_base_meta(row0: pd.Series, W0: np.ndarray, omega: np.ndarray, graph_type: str) -> Dict[str, Any]:
    # Match new possible column names in networks_with_g.csv
    seed_child = int(row0.get("seed_child", row0.get("child_seed", -1)))
    seed_root = int(row0.get("seed_root", row0.get("root_seed", -1)))
    return {
        "run_id": row0["run_id"],
        "graph_type": row0["graph_type"],
        "n": int(row0["n"]),
        "K_start": float(row0["K_start"]),
        "K_min": float(row0["K_min"]),
        "K_step": float(row0["K_step"]),
        "onset_eps": float(row0.get("onset_eps", row0.get("onset_epsilon", 0.05))),
        "onset_minrun": int(row0.get("onset_minrun", 3)),
        "add_weight": float(row0.get("add_weight", row0.get("added_weight", 1.0))),
        "edges_count": int((W0!=0).sum()//2),
        "seed_child": seed_child,
        "seed_root": seed_root,
        "code_version": row0.get("code_version",""),
        "timestamp": row0.get("timestamp",""),
        "graph_params_json": row0.get("graph_params_json", row0.get("graph_params", "")),
    }

def write_meta_txt(path: str, meta: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")

def predict_braess_derivative_sign(
    omega: np.ndarray, W0: np.ndarray, u: int, v: int,
    *, K_start: float, K_min: float, K_step: float,
    onset_eps: float, onset_minrun: int, add_weight: float, eps_frac: float = 0.1
) -> Dict[str, Any]:
    """Finite-difference sign test near baseline Kc.

    Steps:
      1) Compute baseline Kc and r(K).
      2) Add a *small* fraction of add_weight (eps_frac * add_weight).
      3) Recompute r(K) on same grid.
      4) Align by index; take a window of points starting at baseline Kc index.
      5) Compute mean dR/dw ≈ (R_plus - R_base) / (eps_weight). If > 0 ⇒ predict Braess (per your rule).
    """
    Kb, Rb = continuation_curve(omega, W0, K_start, K_min, K_step, rng=np.random.default_rng(123))
    Kc_base = kc_first_onset(Kb, Rb, eps=onset_eps, min_run=onset_minrun)
    if Kc_base is None:
        return {"pred": False, "deriv_mean": None, "Kc_base": None}

    eps_w = max(1e-6, float(eps_frac) * float(add_weight))
    Wp = W0.copy()
    Wp[u, v] += eps_w
    Wp[v, u] += eps_w
    Kp, Rp = continuation_curve(omega, Wp, K_start, K_min, K_step, rng=np.random.default_rng(124))

    # nearest index to Kc_base
    idx = np.argmin(np.abs(Kb - Kc_base))
    j1 = idx
    j2 = min(len(Kb), idx + 5)
    if j2 <= j1:
        j2 = min(len(Kb), j1 + 1)
    dR = (Rp[j1:j2] - Rb[j1:j2]) / eps_w
    deriv_mean = float(np.mean(dR)) if dR.size > 0 else 0.0

    pred = bool(deriv_mean > 0.0)
    return {"pred": pred, "deriv_mean": deriv_mean, "Kc_base": Kc_base}

def process_run(run_rows: pd.DataFrame, outdir: str,
                add_weight: float, K_start: float, K_min: float, K_step: float,
                onset_eps: float, onset_minrun: int,
                global_acc: Dict[str, Any]) -> None:
    # reconstruct baseline
    row0 = run_rows.iloc[0]
    n = int(row0["n"])
    edges = parse_json_field(row0.get("edges_json", row0.get("edges", ""))) or []
    pos = parse_json_field(row0.get("pos_json", row0.get("pos", ""))) or None
    omega_json = row0.get("omega_json", row0.get("omega", ""))
    omega = np.array(parse_json_field(omega_json), float)
    graph_type = row0["graph_type"]
    layout = "geo" if (graph_type == "rgn" and pos is not None) else "spring"

    W0 = W_from_edges(n, edges)

    # baseline curve and first-onset Kc
    Kb, Rb = continuation_curve(omega, W0, K_start, K_min, K_step, rng=np.random.default_rng(0))
    Kc_base = kc_first_onset(Kb, Rb, eps=onset_eps, min_run=onset_minrun)

    # out dirs
    run_id = str(row0["run_id"])
    net_dir = os.path.join(outdir, run_id)
    ensure_dir(net_dir)
    base_dir = os.path.join(net_dir, "baseline")
    ensure_dir(base_dir)

    # baseline plots
    path_rK_base = os.path.join(base_dir, "r_vs_K_baseline.png")
    path_net_base = os.path.join(base_dir, "network_baseline.png")
    plot_r_vs_K(Kb, Rb, path_rK_base,
                Kc=Kc_base, title="Baseline r(K) [first-onset Kc]", K_min=K_min, K_start=K_start)
    draw_network(W0, omega, path_net_base,
                 "Baseline network", layout=layout, pos=pos)

    # per-run meta
    meta = build_base_meta(row0, W0, omega, graph_type)
    write_meta_txt(os.path.join(net_dir, "summary.txt"), meta)

    # find braess edges (from 'is_braess_add_edge' or similar column if present)
    braess_add_edges = set()
    braess_col = "is_braess"
    if "is_braess_add_edge" in run_rows.columns:
        braess_col = "is_braess_add_edge"
    elif "braess_w" in run_rows.columns:
        # Assume nonzero/True means it's a Braess edge
        braess_col = "braess_w"

    edge_u_col = "edge_u" if "edge_u" in run_rows.columns else "added_u"
    edge_v_col = "edge_v" if "edge_v" in run_rows.columns else "added_v"

    for _, r in run_rows.iterrows():
        is_braess = bool(r.get(braess_col, False))
        if is_braess:
            u, v = int(r[edge_u_col]), int(r[edge_v_col])
            if u > v:
                u, v = v, u
            braess_add_edges.add((u, v))
    if not braess_add_edges:
        return

    # per-edge
    rows_out = []
    for (u, v) in sorted(braess_add_edges):
        Wm = W0.copy()
        Wm[u, v] = add_weight
        Wm[v, u] = add_weight

        Km, Rm = continuation_curve(omega, Wm, K_start, K_min, K_step, rng=np.random.default_rng(1))
        Kc_new = kc_first_onset(Km, Rm, eps=onset_eps, min_run=onset_minrun)

        edge_tag = f"edge_{u}_{v}"
        e_dir = os.path.join(net_dir, edge_tag)
        ensure_dir(e_dir)

        # plots with annotations
        plot_overlay(Kb, Rb, Km, Rm,
                     os.path.join(e_dir, "overlay.png"),
                     Kc_base, Kc_new, K_min, K_start, labels=("baseline","added"))
        draw_network(Wm, omega, os.path.join(e_dir, "network_added.png"),
                     f"Added edge ({u},{v})", layout=layout, pos=pos, highlight_edge=(u,v))

        # prediction via finite-difference derivative sign (your rule)
        pred_info = predict_braess_derivative_sign(
            omega, W0, u, v,
            K_start=K_start, K_min=K_min, K_step=K_step,
            onset_eps=onset_eps, onset_minrun=onset_minrun,
            add_weight=add_weight, eps_frac=0.1
        )
        pred = bool(pred_info["pred"])
        deriv_mean = pred_info["deriv_mean"]

        # ground truth
        is_braess_truth = (Kc_base is not None and Kc_new is not None and (Kc_new - Kc_base) > 1e-3)
        correct = (pred == is_braess_truth)

        # write edge_info.txt
        with open(os.path.join(e_dir, "edge_info.txt"), "w") as f:
            f.write(f"edge: ({u},{v})\n")
            f.write(f"add_weight: {add_weight}\n")
            f.write(f"Kc_base: {Kc_base}\n")
            f.write(f"Kc_new: {Kc_new}\n")
            f.write(f"delta: {None if (Kc_base is None or Kc_new is None) else (Kc_new - Kc_base)}\n")
            f.write(f"predict_deriv_mean: {deriv_mean}\n")
            f.write(f"predict_is_braess (sign>0): {pred}\n")
            f.write(f"ground_truth_is_braess: {is_braess_truth}\n")
            f.write(f"prediction_correct: {correct}\n")

        # also drop a copy of run meta here
        write_meta_txt(os.path.join(e_dir, "run_meta.txt"), meta)

        rows_out.append({
            "u": u, "v": v,
            "Kc_base": Kc_base, "Kc_new": Kc_new,
            "delta": None if (Kc_base is None or Kc_new is None) else (Kc_new - Kc_base),
            "pred": pred, "deriv_mean": deriv_mean, "is_braess": is_braess_truth, "correct": correct
        })

        # update global counters
        global_acc["n"] += 1
        if pred: global_acc["pred_pos"] += 1
        if is_braess_truth: global_acc["gt_pos"] += 1
        if correct: global_acc["correct"] += 1

    # save kc table + predictions
    import csv
    with open(os.path.join(net_dir, "kc_values.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["u","v","Kc_base","Kc_new","delta","pred","deriv_mean","is_braess","correct"])
        w.writeheader()
        w.writerows(rows_out)

def main():
    ap = argparse.ArgumentParser(description="Visualize networks_with_g.csv runs + predictions.")
    ap.add_argument("--csv", default="networks_with_g.csv")
    ap.add_argument("--outdir", default="viz_networks_with_g")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Work around headers: try to ensure is_braess column is present and boolean
    # Different csvs may call it 'is_braess' or 'is_braess_add_edge', etc.
    if "is_braess" in df.columns:
        df["is_braess"] = df["is_braess"].astype(bool)
    if "is_braess_add_edge" in df.columns:
        df["is_braess_add_edge"] = df["is_braess_add_edge"].astype(bool)
    if "braess_w" in df.columns:
        # treat as binary
        df["braess_w"] = df["braess_w"].astype(bool)

    # global accuracy accumulator
    global_acc = {"n": 0, "pred_pos": 0, "gt_pos": 0, "correct": 0}

    # Identify grouping column
    group_col = "run_id"
    if "run_id" not in df.columns and "id" in df.columns:
        group_col = "id"

    required_K_grid_cols = ["K_start", "K_min", "K_step"]
    for col in required_K_grid_cols:
        if col not in df.columns:
            raise RuntimeError(f"Column '{col}' is required in the input csv for K grid")

    onset_eps_col = "onset_eps"
    if "onset_eps" not in df.columns:
        onset_eps_col = "onset_epsilon"
    onset_minrun_col = "onset_minrun"
    add_weight_col = "add_weight"
    if "onset_minrun" not in df.columns and "minrun" in df.columns:
        onset_minrun_col = "minrun"
    if "add_weight" not in df.columns and "added_weight" in df.columns:
        add_weight_col = "added_weight"

    for run_id, group in df.groupby(group_col):
        K_start = float(group["K_start"].iloc[0])
        K_min = float(group["K_min"].iloc[0])
        K_step = float(group["K_step"].iloc[0])
        onset_eps = float(group[onset_eps_col].iloc[0]) if onset_eps_col in group.columns else 0.05
        onset_minrun = int(group[onset_minrun_col].iloc[0]) if onset_minrun_col in group.columns else 3
        add_weight = float(group[add_weight_col].iloc[0]) if add_weight_col in group.columns else 1.0

        process_run(group, args.outdir, add_weight, K_start, K_min, K_step, onset_eps, onset_minrun, global_acc)

    # write global summary
    out_summary = os.path.join(args.outdir, "_predict_summary.txt")
    ensure_dir(args.outdir)
    with open(out_summary, "w") as f:
        n = global_acc["n"]
        acc = (global_acc["correct"]/n) if n else 0.0
        f.write("=== Predict Braess Summary ===\n")
        f.write(f"total_edges_evaluated: {n}\n")
        f.write(f"predicted_positive: {global_acc['pred_pos']}\n")
        f.write(f"groundtruth_positive: {global_acc['gt_pos']}\n")
        f.write(f"num_correct: {global_acc['correct']}\n")
        f.write(f"accuracy: {acc:.4f}\n")

if __name__ == "__main__":
    main()