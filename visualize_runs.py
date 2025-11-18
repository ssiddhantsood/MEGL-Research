#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scatterplots to explore relationship between derivative-based g and true Kc change.

Assumes input CSV has at least the following columns (like braess_runs_with_g.csv):

  pred_is_braess_g, deriv_g,
  is_braess, label, baseline_Kc, new_Kc, delta,
  braess_tol, edge_u, edge_v, n, graph_type,
  timestamp, code_version, seed_root, seed_child, run_id,
  graph_params_json, K_start, K_min, K_step,
  onset_eps, onset_minrun, add_weight,
  omega_json, edges_json, pos_json

This script will:
  - Filter to rows where deriv_g and delta are finite.
  - Produce multiple scatterplots:
      1) deriv_g vs delta (all, colored by is_braess).
      2) deriv_g vs delta (zoomed region).
      3) baseline_Kc vs deriv_g (colored by label).
      4) baseline_Kc vs delta (colored by is_braess).

Saved into an output directory (default: g_scatter_plots).
"""

from __future__ import annotations
import argparse
import os
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def scatter_deriv_vs_delta_all(df: pd.DataFrame, outdir: str) -> None:
    """Scatter plot: deriv_g vs delta, colored by is_braess."""
    out_path = os.path.join(outdir, "deriv_g_vs_delta_all.png")

    # Split by braess ground truth
    mask_b = df["is_braess"] == True
    mask_nb = df["is_braess"] == False

    plt.figure()
    plt.scatter(
        df.loc[mask_nb, "delta"],
        df.loc[mask_nb, "deriv_g"],
        s=10,
        alpha=0.6,
        label="non_braess",
    )
    plt.scatter(
        df.loc[mask_b, "delta"],
        df.loc[mask_b, "deriv_g"],
        s=10,
        alpha=0.8,
        label="braess",
    )
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.axvline(0.0, linestyle="--", linewidth=1.0)

    plt.xlabel("true delta = new_Kc - baseline_Kc")
    plt.ylabel("deriv_g (finite-diff near Kc)")
    plt.title("deriv_g vs delta (all edges)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def scatter_deriv_vs_delta_zoom(df: pd.DataFrame, outdir: str,
                                delta_lim: float = 0.05,
                                deriv_lim: float = 1.0) -> None:
    """Zoomed scatter: deriv_g vs delta in small box around (0,0)."""
    out_path = os.path.join(outdir, "deriv_g_vs_delta_zoom.png")

    # Filter to zoom window
    mask = (
        df["delta"].abs() <= delta_lim
    ) & (
        df["deriv_g"].abs() <= deriv_lim
    )

    df_zoom = df[mask]
    if df_zoom.empty:
        print("[scatter] Warning: zoom window had no points; skipping zoom plot.")
        return

    mask_b = df_zoom["is_braess"] == True
    mask_nb = df_zoom["is_braess"] == False

    plt.figure()
    plt.scatter(
        df_zoom.loc[mask_nb, "delta"],
        df_zoom.loc[mask_nb, "deriv_g"],
        s=10,
        alpha=0.6,
        label="non_braess",
    )
    plt.scatter(
        df_zoom.loc[mask_b, "delta"],
        df_zoom.loc[mask_b, "deriv_g"],
        s=10,
        alpha=0.8,
        label="braess",
    )
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.axvline(0.0, linestyle="--", linewidth=1.0)

    plt.xlabel("true delta (zoomed)")
    plt.ylabel("deriv_g (zoomed)")
    plt.title(f"deriv_g vs delta (|delta| ≤ {delta_lim}, |deriv_g| ≤ {deriv_lim})")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def scatter_baselineKc_vs_deriv(df: pd.DataFrame, outdir: str) -> None:
    """Scatter: baseline_Kc vs deriv_g, colored by label."""
    out_path = os.path.join(outdir, "baselineKc_vs_deriv_g.png")

    plt.figure()
    labels = df["label"].unique()

    for lab in labels:
        mask = df["label"] == lab
        plt.scatter(
            df.loc[mask, "baseline_Kc"],
            df.loc[mask, "deriv_g"],
            s=8,
            alpha=0.7,
            label=str(lab),
        )

    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("baseline_Kc")
    plt.ylabel("deriv_g")
    plt.title("baseline_Kc vs deriv_g (colored by label)")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def scatter_baselineKc_vs_delta(df: pd.DataFrame, outdir: str) -> None:
    """Scatter: baseline_Kc vs delta, colored by is_braess."""
    out_path = os.path.join(outdir, "baselineKc_vs_delta.png")

    mask_b = df["is_braess"] == True
    mask_nb = df["is_braess"] == False

    plt.figure()
    plt.scatter(
        df.loc[mask_nb, "baseline_Kc"],
        df.loc[mask_nb, "delta"],
        s=10,
        alpha=0.6,
        label="non_braess",
    )
    plt.scatter(
        df.loc[mask_b, "baseline_Kc"],
        df.loc[mask_b, "delta"],
        s=10,
        alpha=0.8,
        label="braess",
    )
    plt.axhline(0.0, linestyle="--", linewidth=1.0)

    plt.xlabel("baseline_Kc")
    plt.ylabel("true delta = new_Kc - baseline_Kc")
    plt.title("baseline_Kc vs delta (colored by braess)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Make scatterplots of deriv_g vs true delta, etc."
    )
    ap.add_argument(
        "--csv",
        default="braess_runs_with_g.csv",
        help="Input CSV with deriv_g and is_braess (default: braess_runs_with_g.csv)",
    )
    ap.add_argument(
        "--outdir",
        default="g_scatter_plots",
        help="Directory to save scatterplots (default: g_scatter_plots)",
    )
    args = ap.parse_args()

    ensure_dir(args.outdir)

    print(f"[scatter] reading: {os.path.abspath(args.csv)}")
    df = pd.read_csv(args.csv)

    # Ensure the types are correct
    if "is_braess" in df.columns:
        df["is_braess"] = df["is_braess"].astype(bool)

    # Filter rows with valid deriv_g and delta
    mask_valid = np.isfinite(df.get("deriv_g", np.nan)) & np.isfinite(df.get("delta", np.nan))
    df_valid = df[mask_valid].copy()

    if df_valid.empty:
        print("[scatter] No valid rows with finite deriv_g and delta; nothing to plot.")
        return

    print(f"[scatter] using {len(df_valid)} rows with finite deriv_g and delta.")

    # Plot 1: deriv_g vs delta (all)
    scatter_deriv_vs_delta_all(df_valid, args.outdir)

    # Plot 2: deriv_g vs delta (zoom)
    scatter_deriv_vs_delta_zoom(df_valid, args.outdir, delta_lim=0.05, deriv_lim=1.0)

    # Plot 3: baseline_Kc vs deriv_g
    if "baseline_Kc" in df_valid.columns:
        scatter_baselineKc_vs_deriv(df_valid, args.outdir)
    else:
        print("[scatter] baseline_Kc column missing; skipping baselineKc_vs_deriv_g plot.")

    # Plot 4: baseline_Kc vs delta
    if "baseline_Kc" in df_valid.columns:
        scatter_baselineKc_vs_delta(df_valid, args.outdir)
    else:
        print("[scatter] baseline_Kc column missing; skipping baselineKc_vs_delta plot.")

    print(f"[scatter] done. Plots in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
