from __future__ import annotations
import os
import numpy as np
import networkx as nx
from scipy.optimize import fsolve
from typing import List, Tuple, Dict, Optional

# --- plotting (headless safe) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Basic helpers
# -----------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def create_output_folder(root: str = "runs") -> str:
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"{root}/results_{ts}"
    _ensure_dir(folder)
    return folder

def order_parameter(theta: np.ndarray) -> float:
    r"""Kuramoto order parameter r ∈ [0,1]."""
    z = np.exp(1j * theta).mean()
    return float(abs(z))

def _center_omega(omega: np.ndarray) -> np.ndarray:
    w = np.asarray(omega, dtype=float)
    return w - w.mean()


# -----------------------------
# Kuramoto residual (gauge-fixed)
# -----------------------------
def _residual_gauge(theta_red: np.ndarray, omega: np.ndarray, W: np.ndarray, K: float) -> np.ndarray:
    """
    Gauge-fix θ0 = 0. We solve for θ1..θ(n-1) and return the residuals for i=1..n-1:
        0 = ω_i + K * sum_j W_ij * sin(θ_j - θ_i)

    This version is vectorized (no Python loops over i,j), so fsolve calls are much faster.
    """
    n = len(omega)

    # Rebuild full angle vector with θ0 fixed to 0
    theta = np.zeros(n, dtype=float)
    theta[1:] = theta_red

    # Pairwise differences θ_j - θ_i  (shape n×n via broadcasting)
    dtheta = theta[np.newaxis, :] - theta[:, np.newaxis]

    # For each i: s_i = sum_j W_ij * sin(θ_j - θ_i)
    s = (W * np.sin(dtheta)).sum(axis=1)

    # Full residuals, then drop equation i=0 to match reduced unknowns
    res_full = omega + K * s
    return res_full[1:]



def solve_locked(
    omega: np.ndarray,
    W: np.ndarray,
    K: float,
    theta0: Optional[np.ndarray] = None,
    xtol: float = 1e-9,
    maxfev: int = 4000,
    retries: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, bool, float]:
    """
    Solve gauge-fixed Kuramoto steady state with fsolve.
    Returns (theta_full, success, residual_norm_full).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(omega)

    # Initial guess in reduced space (n-1 variables)
    if theta0 is None:
        th0_red = np.full(n-1, 5.0, dtype=float)   # default guess = 5
    else:
        th0 = np.asarray(theta0, dtype=float).reshape(-1)
        th0_red = th0[1:] if len(th0) == n else np.full(n-1, 5.0, dtype=float)

    last_res_norm = np.inf
    for _ in range(retries + 1):
        th_red, info, ier, msg = fsolve(
            _residual_gauge, th0_red, args=(omega, W, K),
            full_output=True, xtol=xtol, maxfev=maxfev
        )
        th_full = np.zeros(n, dtype=float)
        th_full[1:] = th_red

        # Check full-system residual (n equations)
        res = np.zeros(n, dtype=float)
        for i in range(n):
            Wi = W[i]
            ti = th_full[i]
            s = 0.0
            for j in range(n):
                w = Wi[j]
                if w != 0.0:
                    s += w * np.sin(th_full[j] - ti)
            res[i] = omega[i] + K * s
        res_norm = float(np.linalg.norm(res, ord=2))
        last_res_norm = res_norm
        if ier == 1 and res_norm < 1e-6:
            return th_full, True, res_norm

        # Next attempt: randomize reduced-space initial guess
        th0_red = rng.uniform(0.0, 2*np.pi, size=n-1)

    return th_full, False, last_res_norm


# -----------------------------
# Linearized (Laplacian) seed
# -----------------------------
def _laplacian_initial_guess(omega: np.ndarray, W: np.ndarray, K: float) -> np.ndarray:
    """
    Small-angle linear seed: L theta ≈ omega / K with theta[0]=0.
    Solve reduced system (remove first row/col), then reinsert theta0=0.
    """
    n = len(omega)
    L = np.diag(W.sum(axis=1)) - W
    Lrr = L[1:, 1:]
    b = (omega[1:] / K).astype(float)

    # Least-squares for robustness
    theta_r, *_ = np.linalg.lstsq(Lrr, b, rcond=None)
    theta = np.zeros(n, dtype=float)
    theta[1:] = theta_r

    # Wrap to [-pi, pi] to keep angles tame
    theta = (theta + np.pi) % (2*np.pi) - np.pi
    return theta


# -----------------------------
# Continuation: high K -> low K
# -----------------------------
def continuation_descend_K(
    omega: np.ndarray,
    W: np.ndarray,
    K_start: float = 10.0,
    K_min: float = 0.02,
    K_step: float = 0.1,
    r_thresh: float = 0.7,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[float], np.ndarray, np.ndarray]:
    """
    Start at a large K, step down by K_step. Use previous solution as warm-start.
    Return (Kc, K_values (ascending), r_values).
    Kc = first K where r >= r_thresh (fallback: argmax r).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Co-rotating frame
    omega = _center_omega(omega)

    K = float(K_start)

    # Lock at high K with a cascade of seeds
    theta0 = None
    theta, ok, _ = solve_locked(omega, W, K, theta0=theta0, retries=3, rng=rng)

    if not ok:
        theta0 = _laplacian_initial_guess(omega, W, K)
        theta, ok, _ = solve_locked(omega, W, K, theta0=theta0, retries=2, rng=rng)

    if not ok:
        Kh = K * 1.5
        theta, ok, _ = solve_locked(omega, W, Kh, theta0=None, retries=2, rng=rng)
        if not ok:
            theta0 = _laplacian_initial_guess(omega, W, Kh)
            theta, ok, _ = solve_locked(omega, W, Kh, theta0=theta0, retries=2, rng=rng)
        if ok:
            K = Kh

    if not ok:
        return None, np.array([]), np.array([])

    # Walk downward
    Ks, Rs = [], []
    while K >= K_min:
        theta, ok, _ = solve_locked(omega, W, K, theta0=theta, retries=2, rng=rng)
        r = order_parameter(theta) if ok else 0.0
        Ks.append(K)
        Rs.append(r)
        K -= K_step

        if len(Rs) >= 3 and Rs[-1] < 0.05 and Rs[-2] < 0.05:
            break

    Ks = np.array(Ks[::-1], dtype=float)
    Rs = np.array(Rs[::-1], dtype=float)

    Kc = None
    for Ki, ri in zip(Ks, Rs):
        if ri >= r_thresh:
            Kc = float(Ki)
            break
    if Kc is None and len(Ks) > 0:
        Kc = float(Ks[np.argmax(Rs)])
    return Kc, Ks, Rs


# -----------------------------
# Custom sweep (if you want)
# -----------------------------
def sweep_K_custom(
    omega: np.ndarray,
    W: np.ndarray,
    K_values: np.ndarray,
    r_thresh: float = 0.7,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[float], np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    omega = _center_omega(omega)
    K_values = np.asarray(K_values, dtype=float)
    if K_values.ndim != 1 or K_values.size == 0:
        return None, np.array([]), np.array([])

    K_desc = np.sort(K_values)[::-1]
    theta0 = None
    theta, ok, _ = solve_locked(omega, W, K_desc[0], theta0=theta0, retries=3, rng=rng)
    if not ok:
        theta0 = _laplacian_initial_guess(omega, W, K_desc[0])
        theta, ok, _ = solve_locked(omega, W, K_desc[0], theta0=theta0, retries=2, rng=rng)
    if not ok:
        return None, np.array([]), np.array([])

    R_desc = []
    for K in K_desc:
        theta, ok, _ = solve_locked(omega, W, K, theta0=theta, retries=2, rng=rng)
        r = order_parameter(theta) if ok else 0.0
        R_desc.append(r)

    Ks = K_desc[::-1]
    Rs = np.array(R_desc[::-1], dtype=float)

    Kc = None
    for Ki, ri in zip(Ks, Rs):
        if ri >= r_thresh:
            Kc = float(Ki); break
    if Kc is None:
        Kc = float(Ks[np.argmax(Rs)])
    return Kc, Ks, Rs


# -----------------------------
# Graph building
# -----------------------------
def random_connected_graph(
    n: int, p: float = 0.3,
    weight_low: float = 1.0, weight_high: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    max_tries: int = 100
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(max_tries):
        G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(0, 1 << 32)))
        if nx.is_connected(G) and G.number_of_edges() >= n - 1:
            W = np.zeros((n, n), dtype=float)
            for u, v in G.edges():
                w = rng.uniform(weight_low, weight_high)
                W[u, v] = w
                W[v, u] = w
            return W
    raise RuntimeError("Failed to generate a connected graph. Try larger p.")

def random_omega(n: int, mean: float = 0.0, std: float = 1.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(loc=mean, scale=std, size=n).astype(float)

def removable_edges(W: np.ndarray) -> List[Tuple[int, int]]:
    n = W.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] != 0.0:
                out.append((i, j))
    return out

def missing_edges(W: np.ndarray) -> List[Tuple[int, int]]:
    n = W.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] == 0.0:
                out.append((i, j))
    return out

def try_remove_edge_keep_connected(W: np.ndarray, e: Tuple[int, int]) -> Optional[np.ndarray]:
    i, j = e
    if W[i, j] == 0.0:
        return None
    W2 = W.copy()
    W2[i, j] = 0.0
    W2[j, i] = 0.0
    if nx.is_connected(nx.from_numpy_array(W2)):
        return W2
    return None


# -----------------------------
# Plotting helpers
# -----------------------------
def draw_network_graph(W: np.ndarray, omega: np.ndarray, path: str,
                       layout: str = "spring", title: str = "Network",
                       highlight_edge: Optional[Tuple[int, int]] = None,
                       highlight_color: str = "orange"):
    """
    Draw W (undirected). Node color: sign of omega (red: >0, blue: <0).
    Optionally highlight one undirected edge.
    """
    n = W.shape[0]
    G = nx.from_numpy_array(W)

    if layout == "spring":
        pos = nx.spring_layout(G, seed=0)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.random_layout(G, seed=0)

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5)
    colors = ["red" if w > 0 else "blue" for w in omega]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=300, alpha=0.9, linewidths=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)

    if highlight_edge is not None:
        u, v = highlight_edge
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=3.0, edge_color=highlight_color)

    plt.title(f"{title}\nRed: ω>0, Blue: ω<0")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_r_vs_K_overlay(
    K_base: np.ndarray, r_base: np.ndarray,
    K_new: np.ndarray, r_new: np.ndarray,
    out_path: str,
    Kc_base: Optional[float] = None,
    Kc_new: Optional[float] = None,
    title: str = "Baseline vs. modified r(K)",
    r_threshold: float = 0.7,
    labels: Tuple[str, str] = ("baseline", "modified"),
):
    """
    Overlay r(K) for a modified network on top of the baseline curve.
    Handles different K grids by just plotting both series as-is.
    """
    if (K_base is None or len(K_base) == 0) or (K_new is None or len(K_new) == 0):
        return
    _ensure_dir(os.path.dirname(out_path) or ".")

    plt.figure(figsize=(8, 6))
    # baseline
    plt.plot(K_base, r_base, "o-", linewidth=2, markersize=4, label=labels[0])
    if Kc_base is not None:
        plt.axvline(Kc_base, linestyle="--", color="g", alpha=0.6, label=f"{labels[0]} Kc ≈ {Kc_base:.3f}")
    # modified
    plt.plot(K_new, r_new, "s--", linewidth=2, markersize=4, label=labels[1])
    if Kc_new is not None:
        plt.axvline(Kc_new, linestyle=":", color="m", alpha=0.7, label=f"{labels[1]} Kc ≈ {Kc_new:.3f}")

    if r_threshold is not None:
        plt.axhline(r_threshold, linestyle="--", color="r", alpha=0.35, label=f"threshold {r_threshold}")

    plt.xlabel("Coupling strength K")
    plt.ylabel("Order parameter r")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_r_vs_K(K_values: np.ndarray, r_values: np.ndarray, out_path: str,
                Kc: Optional[float] = None, title: str = "Synchronization curve",
                r_threshold: float = 0.7):
    """Plot r(K) and save."""
    if K_values is None or len(K_values) == 0:
        return
    _ensure_dir(os.path.dirname(out_path) or ".")
    plt.figure(figsize=(8, 6))
    plt.plot(K_values, r_values, "o-", linewidth=2, markersize=4, label="r(K)")
    if Kc is not None:
        plt.axvline(Kc, linestyle="--", color="g", alpha=0.6, label=f"Kc ≈ {Kc:.3f}")
    if r_threshold is not None:
        plt.axhline(r_threshold, linestyle="--", color="r", alpha=0.4, label=f"threshold {r_threshold}")
    plt.xlabel("Coupling strength K")
    plt.ylabel("Order parameter r")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_edge_deltas(rows: List[Dict], out_path: str, title: str = "ΔKc per edge change"):
    """Horizontal bar plot of delta = new_Kc - baseline_Kc (sorted)."""
    x, labels = [], []
    for r in rows:
        d = r.get("delta", None)
        if isinstance(d, (int, float)):
            x.append(float(d))
            labels.append(str(r.get("edge")))
    if not x:
        return
    idx = np.argsort(x)
    x = np.array(x)[idx]
    labels = [labels[i] for i in idx]

    _ensure_dir(os.path.dirname(out_path) or ".")
    plt.figure(figsize=(10, max(4, 0.3 * len(x))))
    y = np.arange(len(x))
    plt.barh(y, x)
    plt.axvline(0.0, color="k", linewidth=1)
    plt.xlabel("ΔKc = new_Kc - baseline_Kc  (negative ⇒ improvement)")
    plt.yticks(y, labels, fontsize=7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# -----------------------------
# Braess scans WITH per-edge plots
# -----------------------------
def braess_scan_single_add_edges(
    omega: np.ndarray,
    W: np.ndarray,
    add_weight: float = 1.0,
    K_start: float = 10.0,
    K_min: float = 0.02,
    K_step: float = 0.1,
    r_thresh: float = 0.7,
    rng: Optional[np.random.Generator] = None,
    outdir: Optional[str] = None,
    layout: str = "spring",
) -> List[Dict]:
    """
    Baseline network is fixed. For each missing edge (u,v), add it temporarily,
    compute Kc, then discard. Each iteration uses the original baseline + 1 new edge.
    
    Returns rows with:
      edge, baseline_Kc, new_Kc, delta, is_braess
    """
    if rng is None:
        rng = np.random.default_rng()
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    # Baseline
    omega_c = _center_omega(omega)
    base_Kc, base_Ks, base_Rs = continuation_descend_K(
        omega_c, W, K_start=K_start, K_min=K_min, K_step=K_step,
        r_thresh=r_thresh, rng=rng
    )
    if outdir is not None:
        plot_r_vs_K(
            base_Ks, base_Rs,
            out_path=os.path.join(outdir, "r_vs_K_baseline.png"),
            Kc=base_Kc, title="Baseline r(K)", r_threshold=r_thresh
        )
        draw_network_graph(
            W, omega,
            path=os.path.join(outdir, "network_baseline.png"),
            layout=layout, title="Baseline network"
        )

    if base_Kc is None:
        return [{
            'edge': 'baseline',
            'baseline_Kc': None,
            'new_Kc': None,
            'delta': None,
            'is_braess': False,
        }]

    rows = []
    for (i, j) in missing_edges(W):
        # fresh copy of baseline each time
        W2 = W.copy()
        W2[i, j] = add_weight
        W2[j, i] = add_weight

        # recompute
        Kc2, Ks2, Rs2 = continuation_descend_K(
            omega_c, W2, K_start=K_start, K_min=K_min, K_step=K_step,
            r_thresh=r_thresh, rng=rng
        )
        delta = None if Kc2 is None else float(Kc2 - base_Kc)
        is_braess = (delta is not None) and (delta > 1e-3)

        rows.append({
            'edge': (i, j),
            'baseline_Kc': float(base_Kc),
            'new_Kc': None if Kc2 is None else float(Kc2),
            'delta': delta,
            'is_braess': bool(is_braess),
        })

        if outdir is not None:
            subdir = os.path.join(outdir, f"add_{i}_{j}")
            os.makedirs(subdir, exist_ok=True)

            plot_r_vs_K(
                Ks2, Rs2,
                out_path=os.path.join(subdir, "r_vs_K.png"),
                Kc=Kc2, title=f"Add edge {(i,j)}: r(K)", r_threshold=r_thresh
            )
            plot_r_vs_K_overlay(
                base_Ks, base_Rs,
                Ks2, Rs2,
                out_path=os.path.join(subdir, "r_vs_K_overlay.png"),
                Kc_base=base_Kc, Kc_new=Kc2,
                title=f"Overlay: baseline vs add {(i,j)}",
                r_threshold=r_thresh,
                labels=("baseline", f"add {(i,j)}")
            )
            draw_network_graph(
                W2, omega,
                path=os.path.join(subdir, "network.png"),
                layout=layout, title=f"Network with added edge {(i,j)}",
                highlight_edge=(i, j)
            )

    return rows

def braess_scan_remove_edges(
    omega: np.ndarray,
    W: np.ndarray,
    K_start: float = 10.0,
    K_min: float = 0.02,
    K_step: float = 0.1,
    r_thresh: float = 0.7,
    mode: str = "random",
    rng: Optional[np.random.Generator] = None,
    outdir: Optional[str] = None,
    layout: str = "spring",
) -> List[Dict]:
    """
    Compute baseline Kc, then remove edges one-by-one (keeping graph connected),
    recompute Kc, and SAVE per-edge r(K) plots + baseline overlays if outdir is given.
    Returns result rows with uniform keys.
    """
    if rng is None:
        rng = np.random.default_rng()
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    # Baseline (center omega defensively)
    omega_c = _center_omega(omega)
    base_Kc, base_Ks, base_Rs = continuation_descend_K(
        omega_c, W, K_start=K_start, K_min=K_min, K_step=K_step, r_thresh=r_thresh, rng=rng
    )
    if outdir is not None:
        plot_r_vs_K(
            base_Ks, base_Rs,
            out_path=os.path.join(outdir, "r_vs_K_baseline.png"),
            Kc=base_Kc, title="Baseline r(K)", r_threshold=r_thresh
        )
        draw_network_graph(
            W, omega,
            path=os.path.join(outdir, "network_baseline.png"),
            layout=layout, title="Baseline network"
        )

    if base_Kc is None:
        return [{
            'edge': 'baseline',
            'baseline_Kc': None,
            'new_Kc': None,
            'delta': None,
            'is_braess': False,
        }]

    # Edge order
    edges = removable_edges(W)
    if mode == "betweenness":
        G = nx.from_numpy_array(W)
        eb = nx.edge_betweenness_centrality(G, normalized=True)
        edges = sorted(eb.keys(), key=lambda e: eb[e], reverse=True)
    else:
        rng.shuffle(edges)

    rows = []
    for e in edges:
        W2 = try_remove_edge_keep_connected(W, e)
        if W2 is None:
            continue

        Kc2, Ks2, Rs2 = continuation_descend_K(
            omega_c, W2, K_start=K_start, K_min=K_min, K_step=K_step, r_thresh=r_thresh, rng=rng
        )
        delta = None if Kc2 is None else float(Kc2 - base_Kc)
        is_braess = (delta is not None) and (delta < -1e-3)

        rows.append({
            'edge': e,
            'baseline_Kc': float(base_Kc),
            'new_Kc': None if Kc2 is None else float(Kc2),
            'delta': delta,
            'is_braess': bool(is_braess),
        })

        if outdir is not None:
            i, j = e
            # Modified-only curve
            plot_r_vs_K(
                Ks2, Rs2,
                out_path=os.path.join(outdir, f"r_vs_K_remove_{i}_{j}.png"),
                Kc=Kc2, title=f"Remove edge {e}: r(K)", r_threshold=r_thresh
            )
            # Overlay with baseline
            plot_r_vs_K_overlay(
                base_Ks, base_Rs,
                Ks2, Rs2,
                out_path=os.path.join(outdir, f"r_vs_K_remove_{i}_{j}_overlay.png"),
                Kc_base=base_Kc, Kc_new=Kc2,
                title=f"Overlay: baseline vs remove {e}",
                r_threshold=r_thresh,
                labels=("baseline", f"remove {e}")
            )
            # Snapshot of modified network
            draw_network_graph(
                W2, omega,
                path=os.path.join(outdir, f"network_remove_{i}_{j}.png"),
                layout=layout, title=f"Network after removing {e}",
                highlight_edge=None
            )

    if outdir is not None:
        plot_edge_deltas(
            rows, os.path.join(outdir, "edge_deltas_remove.png"),
            title="ΔKc for removed edges (negative ⇒ improvement)"
        )

    return rows



def braess_scan_add_edges(
    omega: np.ndarray,
    W: np.ndarray,
    add_weight: float = 1.0,
    K_start: float = 10.0,
    K_min: float = 0.02,
    K_step: float = 0.1,
    r_thresh: float = 0.7,
    rng: Optional[np.random.Generator] = None,
    outdir: Optional[str] = None,
    layout: str = "spring",
) -> List[Dict]:
    """
    Add each missing edge (weight = add_weight), recompute Kc, and SAVE per-edge r(K) plots
    + baseline overlays if outdir is given. Baseline is the original W.
    'Braess' here means delta > 0 (adding an edge worsens synchronizability).
    """
    if rng is None:
        rng = np.random.default_rng()
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    # Baseline
    omega_c = _center_omega(omega)
    base_Kc, base_Ks, base_Rs = continuation_descend_K(
        omega_c, W, K_start=K_start, K_min=K_min, K_step=K_step, r_thresh=r_thresh, rng=rng
    )
    if outdir is not None:
        plot_r_vs_K(
            base_Ks, base_Rs,
            out_path=os.path.join(outdir, "r_vs_K_baseline.png"),
            Kc=base_Kc, title="Baseline r(K)", r_threshold=r_thresh
        )
        draw_network_graph(
            W, omega,
            path=os.path.join(outdir, "network_baseline.png"),
            layout=layout, title="Baseline network"
        )

    if base_Kc is None:
        return [{
            'edge': 'baseline',
            'baseline_Kc': None,
            'new_Kc': None,
            'delta': None,
            'is_braess': False,
        }]

    rows = []
    for (i, j) in missing_edges(W):
        W2 = W.copy()
        W2[i, j] = add_weight
        W2[j, i] = add_weight

        Kc2, Ks2, Rs2 = continuation_descend_K(
            omega_c, W2, K_start=K_start, K_min=K_min, K_step=K_step, r_thresh=r_thresh, rng=rng
        )
        delta = None if Kc2 is None else float(Kc2 - base_Kc)
        # For additions, paradox means delta > 0
        is_braess = (delta is not None) and (delta > 1e-3)

        rows.append({
            'edge': (i, j),
            'baseline_Kc': float(base_Kc),
            'new_Kc': None if Kc2 is None else float(Kc2),
            'delta': delta,
            'is_braess': bool(is_braess),
        })

        if outdir is not None:
            # Modified-only curve
            plot_r_vs_K(
                Ks2, Rs2,
                out_path=os.path.join(outdir, f"r_vs_K_add_{i}_{j}.png"),
                Kc=Kc2, title=f"Add edge {(i,j)}: r(K)", r_threshold=r_thresh
            )
            # Overlay with baseline
            plot_r_vs_K_overlay(
                base_Ks, base_Rs,
                Ks2, Rs2,
                out_path=os.path.join(outdir, f"r_vs_K_add_{i}_{j}_overlay.png"),
                Kc_base=base_Kc, Kc_new=Kc2,
                title=f"Overlay: baseline vs add {(i,j)}",
                r_threshold=r_thresh,
                labels=("baseline", f"add {(i,j)}")
            )
            # Snapshot of modified network (highlight new edge)
            draw_network_graph(
                W2, omega,
                path=os.path.join(outdir, f"network_add_{i}_{j}.png"),
                layout=layout, title=f"Network after adding {(i,j)}",
                highlight_edge=(i, j), highlight_color="orange"
            )

    if outdir is not None:
        plot_edge_deltas(
            rows, os.path.join(outdir, "edge_deltas_add.png"),
            title="ΔKc for added edges (positive ⇒ paradox)"
        )

    return rows

def kuramoto_jacobian(theta: np.ndarray, W: np.ndarray, K: float) -> np.ndarray:
    """
    Jacobian J of the Kuramoto steady-state dynamics at a fixed point theta.
    dtheta_i/dt = omega_i + K * sum_j W_ij * sin(theta_j - theta_i)
    J_ij = ∂(dtheta_i/dt)/∂theta_j.
    """
    n = len(theta)
    J = np.zeros((n, n), dtype=float)

    # Off-diagonal: K * W_ij * cos(theta_j - theta_i)
    for i in range(n):
        ti = theta[i]
        for j in range(n):
            wij = W[i, j]
            if i != j and wij != 0.0:
                J[i, j] = K * wij * np.cos(theta[j] - ti)

    # Diagonal: -sum_{j≠i} off-diagonals (graph Laplacian structure)
    for i in range(n):
        J[i, i] = -np.sum(J[i, :])  # ensures row sums ~ 0 (gauge mode)

    # Numerical symmetrization for stability (should already be symmetric)
    J = 0.5 * (J + J.T)
    return J


def linear_stability(theta: np.ndarray, W: np.ndarray, K: float,
                     drop_zero_mode: bool = True,
                     zero_tol: float = 1e-9) -> dict:
    """
    Compute eigen-spectrum of Jacobian at (theta, W, K).
    Returns:
      {
        'eigvals': np.ndarray (sorted ascending),
        'max_real': float (largest real part excluding gauge mode if drop_zero_mode),
        'num_unstable': int (# eigenvalues with real part > 0 excluding gauge mode)
      }
    """
    J = kuramoto_jacobian(theta, W, K)
    # J is symmetric -> real spectrum
    eigvals = np.linalg.eigvalsh(J)  # ascending
    ev = eigvals.copy()

    if drop_zero_mode and ev.size > 0:
        # remove the eigenvalue closest to 0 (gauge mode from rotational symmetry)
        idx0 = int(np.argmin(np.abs(ev)))
        ev = np.delete(ev, idx0)

    if ev.size == 0:
        max_real = 0.0
        num_unstable = 0
    else:
        max_real = float(np.max(ev))
        num_unstable = int(np.sum(ev > zero_tol))

    return {
        "eigvals": eigvals,         # full spectrum incl. gauge mode
        "max_real": max_real,       # gauge mode excluded
        "num_unstable": num_unstable
    }

def sweep_linear_stability(
    omega: np.ndarray,
    W: np.ndarray,
    K_values: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    return_thetas: bool = False,
    include_eigs: bool = False,
) -> dict:
    """
    For each K in K_values (any order), lock a fixed point and compute stability.
    Returns a dict with arrays sorted by ascending K:
      {
        'K': np.ndarray,
        'max_real': np.ndarray,
        'num_unstable': np.ndarray,
        'theta_list': Optional[List[np.ndarray]],
        'eigvals_list': Optional[List[np.ndarray]],
      }
    Notes:
      - Internally follows a descending-K warm start (robust), then reorders to ascending K.
      - Works even if your baseline continuation didn’t store thetas.
    """
    if rng is None:
        rng = np.random.default_rng()
    K_values = np.asarray(K_values, dtype=float)
    if K_values.ndim != 1 or K_values.size == 0:
        return {"K": np.array([]), "max_real": np.array([]), "num_unstable": np.array([])}

    # Work in co-rotating frame to ensure solvability
    omega_c = _center_omega(omega)

    # Follow descending K for robust warm starts
    K_desc = np.sort(K_values)[::-1]
    out = []
    theta = None

    # Lock first K
    K0 = float(K_desc[0])
    theta, ok, _ = solve_locked(omega_c, W, K0, theta0=None, retries=3, rng=rng)
    if not ok:
        theta0 = _laplacian_initial_guess(omega_c, W, K0)
        theta, ok, _ = solve_locked(omega_c, W, K0, theta0=theta0, retries=3, rng=rng)
        if not ok:
            # Give up gracefully: empty result
            return {"K": np.array([]), "max_real": np.array([]), "num_unstable": np.array([])}

    # Walk the rest
    for K in K_desc:
        theta, ok, _ = solve_locked(omega_c, W, K, theta0=theta, retries=2, rng=rng)
        if not ok:
            # try a fresh linearized seed at this K
            theta0 = _laplacian_initial_guess(omega_c, W, K)
            theta, ok, _ = solve_locked(omega_c, W, K, theta0=theta0, retries=2, rng=rng)
        if not ok:
            # No fixed point found: record NaNs for this K
            entry = {"K": float(K), "max_real": np.nan, "num_unstable": np.nan}
            if return_thetas: entry["theta"] = None
            if include_eigs: entry["eigvals"] = None
            out.append(entry)
            continue

        stab = linear_stability(theta, W, K, drop_zero_mode=True)
        entry = {
            "K": float(K),
            "max_real": stab["max_real"],
            "num_unstable": stab["num_unstable"],
        }
        if return_thetas:
            entry["theta"] = theta.copy()
        if include_eigs:
            entry["eigvals"] = stab["eigvals"].copy()
        out.append(entry)

    # Reorder to ascending K
    out_sorted = sorted(out, key=lambda d: d["K"])
    K_asc = np.array([d["K"] for d in out_sorted], dtype=float)
    max_real = np.array([d["max_real"] for d in out_sorted], dtype=float)
    num_unst = np.array([d["num_unstable"] for d in out_sorted], dtype=float)

    result = {"K": K_asc, "max_real": max_real, "num_unstable": num_unst}
    if return_thetas:
        result["theta_list"] = [d.get("theta", None) for d in out_sorted]
    if include_eigs:
        result["eigvals_list"] = [d.get("eigvals", None) for d in out_sorted]
    return result


def plot_linear_stability(
    K: np.ndarray,
    max_real: np.ndarray,
    out_path: str,
    bifurcations: Optional[List[dict]] = None,
    title: str = "Linear stability: max Re(λ(J)) vs K"
):
    """Plot max real-part eigenvalue vs K and mark detected bifurcations."""
    if K is None or len(K) == 0:
        return
    _ensure_dir(os.path.dirname(out_path) or ".")
    plt.figure(figsize=(8, 6))
    plt.plot(K, max_real, "o-", linewidth=2, markersize=4, label="max Re(λ(J))")
    plt.axhline(0.0, color="r", linestyle="--", alpha=0.5, label="Re(λ)=0")
    if bifurcations:
        for ev in bifurcations:
            Kb = ev.get("K_bif", None)
            if Kb is not None:
                plt.axvline(Kb, color="orange", linestyle=":", alpha=0.7)
    plt.xlabel("K")
    plt.ylabel("max Re(λ(J))")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_stability_csv(K: np.ndarray, max_real: np.ndarray, num_unstable: np.ndarray, out_path: str):
    """Save stability sweep as CSV with columns K, max_real, num_unstable."""
    import pandas as pd
    _ensure_dir(os.path.dirname(out_path) or ".")
    pd.DataFrame({
        "K": np.asarray(K, dtype=float),
        "max_real": np.asarray(max_real, dtype=float),
        "num_unstable": np.asarray(num_unstable, dtype=float),
    }).to_csv(out_path, index=False)

def detect_bifurcations_from_stability(
    K: np.ndarray,
    max_real: np.ndarray,
    tol: float = 1e-6
) -> List[dict]:
    """
    Detect bifurcation candidates as zero-crossings of the largest real part.
    Returns a list of events:
      [{'K_bif': float, 'type': 'eig_zero_cross', 'left': (K_i, m_i), 'right': (K_{i+1}, m_{i+1})}, ...]
    """
    K = np.asarray(K, dtype=float)
    mr = np.asarray(max_real, dtype=float)
    out = []
    if K.size < 2:
        return out

    for i in range(len(K) - 1):
        a, b = mr[i], mr[i + 1]
        if np.isnan(a) or np.isnan(b):
            continue
        if (a > tol and b < -tol) or (a < -tol and b > tol) or (abs(a) <= tol) or (abs(b) <= tol):
            # Linear interpolation for crossing if denom != 0
            if (b - a) != 0:
                t = (0.0 - a) / (b - a)
                t = np.clip(t, 0.0, 1.0)
                Kb = float(K[i] + t * (K[i + 1] - K[i]))
            else:
                Kb = float(0.5 * (K[i] + K[i + 1]))
            out.append({
                "K_bif": Kb,
                "type": "eig_zero_cross",
                "left": (float(K[i]), float(a)),
                "right": (float(K[i + 1]), float(b)),
            })
    return out

def save_eigs_long_csv(stab: dict, out_path: str):
    """
    Write eigenvalues for each K to a long CSV with columns (K, eig_index, eigval).
    Requires sweep_linear_stability(..., include_eigs=True).
    """
    import pandas as pd, os
    K = np.asarray(stab.get("K", []), dtype=float)
    eigs_list = stab.get("eigvals_list", None)
    if eigs_list is None or len(eigs_list) != len(K):
        return
    rows = []
    for Ki, ev in zip(K, eigs_list):
        if ev is None:
            continue
        ev = np.asarray(ev, dtype=float)
        for j, val in enumerate(ev):
            rows.append((float(Ki), int(j), float(val)))
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pd.DataFrame(rows, columns=["K", "eig_index", "eigval"]).to_csv(out_path, index=False)


# Random Geometrix Graph 


def random_geometric_graph_2d(
    n: int,
    radius: float = 20.0,
    box_size: float = 100.0,
    weight_low: float = 1.0,
    weight_high: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    return_pos: bool = False,
) -> Tuple[np.ndarray, Dict[int, Tuple[float, float]]]:


    if rng is None:
        rng = np.random.default_rng()

    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")

    # clamp radius (interpreting user requirement "between 1 and 99")
    radius = float(radius)
    radius = max(1.0, min(99.0, radius))

    # sample positions
    xs = rng.uniform(0.0, box_size, size=n)
    ys = rng.uniform(0.0, box_size, size=n)
    pos = {i: (float(xs[i]), float(ys[i])) for i in range(n)}

    # initialize empty weight matrix
    W = np.zeros((n, n), dtype=float)

    # helper: squared distance to avoid extra sqrt in comparisons
    def dist2(i: int, j: int) -> float:
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        return dx * dx + dy * dy

    r2 = radius * radius

    # add edges for pairs within radius
    for i in range(n):
        for j in range(i + 1, n):
            if dist2(i, j) <= r2:
                w = float(rng.uniform(weight_low, weight_high))
                W[i, j] = w
                W[j, i] = w

    # --- connectivity repair step 1: fix isolated nodes (degree 0) ---
    # if a node has no neighbors, connect it to its nearest neighbor
    deg = (W > 0).sum(axis=1)
    isolated = np.where(deg == 0)[0]
    for i in isolated:
        # find nearest j != i
        j_best = None
        d_best = float("inf")
        for j in range(n):
            if i == j:
                continue
            d = dist2(i, j)
            if d < d_best:
                d_best = d
                j_best = j
        if j_best is not None:
            w = float(rng.uniform(weight_low, weight_high))
            W[i, j_best] = w
            W[j_best, i] = w

    # --- connectivity repair step 2: connect components greedily ---
    G = nx.from_numpy_array(W)
    if not nx.is_connected(G):
        # While there are multiple components, add the shortest cross-component edge
        comps = [list(c) for c in nx.connected_components(G)]
        while len(comps) > 1:
            # pick the globally closest pair across ANY two components
            best_pair = None
            best_d = float("inf")
            # (Optionally, you could connect to the largest component each time;
            # here we choose the globally closest across all pairs for a shorter total wire length.)
            for a_idx in range(len(comps)):
                A = comps[a_idx]
                for b_idx in range(a_idx + 1, len(comps)):
                    B = comps[b_idx]
                    for i in A:
                        for j in B:
                            d = dist2(i, j)
                            if d < best_d:
                                best_d = d
                                best_pair = (i, j)
            # add that bridge
            i, j = best_pair
            w = float(rng.uniform(weight_low, weight_high))
            W[i, j] = w
            W[j, i] = w

            # update components
            G = nx.from_numpy_array(W)
            comps = [list(c) for c in nx.connected_components(G)]

    if return_pos:
        return W, pos
    return W


def draw_geometric_network_graph(
    W: np.ndarray,
    omega: np.ndarray,
    pos: Dict[int, Tuple[float, float]],
    path: str,
    title: str = "Random Geometric Network (2D)",
    highlight_edge: Optional[Tuple[int, int]] = None,
    highlight_color: str = "orange",
):
    # Plot the geometric graph using given 2D positions (no layout distortion).

    _ensure_dir(os.path.dirname(path) or ".")
    n = W.shape[0]
    G = nx.from_numpy_array(W)

    # ensure pos has all nodes; if not, fall back gracefully
    if not all(i in pos for i in range(n)):
        pos = {i: pos.get(i, (float(i), 0.0)) for i in range(n)}  # minimal fallback

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5)
    colors = ["red" if float(w) > 0.0 else "blue" for w in omega]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=300, alpha=0.9, linewidths=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)

    if highlight_edge is not None:
        u, v = highlight_edge
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=3.0, edge_color=highlight_color)

    plt.title(f"{title}\nRed: ω>0, Blue: ω<0")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


