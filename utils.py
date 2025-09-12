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
    Gauge-fix θ0 = 0. Solve for θ1..θ(n-1) using equations for i=1..n-1:
      0 = ω_i + K * sum_j W_ij * sin(θ_j - θ_i)
    """
    n = len(omega)
    th = np.zeros(n, dtype=float)
    th[1:] = theta_red

    res = np.empty(n-1, dtype=float)
    for i in range(1, n):
        Wi = W[i]
        ti = th[i]
        s = 0.0
        for j in range(n):
            w = Wi[j]
            if w != 0.0:
                s += w * np.sin(th[j] - ti)
        res[i-1] = omega[i] + K * s
    return res


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
        th0_red = np.zeros(n-1, dtype=float)
    else:
        th0 = np.asarray(theta0, dtype=float).reshape(-1)
        th0_red = th0[1:] if len(th0) == n else np.zeros(n-1, dtype=float)

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
    recompute Kc, and SAVE per-edge r(K) plots + network snapshots if outdir is given.
    Returns result rows with uniform keys.
    """
    if rng is None:
        rng = np.random.default_rng()
    if outdir is not None:
        _ensure_dir(outdir)

    # Baseline
    omega_c = _center_omega(omega)
    base_Kc, base_Ks, base_Rs = continuation_descend_K(
        omega_c, W, K_start=K_start, K_min=K_min, K_step=K_step, r_thresh=r_thresh, rng=rng
    )
    if outdir is not None:
        plot_r_vs_K(base_Ks, base_Rs, os.path.join(outdir, "r_vs_K_baseline.png"),
                    Kc=base_Kc, title="Baseline r(K)", r_threshold=r_thresh)
        draw_network_graph(W, omega, os.path.join(outdir, "network_baseline.png"),
                           layout=layout, title="Baseline network")

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
            plot_r_vs_K(Ks2, Rs2, os.path.join(outdir, f"r_vs_K_remove_{i}_{j}.png"),
                        Kc=Kc2, title=f"Remove edge {e}: r(K)", r_threshold=r_thresh)
            draw_network_graph(W2, omega, os.path.join(outdir, f"network_remove_{i}_{j}.png"),
                               layout=layout, title=f"Network after removing {e}",
                               highlight_edge=None)

    if outdir is not None:
        plot_edge_deltas(rows, os.path.join(outdir, "edge_deltas_remove.png"),
                         title="ΔKc for removed edges")

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
    + network snapshots if outdir is given. Baseline is the original W.
    """
    if rng is None:
        rng = np.random.default_rng()
    if outdir is not None:
        _ensure_dir(outdir)

    # Baseline
    omega_c = _center_omega(omega)
    base_Kc, base_Ks, base_Rs = continuation_descend_K(
        omega_c, W, K_start=K_start, K_min=K_min, K_step=K_step, r_thresh=r_thresh, rng=rng
    )
    if outdir is not None:
        # If remove+add are both called, baseline file may already exist; it's fine to overwrite.
        plot_r_vs_K(base_Ks, base_Rs, os.path.join(outdir, "r_vs_K_baseline.png"),
                    Kc=base_Kc, title="Baseline r(K)", r_threshold=r_thresh)
        draw_network_graph(W, omega, os.path.join(outdir, "network_baseline.png"),
                           layout=layout, title="Baseline network")

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
        # Here, "Braess" in the classical sense is delta > 0 (adding an edge harms synchronizability)
        is_braess = (delta is not None) and (delta > 1e-3)

        rows.append({
            'edge': (i, j),
            'baseline_Kc': float(base_Kc),
            'new_Kc': None if Kc2 is None else float(Kc2),
            'delta': delta,
            'is_braess': bool(is_braess),
        })

        if outdir is not None:
            plot_r_vs_K(Ks2, Rs2, os.path.join(outdir, f"r_vs_K_add_{i}_{j}.png"),
                        Kc=Kc2, title=f"Add edge {(i,j)}: r(K)", r_threshold=r_thresh)
            draw_network_graph(W2, omega, os.path.join(outdir, f"network_add_{i}_{j}.png"),
                               layout=layout, title=f"Network after adding {(i,j)}",
                               highlight_edge=(i, j), highlight_color="orange")

    if outdir is not None:
        plot_edge_deltas(rows, os.path.join(outdir, "edge_deltas_add.png"),
                         title="ΔKc for added edges (positive ⇒ paradox)")

    return rows
