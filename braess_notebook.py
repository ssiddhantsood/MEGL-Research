#!/usr/bin/env python3
"""
Braess Paradox Analysis - Clean Notebook Version

Core functionality:
1. Define a hardcoded network (W, omega, K parameters)
2. Compute baseline Kc using continuation method
3. For each missing edge: add it and compute new Kc
4. Use g.py predictor to predict Braess paradox
5. Compare predictions vs actual results
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from typing import Optional, Tuple, List, Dict
import networkx as nx
import os
import sys
import csv

# =============================================================================
# NETWORK DEFINITION - EDIT THIS SECTION
# =============================================================================

# Number of nodes
n = 12

# Adjacency matrix (symmetric, weighted)
# Explicitly define a connected 12-node network
W = np.array([
    [0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
    [1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
    [0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0.],
    [0., 0., 0., 1., 1., 0., 1., 0., 1., 0., 0., 0.],
    [1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0.],
    [0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
    [0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1.],
    [0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.]
])

# Natural frequencies for n=12, example values
omega = np.array([0.2, -0.3, 1.2, -0.8, 0.4, -0.6, 0.9, -0.2, 0.7, -1.0, 0.5, -0.7])

# K sweep parameters
K_start = 3.0    # Start at high coupling
K_min = 0.01      # End at low coupling
K_step = 0.001    # Step size for K

# Onset detection parameters
onset_eps = 0.05      # Threshold for r to be considered "synchronized"
onset_minrun = 3      # Number of consecutive increasing points needed

# Edge addition weight
add_weight = 1.0

# Output directory for saved plots and results
output_dir = "braess_output"

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def center_omega(omega: np.ndarray) -> np.ndarray:
    """Center frequencies to co-rotating frame."""
    return omega - omega.mean()

def order_parameter(theta: np.ndarray) -> float:
    """Compute Kuramoto order parameter r ∈ [0,1]."""
    z = np.exp(1j * theta).mean()
    return abs(z)

def kuramoto_jacobian(theta: np.ndarray, W: np.ndarray, K: float) -> np.ndarray:
    """
    Compute Jacobian J of the Kuramoto steady-state dynamics at fixed point theta.
    J_ij = ∂(dtheta_i/dt)/∂theta_j
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
        J[i, i] = -np.sum(J[i, :])
    
    return J

def compute_eigenvalues(theta: np.ndarray, W: np.ndarray, K: float) -> Dict:
    """
    Compute eigenvalue spectrum of Jacobian at (theta, W, K).
    Returns dict with eigenvalues and stability info.
    """
    J = kuramoto_jacobian(theta, W, K)
    eigvals = np.linalg.eigvalsh(J)  # ascending, real (symmetric matrix)
    
    # Remove gauge mode (eigenvalue closest to 0)
    ev_no_gauge = eigvals.copy()
    if len(ev_no_gauge) > 0:
        idx0 = int(np.argmin(np.abs(ev_no_gauge)))
        ev_no_gauge = np.delete(ev_no_gauge, idx0)
    
    max_real = float(np.max(ev_no_gauge)) if len(ev_no_gauge) > 0 else 0.0
    num_unstable = int(np.sum(ev_no_gauge > 1e-9)) if len(ev_no_gauge) > 0 else 0
    
    return {
        'eigvals': eigvals,  # full spectrum including gauge mode
        'eigvals_no_gauge': ev_no_gauge,  # without gauge mode
        'max_real': max_real,
        'num_unstable': num_unstable,
        'min_eigval': float(np.min(eigvals)),
        'max_eigval': float(np.max(eigvals))
    }

def residual_gauge(theta_red: np.ndarray, omega: np.ndarray, W: np.ndarray, K: float) -> np.ndarray:
    """
    Gauge-fixed Kuramoto residual (θ₀ = 0).
    Solves: 0 = ωᵢ + K * Σⱼ Wᵢⱼ sin(θⱼ - θᵢ)
    """
    n = len(omega)
    theta = np.zeros(n)
    theta[1:] = theta_red
    
    dtheta = theta[np.newaxis, :] - theta[:, np.newaxis]
    s = (W * np.sin(dtheta)).sum(axis=1)
    res_full = omega + K * s
    
    return res_full[1:]

def solve_locked(omega: np.ndarray, W: np.ndarray, K: float, 
                 theta0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
    """Solve for locked state at given K."""
    n = len(omega)
    
    if theta0 is None:
        th0_red = np.zeros(n-1)
    else:
        th0_red = theta0[1:] if len(theta0) == n else np.zeros(n-1)
    
    th_red, info, ier, msg = fsolve(
        residual_gauge, th0_red, args=(omega, W, K),
        full_output=True, xtol=1e-9, maxfev=4000
    )
    
    th_full = np.zeros(n)
    th_full[1:] = th_red
    
    # Check residual
    res = np.zeros(n)
    for i in range(n):
        s = sum(W[i,j] * np.sin(th_full[j] - th_full[i]) 
                for j in range(n) if W[i,j] != 0)
        res[i] = omega[i] + K * s
    
    success = (ier == 1) and (np.linalg.norm(res) < 1e-6)
    return th_full, success

def continuation_curve(omega_c: np.ndarray, W: np.ndarray, 
                       K_start: float, K_min: float, K_step: float,
                       theta0_high: Optional[np.ndarray] = None,
                       capture_full_data: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict]]]:
    """
    Compute r(K) by descending from high K to low K.
    
    Args:
        omega_c: Pre-centered natural frequencies (must be centered)
        W: Adjacency matrix
        K_start: Starting coupling strength (high K)
        K_min: Minimum coupling strength
        K_step: Step size for K descent
        theta0_high: Optional warm start state at high K (for branch tracking)
        capture_full_data: If True, capture all intermediate data (theta, eigenvalues, etc.)
    
    Returns: 
        - (K_values, r_values) both in ascending order
        - If capture_full_data=True, also returns list of dicts with full data
    """
    K = K_start
    
    # Lock at high K, using warm start if provided
    theta, ok = solve_locked(omega_c, W, K, theta0=theta0_high)
    if not ok:
        return np.array([]), np.array([]), [] if capture_full_data else None
    
    # Walk downward
    Ks, Rs = [], []
    full_data = [] if capture_full_data else None
    
    while K >= K_min:
        theta, ok = solve_locked(omega_c, W, K, theta0=theta)
        r = order_parameter(theta) if ok else 0.0
        Ks.append(K)
        Rs.append(r)
        
        # Capture full data if requested
        if capture_full_data and ok:
            eig_data = compute_eigenvalues(theta, W, K)
            data_point = {
                'K': float(K),
                'r': float(r),
                'theta': theta.copy(),
                'omega': omega_c.copy(),
                'eigvals': eig_data['eigvals'].copy(),
                'eigvals_no_gauge': eig_data['eigvals_no_gauge'].copy(),
                'max_real': eig_data['max_real'],
                'num_unstable': eig_data['num_unstable'],
                'min_eigval': eig_data['min_eigval'],
                'max_eigval': eig_data['max_eigval'],
                'converged': True
            }
            full_data.append(data_point)
        elif capture_full_data:
            # Failed to converge
            data_point = {
                'K': float(K),
                'r': 0.0,
                'theta': np.full(len(omega_c), np.nan),
                'omega': omega_c.copy(),
                'eigvals': np.full(len(omega_c), np.nan),
                'eigvals_no_gauge': np.full(len(omega_c)-1, np.nan),
                'max_real': np.nan,
                'num_unstable': np.nan,
                'min_eigval': np.nan,
                'max_eigval': np.nan,
                'converged': False
            }
            full_data.append(data_point)
        
        K -= K_step
        
        # Early stop if clearly desynchronized
        if len(Rs) >= 3 and Rs[-1] < 0.05 and Rs[-2] < 0.05:
            break
    
    # Reverse to ascending order
    Ks_reversed = np.array(Ks[::-1])
    Rs_reversed = np.array(Rs[::-1])
    
    if capture_full_data:
        full_data_reversed = full_data[::-1]  # Reverse to match ascending K order
        return Ks_reversed, Rs_reversed, full_data_reversed
    
    return Ks_reversed, Rs_reversed, None

def kc_first_onset(Ks: np.ndarray, Rs: np.ndarray, 
                   eps: float = 0.05, min_run: int = 3) -> Optional[float]:
    """
    Find first K where r crosses eps and stays increasing for min_run points.
    This is our definition of the critical coupling Kc.
    """
    n = len(Rs)
    for i in range(n):
        if Rs[i] > eps:
            j_end = min(n, i + min_run)
            if j_end - i < min_run:
                return None
            # Check if r is non-decreasing
            if np.all(np.diff(Rs[i:j_end]) >= -1e-10):
                return float(Ks[i])
    return None

def missing_edges(W: np.ndarray) -> List[Tuple[int, int]]:
    """Find all edges not in the graph."""
    n = W.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] == 0.0:
                edges.append((i, j))
    return edges

# =============================================================================
# G.PY PREDICTOR FUNCTIONS
# =============================================================================

def compute_g_theta(theta: np.ndarray, edge: Tuple[int, int]) -> np.ndarray:
    """
    Compute g(θ) = effect of adding edge.
    g(θ) is only non-zero at nodes i and j.
    """
    n = len(theta)
    g = np.zeros(n)
    i, j = edge
    g[i] = np.sin(theta[j] - theta[i])
    g[j] = np.sin(theta[i] - theta[j])
    return g

def compute_s_theta(theta: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Compute s(θ) = Σⱼ Wᵢⱼ sin(θⱼ - θᵢ)
    """
    dth = theta[np.newaxis, :] - theta[:, np.newaxis]
    return (W * np.sin(dth)).sum(axis=1)

def predict_braess_g(theta_0: np.ndarray, omega: np.ndarray, W: np.ndarray,
                     K_0: float, edge: Tuple[int, int],
                     omega_c: Optional[np.ndarray] = None) -> Tuple[bool, float]:
    """
    Predict Braess paradox using the g.py derivative formula.
    
    Returns k'(0) = -v₀ᵀ·g(θ₀) / [v₀ᵀ·s(θ₀)]
    
    For our purposes, we approximate v₀ ≈ ω (centered).
    If k'(0) > 0: predict Braess (adding edge increases Kc)
    
    Uses pre-centered omega_c if provided, otherwise centers omega.
    """
    if omega_c is None:
        omega_c = center_omega(omega)
    
    g_theta = compute_g_theta(theta_0, edge)
    s_theta = compute_s_theta(theta_0, W)
    
    # eig_data = compute_eigenvalues(theta_0, W, K_0)
    # # Compute v0: eigenvector associated with smallest eigenvalue (no gauge mode)
    J = kuramoto_jacobian(theta_0, W, K_0)
    # eigvals, eigvecs = np.linalg.eigh(J)
    # eigvals_no_gauge = eig_data['eigvals_no_gauge']
    # eigvals_full = eig_data['eigvals']
    # idx_gauge = int(np.argmin(np.abs(eigvals)))
    # other_idxs = [i for i in range(len(eigvals)) if i != idx_gauge]
    # min_idx_no_gauge = other_idxs[np.argmin(eigvals[other_idxs])]
    # v_0 = eigvecs[:, min_idx_no_gauge]

    vals, vecs = np.linalg.eigh(J)

    second_largest_eigenvalue = vals[-2]
    v_0 = vecs[:, -2]
    
    
    numerator = -np.dot(v_0, g_theta)
    denominator = np.dot(v_0, s_theta)
    
    if abs(denominator) < 1e-10:
        k_prime = 0.0
    else:
        k_prime = numerator / denominator
    
    pred_braess = (k_prime > 0)
    return pred_braess, k_prime

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_r_vs_K(Ks: np.ndarray, Rs: np.ndarray, Kc: Optional[float] = None, 
                title: str = "r(K) curve", output_path: str = None):
    """Plot order parameter vs coupling strength with proper scaling and save to file."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(Ks, Rs, 'o-', linewidth=2, markersize=4, label='r(K)')
    
    if Kc is not None:
        ax.axvline(Kc, linestyle='--', color='red', alpha=0.7, 
                   label=f'Kc ≈ {Kc:.3f}')
    
    ax.axhline(onset_eps, linestyle=':', color='gray', alpha=0.5, 
               label=f'onset threshold ({onset_eps})')
    
    ax.set_xlabel('Coupling strength K', fontsize=12)
    ax.set_ylabel('Order parameter r', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(K_min, K_start)
    
    # Set explicit x-axis ticks at 0, 1, 2, 3, 4, 5 for clarity
    x_ticks = np.arange(0, 6, 1)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{x:.0f}' for x in x_ticks])
    
    # Set explicit y-axis limits and ticks for proper scaling
    ax.set_ylim(0.0, 1.0)
    y_ticks = np.arange(0.0, 1.1, 0.1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y:.1f}' for y in y_ticks])
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_network(W: np.ndarray, title: str = "Network", 
                highlight_edge: Optional[Tuple[int, int]] = None,
                output_path: str = None):
    """Visualize the network graph and save to file."""
    G = nx.from_numpy_array(W)
    pos = nx.spring_layout(G, seed=42)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=2, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', 
                          edgecolors='black', linewidths=2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    
    if highlight_edge is not None:
        nx.draw_networkx_edges(G, pos, edgelist=[highlight_edge], 
                              width=4, edge_color='red', ax=ax)
    
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_overlay(Kb: np.ndarray, Rb: np.ndarray, Kn: np.ndarray, Rn: np.ndarray,
                Kc_base: Optional[float], Kc_new: Optional[float], 
                edge: Tuple[int, int], output_path: str = None):
    """Plot baseline vs modified r(K) curves with proper scaling and save to file."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(Kb, Rb, 'o-', linewidth=2, markersize=4, label='Baseline')
    ax.plot(Kn, Rn, 's--', linewidth=2, markersize=4, 
            label=f'Added edge {edge}')
    
    if Kc_base is not None:
        ax.axvline(Kc_base, linestyle='--', color='blue', alpha=0.6, 
                   label=f'Baseline Kc ≈ {Kc_base:.3f}')
    if Kc_new is not None:
        ax.axvline(Kc_new, linestyle=':', color='orange', alpha=0.7, 
                   label=f'New Kc ≈ {Kc_new:.3f}')
    
    ax.set_xlabel('Coupling strength K', fontsize=12)
    ax.set_ylabel('Order parameter r', fontsize=12)
    ax.set_title(f'Baseline vs Edge {edge} Added', fontsize=14)
    ax.set_xlim(K_min, K_start)
    
    # Set explicit x-axis ticks at 0, 1, 2, 3, 4, 5 for clarity
    x_ticks = np.arange(0, 6, 1)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{x:.0f}' for x in x_ticks])
    
    # Set explicit y-axis limits and ticks for proper scaling
    ax.set_ylim(0.0, 1.0)
    y_ticks = np.arange(0.0, 1.1, 0.1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y:.1f}' for y in y_ticks])
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_scatter_kprime_vs_delta(results: List[dict], output_path: str = None):
    """Scatter plot: k'(0) vs delta, colored by Braess status."""
    # Filter to edges with valid delta and k_prime
    valid_results = [r for r in results if r['k_prime'] is not None and r['delta'] is not None]
    
    if not valid_results:
        print("      Warning: No valid data points for scatter plot (k'(0) vs delta)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate Braess and non-Braess edges
    braess_edges = [r for r in valid_results if r['is_braess_actual']]
    non_braess_edges = [r for r in valid_results if not r['is_braess_actual']]
    
    # Plot non-Braess edges
    if non_braess_edges:
        delta_nb = [r['delta'] for r in non_braess_edges]
        kprime_nb = [r['k_prime'] for r in non_braess_edges]
        ax.scatter(delta_nb, kprime_nb, s=50, alpha=0.6, label='Non-Braess', color='blue')
    
    # Plot Braess edges
    if braess_edges:
        delta_b = [r['delta'] for r in braess_edges]
        kprime_b = [r['k_prime'] for r in braess_edges]
        ax.scatter(delta_b, kprime_b, s=100, alpha=0.8, label='Braess', color='red', marker='*', edgecolors='black', linewidths=1)
    
    # Add reference lines
    ax.axhline(0.0, linestyle='--', linewidth=1.0, color='gray', alpha=0.5)
    ax.axvline(0.0, linestyle='--', linewidth=1.0, color='gray', alpha=0.5)
    
    ax.set_xlabel('Delta (Kc_new - Kc_base)', fontsize=12)
    ax.set_ylabel("k'(0) (derivative)", fontsize=12)
    ax.set_title("k'(0) vs Delta: Braess Edge Analysis", fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_scatter_kprime_vs_baselineKc(results: List[dict], output_path: str = None):
    """Scatter plot: k'(0) vs baseline_Kc, colored by Braess status."""
    # Filter to edges with valid Kc_base and k_prime
    valid_results = [r for r in results if r['k_prime'] is not None and r['Kc_base'] is not None]
    
    if not valid_results:
        print("      Warning: No valid data points for scatter plot (k'(0) vs baseline_Kc)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate Braess and non-Braess edges
    braess_edges = [r for r in valid_results if r['is_braess_actual']]
    non_braess_edges = [r for r in valid_results if not r['is_braess_actual']]
    
    # Plot non-Braess edges
    if non_braess_edges:
        kc_nb = [r['Kc_base'] for r in non_braess_edges]
        kprime_nb = [r['k_prime'] for r in non_braess_edges]
        ax.scatter(kc_nb, kprime_nb, s=50, alpha=0.6, label='Non-Braess', color='blue')
    
    # Plot Braess edges
    if braess_edges:
        kc_b = [r['Kc_base'] for r in braess_edges]
        kprime_b = [r['k_prime'] for r in braess_edges]
        ax.scatter(kc_b, kprime_b, s=100, alpha=0.8, label='Braess', color='red', marker='*', edgecolors='black', linewidths=1)
    
    # Add reference line
    ax.axhline(0.0, linestyle='--', linewidth=1.0, color='gray', alpha=0.5)
    
    ax.set_xlabel('Baseline Kc', fontsize=12)
    ax.set_ylabel("k'(0) (derivative)", fontsize=12)
    ax.set_title("k'(0) vs Baseline Kc: Braess Edge Analysis", fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# CSV DATA EXPORT FUNCTIONS
# =============================================================================

def save_continuation_data_to_csv(full_data: List[Dict], output_path: str, run_name: str, 
                                  omega_original: np.ndarray, W: np.ndarray):
    """
    Save all continuation data to CSV files.
    
    Creates multiple CSV files:
    1. Main continuation data (K, r, theta values, eigenvalues summary)
    2. Detailed eigenvalues (all eigenvalues at each K)
    3. Theta values (all phase angles at each K)
    4. Omega values (natural frequencies - same for all K)
    """
    if not full_data:
        print(f"      Warning: No data to save for {run_name}")
        return
    
    n = len(omega_original)
    
    # 1. Main continuation CSV: K, r, theta values, eigenvalue summary
    main_csv_path = os.path.join(output_path, f"{run_name}_continuation_main.csv")
    with open(main_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: K, r, theta_0, theta_1, ..., theta_n-1, min_eigval, max_eigval, max_real, num_unstable, converged
        header = ['K', 'r', 'converged', 'min_eigval', 'max_eigval', 'max_real', 'num_unstable']
        header.extend([f'theta_{i}' for i in range(n)])
        writer.writerow(header)
        
        for data_point in full_data:
            row = [
                data_point['K'],
                data_point['r'],
                data_point['converged'],
                data_point['min_eigval'],
                data_point['max_eigval'],
                data_point['max_real'],
                data_point['num_unstable']
            ]
            row.extend(data_point['theta'].tolist())
            writer.writerow(row)
    
    # 2. Detailed eigenvalues CSV: K, r, all eigenvalues
    eig_csv_path = os.path.join(output_path, f"{run_name}_eigenvalues.csv")
    with open(eig_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: K, r, eigval_0, eigval_1, ..., eigval_n-1 (full spectrum)
        header = ['K', 'r']
        header.extend([f'eigval_{i}' for i in range(n)])
        writer.writerow(header)
        
        for data_point in full_data:
            row = [data_point['K'], data_point['r']]
            row.extend(data_point['eigvals'].tolist())
            writer.writerow(row)
    
    # 3. Eigenvalues without gauge mode CSV
    eig_no_gauge_csv_path = os.path.join(output_path, f"{run_name}_eigenvalues_no_gauge.csv")
    with open(eig_no_gauge_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: K, r, eigval_0, ..., eigval_n-2 (no gauge mode)
        header = ['K', 'r']
        header.extend([f'eigval_no_gauge_{i}' for i in range(n-1)])
        writer.writerow(header)
        
        for data_point in full_data:
            row = [data_point['K'], data_point['r']]
            eigvals_ng = data_point['eigvals_no_gauge']
            row.extend(eigvals_ng.tolist())
            # Pad with NaN if needed (shouldn't happen, but safety check)
            while len(row) < len(header):
                row.append(np.nan)
            writer.writerow(row)
    
    # 4. Theta values CSV (already in main, but separate for convenience)
    theta_csv_path = os.path.join(output_path, f"{run_name}_theta_values.csv")
    with open(theta_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['K', 'r']
        header.extend([f'theta_{i}' for i in range(n)])
        writer.writerow(header)
        
        for data_point in full_data:
            row = [data_point['K'], data_point['r']]
            row.extend(data_point['theta'].tolist())
            writer.writerow(row)
    
    # 5. Omega values CSV (same for all K, but include for completeness)
    omega_csv_path = os.path.join(output_path, f"{run_name}_omega_values.csv")
    with open(omega_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['node_index', 'omega_original', 'omega_centered'])
        omega_c = center_omega(omega_original)
        for i in range(n):
            writer.writerow([i, omega_original[i], omega_c[i]])
    
    # 6. Network adjacency matrix CSV
    network_csv_path = os.path.join(output_path, f"{run_name}_network_W.csv")
    with open(network_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header row with node indices
        header = [''] + [f'node_{i}' for i in range(n)]
        writer.writerow(header)
        # Data rows
        for i in range(n):
            row = [f'node_{i}'] + W[i, :].tolist()
            writer.writerow(row)
    
    print(f"      Saved CSV files for {run_name}:")
    print(f"        - {run_name}_continuation_main.csv (K, r, theta, eigenvalue summary)")
    print(f"        - {run_name}_eigenvalues.csv (all eigenvalues)")
    print(f"        - {run_name}_eigenvalues_no_gauge.csv (eigenvalues without gauge mode)")
    print(f"        - {run_name}_theta_values.csv (phase angles)")
    print(f"        - {run_name}_omega_values.csv (natural frequencies)")
    print(f"        - {run_name}_network_W.csv (adjacency matrix)")

# =============================================================================
# OUTPUT CAPTURE
# =============================================================================

class Tee:
    """Write to both stdout and a file."""
    def __init__(self, file_path):
        self.file = open(file_path, 'w')
        self.stdout = sys.stdout
    
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()

class FileOnly:
    """Write only to file, not console."""
    def __init__(self, file):
        self.file = file
    
    def write(self, text):
        self.file.write(text)
        self.file.flush()
    
    def flush(self):
        self.file.flush()

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

# Create output directory
os.makedirs(output_dir, exist_ok=True)
log_file_path = os.path.join(output_dir, "analysis_results.txt")

# Redirect all output to file only (not console)
original_stdout = sys.stdout
log_file = open(log_file_path, 'w')
file_only = FileOnly(log_file)
sys.stdout = file_only

print("=" * 70)
print("BRAESS PARADOX ANALYSIS")
print("=" * 70)

# Step 1: Compute baseline
# KEY: Center omega ONCE and reuse for all computations
print("\n1. Computing baseline r(K) curve...")
omega_c = center_omega(omega)  # Center ONCE and reuse

Kb, Rb, baseline_full_data = continuation_curve(omega_c, W, K_start, K_min, K_step, 
                                                  capture_full_data=True)
Kc_base = kc_first_onset(Kb, Rb, eps=onset_eps, min_run=onset_minrun)

# Save baseline data to CSV
print("\n   Saving baseline data to CSV files...")
save_continuation_data_to_csv(baseline_full_data, output_dir, "baseline", omega, W)

print(f"   Baseline Kc = {Kc_base:.4f}" if Kc_base else "   Failed to find baseline Kc")
print(f"   Number of K points: {len(Kb)}")

# Visualize baseline
baseline_r_path = os.path.join(output_dir, "baseline_r_vs_K.png")
baseline_network_path = os.path.join(output_dir, "baseline_network.png")
plot_r_vs_K(Kb, Rb, Kc_base, title="Baseline r(K)", output_path=baseline_r_path)
plot_network(W, title="Baseline Network", output_path=baseline_network_path)

# Step 2: Find baseline locked state at Kc and high K for warm starting
theta_base = None
theta_base_high = None
if Kc_base is not None:
    # Get locked state at Kc for g predictor
    theta_base, ok = solve_locked(omega_c, W, Kc_base)
    if not ok:
        print("   Warning: Could not lock baseline state at Kc")
        theta_base = None
    
    # Get locked state at high K for warm starting modified networks
    theta_base_high, ok_high = solve_locked(omega_c, W, K_start)
    if not ok_high:
        print("   Warning: Could not lock baseline state at high K")
        theta_base_high = None

# Step 3: Analyze each missing edge
print("\n2. Testing edge additions...")
edges_to_test = missing_edges(W)
print(f"   Found {len(edges_to_test)} missing edges to test")

results = []

for idx, (u, v) in enumerate(edges_to_test):
    print(f"\n   [{idx+1}/{len(edges_to_test)}] Testing edge ({u}, {v})...")
    
    # Add edge
    W_new = W.copy()
    W_new[u, v] = add_weight
    W_new[v, u] = add_weight
    
    # Compute new Kc using SAME omega_c and warm start from baseline
    # This ensures we track the same solution branch
    Kn, Rn, edge_full_data = continuation_curve(omega_c, W_new, K_start, K_min, K_step, 
                                                 theta0_high=theta_base_high,
                                                 capture_full_data=True)
    
    # Save edge-specific data to CSV
    edge_name = f"edge_{u}_{v}"
    print(f"      Saving data for edge ({u}, {v}) to CSV files...")
    save_continuation_data_to_csv(edge_full_data, output_dir, edge_name, omega, W_new)
    
    # Consistency check: verify we're tracking the same solution branch
    # Check order parameter at high K (should be close to baseline)
    branch_consistent = True  # Assume consistent unless check fails
    if len(Kn) > 0 and len(Kb) > 0 and theta_base_high is not None:
        # Get order parameter at high K for both baseline and modified
        r_baseline_high = Rb[-1] if len(Rb) > 0 else None
        r_modified_high = Rn[-1] if len(Rn) > 0 else None
        
        # Also check the actual locked state at high K
        theta_new_high, ok_new = solve_locked(omega_c, W_new, K_start, theta0=theta_base_high)
        if ok_new and theta_base_high is not None:
            # Check phase difference (should be small if same branch)
            phase_diff = np.abs(theta_new_high - theta_base_high)
            phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)  # Account for 2π periodicity
            max_phase_diff = np.max(phase_diff)
            
            # If phase difference is too large, might be on different branch
            if max_phase_diff > 0.5:  # ~30 degrees
                branch_consistent = False
                print(f"      Warning: Large phase difference at high K ({max_phase_diff:.3f} rad)")
        
        # Check order parameter consistency
        if r_baseline_high is not None and r_modified_high is not None:
            r_diff = abs(r_baseline_high - r_modified_high)
            if r_diff > 0.1:  # Order parameter should be similar
                branch_consistent = False
                print(f"      Warning: Order parameter mismatch at high K (baseline: {r_baseline_high:.3f}, modified: {r_modified_high:.3f})")
    
    Kc_new = kc_first_onset(Kn, Rn, eps=onset_eps, min_run=onset_minrun)
    
    # Determine if Braess (Kc increased)
    if Kc_base is not None and Kc_new is not None:
        delta = Kc_new - Kc_base
        is_braess_actual = (delta > 1e-3)
    else:
        delta = None
        is_braess_actual = False
    
    # Predict using g.py (using SAME omega_c)
    if theta_base is not None and Kc_base is not None:
        pred_braess, k_prime = predict_braess_g(theta_base, omega, W, Kc_base, (u, v), omega_c=omega_c)
    else:
        pred_braess = False
        k_prime = None
    
    # Check if prediction is correct
    correct = (pred_braess == is_braess_actual)
    
    results.append({
        'edge': (u, v),
        'Kc_base': Kc_base,
        'Kc_new': Kc_new,
        'delta': delta,
        'is_braess_actual': is_braess_actual,
        'pred_braess': pred_braess,
        'k_prime': k_prime,
        'correct': correct,
        'branch_consistent': branch_consistent
    })
    
    print(f"      Kc_new = {Kc_new:.4f}" if Kc_new else "      Failed to find Kc_new")
    print(f"      Delta = {delta:.4f}" if delta else "      Delta = N/A")
    print(f"      Braess (actual): {is_braess_actual}")
    print(f"      Braess (predicted): {pred_braess}")
    print(f"      k'(0) = {k_prime:.4f}" if k_prime else "      k'(0) = N/A")
    print(f"      Correct: {correct}")
    
    # Save detailed plots for all edges
    if is_braess_actual:
        print(f"      >>> BRAESS PARADOX DETECTED! <<<")
    
    overlay_path = os.path.join(output_dir, f"edge_{u}_{v}_overlay.png")
    network_path = os.path.join(output_dir, f"edge_{u}_{v}_network.png")
    plot_overlay(Kb, Rb, Kn, Rn, Kc_base, Kc_new, (u, v), output_path=overlay_path)
    plot_network(W_new, title=f"Network with Added Edge {(u, v)}", 
                highlight_edge=(u, v), output_path=network_path)

# Step 4: Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

total = len(results)
braess_actual = sum(r['is_braess_actual'] for r in results)
braess_predicted = sum(r['pred_braess'] for r in results)
correct = sum(r['correct'] for r in results)
accuracy = correct/total*100 if total > 0 else 0.0

print(f"\nTotal edges tested: {total}")
print(f"Braess edges (actual): {braess_actual}")
print(f"Braess edges (predicted): {braess_predicted}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.1f}%" if total > 0 else "Accuracy: N/A")

print("\nDetailed results:")
print(f"{'Edge':<10} {'Kc_base':<10} {'Kc_new':<10} {'Delta':<10} {'Actual':<10} {'Predicted':<10} {'k_prime':<10}")
print("-" * 70)
for r in results:
    edge_str = f"{r['edge']}"
    kc_base_str = f"{r['Kc_base']:.4f}" if r['Kc_base'] else "N/A"
    kc_new_str = f"{r['Kc_new']:.4f}" if r['Kc_new'] else "N/A"
    delta_str = f"{r['delta']:.4f}" if r['delta'] else "N/A"
    actual_str = "Yes" if r['is_braess_actual'] else "No"
    pred_str = "Yes" if r['pred_braess'] else "No"
    kprime_str = f"{r['k_prime']:.4f}" if r['k_prime'] else "N/A"
    
    print(f"{edge_str:<10} {kc_base_str:<10} {kc_new_str:<10} {delta_str:<10} {actual_str:<10} {pred_str:<10} {kprime_str:<10}")

print("\n" + "=" * 70)
print(f"\nAll plots saved to: {output_dir}/")
print(f"Full results saved to: {log_file_path}")
print("=" * 70)

# Generate scatter plots
print("\nGenerating scatter plots...")
scatter_delta_path = os.path.join(output_dir, "scatter_kprime_vs_delta.png")
scatter_kc_path = os.path.join(output_dir, "scatter_kprime_vs_baselineKc.png")
plot_scatter_kprime_vs_delta(results, output_path=scatter_delta_path)
plot_scatter_kprime_vs_baselineKc(results, output_path=scatter_kc_path)
print("Scatter plots saved.")

# Restore stdout and close file
sys.stdout = original_stdout
log_file.close()

# Print only overall accuracy to console
print(f"\n{'='*70}")
print(f"OVERALL ACCURACY: {accuracy:.1f}%")
print(f"{'='*70}")
print(f"\nFull results saved to: {log_file_path}")
print(f"All plots saved to: {output_dir}/")