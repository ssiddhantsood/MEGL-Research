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
from typing import Optional, Tuple, List
import networkx as nx
import os
import sys
from io import StringIO

# =============================================================================
# NETWORK DEFINITION - EDIT THIS SECTION
# =============================================================================

# Number of nodes
n = 10

# Adjacency matrix (symmetric, weighted)
# Example: random connected graph
np.random.seed(42)
W = np.array([
    [0., 1., 1., 0., 0., 0., 1., 0., 0., 0.],
    [1., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
    [1., 1., 0., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 0., 1., 1., 0., 0., 0., 0.],
    [0., 0., 1., 1., 0., 1., 0., 1., 0., 0.],
    [0., 0., 0., 1., 1., 0., 1., 0., 0., 0.],
    [1., 0., 0., 0., 0., 1., 0., 1., 1., 0.],
    [0., 0., 0., 0., 1., 0., 1., 0., 1., 1.],
    [0., 0., 0., 0., 0., 0., 1., 1., 0., 1.],
    [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.]
])

# Natural frequencies
omega = np.array([0.5, -0.3, 1.2, -0.8, 0.4, -0.6, 0.9, -0.2, 0.7, -1.0])

# K sweep parameters
K_start = 5.0    # Start at high coupling
K_min = 0.4      # End at low coupling
K_step = 0.01    # Step size for K

# Onset detection parameters
onset_eps = 0.05      # Threshold for r to be considered "synchronized"
onset_minrun = 3      # Number of consecutive increasing points needed

# Edge addition weight
add_weight = 1.0

# Output directory for saved plots
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

def continuation_curve(omega: np.ndarray, W: np.ndarray, 
                       K_start: float, K_min: float, K_step: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute r(K) by descending from high K to low K.
    Returns: (K_values, r_values) both in ascending order.
    """
    omega_c = center_omega(omega)
    K = K_start
    
    # Lock at high K
    theta, ok = solve_locked(omega_c, W, K)
    if not ok:
        return np.array([]), np.array([])
    
    # Walk downward
    Ks, Rs = [], []
    while K >= K_min:
        theta, ok = solve_locked(omega_c, W, K, theta0=theta)
        r = order_parameter(theta) if ok else 0.0
        Ks.append(K)
        Rs.append(r)
        K -= K_step
        
        # Early stop if clearly desynchronized
        if len(Rs) >= 3 and Rs[-1] < 0.05 and Rs[-2] < 0.05:
            break
    
    # Reverse to ascending order
    return np.array(Ks[::-1]), np.array(Rs[::-1])

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
                     K_0: float, edge: Tuple[int, int]) -> Tuple[bool, float]:
    """
    Predict Braess paradox using the g.py derivative formula.
    
    Returns k'(0) = -v₀ᵀ·g(θ₀) / [v₀ᵀ·s(θ₀)]
    
    For our purposes, we approximate v₀ ≈ ω (centered).
    If k'(0) > 0: predict Braess (adding edge increases Kc)
    """
    omega_c = center_omega(omega)
    
    g_theta = compute_g_theta(theta_0, edge)
    s_theta = compute_s_theta(theta_0, W)
    
    # Use centered omega as approximation of critical eigenvector
    v_0 = omega_c / np.linalg.norm(omega_c)
    
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
    """Plot order parameter vs coupling strength and save to file."""
    plt.figure(figsize=(10, 6))
    plt.plot(Ks, Rs, 'o-', linewidth=2, markersize=4, label='r(K)')
    
    if Kc is not None:
        plt.axvline(Kc, linestyle='--', color='red', alpha=0.7, 
                   label=f'Kc ≈ {Kc:.3f}')
    
    plt.axhline(onset_eps, linestyle=':', color='gray', alpha=0.5, 
               label=f'onset threshold ({onset_eps})')
    
    plt.xlabel('Coupling strength K', fontsize=12)
    plt.ylabel('Order parameter r', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xlim(K_min, K_start)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"      Saved: {output_path}")
    plt.close()

def plot_network(W: np.ndarray, title: str = "Network", 
                highlight_edge: Optional[Tuple[int, int]] = None,
                output_path: str = None):
    """Visualize the network graph and save to file."""
    G = nx.from_numpy_array(W)
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=2)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', 
                          edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    if highlight_edge is not None:
        nx.draw_networkx_edges(G, pos, edgelist=[highlight_edge], 
                              width=4, edge_color='red')
    
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"      Saved: {output_path}")
    plt.close()

def plot_overlay(Kb: np.ndarray, Rb: np.ndarray, Kn: np.ndarray, Rn: np.ndarray,
                Kc_base: Optional[float], Kc_new: Optional[float], 
                edge: Tuple[int, int], output_path: str = None):
    """Plot baseline vs modified r(K) curves and save to file."""
    plt.figure(figsize=(10, 6))
    plt.plot(Kb, Rb, 'o-', linewidth=2, markersize=4, label='Baseline')
    plt.plot(Kn, Rn, 's--', linewidth=2, markersize=4, 
            label=f'Added edge {edge}')
    
    if Kc_base is not None:
        plt.axvline(Kc_base, linestyle='--', color='blue', alpha=0.6, 
                   label=f'Baseline Kc ≈ {Kc_base:.3f}')
    if Kc_new is not None:
        plt.axvline(Kc_new, linestyle=':', color='orange', alpha=0.7, 
                   label=f'New Kc ≈ {Kc_new:.3f}')
    
    plt.xlabel('Coupling strength K', fontsize=12)
    plt.ylabel('Order parameter r', fontsize=12)
    plt.title(f'Baseline vs Edge {edge} Added', fontsize=14)
    plt.xlim(K_min, K_start)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"      Saved: {output_path}")
    plt.close()

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

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

# Create output directory
os.makedirs(output_dir, exist_ok=True)
log_file_path = os.path.join(output_dir, "analysis_results.txt")

# Redirect all output to both console and file
original_stdout = sys.stdout
tee = Tee(log_file_path)
sys.stdout = tee

print("=" * 70)
print("BRAESS PARADOX ANALYSIS")
print("=" * 70)

print(f"\nOutput directory: {output_dir}/")

# Step 1: Compute baseline
print("\n1. Computing baseline r(K) curve...")
Kb, Rb = continuation_curve(omega, W, K_start, K_min, K_step)
Kc_base = kc_first_onset(Kb, Rb, eps=onset_eps, min_run=onset_minrun)

print(f"   Baseline Kc = {Kc_base:.4f}" if Kc_base else "   Failed to find baseline Kc")
print(f"   Number of K points: {len(Kb)}")

# Visualize baseline
baseline_r_path = os.path.join(output_dir, "baseline_r_vs_K.png")
baseline_network_path = os.path.join(output_dir, "baseline_network.png")
plot_r_vs_K(Kb, Rb, Kc_base, title="Baseline r(K)", output_path=baseline_r_path)
plot_network(W, title="Baseline Network", output_path=baseline_network_path)

# Step 2: Find baseline locked state at Kc for g predictor
if Kc_base is not None:
    omega_c = center_omega(omega)
    theta_base, ok = solve_locked(omega_c, W, Kc_base)
    if not ok:
        print("   Warning: Could not lock baseline state at Kc")
        theta_base = None
else:
    theta_base = None

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
    
    # Compute new Kc
    Kn, Rn = continuation_curve(omega, W_new, K_start, K_min, K_step)
    Kc_new = kc_first_onset(Kn, Rn, eps=onset_eps, min_run=onset_minrun)
    
    # Determine if Braess (Kc increased)
    if Kc_base is not None and Kc_new is not None:
        delta = Kc_new - Kc_base
        is_braess_actual = (delta > 1e-3)
    else:
        delta = None
        is_braess_actual = False
    
    # Predict using g.py
    if theta_base is not None and Kc_base is not None:
        pred_braess, k_prime = predict_braess_g(theta_base, omega, W, Kc_base, (u, v))
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
        'correct': correct
    })
    
    print(f"      Kc_new = {Kc_new:.4f}" if Kc_new else "      Failed to find Kc_new")
    print(f"      Delta = {delta:.4f}" if delta else "      Delta = N/A")
    print(f"      Braess (actual): {is_braess_actual}")
    print(f"      Braess (predicted): {pred_braess}")
    print(f"      k'(0) = {k_prime:.4f}" if k_prime else "      k'(0) = N/A")
    print(f"      Correct: {correct}")
    
    # Save detailed plots for Braess edges only
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

# Write summary and detailed results only to file (not console)
# Temporarily redirect stdout to only file for this section
class FileOnly:
    """Write only to file, not console."""
    def __init__(self, file):
        self.file = file
    
    def write(self, text):
        self.file.write(text)
        self.file.flush()
    
    def flush(self):
        self.file.flush()

file_only = FileOnly(tee.file)
sys.stdout = file_only

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

# Restore stdout and close file
sys.stdout = original_stdout
tee.close()

# Print only overall accuracy to console
print(f"\n{'='*70}")
print(f"OVERALL ACCURACY: {accuracy:.1f}%")
print(f"{'='*70}")
print(f"\nFull results saved to: {log_file_path}")
print(f"All plots saved to: {output_dir}/")
