import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import networkx as nx
import os
import pandas as pd
from datetime import datetime

def create_output_folder():
    """Create output folder with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"runs/results_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def kuramoto_dynamics(theta, omega, W, K):
    """Compute the time derivatives for Kuramoto system"""
    n = len(theta)
    dtheta = np.zeros(n)
    for i in range(n):
        coupling = 0.0
        for j in range(n):
            if W[i,j] > 0:
                coupling += W[i,j] * np.sin(theta[j] - theta[i])
        dtheta[i] = omega[i] + K * coupling
    return dtheta

def time_integration_solver(omega, W, K, theta_init=None, dt=0.01, max_time=30, tol=1e-6):
    """Time integration solver for Kuramoto steady state"""
    n = len(omega)
    if theta_init is None:
        theta_init = np.zeros(n)
    
    theta = theta_init.copy()
    t = 0.0
    
    theta_history = []
    derivative_history = []
    
    while t < max_time:
        dtheta = kuramoto_dynamics(theta, omega, W, K)
        theta_new = theta + dt * dtheta
        
        theta_history.append(theta.copy())
        derivative_history.append(np.linalg.norm(dtheta))
        
        if len(theta_history) > 10:
            if np.linalg.norm(dtheta) < tol:
                return theta_new, True, t
            
            recent_theta_changes = [np.linalg.norm(theta_history[-i] - theta_history[-i-1]) 
                                   for i in range(1, min(11, len(theta_history)))]
            if np.mean(recent_theta_changes) < tol:
                return theta_new, True, t
            
            recent_derivatives = derivative_history[-10:]
            if np.mean(recent_derivatives) < 1e-3 and np.std(recent_derivatives) < 1e-4:
                return theta_new, True, t
        
        theta = theta_new
        t += dt
    
    final_derivative_norm = np.linalg.norm(kuramoto_dynamics(theta, omega, W, K))
    if final_derivative_norm < 1e-1:
        return theta, True, t
    
    return theta, False, t

def solve_kuramoto(theta_guess, omega, W, K):
    """Solve Kuramoto steady state using time integration."""
    theta_sol, success, final_time = time_integration_solver(omega, W, K, theta_guess)
    r = compute_order_parameter(theta_sol)
    
    if success:
        residual_norm = np.linalg.norm(kuramoto_dynamics(theta_sol, omega, W, K))
        if residual_norm < 1e-2:
            return theta_sol, True
    elif r > 0.5:
        return theta_sol, True
    
    return theta_sol, False

def compute_order_parameter(theta):
    """Compute the Kuramoto order parameter"""
    return np.abs(np.exp(1j * theta).mean())

def solve_locked(omega, W, K, theta0=None, pin=0, max_iter=800, tol=1e-11):
    """Wrapper for solve_kuramoto"""
    theta_guess = theta0 if theta0 is not None else np.zeros(len(omega))
    theta_sol, ok = solve_kuramoto(theta_guess, omega, W, K)
    return theta_sol, bool(ok), {"method": "time_integration"}

def sweep_K_and_find_Kc(omega, W, K_values, theta_guess=None, r_threshold=0.7, verbose=True):
    """Sweep K values and find critical coupling Kc"""
    n = len(omega)
    if theta_guess is None:
        theta_guess = np.zeros(n)

    r_values = []
    last_theta = theta_guess.copy()
    crossed = False
    Kc_idx = None

    if verbose:
        print(f"Sweeping K values from {K_values[0]:.3f} to {K_values[-1]:.3f}")
    
    for idx, K in enumerate(K_values):
        theta_sol, ok, _ = solve_locked(omega, W, K, theta0=last_theta)
        
        if ok:
            r = compute_order_parameter(theta_sol)
            last_theta = theta_sol
        else:
            r = 0.0
            last_theta = None

        if verbose and (idx < 3 or idx % 5 == 0):
            residual_norm = np.linalg.norm(kuramoto_dynamics(theta_sol, omega, W, K)) if ok else np.inf
            print(f"  K={K:.3f}: r={r:.4f}, ok={ok}, residual={residual_norm:.2e}")
        
        r_values.append(r)

        if not crossed and r >= r_threshold:
            crossed = True
            Kc_idx = idx

    r_values = np.array(r_values)
    K_values = np.array(K_values)

    if crossed:
        stable_Kc_idx = Kc_idx
        for idx in range(Kc_idx + 1, len(r_values)):
            if r_values[idx] < r_threshold:
                stable_Kc_idx = idx
                for j in range(idx + 1, len(r_values)):
                    if r_values[j] >= r_threshold:
                        stable_Kc_idx = j
                        break
                else:
                    break
        
        Kc_basic = float(K_values[stable_Kc_idx])
        return Kc_basic, K_values, r_values

    r_min = np.min(r_values)
    r_max = np.max(r_values)
    if r_max - r_min > 0.1:
        threshold_r = r_min + 0.5 * (r_max - r_min)
        for idx, r in enumerate(r_values):
            if r > threshold_r:
                Kc_out = float(K_values[idx])
                break
        else:
            Kc_out = float(K_values[int(np.argmax(r_values))])
    else:
        Kc_out = float(K_values[int(np.argmax(r_values))])
    return Kc_out, K_values, r_values

def list_missing_edges(W):
    """List all missing edges in the network"""
    n = W.shape[0]
    missing = []
    for i in range(n):
        for j in range(i+1, n):
            if W[i,j] == 0:
                missing.append((i, j))
    return missing

def draw_network_graph(W, omega, output_folder, title="Network", layout_type="circular"):
    """Draw and save a network graph"""
    n = W.shape[0]
    G = nx.Graph()
    
    # Add nodes
    for i in range(n):
        G.add_node(i, omega=omega[i])
    
    # Add edges
    for i in range(n):
        for j in range(i+1, n):
            if W[i,j] > 0:
                G.add_edge(i, j, weight=W[i,j])
    
    # Create layout
    if layout_type == "circular":
        pos = {}
        for i in range(n):
            angle = 2 * np.pi * i / n
            pos[i] = (np.cos(angle), np.sin(angle))
    elif layout_type == "spring":
        pos = nx.spring_layout(G)
    else:  # random
        pos = nx.random_layout(G)
    
    # Draw the graph
    plt.figure(figsize=(10, 10))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.6, width=2)
    
    # Draw nodes with colors based on omega values
    node_colors = ['red' if omega[i] > 0 else 'blue' for i in range(n)]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, alpha=0.8)
    
    # Draw labels
    labels = {i: f'{i}\nω={omega[i]:.1f}' for i in range(n)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    plt.title(f'{title}\nRed: ω>0, Blue: ω<0')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{title.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_synchronization_curve(K_values, r_values, Kc, title, output_folder, filename):
    """Plot synchronization curve and save to file"""
    plt.figure(figsize=(12, 8))
    plt.semilogx(K_values, r_values, 'o-', label=f'{title} (Kc={Kc:.3f})', markersize=4)
    plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Threshold')
    plt.axvline(x=Kc, color='g', linestyle='--', alpha=0.5, label=f'Kc={Kc:.3f}')
    plt.xlabel('Coupling Strength K')
    plt.ylabel('Order Parameter r')
    plt.title(f'Synchronization Transition - {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(K_values[0], K_values[-1])
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()

def plot_edge_comparison(K_values, r_baseline, r_with_edge, Kc_baseline, Kc_with_edge, 
                        edge, output_folder, timestamp):
    """Plot comparison between baseline and with specific edge"""
    plt.figure(figsize=(12, 8))
    
    # Plot baseline
    plt.semilogx(K_values, r_baseline, 'o-', label=f'Baseline (Kc={Kc_baseline:.3f})', 
                linewidth=2, markersize=4, color='blue')
    
    # Plot with edge
    plt.semilogx(K_values, r_with_edge, 's-', label=f'With edge {edge} (Kc={Kc_with_edge:.3f})', 
                linewidth=2, markersize=4, color='red')
    
    plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Threshold')
    plt.axvline(x=Kc_baseline, color='blue', linestyle='--', alpha=0.5, label=f'Baseline Kc={Kc_baseline:.3f}')
    plt.axvline(x=Kc_with_edge, color='red', linestyle='--', alpha=0.5, label=f'Edge Kc={Kc_with_edge:.3f}')
    
    delta = Kc_with_edge - Kc_baseline
    plt.xlabel('Coupling Strength K')
    plt.ylabel('Order Parameter r')
    plt.title(f'Edge {edge} Analysis: ΔKc = {delta:+.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(K_values[0], K_values[-1])
    plt.tight_layout()
    
    # Save with timestamp and edge name
    filename = f"{timestamp}_edge_{edge[0]}_{edge[1]}.png"
    plt.savefig(f'{output_folder}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()

def test_braess_paradox(omega, W, K_values, output_folder, network_name="Network", r_threshold=0.7):
    """Test Braess paradox by adding missing edges - ORGANIZED VERSION"""
    print(f"\n=== Testing Braess Paradox in {network_name} ===")
    
    # Get timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Draw baseline graph
    draw_network_graph(W, omega, output_folder, f"baseline_{network_name}", layout_type="circular")
    
    # Find baseline Kc
    Kc_baseline, K_grid, r_baseline = sweep_K_and_find_Kc(omega, W, K_values, r_threshold=r_threshold)
    
    print(f"\nBaseline Kc ≈ {Kc_baseline:.3f}")
    print(f"Baseline max r = {np.max(r_baseline):.4f}")
    
    missing_edges = list_missing_edges(W)
    print(f"\nTesting ALL {len(missing_edges)} missing edges for Braess paradox...")
    
    # Initialize results list for CSV
    results_data = []
    braess_candidates = []
    
    # Test all missing edges
    for i, edge in enumerate(missing_edges):
        print(f"\nTesting edge {i+1}/{len(missing_edges)}: {edge}")
        
        W_test = W.copy()
        W_test[edge[0], edge[1]] = 1.0
        W_test[edge[1], edge[0]] = 1.0
        
        # Draw graph with new edge
        draw_network_graph(W_test, omega, output_folder, f"graph_with_edge_{edge[0]}_{edge[1]}", layout_type="circular")
        
        Kc_after, _, r_after = sweep_K_and_find_Kc(omega, W_test, K_values, r_threshold=r_threshold, verbose=False)
        
        delta = Kc_after - Kc_baseline
        max_r = np.max(r_after)
        
        print(f"  Edge {edge}: Kc_after={Kc_after:.3f}, delta={delta:+.3f}, max_r={max_r:.4f}")
        
        # Add to results data for CSV
        results_data.append({
            'edge': f"{edge[0]}_{edge[1]}",
            'baseline_Kc': Kc_baseline,
            'new_Kc': Kc_after,
            'delta_Kc': delta,
            'max_r': max_r,
            'is_braess': delta > 0.01
        })
        
        # Create individual comparison plot for this edge
        plot_edge_comparison(K_grid, r_baseline, r_after, Kc_baseline, Kc_after, 
                           edge, output_folder, timestamp)
        
        if delta > 0.01:
            braess_candidates.append((edge, delta, Kc_after))
    
    # Save results to CSV
    df = pd.DataFrame(results_data)
    csv_filename = f"{timestamp}_braess_results.csv"
    df.to_csv(f'{output_folder}/{csv_filename}', index=False)
    print(f"\nResults saved to CSV: {csv_filename}")
    
    # Save detailed results as numpy array too
    np.save(f'{output_folder}/braess_results.npy', results_data)
    
    # Create summary plot
    if braess_candidates:
        print(f"\nBraess paradox candidates found:")
        for edge, delta, Kc_after in sorted(braess_candidates, key=lambda x: x[1], reverse=True):
            print(f"  Edge {edge}: ΔKc = +{delta:.3f}")
        
        # Plot all results
        plt.figure(figsize=(15, 10))
        
        # Plot baseline
        plt.semilogx(K_grid, r_baseline, 'o-', label=f'Baseline (Kc={Kc_baseline:.3f})', linewidth=2, markersize=4)
        
        # Plot best Braess candidates
        colors = ['red', 'orange', 'purple', 'brown', 'pink']
        for i, (edge, delta, Kc_after) in enumerate(sorted(braess_candidates, key=lambda x: x[1], reverse=True)[:5]):
            W_test = W.copy()
            W_test[edge[0], edge[1]] = 1.0
            W_test[edge[1], edge[0]] = 1.0
            _, _, r_after = sweep_K_and_find_Kc(omega, W_test, K_values, r_threshold=r_threshold, verbose=False)
            plt.semilogx(K_grid, r_after, 's-', label=f'Edge {edge} (Kc={Kc_after:.3f}, Δ={delta:+.3f})', 
                        color=colors[i % len(colors)], linewidth=1, markersize=3)
        
        plt.axhline(y=r_threshold, color='r', linestyle='--', alpha=0.5, label='Threshold')
        plt.xlabel('Coupling Strength K')
        plt.ylabel('Order Parameter r')
        plt.title(f'Braess Paradox in {network_name}: Adding Edges Can Increase Kc')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xlim(K_values[0], K_values[-1])
        plt.tight_layout()
        plt.savefig(f'{output_folder}/{timestamp}_braess_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    else:
        print("\nNo Braess paradox found in this configuration.")
    
    print(f"\n=== Braess Paradox Test Complete ===")
    return results_data

def save_experiment_summary(output_folder, sync_results, braess_results, network_name="Network"):
    """Save experiment summary"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'network_name': network_name,
        'sync_results': sync_results,
        'braess_results': braess_results
    }
    
    np.save(f'{output_folder}/summary.npy', summary)
    
    print(f"\nAll tests completed! Results saved to {output_folder}")
    print(f"Generated {len(os.listdir(output_folder))} files in the output folder.")
