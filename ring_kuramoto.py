import numpy as np
from experiment_utils import *

def sample_ring_network(n=6, capacity=1.0):
    """Create a ring network with n nodes"""
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i+1) % n] = capacity
        W[(i+1) % n, i] = capacity
    return W

def test_ring_synchronization(output_folder):
    """Test the ring network with different omega configurations"""
    print("=== Testing 6-Node Ring Network ===")
    
    n = 6
    W = sample_ring_network(n=6, capacity=1.0)
    
    print(f"Ring network created:")
    print(f"  Number of nodes: {n}")
    print(f"  Number of edges: {np.sum(W > 0) // 2}")
    
    omega_configs = [
        ("Balanced", np.array([1, 1, 1, -1, -1, -1])),
        ("Unbalanced", np.array([1, 1, 1, 1, -1, -1])),
        ("Small variation", np.array([0.1, 0.1, 0.1, -0.1, -0.1, -0.1]))
    ]
    
    # Extended range with more points
    K_values = np.geomspace(0.05, 10.0, num=40)
    results = {}
    
    for name, omega in omega_configs:
        print(f"\n--- Testing {name} omega configuration ---")
        print(f"omega = {omega}")
        
        # Draw and save the graph
        draw_network_graph(W, omega, output_folder, f"Ring_{name}", layout_type="circular")
        
        Kc, K_grid, r_values = sweep_K_and_find_Kc(omega, W, K_values, r_threshold=0.7)
        
        print(f"Results for {name}:")
        print(f"  Kc ≈ {Kc:.3f}")
        print(f"  Max r = {np.max(r_values):.4f}")
        print(f"  Min r = {np.min(r_values):.4f}")
        
        results[name] = {
            'omega': omega,
            'Kc': Kc,
            'K_values': K_grid,
            'r_values': r_values,
            'max_r': np.max(r_values),
            'min_r': np.min(r_values)
        }
        
        # Plot synchronization curve
        plot_synchronization_curve(K_grid, r_values, Kc, name, output_folder, 
                                 f"ring_sync_{name.lower().replace(' ', '_')}.png")
    
    print("\n=== Ring Network Tests Complete ===")
    return results

def test_ring_braess_paradox(output_folder):
    """Test Braess paradox in the ring network"""
    n = 6
    W = sample_ring_network(n=6, capacity=1.0)
    omega = np.array([1, 1, 1, -1, -1, -1])
    
    print(f"Testing with balanced omega: {omega}")
    
    # Extended range with more points
    K_values = np.geomspace(0.05, 10.0, num=40)
    
    # Use the generic Braess paradox test function
    braess_results = test_braess_paradox(omega, W, K_values, output_folder, "Ring", r_threshold=0.7)
    
    return braess_results

if __name__ == "__main__":
    np.random.seed(42)
    
    output_folder = create_output_folder()
    print(f"Output will be saved to: {output_folder}")
    
    # Test ring synchronization
    sync_results = test_ring_synchronization(output_folder)
    
    # Test Braess paradox
    braess_results = test_ring_braess_paradox(output_folder)
    
    # Save summary
    save_experiment_summary(output_folder, sync_results, braess_results, "Ring")
    
    print("This simulation tests the 6-node ring Kuramoto network.")
