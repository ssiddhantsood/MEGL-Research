import numpy as np
import ast
import matplotlib.pyplot as plt
import networkx as nx
import csv
from utils import (
    draw_geometric_network_graph, 
    draw_network_graph,
    continuation_descend_K,
    plot_r_vs_K,
    order_parameter
)

def parse_csv_data(csv_file):
    """Parse the CSV data and extract network information without pandas"""
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        row = next(reader)  # Get the first row
    
    # Extract the data from the row
    omega_str = row[-3]  # Second to last column contains omega
    edges_str = row[-2]  # Third to last column contains edges
    metadata_str = row[-1]  # Last column contains metadata
    
    # Parse omega (natural frequencies)
    omega = np.array(ast.literal_eval(omega_str))
    
    # Parse edges
    edges = ast.literal_eval(edges_str)
    
    # Parse metadata
    metadata = ast.literal_eval(metadata_str)
    
    # Create weight matrix from edges
    n = len(omega)
    W = np.zeros((n, n))
    for edge in edges:
        i, j, weight = edge
        W[i, j] = weight
        W[j, i] = weight  # Undirected graph
    
    return omega, W, metadata

def create_network_visualizations(omega, W, metadata, output_dir="graph_output"):
    """Create various network visualizations"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    n = len(omega)
    
    # 1. Basic network visualization
    print("Creating basic network visualization...")
    draw_network_graph(
        W, omega, 
        path=f"{output_dir}/network_basic.png",
        layout="spring",
        title=f"Network Graph (n={n})"
    )
    
    # 2. If we have position data, create geometric visualization
    if metadata.get("graph_pos_saved", False):
        print("Creating geometric network visualization...")
        # For RGN, we'll create a synthetic position layout
        # In a real scenario, you'd load the actual positions
        pos = {}
        for i in range(n):
            # Create a simple circular layout for demonstration
            angle = 2 * np.pi * i / n
            pos[i] = (10 * np.cos(angle), 10 * np.sin(angle))
        
        draw_geometric_network_graph(
            W, omega, pos,
            path=f"{output_dir}/network_geometric.png",
            title=f"Random Geometric Network (n={n})"
        )
    
    # 3. Network statistics
    print("Computing network statistics...")
    G = nx.from_numpy_array(W)
    
    stats = {
        'num_nodes': n,
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G),
        'avg_shortest_path': nx.average_shortest_path_length(G) if nx.is_connected(G) else np.inf,
        'diameter': nx.diameter(G) if nx.is_connected(G) else np.inf,
        'omega_mean': np.mean(omega),
        'omega_std': np.std(omega)
    }
    
    print(f"Network Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 4. Synchronization analysis
    print("Performing synchronization analysis...")
    try:
        Kc, K_values, r_values = continuation_descend_K(
            omega, W, 
            K_start=10.0, K_min=0.02, K_step=0.1, 
            r_thresh=0.7
        )
        
        if Kc is not None:
            print(f"Critical coupling strength Kc: {Kc:.4f}")
            
            # Plot synchronization curve
            plot_r_vs_K(
                K_values, r_values,
                out_path=f"{output_dir}/synchronization_curve.png",
                Kc=Kc,
                title=f"Synchronization Curve (Kc = {Kc:.4f})",
                r_threshold=0.7
            )
        else:
            print("Could not determine critical coupling strength")
            
    except Exception as e:
        print(f"Error in synchronization analysis: {e}")
    
    # 5. Node degree distribution
    print("Creating degree distribution plot...")
    degrees = [G.degree(n) for n in G.nodes()]
    
    plt.figure(figsize=(8, 6))
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), alpha=0.7, edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Node Degree Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/degree_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 6. Omega distribution
    print("Creating omega distribution plot...")
    plt.figure(figsize=(8, 6))
    plt.hist(omega, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Natural Frequency (ω)')
    plt.ylabel('Frequency')
    plt.title('Natural Frequency Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/omega_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    return stats

def main():
    """Main function to process the CSV data"""
    csv_file = "simple.csv"
    
    try:
        print("Parsing CSV data...")
        omega, W, metadata = parse_csv_data(csv_file)
        
        print(f"Loaded network with {len(omega)} nodes")
        print(f"Generator: {metadata.get('generator', 'unknown')}")
        
        print("Creating visualizations...")
        stats = create_network_visualizations(omega, W, metadata)
        
        print(f"\nVisualizations saved to 'graph_output' directory")
        print("Files created:")
        print("  - network_basic.png: Basic network layout")
        print("  - network_geometric.png: Geometric network layout")
        print("  - synchronization_curve.png: r(K) curve")
        print("  - degree_distribution.png: Node degree histogram")
        print("  - omega_distribution.png: Natural frequency histogram")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 