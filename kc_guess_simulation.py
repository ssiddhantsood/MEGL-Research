import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


# ---------------------------
# Graph utilities
# ---------------------------

def generate_connected_random_graph(n=10, p=0.3, capacity=1.0, seed=None):
    # If no seed provided → use current time in microseconds
    if seed is None:
        seed = int(time.time() * 1e6)
    rng = np.random.default_rng(seed)

    while True:
        # Use a unique random seed for NetworkX every time
        rand_seed = int(rng.integers(0, 1e9))
        G = nx.erdos_renyi_graph(n=n, p=p, seed=rand_seed)

        # Ensure the graph is connected
        if nx.is_connected(G):
            # Build symmetric adjacency/capacity matrix
            W = np.zeros((n, n), dtype=float)
            for u, v in G.edges():
                W[u, v] = capacity
                W[v, u] = capacity
            return W, G


def draw_graph(G, node_labels=True, capacity_labels=False, W=None, seed=0):
    pos = nx.spring_layout(G, seed=seed)
    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_size=400)
    nx.draw_networkx_edges(G, pos, width=1.5)
    if node_labels:
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

    # If requested, label edges with capacity from W
    if capacity_labels and W is not None:
        labels = {(i, j): f"{W[i,j]:.2g}" for i, j in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)

    plt.axis("off")
    plt.title("Random Connected Graph")
    plt.show()


def add_edge(W, i, j, capacity=1.0):
    """Add or update an undirected edge (i,j) with given capacity."""
    W_new = W.copy()
    W_new[i, j] = capacity
    W_new[j, i] = capacity
    return W_new


def remove_edge(W, i, j):
    """Remove an undirected edge (i,j) by setting capacity to 0."""
    W_new = W.copy()
    W_new[i, j] = 0.0
    W_new[j, i] = 0.0
    return W_new


def list_missing_edges(W):
    """List all possible edges NOT in the current graph."""
    n = W.shape[0]
    missing = []
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] == 0.0:
                missing.append((i, j))
    return missing


def list_existing_edges(W):
    """List all existing edges."""
    n = W.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0:
                edges.append((i, j))
    return edges


# ---------------------------
# Kuramoto steady-state solver
# ---------------------------

def kuramoto_residual(theta, omega, W, K):
    """
    Residual for steady-state Kuramoto equations:
      0 = omega_i + K * sum_j W_ij * sin(theta_j - theta_i)
    """
    n = len(theta)
    residual = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if W[i, j] > 0:
                s += W[i, j] * np.sin(theta[j] - theta[i])
        residual[i] = omega[i] + K * s
    return residual


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

def time_integration_solver(omega, W, K, theta_init=None, dt=0.01, max_time=100, tol=1e-6):
    """Time integration solver for Kuramoto steady state"""
    n = len(omega)
    if theta_init is None:
        theta_init = np.zeros(n)
    
    theta = theta_init.copy()
    t = 0.0
    
    # Store history for convergence check
    theta_history = []
    
    while t < max_time:
        # Compute derivatives
        dtheta = kuramoto_dynamics(theta, omega, W, K)
        
        # Euler integration
        theta_new = theta + dt * dtheta
        
        # Check for convergence
        if len(theta_history) > 10:
            # Check if the last 10 steps show little change
            recent_changes = [np.linalg.norm(theta_history[-i] - theta_history[-i-1]) 
                            for i in range(1, min(11, len(theta_history)))]
            if np.mean(recent_changes) < tol:
                return theta_new, True, t
        
        theta_history.append(theta.copy())
        theta = theta_new
        t += dt
    
    return theta, False, t

def solve_kuramoto(theta_guess, omega, W, K):
    """
    Solve Kuramoto steady state using time integration.
    NOTE: theta_guess is the FIRST positional arg (keep this order).
    Returns: (theta_solution, success_flag)
    """
    # Use time integration solver
    theta_sol, success, final_time = time_integration_solver(omega, W, K, theta_guess)
    
    # Check if solution is reasonable
    if success:
        residual_norm = np.linalg.norm(kuramoto_dynamics(theta_sol, omega, W, K))
        if residual_norm < 1e-2:  # Accept if residual is small enough
            return theta_sol, True
    
    # If time integration failed or residual too large, return the best guess
    return theta_sol, False


def compute_order_parameter(theta):
    """
    Kuramoto order parameter r in [0,1].
    r ≈ 1 → synchronized; r ≈ 0 → incoherent.
    """
    return np.abs(np.exp(1j * theta).mean())


# Compatibility wrapper expected by sweep_K_and_find_Kc
def solve_locked(omega, W, K, theta0=None, pin=0, max_iter=800, tol=1e-11):
    """
    Thin wrapper that adapts to the expected signature in the sweep.
    Calls your existing solve_kuramoto with correct argument ORDER.
    Returns: (theta, ok_flag, meta_dict)
    """
    theta_guess = theta0 if theta0 is not None else np.zeros(len(omega))
    # IMPORTANT FIX: theta_guess must be FIRST positional argument
    theta_sol, ok = solve_kuramoto(theta_guess, omega, W, K)
    return theta_sol, bool(ok), {"method": "fsolve"}


# ---------------------------
# Sweep K and find critical Kc
# ---------------------------

def sweep_K_and_find_Kc(
    omega, W, K_values, theta_guess=None, r_threshold=0.9, refine=True
):
    n = len(omega)
    if theta_guess is None:
        theta_guess = np.zeros(n)

    r_values = []
    last_theta = theta_guess.copy()
    crossed = False
    Kc_idx = None

    # 1) Sweep with adiabatic seeding
    for idx, K in enumerate(K_values):
        theta_sol, ok, _ = solve_locked(
            omega, W, K, theta0=last_theta, pin=0, max_iter=800, tol=1e-11
        )
        
        # Compute order parameter only if solver succeeded
        if ok:
            r = compute_order_parameter(theta_sol)
            last_theta = theta_sol
        else:
            r = 0.0  # Set r=0 when solver fails
            last_theta = None  # reset if solve failed

        # Debug: print first few K values to see what's happening
        if idx < 5 or idx % 20 == 0:
            residual_norm = np.linalg.norm(kuramoto_residual(theta_sol, omega, W, K)) if ok else np.inf
            print(f"  K={K:.3f}: r={r:.4f}, ok={ok}, residual_norm={residual_norm:.2e}")
        
        r_values.append(r)

        if not crossed and r >= r_threshold:
            crossed = True
            Kc_idx = idx  # first index where we cross

    r_values = np.array(r_values)
    K_values = np.array(K_values)

    # 2) If no crossing: fallback to argmax r
    if not crossed:
        # Find the K value where r starts to increase significantly
        # Look for the point where r increases by more than 0.1 from its minimum
        r_min = np.min(r_values)
        r_max = np.max(r_values)
        if r_max - r_min > 0.1:  # Only if there's significant variation
            # Find first point where r > r_min + 0.5*(r_max - r_min)
            threshold_r = r_min + 0.5 * (r_max - r_min)
            for idx, r in enumerate(r_values):
                if r > threshold_r:
                    Kc_out = float(K_values[idx])
                    break
            else:
                Kc_out = float(K_values[int(np.argmax(r_values))])
        else:
            # If no significant variation, use the K where r is highest
            Kc_out = float(K_values[int(np.argmax(r_values))])
        return Kc_out, K_values, r_values

    # 3) Crossing found: basic Kc is first crossing point
    Kc_basic = float(K_values[Kc_idx])
    if not refine:
        return Kc_basic, K_values, r_values

    # 4) Optional refinement via log-bisection around the crossing
    if Kc_idx == 0:
        # crossed at the first point — nothing to refine
        return Kc_basic, K_values, r_values

    K_lo = float(K_values[Kc_idx - 1])
    K_hi = float(K_values[Kc_idx])

    seed_theta = None
    max_refine = 18
    for _ in range(max_refine):
        K_mid = (K_lo * K_hi) ** 0.5  # geometric midpoint (log-bisection)
        theta_mid, ok_mid, _ = solve_locked(
            omega, W, K_mid, theta0=seed_theta, pin=0, max_iter=800, tol=1e-11
        )
        r_mid = compute_order_parameter(theta_mid) if ok_mid else 0.0

        if ok_mid and r_mid >= r_threshold:
            K_hi = K_mid
            seed_theta = theta_mid
        else:
            K_lo = K_mid
            seed_theta = None

    return float(K_hi), K_values, r_values


# ---------------------------
# Example driver / demo
# ---------------------------

if __name__ == "__main__":
    # Test the Kuramoto solver with a simple case first
    print("Testing Kuramoto solver with simple case...")
    n_test = 3
    omega_test = np.array([0.1, -0.1, 0.0])
    W_test = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    K_test = 2.0
    
    theta_test, ok_test = solve_kuramoto(np.zeros(n_test), omega_test, W_test, K_test)
    if ok_test:
        r_test = compute_order_parameter(theta_test)
        residual_test = kuramoto_residual(theta_test, omega_test, W_test, K_test)
        print(f"  Test case: r={r_test:.4f}, residual_norm={np.linalg.norm(residual_test):.2e}")
        print(f"  Theta values: {theta_test}")
    else:
        print("  Test case failed!")
    
    print("\n" + "="*50 + "\n")
    
    # 1) Build a connected random graph
    n = 12
    p = 0.25
    capacity = 1.0
    W, G = generate_connected_random_graph(n=n, p=p, capacity=capacity)
    draw_graph(G, node_labels=True, capacity_labels=True, W=W, seed=7)
    
    # Debug: check graph properties
    print(f"Graph properties:")
    print(f"  Number of nodes: {G.number_of_nodes()}")
    print(f"  Number of edges: {G.number_of_edges()}")
    print(f"  Connected: {nx.is_connected(G)}")
    print(f"  Average degree: {2*G.number_of_edges()/G.number_of_nodes():.2f}")
    print(f"  Edge density: {nx.density(G):.3f}")
    
    # Test a simple synchronization case with this graph
    print(f"\nTesting synchronization with current graph...")
    omega_test = np.ones(n) * 0.1  # All same frequency
    K_test = 1.0
    theta_test, ok_test = solve_kuramoto(np.zeros(n), omega_test, W, K_test)
    if ok_test:
        r_test = compute_order_parameter(theta_test)
        print(f"  Simple sync test: r={r_test:.4f}")
    else:
        print(f"  Simple sync test: FAILED")
    
    # Test with improved solver
    print(f"Testing with improved solver...")
    theta_test2, ok_test2 = solve_kuramoto(np.zeros(n), omega_test, W, K_test)
    r_test2 = compute_order_parameter(theta_test2) if ok_test2 else 0.0
    print(f"  Improved solver: r={r_test2:.4f}, ok={ok_test2}")
    
    print("\n" + "="*50 + "\n")

    # 2) Choose omega (heterogeneous for clearer effects)
    # To make Braess's paradox/desynchronization more likely, use a more "imbalanced" omega
    P = 0.2  # Reduced from 1.0 to make synchronization more achievable
    omega = np.zeros(n, dtype=float)
    idx = np.arange(n)
    np.random.shuffle(idx)
    # Create more imbalance: 70% positive, 30% negative for stronger effects
    positive_count = int(0.7 * n)
    omega[idx[:positive_count]] = +P
    omega[idx[positive_count:]] = -P
    omega = omega + 0.02 * np.random.randn(n)  # reduced jitter
    
    print(f"\nNow testing with heterogeneous omega...")

    # 3) Sweep K and find Kc (baseline)
    K_vals = np.geomspace(0.05, 2.0, num=100)  # adjusted range for smaller omega
    
    # Test with more homogeneous omega first
    print(f"Testing with homogeneous omega...")
    omega_homog = np.ones(n) * 0.1  # All same frequency
    Kc_homog, _, r_homog = sweep_K_and_find_Kc(
        omega_homog, W, K_vals, theta_guess=np.zeros(len(omega_homog)), r_threshold=0.7
    )
    print(f"  Homogeneous omega: Kc={Kc_homog:.6g}, max_r={np.max(r_homog):.4f}")
    
    print(f"\nNow testing with heterogeneous omega...")
    
    # IMPORTANT FIX: avoid relying on 'n' in outer scopes; tie to omega length
    Kc_base, K_grid, r_base = sweep_K_and_find_Kc(
        omega, W, K_vals, theta_guess=np.zeros(len(omega)), r_threshold=0.7  # lower threshold
    )
    print(f"Baseline Kc ≈ {Kc_base:.6g}")
    print(f"Max r value in baseline: {np.max(r_base):.4f}")
    print(f"Min r value in baseline: {np.min(r_base):.4f}")
    print(f"Omega distribution: {np.sum(omega > 0)} positive, {np.sum(omega < 0)} negative")
    print(f"Omega values: {omega}")
    print(f"Omega range: [{np.min(omega):.3f}, {np.max(omega):.3f}]")

    # 4) Try adding a missing edge that increases Kc (Braess-like effect)
    missing = list_missing_edges(W)
    if not missing:
        raise RuntimeError("Graph is complete; rerun with lower p or larger n to have missing edges.")

    print(f"Testing {len(missing)} missing edges for Braess paradox...")
    
    best_edge = None
    best_delta = -np.inf
    best_Kc_after = None
    best_r_after = None
    best_W_plus = None
    
    # Test all missing edges (not just random sample)
    for (i, j) in missing:
        W_plus = add_edge(W, i, j, capacity=1.0)
        Kc_after, _, r_after = sweep_K_and_find_Kc(
            omega, W_plus, K_vals, theta_guess=np.zeros(len(omega)), r_threshold=0.7
        )
        delta = (Kc_after if Kc_after is not None else np.inf) - (Kc_base if Kc_base is not None else np.inf)
        
        # Only print significant changes to avoid spam
        if abs(delta) > 0.01:
            print(f"Edge ({i},{j}): Kc_after={Kc_after:.6g}, delta={delta:+.6g}, max_r={np.max(r_after):.4f}")
        
        if (Kc_after is not None) and (Kc_base is not None) and (delta > best_delta):
            best_delta = delta
            best_edge = (i, j)
            best_Kc_after = Kc_after
            best_r_after = r_after
            best_W_plus = W_plus

    if best_edge is None:
        print("Did not find a Braess edge among sampled candidates. Try a different random seed or increase M.")
    else:
        print(f"Braess candidate: add edge {best_edge[0]}–{best_edge[1]}")
        print(f"Kc (base)  ≈ {Kc_base:.6g}")
        print(f"Kc (after) ≈ {best_Kc_after:.6g}   ΔKc = {best_Kc_after - Kc_base:+.6g}")

        # Draw modified network
        G_plus = G.copy()
        G_plus.add_edge(*best_edge)
        draw_graph(G_plus, node_labels=True, capacity_labels=True, W=best_W_plus, seed=7)

        # Plot r(K) curves + vertical Kc lines
        plt.figure()
        plt.plot(K_grid, r_base, label="original")
        plt.plot(K_grid, best_r_after, label=f"add edge {best_edge[0]}-{best_edge[1]}")
        if Kc_base is not None:
            plt.axvline(Kc_base, ls="--", label=f"Kc base ≈ {Kc_base:.3g}")
        if best_Kc_after is not None:
            plt.axvline(best_Kc_after, ls="--", label=f"Kc after ≈ {best_Kc_after:.3g}")
        plt.xscale("log")
        plt.xlabel("K (log scale)")
        plt.ylabel("steady-state r")
        plt.title("Braess paradox: r(K) before vs after adding one edge (desynchronization)")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend()
        plt.show()

    print("Connected:", nx.is_connected(G))
    print("Adjacency matrix:\n", W)
