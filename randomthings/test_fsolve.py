import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def test_simple_fsolve():
    """Test fsolve with a simple function"""
    print("=== Testing Simple fsolve ===")
    
    # Simple function: f(x) = x^2 - 4
    def simple_func(x):
        return x**2 - 4
    
    # Should find x = 2 or x = -2
    x0 = 1.0
    x_sol, info, ier, msg = fsolve(simple_func, x0, full_output=True)
    print(f"Simple function: x = {x_sol}, success = {ier == 1}, message = {msg}")
    print(f"Function value at solution: {simple_func(x_sol)}")
    print()

def test_kuramoto_simple():
    """Test Kuramoto solver with a very simple 2-node case"""
    print("=== Testing Simple Kuramoto (2 nodes) ===")
    
    def kuramoto_residual_2node(theta, omega, W, K):
        """Residual for 2-node Kuramoto system"""
        residual = np.zeros(2)
        residual[0] = omega[0] + K * W[0,1] * np.sin(theta[1] - theta[0])
        residual[1] = omega[1] + K * W[1,0] * np.sin(theta[0] - theta[1])
        return residual
    
    # Simple 2-node system
    omega = np.array([0.1, -0.1])  # Different frequencies
    W = np.array([[0, 1], [1, 0]])  # Connected
    K = 1.0
    
    print(f"omega = {omega}")
    print(f"W = \n{W}")
    print(f"K = {K}")
    
    # Try different initial conditions
    initial_guesses = [
        np.zeros(2),
        np.array([0, np.pi]),
        np.array([np.pi/2, -np.pi/2]),
        np.random.uniform(0, 2*np.pi, 2)
    ]
    
    for i, theta_init in enumerate(initial_guesses):
        print(f"\nTrying initial guess {i+1}: {theta_init}")
        try:
            theta_sol, info, ier, msg = fsolve(
                kuramoto_residual_2node, theta_init, args=(omega, W, K), 
                full_output=True, xtol=1e-8, maxfev=1000
            )
            residual = kuramoto_residual_2node(theta_sol, omega, W, K)
            residual_norm = np.linalg.norm(residual)
            print(f"  Success: {ier == 1}")
            print(f"  Solution: {theta_sol}")
            print(f"  Residual norm: {residual_norm:.2e}")
            print(f"  Message: {msg}")
            
            if ier == 1 and residual_norm < 1e-6:
                print("  ✓ Good solution found!")
                break
        except Exception as e:
            print(f"  Error: {e}")
    print()

def test_kuramoto_3node():
    """Test Kuramoto solver with 3-node case"""
    print("=== Testing Kuramoto (3 nodes) ===")
    
    def kuramoto_residual_3node(theta, omega, W, K):
        """Residual for 3-node Kuramoto system"""
        residual = np.zeros(3)
        for i in range(3):
            s = 0.0
            for j in range(3):
                if W[i,j] > 0:
                    s += W[i,j] * np.sin(theta[j] - theta[i])
            residual[i] = omega[i] + K * s
        return residual
    
    # 3-node system
    omega = np.array([0.1, -0.1, 0.0])
    W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    K = 1.0
    
    print(f"omega = {omega}")
    print(f"W = \n{W}")
    print(f"K = {K}")
    
    # Try different initial conditions
    initial_guesses = [
        np.zeros(3),
        np.array([0, np.pi, 0]),
        np.random.uniform(0, 2*np.pi, 3),
        np.linspace(0, 2*np.pi, 3, endpoint=False)
    ]
    
    for i, theta_init in enumerate(initial_guesses):
        print(f"\nTrying initial guess {i+1}: {theta_init}")
        try:
            theta_sol, info, ier, msg = fsolve(
                kuramoto_residual_3node, theta_init, args=(omega, W, K), 
                full_output=True, xtol=1e-8, maxfev=1000
            )
            residual = kuramoto_residual_3node(theta_sol, omega, W, K)
            residual_norm = np.linalg.norm(residual)
            print(f"  Success: {ier == 1}")
            print(f"  Solution: {theta_sol}")
            print(f"  Residual norm: {residual_norm:.2e}")
            print(f"  Message: {msg}")
            
            if ier == 1 and residual_norm < 1e-6:
                print("  ✓ Good solution found!")
                break
        except Exception as e:
            print(f"  Error: {e}")
    print()

def test_kuramoto_large():
    """Test Kuramoto solver with larger system"""
    print("=== Testing Kuramoto (12 nodes) ===")
    
    def kuramoto_residual(theta, omega, W, K):
        """Residual for Kuramoto system"""
        n = len(theta)
        residual = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                if W[i,j] > 0:
                    s += W[i,j] * np.sin(theta[j] - theta[i])
            residual[i] = omega[i] + K * s
        return residual
    
    # 12-node system (like in main code)
    n = 12
    omega = np.random.uniform(-0.2, 0.2, n)
    W = np.random.choice([0, 1], size=(n, n), p=[0.7, 0.3])
    W = (W + W.T) / 2  # Make symmetric
    np.fill_diagonal(W, 0)  # No self-loops
    K = 1.0
    
    print(f"omega range: [{omega.min():.3f}, {omega.max():.3f}]")
    print(f"W has {np.sum(W > 0)} edges")
    print(f"K = {K}")
    
    # Try different initial conditions
    initial_guesses = [
        np.zeros(n),
        np.random.uniform(0, 2*np.pi, n),
        np.linspace(0, 2*np.pi, n, endpoint=False)
    ]
    
    for i, theta_init in enumerate(initial_guesses):
        print(f"\nTrying initial guess {i+1}")
        try:
            theta_sol, info, ier, msg = fsolve(
                kuramoto_residual, theta_init, args=(omega, W, K), 
                full_output=True, xtol=1e-6, maxfev=2000
            )
            residual = kuramoto_residual(theta_sol, omega, W, K)
            residual_norm = np.linalg.norm(residual)
            print(f"  Success: {ier == 1}")
            print(f"  Residual norm: {residual_norm:.2e}")
            print(f"  Message: {msg}")
            
            if ier == 1 and residual_norm < 1e-3:
                print("  ✓ Good solution found!")
                break
        except Exception as e:
            print(f"  Error: {e}")
    print()

def test_scipy_version():
    """Check scipy version and available solvers"""
    print("=== System Information ===")
    import scipy
    print(f"SciPy version: {scipy.__version__}")
    print(f"NumPy version: {np.__version__}")
    print()

def test_custom_solver():
    """Test custom iterative solver for Kuramoto system"""
    print("=== Testing Custom Iterative Solver ===")
    
    def kuramoto_residual(theta, omega, W, K):
        """Residual for Kuramoto system"""
        n = len(theta)
        residual = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                if W[i,j] > 0:
                    s += W[i,j] * np.sin(theta[j] - theta[i])
            residual[i] = omega[i] + K * s
        return residual
    
    def custom_kuramoto_solver(omega, W, K, theta_init=None, max_iter=1000, tol=1e-6):
        """Custom iterative solver for Kuramoto steady state"""
        n = len(omega)
        if theta_init is None:
            theta_init = np.zeros(n)
        
        theta = theta_init.copy()
        
        for iter in range(max_iter):
            theta_old = theta.copy()
            
            # Update each phase using a simple iteration
            for i in range(n):
                # Compute the coupling term
                coupling = 0.0
                for j in range(n):
                    if W[i,j] > 0:
                        coupling += W[i,j] * np.sin(theta[j] - theta[i])
                
                # Update phase (this is a simplified approach)
                theta[i] = theta[i] + 0.1 * (omega[i] + K * coupling)
            
            # Check convergence
            if np.linalg.norm(theta - theta_old) < tol:
                return theta, True, iter
        
        return theta, False, max_iter
    
    # Test with 12-node system
    n = 12
    omega = np.random.uniform(-0.2, 0.2, n)
    W = np.random.choice([0, 1], size=(n, n), p=[0.7, 0.3])
    W = (W + W.T) / 2  # Make symmetric
    np.fill_diagonal(W, 0)  # No self-loops
    K = 1.0
    
    print(f"omega range: [{omega.min():.3f}, {omega.max():.3f}]")
    print(f"W has {np.sum(W > 0)} edges")
    print(f"K = {K}")
    
    # Try custom solver
    theta_sol, success, iterations = custom_kuramoto_solver(omega, W, K)
    residual = kuramoto_residual(theta_sol, omega, W, K)
    residual_norm = np.linalg.norm(residual)
    
    print(f"\nCustom solver results:")
    print(f"  Success: {success}")
    print(f"  Iterations: {iterations}")
    print(f"  Residual norm: {residual_norm:.2e}")
    
    if success and residual_norm < 1e-3:
        print("  ✓ Good solution found!")
    print()

def test_time_integration_solver():
    """Test time integration solver for Kuramoto system"""
    print("=== Testing Time Integration Solver ===")
    
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
    
    def compute_order_parameter(theta):
        """Compute the Kuramoto order parameter"""
        return np.abs(np.exp(1j * theta).mean())
    
    # Test with 12-node system
    n = 12
    omega = np.random.uniform(-0.2, 0.2, n)
    W = np.random.choice([0, 1], size=(n, n), p=[0.7, 0.3])
    W = (W + W.T) / 2  # Make symmetric
    np.fill_diagonal(W, 0)  # No self-loops
    K = 1.0
    
    print(f"omega range: [{omega.min():.3f}, {omega.max():.3f}]")
    print(f"W has {np.sum(W > 0)} edges")
    print(f"K = {K}")
    
    # Try time integration solver
    theta_sol, success, final_time = time_integration_solver(omega, W, K)
    r = compute_order_parameter(theta_sol)
    
    print(f"\nTime integration results:")
    print(f"  Success: {success}")
    print(f"  Final time: {final_time:.2f}")
    print(f"  Order parameter r: {r:.4f}")
    
    # Check residual
    residual = kuramoto_dynamics(theta_sol, omega, W, K)
    residual_norm = np.linalg.norm(residual)
    print(f"  Residual norm: {residual_norm:.2e}")
    
    if success and residual_norm < 1e-2:
        print("  ✓ Good solution found!")
    print()

def test_time_integration_vs_fsolve():
    """Compare time integration with fsolve for small systems"""
    print("=== Comparing Time Integration vs fsolve ===")
    
    def kuramoto_residual_3node(theta, omega, W, K):
        """Residual for 3-node Kuramoto system"""
        residual = np.zeros(3)
        for i in range(3):
            s = 0.0
            for j in range(3):
                if W[i,j] > 0:
                    s += W[i,j] * np.sin(theta[j] - theta[i])
            residual[i] = omega[i] + K * s
        return residual
    
    def kuramoto_dynamics_3node(theta, omega, W, K):
        """Compute the time derivatives for 3-node Kuramoto system"""
        n = len(theta)
        dtheta = np.zeros(n)
        for i in range(n):
            coupling = 0.0
            for j in range(n):
                if W[i,j] > 0:
                    coupling += W[i,j] * np.sin(theta[j] - theta[i])
            dtheta[i] = omega[i] + K * coupling
        return dtheta
    
    def time_integration_solver_3node(omega, W, K, theta_init=None, dt=0.01, max_time=50, tol=1e-6):
        """Time integration solver for 3-node Kuramoto system"""
        n = len(omega)
        if theta_init is None:
            theta_init = np.zeros(n)
        
        theta = theta_init.copy()
        t = 0.0
        
        # Store history for convergence check
        theta_history = []
        
        while t < max_time:
            # Compute derivatives
            dtheta = kuramoto_dynamics_3node(theta, omega, W, K)
            
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
    
    # Test with 3-node system
    omega = np.array([0.1, -0.1, 0.0])
    W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    K = 1.0
    
    print(f"omega = {omega}")
    print(f"W = \n{W}")
    print(f"K = {K}")
    
    # Try fsolve
    print("\n--- fsolve method ---")
    try:
        theta_fsolve, info, ier, msg = fsolve(
            kuramoto_residual_3node, np.zeros(3), args=(omega, W, K), 
            full_output=True, xtol=1e-8, maxfev=1000
        )
        residual_fsolve = kuramoto_residual_3node(theta_fsolve, omega, W, K)
        residual_norm_fsolve = np.linalg.norm(residual_fsolve)
        print(f"  Success: {ier == 1}")
        print(f"  Solution: {theta_fsolve}")
        print(f"  Residual norm: {residual_norm_fsolve:.2e}")
        print(f"  Message: {msg}")
    except Exception as e:
        print(f"  Error: {e}")
        residual_norm_fsolve = np.inf
    
    # Try time integration
    print("\n--- Time integration method ---")
    theta_time, success_time, final_time = time_integration_solver_3node(omega, W, K)
    residual_time = kuramoto_residual_3node(theta_time, omega, W, K)
    residual_norm_time = np.linalg.norm(residual_time)
    print(f"  Success: {success_time}")
    print(f"  Solution: {theta_time}")
    print(f"  Residual norm: {residual_norm_time:.2e}")
    print(f"  Final time: {final_time:.2f}")
    
    # Compare solutions
    if residual_norm_fsolve < np.inf and residual_norm_time < np.inf:
        solution_diff = np.linalg.norm(theta_fsolve - theta_time)
        print(f"\n--- Comparison ---")
        print(f"  Solution difference: {solution_diff:.2e}")
        print(f"  fsolve residual: {residual_norm_fsolve:.2e}")
        print(f"  Time integration residual: {residual_norm_time:.2e}")
    
    print()

def test_main_simulation_parameters():
    """Test time integration with the exact parameters from main simulation"""
    print("=== Testing Main Simulation Parameters ===")
    
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
    
    def compute_order_parameter(theta):
        """Compute the Kuramoto order parameter"""
        return np.abs(np.exp(1j * theta).mean())
    
    # Use exact parameters from main simulation
    n = 12
    p = 0.25
    capacity = 1.0
    
    # Generate the same graph
    np.random.seed(7)  # Same seed as main simulation
    W = np.random.choice([0, 1], size=(n, n), p=[1-p, p])
    W = (W + W.T) / 2  # Make symmetric
    np.fill_diagonal(W, 0)  # No self-loops
    
    # Use exact omega from main simulation
    P = 0.2
    omega = np.zeros(n, dtype=float)
    idx = np.arange(n)
    np.random.shuffle(idx)
    positive_count = int(0.7 * n)
    omega[idx[:positive_count]] = +P
    omega[idx[positive_count:]] = -P
    omega = omega + 0.02 * np.random.randn(n)
    
    print(f"omega range: [{omega.min():.3f}, {omega.max():.3f}]")
    print(f"W has {np.sum(W > 0)} edges")
    print(f"omega values: {omega}")
    
    # Test with different K values
    K_values = [0.05, 0.1, 0.5, 1.0, 2.0]
    
    for K in K_values:
        print(f"\nTesting K = {K}")
        theta_sol, success, final_time = time_integration_solver(omega, W, K)
        r = compute_order_parameter(theta_sol)
        residual_norm = np.linalg.norm(kuramoto_dynamics(theta_sol, omega, W, K))
        
        print(f"  Success: {success}")
        print(f"  Final time: {final_time:.2f}")
        print(f"  Order parameter r: {r:.4f}")
        print(f"  Residual norm: {residual_norm:.2e}")
        
        if success and residual_norm < 1e-2:
            print("  ✓ Good solution!")
        print()
    
    print()

if __name__ == "__main__":
    test_scipy_version()
    test_simple_fsolve()
    test_kuramoto_simple()
    test_kuramoto_3node()
    test_kuramoto_large()
    test_custom_solver()
    test_time_integration_solver()
    test_time_integration_vs_fsolve()
    test_main_simulation_parameters() 