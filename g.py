import numpy as np

def G(theta, v, K, a, W, edge_to_add, omega):
    """
    MEGL constraint map for edge addition analysis.
    
    Parameters:
    -----------
    theta : array of shape (n,)
        Phase angles of oscillators
    v : array of shape (n,)
        Eigenvector (must satisfy: ||v||=1, v^T·1=0)
    K : float
        Coupling strength
    a : float
        Edge addition parameter (a=0: no edge, a=1: full edge added)
    W : array of shape (n,n)
        Original adjacency matrix (weighted)
    edge_to_add : tuple (i, j)
        Indices of nodes to connect with new edge
    omega : array of shape (n,)
        Natural frequencies
    
    Returns:
    --------
    G : array of shape (2n+3,)
        Constraint vector [F(θ), DF(θ)·v, θ₁, v^T·1, v^T·v-1]
    
    Structure:
    ----------
    G = [Ω + K·s(θ) + a·g(θ),    # n equations: steady state with edge perturbation
         K·c(θ)·v + a·Dg(θ)·v,   # n equations: DF·v = 0 (eigenvector condition)
         θ₁,                      # 1 equation: gauge fixing (e₁)
         v^T·1,                   # 1 equation: orthogonality to ones
         v^T·v - 1]               # 1 equation: normalization
    
    where:
        s(θ) = Σⱼ Wᵢⱼ sin(θⱼ - θᵢ)  (Kuramoto coupling)
        g(θ) = sin(θⱼ - θᵢ) for edge (i,j), zero elsewhere
        c(θ)ᵢⱼ = Wᵢⱼ cos(θⱼ - θᵢ)   (Jacobian entries)
    """
    n = len(theta)
    
    # Phase differences: dth[i,j] = θⱼ - θᵢ
    dth = theta[np.newaxis, :] - theta[:, np.newaxis]
    
    # s(θ) = Σⱼ Wᵢⱼ sin(θⱼ - θᵢ)
    s_theta = (W * np.sin(dth)).sum(axis=1)
    
    # g(θ) = effect of new edge
    # Only non-zero at nodes i and j where edge is added
    g_theta = np.zeros(n)
    i, j = edge_to_add
    g_theta[i] = np.sin(theta[j] - theta[i])
    g_theta[j] = np.sin(theta[i] - theta[j])
    
    # F(θ) = Ω + K·s(θ) + a·g(θ)
    F = omega + K * s_theta + a * g_theta
    
    # c(θ) = Wᵢⱼ cos(θⱼ - θᵢ) (Jacobian of s)
    C = W * np.cos(dth)
    
    # Dg(θ) = Jacobian of g (new edge effect)
    # Only non-zero for the edge being added
    Dg = np.zeros((n, n))
    i, j = edge_to_add
    Dg[i, j] = np.cos(theta[j] - theta[i])
    Dg[j, i] = np.cos(theta[i] - theta[j])
    Dg[i, i] = -np.cos(theta[j] - theta[i])
    Dg[j, j] = -np.cos(theta[i] - theta[j])
    
    # DF·v = K·c(θ)·v + a·Dg(θ)·v
    # Using: (c·v)ᵢ = Σⱼ cᵢⱼ(vⱼ - vᵢ) (exploiting symmetry)
    DFv = K * (C * (v[np.newaxis, :] - v[:, np.newaxis])).sum(axis=1) + a * Dg.dot(v)
    
    # Constraints
    gauge_fix = theta[0]                    # θ₁ = 0 (e₁)
    orthogonality = np.dot(v, np.ones(n))   # v^T·1 = 0
    normalization = np.dot(v, v) - 1        # v^T·v = 1
    
    return np.concatenate([
        F,                              # n equations
        DFv,                            # n equations  
        [gauge_fix],                    # 1 equation
        [orthogonality],                # 1 equation
        [normalization]                 # 1 equation
    ])

theta_0 = np.array([0.0, 0.0, 0.0])
v_0 = np.array([1.0, 0.0, 0.0])
K_0 = 1.0
edge_to_add = (0, 1)
W = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
omega = np.array([1.0, 1.0, 1.0])
example_g = G(theta_0, v_0, K_0, edge_to_add, W, omega)
print(example_g)


def predict_braess(theta_0, v_0, K_0, edge_to_add, W, omega):
    """
    Predict whether adding an edge causes Braess paradox.
    
    Returns k'(0) = -v(0)^T·g(θ(0)) / [v(0)^T·s(θ(0))]
    
    If k'(0) > 0: adding edge increases K (Braess paradox)
    If k'(0) < 0: adding edge decreases K (normal behavior)
    
    Note: Scale v_0 so that v_0^T·ω > 0 to control denominator sign.
    """
    n = len(theta_0)
    
    # Compute s(θ₀)
    dth = theta_0[np.newaxis, :] - theta_0[:, np.newaxis]
    s_theta_0 = (W * np.sin(dth)).sum(axis=1)
    
    # Compute g(θ₀) for the new edge
    g_theta_0 = np.zeros(n)
    i, j = edge_to_add
    g_theta_0[i] = np.sin(theta_0[j] - theta_0[i])
    g_theta_0[j] = np.sin(theta_0[i] - theta_0[j])
    
    # Scale v_0 so denominator is positive (v^T·ω > 0)
    if np.dot(v_0, omega) < 0:
        v_0 = -v_0
    
    # k'(0) formula
    numerator = -np.dot(v_0, g_theta_0)
    denominator = np.dot(v_0, s_theta_0)
    
    k_prime_0 = numerator / denominator
    
    return k_prime_0