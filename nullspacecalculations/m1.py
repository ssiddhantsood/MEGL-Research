import numpy as np

# -------------------------------------------------------
# Function: Kuramoto Jacobian matrix J(θ)
# -------------------------------------------------------
# Input:
#   theta : array of phase angles (θ₁, θ₂, ..., θₙ)
#   A     : adjacency matrix (symmetric if undirected)
#   K     : coupling strength (scalar)
#
# Output:
#   J : n×n Jacobian matrix, where
#       J[i,j] = ∂F_i / ∂θ_j  for  F_i = Σ_j K*A_ij*sin(θ_j - θ_i)
# -------------------------------------------------------
def compute_J(theta, A, K=1.0):
    n = len(theta)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # derivative of sin(θ_j - θ_i) is cos(θ_j - θ_i)
                J[i, j] = K * A[i, j] * np.cos(theta[j] - theta[i])
        # diagonal chosen so each row sums to zero (Laplacian structure)
        J[i, i] = -np.sum(J[i, :])
    return J


# -------------------------------------------------------
# Function: Numeric computation of M = D_θ ( J(θ) v )
# -------------------------------------------------------
# Input:
#   theta : array of phase angles
#   A     : adjacency matrix
#   v     : vector being multiplied by J(θ)
#   eps   : small perturbation for finite-difference derivative
#
# Output:
#   M : n×n matrix where column l is ∂(Jv)/∂θ_l
# -------------------------------------------------------
def compute_M_numeric(theta, A, v, eps=1e-7):
    n = len(theta)
    M = np.zeros((n, n))

    # baseline product J(θ) v
    Jv_base = compute_J(theta, A) @ v

    # loop over each θ_l coordinate to compute finite differences
    for l in range(n):
        theta_perturbed = theta.copy()
        theta_perturbed[l] += eps  # small change in θ_l

        Jv_perturbed = compute_J(theta_perturbed, A) @ v
        M[:, l] = (Jv_perturbed - Jv_base) / eps  # numerical partial derivative

    return M


# -------------------------------------------------------
# Test: Check if M * 1 = 0  (row sums ≈ 0)
# -------------------------------------------------------

rng = np.random.default_rng(1)
n = 10  # number of oscillators

# Symmetric (undirected) adjacency matrix with random positive weights
A_random = rng.uniform(0.2, 1.0, size=(n, n))
A_symmetric = 0.5 * (A_random + A_random.T)
np.fill_diagonal(A_symmetric, 0.0)  # no self-connections

# Random phase angles θ_i in [-π, π)
theta_random = rng.uniform(-np.pi, np.pi, size=n)

# Random eigenvector candidate (just a test vector)
v_random = rng.normal(size=n)

# Compute M and its product with the all-ones vector
M_matrix = compute_M_numeric(theta_random, A_symmetric, v_random)
ones_vector = np.ones(n)
M_times_one = M_matrix @ ones_vector

# Display results
print(f"Row-sum norm ||M * 1|| = {np.linalg.norm(M_times_one):.3e}")
print("Row sums (each entry of M * 1) =", M_times_one)
