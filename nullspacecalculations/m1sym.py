import sympy as sp

# Number of oscillators (keep small for clarity)
n = 3

# Define symbolic variables
theta = sp.symbols(f"theta0:{n}")
v = sp.symbols(f"v0:{n}")
K = sp.Symbol("K", real=True)
A = sp.MatrixSymbol("A", n, n)  # symbolic adjacency; we’ll treat it as symmetric later

# Build symbolic Jacobian J(θ)
J = sp.MutableDenseMatrix(n, n, [0]*n*n)

for i in range(n):
    for j in range(n):
        if i != j:
            J[i,j] = K * sp.Symbol(f"A{i}{j}") * sp.cos(theta[j] - theta[i])
    J[i,i] = -sum(J[i,k] for k in range(n) if k != i)

# Define M = D_θ (J(θ) v)
v_vec = sp.Matrix(v)
Jv = J * v_vec
M = sp.zeros(n, n)

for ell in range(n):
    M[:, ell] = sp.Matrix([sp.diff(Jv[i], theta[ell]) for i in range(n)])

# Compute M * 1 (row sums)

hb ntdnsones = sp.Matrix([1]*n)
#ones = sp.Matrix(v_vec)
M1 = sp.simplify(M * ones)

sp.pretty_print(M)
sp.pretty_print(M1)
