import numpy as np

# 1: intitial guess (usually with SAD algorithm)
# if N basis functions such that Kohn sham orbital = c_1B_1 + c_2B_2 + ... + c_NB_N (denoting B as the basis function)
# then D = N x N matrix

# 1.1: Real space crystal set

x = np.linspace(-5, 5, 200)
dx = x[1] - x[0]  # difference between each step
n_elec = 2
kbT = 0.02  # Fermi-Dirac smearing width (temperature in Hartree)

#this is where the sigma term comes in

# 1.2 Set up potentials, KE operator

V_ext = -5.0 * (np.exp(-((x + 1.5) ** 2) / 0.2) + np.exp(-((x - 1.5) ** 2) / 0.2))


N = len(x)  # total length of grid
T = (
    -0.5  # shcrodinger single particl coeffs
    * (
        np.diag(-2 * np.ones(N))
        + np.diag(np.ones(N - 1), 1)
        + np.diag(np.ones(N - 1), -1)
    )
    / (dx**2)
)


# Helper function to find chemical potential (Fermi level) and calculate f_i
def find_fermi_occupations(evals, n_elec, kbT):
    target_occ = n_elec / 2.0  # Spatial orbital occupancy target
    mu_min, mu_max = evals[0] - 1.0, evals[-1] + 1.0
    for _ in range(50):  # Bisection root search for Fermi level (mu)
        mu = 0.5 * (mu_min + mu_max)
        x_arg = np.clip((evals - mu) / kbT, -100, 100)
        f = 1.0 / (1.0 + np.exp(x_arg))
        if np.sum(f) > target_occ:
            mu_max = mu
        else:
            mu_min = mu
    return f, mu


# 1.3 Acutal SAD Guess
# Uses Gaussian Approximation for density modeling

n_atom1 = np.exp(-((x + 1.5) ** 2) / 0.5)  # at x = -1.5
n_atom2 = np.exp(-((x - 1.5) ** 2) / 0.5)  # at x = 1.5

n = n_atom1 + n_atom2

n = n * (
    n_elec / (np.sum(n) * dx)
)  # makes sure that n comes under total density of 2
alpha = 0.2  # mixiing 20% new
max_iter = 100
tol = 1e-5


# 2 Main SCF Loop

for scf_step in range(1, max_iter + 1):
    # 2.1 Calculate the potentials using the given density guess

    V_hartree = 0.5 * n  # using simple repulsions constant --> basically V_ee
    V_xc = -((3.0 / np.pi * n) ** (1.0 / 3.0))  # (LCA Fermi but 1D simplified) Dirac local exchange potential
    V_eff = V_ext + V_hartree + V_xc

    # 2.2 solves the matrix equation to get the eigenvalues (energies) and the eigenvectors (coeffs for orbials) respctively
    H_ks = T + np.diag(V_eff)
    evals, evecs = np.linalg.eigh(H_ks)

    print("energies", evals)
    print("orbitals", evecs)
    # for orbitals, there will be a different orbital "coefficient" for every column of the spatial cooredinates
    # ex) x = [1,2,3,4]
    #     e = [3.33, 5.66, 7.89, 9.44]
    #.    o = [(x1,e1). (x1, e2). (x1, e3). (x1, e4)]

    # 2.3 using new orbitals recalculates the electron density (using relation between denisty and orbital squared with Fermi-Dirac smoothing)

    orbitals = evecs / np.sqrt(dx)  # nrmalize orbitals over continuous grid
    f_i, mu = find_fermi_occupations(
        evals, n_elec, kbT
    )  # Fermi-Dirac smoothing factors f_i
    n_new = 2.0 * np.sum(
        f_i * (orbitals**2), axis=1
    )  # factor of 2 accounts for spin pair, weighted by f_i

    # 3 checks for convergence (decides loop continuace)
    diff = np.sum(np.abs(n_new - n)) * dx
    if diff < tol:
        print(f"SCF Converged in {scf_step} iterations!")
        print(f"Fermi Energy (mu): {mu:.4f} Ha")
        break

    n = alpha * n_new + (1 - alpha) * n

# 4 Calculate Density of States (DOS) using Gaussian broadening

E_mesh = np.linspace(evals[0] - 0.5, 0.5, 500)  # Energy grid for DOS plot
sigma = 0.05  # Gaussian smearing factor for delta functions

# Calculate DOS: sum of Gaussians centered at each eigenvalue
dos = np.sum(
    [
        np.exp(-((E_mesh - e) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        for e in evals
    ],
    axis=0,
)