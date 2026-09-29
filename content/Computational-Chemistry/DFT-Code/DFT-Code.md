

### Download the Code

* [Download scf_na_k.py](./scf_na_k.py)

### Code Preview:
```python
import matplotlib.pyplot as plt
import numpy as np

#1. initial setup
x = np.linspace(-12, 12, 400)
dx = x[1] - x[0]
n_elec = 4  # na = 1, k = 1 1 * 4 = 4
kbT = 0.02  # Fermi-Dirac temperature smearing (Hartree)

atom_centers = [-6.0, -2.0, 2.0, 6.0]
atom_types = ["Na", "K", "Na", "K"]
atom_depths = {
    "Na": -5.5,
    "K": -4.2,
}  # Na nucleus binds valence electrons more tightly
atom_widths = {"Na": 0.4, "K": 0.6}  # K core is broader

# Construct V_ext(x) by superposing atomic core potential wells
V_ext = np.zeros_like(x)
for pos, at_type in zip(atom_centers, atom_types):
    V_ext += atom_depths[at_type] * np.exp(-((x - pos) ** 2) / atom_widths[at_type])

# Kinetic Energy Operator T (Finite Difference)
N = len(x)
T = (
    -0.5
    * (
        np.diag(-2 * np.ones(N))
        + np.diag(np.ones(N - 1), 1)
        + np.diag(np.ones(N - 1), -1)
    )
    / (dx**2)
)

# SAD Initial Density Guess
n = np.zeros_like(x)
for pos in atom_centers:
    n += np.exp(-((x - pos) ** 2) / 0.5)
n *= n_elec / (np.sum(n) * dx)


def find_fermi_occupations(evals, n_elec, kbT):
    """Finds Fermi level (mu) and computes Fermi-Dirac factors f_i."""
    target_occ = n_elec / 2.0
    mu_min, mu_max = evals[0] - 1.0, evals[-1] + 1.0
    for _ in range(60):
        mu = 0.5 * (mu_min + mu_max)
        x_arg = np.clip((evals - mu) / kbT, -100, 100)
        f = 1.0 / (1.0 + np.exp(x_arg))
        if np.sum(f) > target_occ:
            mu_max = mu
        else:
            mu_min = mu
    return f, mu

#2. main scf loop
alpha, max_iter, tol = 0.2, 100, 1e-5
for scf_step in range(1, max_iter + 1):
    V_hartree = 0.5 * n
    V_xc = -((3.0 / np.pi * n) ** (1.0 / 3.0))
    V_eff = V_ext + V_hartree + V_xc

    H_ks = T + np.diag(V_eff)
    evals, evecs = np.linalg.eigh(H_ks)

    orbitals = evecs / np.sqrt(dx)
    f_i, mu = find_fermi_occupations(evals, n_elec, kbT)
    n_new = 2.0 * np.sum(f_i * (orbitals**2), axis=1)

    diff = np.sum(np.abs(n_new - n)) * dx
    if diff < tol:
        print(f"SCF Converged in {scf_step} iterations!")
        print(f"Fermi Level (E_F): {mu:.4f} Ha")
        break
    n = alpha * n_new + (1 - alpha) * n

#3 dos
E_mesh = np.linspace(evals[0] - 0.5, evals[8] + 0.5, 500)
sigma = 0.05  #how much to broaen
dos = np.sum(
    [
        np.exp(-((E_mesh - e) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        for e in evals
    ],
    axis=0,
)

#4 visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# Subplot 1: Real-Space System Profile
ax1.plot(x, V_ext, "k--", label="$V_{ext}(x)$")
ax1.plot(x, n, "b-", lw=2, label="Electron Density $n(x)$")
ax1.set_title("Na-K Supercell Real-Space Profile")
ax1.set_xlabel("Position x (bohr)")
ax1.set_ylabel("Energy / Density")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Density of States (DOS)
ax2.plot(E_mesh, dos, "r-", lw=2, label="DOS $g(E)$")
ax2.axvline(
    mu, color="k", linestyle=":", label=f"Fermi Level ($E_F = {mu:.2f}$ Ha)"
)
ax2.fill_between(
    E_mesh,
    0,
    dos,
    where=(E_mesh <= mu),
    color="red",
    alpha=0.3,
    label="Occupied States",
)
ax2.set_title("Density of States (DOS)")
ax2.set_xlabel("Energy E (Hartree)")
ax2.set_ylabel("g(E)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

* Shows full SCF implementation of a toy Na-K alloy with 6 atoms
* Exports DOS projection of system with matplotlib

### DOS Projection Explanation![[Figure_1_na_k.png]]
- This DOS projection shows both the real space (left) profile and the reciprocal space occupation of each state (right). 
- In this alloy, we can see that all occupied sites are below the Fermi Level. This is expected for a non-interacting system. 
