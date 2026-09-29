### Download Code
- *[[dmft_sample_code.py|Download dmft_sample_code.py]]

### Code Preview
```python
import itertools

import math

import numpy as np

import tqdm

  

#01. Lattice work

  

def real_to_reciprocal(a1: float, a2: float, a3: float) -> tuple[float, float, float]:

a1_vec = [a1, 0, 0]

a2_vec = [0, a2, 0]

a3_vec = [0, 0, a3]

vol = np.dot(a1_vec, np.cross(a2_vec, a3_vec))

b1 = 2 * math.pi * (np.cross(a2_vec, a3_vec) / vol)

b2 = 2 * math.pi * (np.cross(a1_vec, a3_vec) / vol)

b3 = 2 * math.pi * (np.cross(a2_vec, a1_vec) / vol)

return b1, b2, b3

  

def establish_k_points(k, b1, b2, b3):

b1_vec = [b1, 0, 0]

b2_vec = [0, b2, 0]

b3_vec = [0, 0, b3]

r_range = [(a - ((k + 1) / 2)) / k for a in range(1, k + 1)] #d Monkhorst-Pack grid

fractions = list(itertools.product(r_range, repeat=3))

kpoints = []

for f in fractions:

x = b1_vec[0] * f[0] + b2_vec[0] * f[1] + b3_vec[0] * f[2]

y = b1_vec[1] * f[0] + b2_vec[1] * f[1] + b3_vec[1] * f[2]

z = b1_vec[2] * f[0] + b2_vec[2] * f[1] + b3_vec[2] * f[2]

kpoints.append([x, y, z])

return kpoints

  

def loc_greens(matsubara, chem_pot, kx, ky, kz, kinetic, self_energy):

epsilon = -2 * kinetic * (np.cos(kx) + np.cos(ky) + np.cos(kz))

g_lattice = 1.0 / (matsubara + chem_pot - epsilon - self_energy)

return np.mean(g_lattice)

  

def compute_overlap(dag_up, ann_up, dag_dn, ann_dn, beta):

def get_occupation_grid(dags, anns):

grid = np.zeros(1000)

tau_axis = np.linspace(0, beta, 1000)

for td, ta in zip(dags, anns):

if td > ta:

grid[(tau_axis >= td) | (tau_axis <= ta)] = 1.0

else:

grid[(tau_axis >= td) & (tau_axis <= ta)] = 1.0

return grid

  

grid_up = get_occupation_grid(dag_up, ann_up)

grid_dn = get_occupation_grid(dag_dn, ann_dn)

overlap_time = np.sum(grid_up * grid_dn) * (beta / 1000.0)

len_up = np.sum(grid_up) * (beta / 1000.0)

len_dn = np.sum(grid_dn) * (beta / 1000.0)

return len_up, len_dn, overlap_time

  

#02. paramters

  

beta = 20.0 # Inverse temperature

num_matsubara = 100 # Number of Matsubara frequencies

t_hopping = 0.2 # Nearest-neighbor hopping (eV)

mu = 2.0 # Chemicl potential

U_interaction = 4.0 # Local Coulomb repulsion (eV)

  

num_tau = 200 # Imaginary-time grid points

mc_steps = 20000 # Total Monte Carlo sweeps per DMFT step

burn_in = 5000 # Thermalization sweeps

max_dmft_iters = 40 # Number of self-consistency iterations

alpha = 0.2 # Linear self-energy mixing factor (make sure it is relatively low)

  
  

b1, b2, b3 = real_to_reciprocal(3.84, 3.84, 3.84)

kpoint_grid = np.array(establish_k_points(8, b1, b2, b3))

kx, ky, kz = kpoint_grid[:, 0], kpoint_grid[:, 1], kpoint_grid[:, 2]

  

n_grid = np.arange(num_matsubara)

iw_array = 1j * (2 * n_grid + 1) * np.pi / beta

tau_grid = np.linspace(0, beta, num_tau)

  

# Initial guess: \Sigma(iw) = 0

self_energy_array = np.zeros(num_matsubara, dtype=complex)

  

#3. loop

for dmft_iter in range(max_dmft_iters):

print(f"Starting DMFT Iteration {dmft_iter + 1}/{max_dmft_iters}")

  

# Step 02: Local Green's Function

g_local_array = np.zeros(num_matsubara, dtype=complex)

for n in range(num_matsubara):

g_local_array[n] = loc_greens(

iw_array[n], mu, kx, ky, kz, t_hopping, self_energy_array[n]

)

  

# Step 03: Weiss Field G_0(iw)

g0_inverse = (1.0 / g_local_array) + self_energy_array

g0 = 1.0 / g0_inverse

  

# Step 04a: Hybridization Delta(iw) -> Delta(tau)

delta_iw = iw_array + mu - g0_inverse

delta_tau_grid = np.zeros(num_tau)

for i, t in enumerate(tau_grid):

delta_tau_grid[i] = (2.0 / beta) * np.real(np.sum(delta_iw * np.exp(-iw_array * t)))

  

def get_delta_tau(tau):

tau_mod = np.mod(tau, beta)

return np.interp(tau_mod, tau_grid, delta_tau_grid)

  

# Step 04b: CT-HYB Impurity Solver

tau_dag = {'up': [], 'dn': []}

tau_ann = {'up': [], 'dn': []}

g_tau_hist = np.zeros(num_tau)

n_measurements = 0

  

for step in tqdm.tqdm(range(mc_steps)):

spin = np.random.choice(['up', 'dn'])

k = len(tau_dag[spin])

move = np.random.choice([0, 1])

  

l_up, l_dn, O_updn = compute_overlap(tau_dag['up'], tau_ann['up'], tau_dag['dn'], tau_ann['dn'], beta)

w_loc_old = np.exp(mu * (l_up + l_dn) - U_interaction * O_updn)

  

if move == 0:#insert

t_dag_new = np.random.uniform(0, beta)

t_ann_new = np.random.uniform(0, beta)

  

trial_dag = tau_dag[spin] + [t_dag_new]

trial_ann = tau_ann[spin] + [t_ann_new]

  

M_inv_new = np.array([[get_delta_tau(td - ta) for ta in trial_ann] for td in trial_dag])

det_new = np.linalg.det(M_inv_new)

  

M_inv_old = np.array([[get_delta_tau(td - ta) for ta in tau_ann[spin]] for td in tau_dag[spin]]) if k > 0 else np.array([[1.0]])

det_old = np.linalg.det(M_inv_old) if k > 0 else 1.0

  

t_dag_spin = trial_dag if spin == 'up' else tau_dag['up']

t_ann_spin = trial_ann if spin == 'up' else tau_ann['up']

t_dag_other = tau_dag['dn'] if spin == 'up' else trial_dag

t_ann_other = tau_ann['dn'] if spin == 'up' else trial_ann

  

l_up_n, l_dn_n, O_n = compute_overlap(t_dag_spin, t_ann_spin, t_dag_other, t_ann_other, beta)

w_loc_new = np.exp(mu * (l_up_n + l_dn_n) - U_interaction * O_n)

  

ratio = (beta**2 / (k + 1)**2) * np.abs(det_new / (det_old + 1e-12)) * (w_loc_new / (w_loc_old + 1e-12))

if np.random.rand() < min(1.0, ratio):

tau_dag[spin].append(t_dag_new)

tau_ann[spin].append(t_ann_new)

  

elif move == 1 and k > 0: # REMOVAL

idx = np.random.randint(0, k)

trial_dag = [tau_dag[spin][i] for i in range(k) if i != idx]

trial_ann = [tau_ann[spin][i] for i in range(k) if i != idx]

  

M_inv_old = np.array([[get_delta_tau(td - ta) for ta in tau_ann[spin]] for td in tau_dag[spin]])

det_old = np.linalg.det(M_inv_old)

  

M_inv_new = np.array([[get_delta_tau(td - ta) for ta in trial_ann] for td in trial_dag]) if k - 1 > 0 else np.array([[1.0]])

det_new = np.linalg.det(M_inv_new) if k - 1 > 0 else 1.0

  

t_dag_spin = trial_dag if spin == 'up' else tau_dag['up']

t_ann_spin = trial_ann if spin == 'up' else tau_ann['up']

t_dag_other = tau_dag['dn'] if spin == 'up' else trial_dag

t_ann_other = tau_ann['dn'] if spin == 'up' else trial_ann

  

l_up_n, l_dn_n, O_n = compute_overlap(t_dag_spin, t_ann_spin, t_dag_other, t_ann_other, beta)

w_loc_new = np.exp(mu * (l_up_n + l_dn_n) - U_interaction * O_n)

  

ratio = (k**2 / beta**2) * np.abs(det_new / (det_old + 1e-12)) * (w_loc_new / (w_loc_old + 1e-12))

if np.random.rand() < min(1.0, ratio):

tau_dag[spin].pop(idx)

tau_ann[spin].pop(idx)

  

# Measurement

if step > burn_in and step % 10 == 0:

for s in ['up', 'dn']:

k_s = len(tau_dag[s])

if k_s > 0:

M_inv = np.array([[get_delta_tau(td - ta) for ta in tau_ann[s]] for td in tau_dag[s]])

M_mat = np.linalg.pinv(M_inv)

for i in range(k_s):

for j in range(k_s):

dt = tau_dag[s][i] - tau_ann[s][j]

sign = 1.0

if dt < 0:

dt += beta

sign = -1.0

bin_idx = int((dt / beta) * num_tau) % num_tau

g_tau_hist[bin_idx] -= sign * M_mat[j, i] / (2.0 * beta)

n_measurements += 1

  

if n_measurements > 0:

g_tau_hist /= n_measurements

  

# Step 04c: Fourier Transform G(tau) -> G_imp(iw)

g_imp_iw = np.zeros(num_matsubara, dtype=complex)

for n in range(num_matsubara):

kernel = np.exp(iw_array[n] * tau_grid)

g_imp_iw[n] = np.trapezoid(g_tau_hist * kernel, tau_grid) #trapezodial integration approximation

  

# Step 05: Update self energy

self_energy_raw = g0_inverse - (1.0 / g_imp_iw)

n_high_cutoff = int(0.6 * num_matsubara)

self_energy_new = np.copy(self_energy_raw)

self_energy_new[n_high_cutoff:] = U_interaction * 0.5 # Static asymptotic limit

  
  

self_energy_array = alpha * self_energy_new + (1.0 - alpha) * self_energy_array

print(f"Img(Sigma[0]) = {self_energy_array[0].imag:.4f}")
```
### Code Explanation
- Shows full NSCF implementation of DMFT cycle
- Uses CT-HYB as impurity solver
- Has max_dmft_iters set to 40 (recommended to set 15~40 for accuracy in case of Mott insulator predictions)
- Uses a toy system of a single band Hubbard Hamiltonian:
	$$epsilon =  -2 \times kinetic \times  [cos(kx) + cos(ky) + cos(kz)]$$
- $U/t$ ratio is set to $20$, making it a strongly correlating Mott insulator.
- The parameters are as follows:
```python
	beta = 20.0 # Inverse temperature
	
	num_matsubara = 100 # Number of Matsubara frequencies
	
	t_hopping = 0.2 # Nearest-neighbor hopping (eV)
	
	mu = 2.0 # Chemicl potential
	
	U_interaction = 4.0 # Local Coulomb repulsion (eV)
```

- such that $\mu = U/2$ while keeping hopping parameter $t$ relatively close to 0 (complete 0 would break the CT-HYB solver).
- The predicted self-energy should be: $$Im[\Sigma\left(i\omega_0\right)]=-\frac{U^2}{4\cdot\omega_0}=\frac{-4.0^2}{4\cdot0.157}=-25.4$$
- Hence, it is a very large negative number. 

### Derivation of Prediction Energy
- If you are interested in the derivation, please look here:
- ![[Derivation-Of-Im.svg]]

### Prediction vs. Code

Prediction: -25.46
Code(Trial 1):  -13.4391 [[out.txt]]
Code (Trial 2): 