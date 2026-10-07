import numpy as np
from triqs.utility import mpi
from triqs.gf import MeshImFreq, BlockGf, inverse
from h5 import HDFArchive
from triqs_dft_tools.sumk_dft import *
from triqs_cthyb import Solver  
from triqs.operators import n

# ---------------------------------------------------------
# 1. Physics & Simulation Parameters
# ---------------------------------------------------------
seed = "ybco"
beta = 128.94             # Inverse temperature (1/eV) ~ 90 K
U = 6.0                 # On-site Coulomb interaction (eV)
max_loops = 50          # Maximum DMFT iterations
mix = 0.4               # Self-energy mixing factor
n_iw = 1025             # Matsubara frequency points
conv_threshold = 0.001  # Convergence threshold (eV) on Im Sigma(iw_0)

# ---------------------------------------------------------
# 2. Initialize SumkDFT & Format Blocks
# ---------------------------------------------------------
SK = SumkDFT(hdf_file=f"{seed}.h5", use_dft_blocks=True, beta=beta, n_iw=n_iw)
n_inequiv = max(SK.corr_to_inequiv) + 1

raw_struct = SK.gf_struct_solver[0]
if isinstance(raw_struct, dict):
    gf_struct = [(k, len(v) if isinstance(v, (list, tuple)) else v) for k, v in raw_struct.items()]
elif isinstance(raw_struct, list):
    gf_struct = [(k, len(v) if isinstance(v, (list, tuple)) else v) for k, v in raw_struct]
else:
    gf_struct = raw_struct

up_block = [b for b, _ in gf_struct if 'up' in b][0]
dn_block = [b for b, _ in gf_struct if 'down' in b or 'dn' in b][0]

if mpi.is_master_node():
    print(f"Detected gf_struct: {gf_struct}")
    print(f"Using spin block names: up='{up_block}', down='{dn_block}'")

# ---------------------------------------------------------
# 3. Initial Zero Self-Energy
# ---------------------------------------------------------
mesh = SK.mesh 

Sigma_init = BlockGf(mesh=mesh, gf_struct=gf_struct)
Sigma_init.zero()

SK.put_Sigma([Sigma_init] * n_inequiv)
SK.put_Sigma([Sigma_init] * n_inequiv)

# ---------------------------------------------------------
# 4. Initialize CTHYB Solver
# ---------------------------------------------------------
S = Solver(beta=beta, gf_struct=gf_struct, n_iw=n_iw)
h_int = U * n(up_block, 0) * n(dn_block, 0)

# ---------------------------------------------------------
# 5. DMFT Loop
# ---------------------------------------------------------
Sigma_current = Sigma_init.copy()
Sigma_prev = None

for iteration in range(max_loops):
    if mpi.is_master_node():
        print(f"\n==========================================")
        print(f"   DMFT Iteration (CT-HYB) {iteration+1} / {max_loops}")
        print(f"==========================================")

    # A. Lattice Green's function
    G_loc = SK.extract_G_loc()

    # B. Compute Weiss field G0(iw)
    S.G0_iw << inverse(Sigma_current + inverse(G_loc[0]))

    # C. Solve Quantum Impurity Problem
    S.solve(
        h_int=h_int,
        n_cycles=100000,
        length_cycle=200,
        n_warmup_cycles=10000
    )

    # D. Linear Mixing
    if iteration > 0 and Sigma_prev is not None:
        Sigma_current << mix * S.Sigma_iw + (1.0 - mix) * Sigma_prev
    else:
        Sigma_current << S.Sigma_iw.copy()

    # E. Convergence check
    if iteration > 0 and Sigma_prev is not None:
        sig_curr_w0 = Sigma_current[up_block].data[0, 0, 0]
        sig_prev_w0 = Sigma_prev[up_block].data[0, 0, 0]
        diff_w0 = abs(sig_curr_w0 - sig_prev_w0)

        if mpi.is_master_node():
            print(f"--> Delta Im Sigma(iw_0): {diff_w0.imag:.6f} eV (Target: < {conv_threshold} eV)")

        if diff_w0.imag < conv_threshold:
            if mpi.is_master_node():
                print(f"\n*** DMFT converged at iteration {iteration+1}! ***")
            SK.put_Sigma([Sigma_current] * n_inequiv)
            SK.calc_mu(precision=0.0001)
            if mpi.is_master_node():
                with HDFArchive(f"{seed}_dmft_results.h5", 'a') as ar:
                    ar[f"iter_{iteration}"] = {
                        "G_loc": G_loc[0],
                        "G_iw": S.G_iw,
                        "Sigma_iw": Sigma_current,
                        "chemical_potential": SK.chemical_potential
                    }
            break

    Sigma_prev = Sigma_current.copy()
    SK.put_Sigma([Sigma_current] * n_inequiv)
    SK.calc_mu(precision=0.0001)

    if mpi.is_master_node():
        with HDFArchive(f"{seed}_dmft_results.h5", 'a') as ar:
            ar[f"iter_{iteration}"] = {
                "G_loc": G_loc[0],
                "G_iw": S.G_iw,
                "Sigma_iw": Sigma_current,
                "chemical_potential": SK.chemical_potential
            }

if mpi.is_master_node():
    print("\nDMFT run completed!")