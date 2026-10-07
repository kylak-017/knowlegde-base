import numpy as np
from triqs.utility import mpi
from triqs.gf import MeshImFreq, BlockGf, inverse, Fourier, iOmega_n
from h5 import HDFArchive
from triqs_dft_tools.sumk_dft import *
from triqs_ctseg import Solver  
from triqs.operators import n

seed = "ybco"
beta = 40.0
U = 4.0
max_loops = 20
mix = 0.4
n_iw = 1025
conv_threshold = 0.005

SK = SumkDFT(hdf_file=f"{seed}.h5", use_dft_blocks=True)
n_inequiv = max(SK.corr_to_inequiv) + 1

raw_struct = SK.gf_struct_solver[0]
gf_struct = [(k, len(v) if isinstance(v, (list, tuple)) else v) for k, v in (raw_struct.items() if isinstance(raw_struct, dict) else raw_struct)]

up_block = [b for b, _ in gf_struct if 'up' in b][0]
dn_block = [b for b, _ in gf_struct if 'down' in b or 'dn' in b][0]

mesh = MeshImFreq(beta=beta, statistic='Fermion', n_iw=n_iw)
Sigma_init = BlockGf(mesh=mesh, gf_struct=gf_struct)
Sigma_init.zero()
SK.put_Sigma([Sigma_init] * n_inequiv)

# Initialize CT-SEG solver
S = Solver(beta=beta, gf_struct=gf_struct)
h_int = U * n(up_block, 0) * n(dn_block, 0)

# Temporary Green's function for Delta(iw)
Delta_iw = BlockGf(mesh=mesh, gf_struct=gf_struct)

Sigma_current = Sigma_init.copy()
Sigma_prev = None

for iteration in range(max_loops):
    if mpi.is_master_node():
        print(f"\n==========================================")
        print(f"   DMFT Iteration (CT-SEG) {iteration+1} / {max_loops}")
        print(f"==========================================")

    G_loc = SK.extract_G_loc()

    # 1. Compute G0_iw = (Sigma + G_loc^-1)^-1
    G0_iw = inverse(Sigma_current + inverse(G_loc[0]))

    # 2. Compute Delta(iw) = iwn - G0_iw^-1
    Delta_iw << iOmega_n - inverse(G0_iw)

    # 3. Fourier transform Delta(iw) -> S.Delta_tau
    S.Delta_tau << Fourier(Delta_iw)

    # 4. Solve impurity problem
    S.solve(
        h_int=h_int,
        n_cycles=100000,
        length_cycle=200,
        n_warmup_cycles=10000
    )

    # Extract G_iw and Sigma_iw from results
    G_iw_solver = S.results.G_iw if hasattr(S, 'results') else S.G_iw
    Sigma_iw_solver = S.results.Sigma_iw if hasattr(S, 'results') else S.Sigma_iw

    # Linear mixing
    if iteration > 0 and Sigma_prev is not None:
        Sigma_current << mix * Sigma_iw_solver + (1.0 - mix) * Sigma_prev
    else:
        Sigma_current << Sigma_iw_solver.copy()

    # Convergence check
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
            break

    Sigma_prev = Sigma_current.copy()
    SK.put_Sigma([Sigma_current] * n_inequiv)
    SK.calc_mu(precision=0.0001)

    if mpi.is_master_node():
        with HDFArchive(f"{seed}_dmft_results.h5", 'a') as ar:
            ar[f"iter_{iteration}"] = {
                "G_loc": G_loc[0],
                "G_iw": G_iw_solver,
                "Sigma_iw": Sigma_current,
                "chemical_potential": SK.chemical_potential
            }

if mpi.is_master_node():
    print("\nDMFT run completed!")