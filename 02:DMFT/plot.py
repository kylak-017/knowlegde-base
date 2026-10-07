import sys
import triqs.gf

# Alias triqs.gf to triqs.gfs for triqs_maxent backwards compatibility
sys.modules['triqs.gfs'] = triqs.gf

import numpy as np
import matplotlib.pyplot as plt
from h5 import HDFArchive
from triqs.gf import MeshImTime, BlockGf, Fourier
from triqs_maxent import TauMaxEnt, LinearOmegaMesh

#we now use noise suprresion (shannon entropy) to choose one A(w) out of a lot of options that were 
#created from noise of the sampling


# 1. Load converged G_loc from ybco_dmft_results.h5
with HDFArchive("ybco_dmft_results.h5", "r") as ar:
    last_iter = list(ar.keys())[-1]
    G_loc_iw = ar[last_iter]["G_loc"]
    mu = ar[last_iter]["chemical_potential"]

print(f"Loaded dataset: {last_iter} | Chemical Potential \u03bc = {mu:.4f} eV")

# 2. Extract beta and gf_struct dynamically
beta = G_loc_iw.mesh.beta
gf_struct = [(name, g.target_shape[0]) for name, g in G_loc_iw]

# 3. Create imaginary-time mesh and BlockGf for G(tau)
mesh_tau = MeshImTime(beta=beta, statistic='Fermion', n_tau=10001)
G_loc_tau = BlockGf(mesh=mesh_tau, gf_struct=gf_struct)

# 4. Perform Fourier transform G(iw) -> G(tau)
G_loc_tau << Fourier(G_loc_iw)

# 5. Configure MaxEnt Solver
tm = TauMaxEnt()
tm.set_omega(LinearOmegaMesh(omega_min=-5.0, omega_max=5.0, n_points=500))


plt.figure(figsize=(8, 5))

# 6. Run MaxEnt for each orbital/spin block
for block_name, g_tau in G_loc_tau:
    print(f"Running MaxEnt continuation for block: {block_name}...")
    tm.set_G_tau(g_tau[0, 0])
    tm.set_error(0.001)
    
    result = tm.run()
    
    omega = result.omega
    if hasattr(result, 'A_out'):
        A_w = result.A_out
    else:
        first_key = list(result.analyzer_results.keys())[0]
        analyzer_res = result.analyzer_results[first_key]
        if isinstance(analyzer_res, dict) and 'A_out' in analyzer_res:
            A_w = analyzer_res['A_out']
        else:
            A_w = getattr(analyzer_res, 'A_out', analyzer_res)
    
    plt.plot(omega, A_w, label=f"Cu $d_{{x^2-y^2}}$ ({block_name})", linewidth=1.8)

# 7. Format and save plot
plt.axvline(0.0, color='black', linestyle='--', alpha=0.7, label=r"$E_F$")
plt.xlabel(r"Energy $\omega - E_F$ (eV)", fontsize=12)
plt.ylabel(r"Spectral Function $A(\omega)$ (states/eV)", fontsize=12)
plt.title("YBCO DMFT Local Spectral Function (MaxEnt)", fontsize=13)
plt.xlim(-3.5, 3.5)
plt.ylim(0.0, None)  # Enforce non-negative density of states
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ybco_maxent_spectral.png", dpi=300)
plt.show()