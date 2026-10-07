from triqs.operators import n, c, c_dag
from triqs_cthyb import Solver
from triqs.gf import GfImFreq, BlockGf, inverse, iOmega_n, SemiCircular


t = 1.0        # Hopping parameter
e_f = 0.0      # Local orbital energy level
mu = 2.0       # Chemical potential 
U = 4.0    
beta = 40.0 #what is the temperature term that is derived from the imaginary time conversion (our sampling depends on this time conversion)
gf_struct = [ ('up', 2), ('down', 2) ] # what is the composition of the inputted Green's function for the non-interacting term?
h_int = U * (n('up',0) * n('down',0) + n('up',1) * n('down',1))

S = Solver(beta = beta, gf_struct = gf_struct) #instance of the CTHYB Solver

    #Herein, we show that the function utilizes Monte Carlo samp;ing and thus a Markov chain to decide upon the optimized
    #configuration of the system.
    #Expectation value of a certain function is that function multpiplied by the weight of the partition function divided nu the netieyu of the 


G = S.G_iw.copy()
for name, g in G:
    g << SemiCircular(2*t)          # initial guess

for it in range(10):
    for name, g0 in S.G0_iw:
        g0 << inverse(iOmega_n + mu - e_f - t**2 * G[name])
    S.solve(h_int=h_int, n_cycles=10000, length_cycle=500,
            n_warmup_cycles=10000)
    G << S.G_iw                     # new impurity G -> new bath
    print(it, S.average_sign)    
 

