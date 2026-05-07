#%%
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from ciss_rpm.core import RPMSystem, get_hyperfine_hamiltonian, get_zeeman_hamiltonian
from ciss_rpm.solver import CISSSolver

# --- 1. CONFIGURATION ---
T_MAX = 1.0  # Matches Fig 4 x-axis

# CRITICAL FIX: Multiply by 2pi to convert MHz -> rad/us
TWO_PI = 2 * np.pi 

# Paper Parameters (Fig 4):
# B = 50 microTesla = 0.05 mT
# A = 0.5 mT (Dotted line in paper)
PARAMS = {
    'B': 0.05,  
    'A': 0.5,   
    'D': 0.0,
    'J': 0.0,
    'k': 0.0      # Set to 0 for coherent dynamics (no decay)
}

SYS_CONFIG = {'d': 0.5, 'a': 0.0}

# --- 2. SETUP ---
# Note: We divide A/B by 2pi in the print statement just for readability
print(f"Initializing System (A={PARAMS['A']/TWO_PI:.1f} mT, k={PARAMS['k']})...")

sys = RPMSystem(donor_spin=SYS_CONFIG['d'], acceptor_spin=SYS_CONFIG['a'])

# Hamiltonian
# Note: GYRO_E in your library is 28.0. 
# Since we multiplied PARAMS by 2pi, the final H is in rad/us.
H_hf = get_hyperfine_hamiltonian(sys, PARAMS['A'], PARAMS['A'], PARAMS['A'], 0, 0)
H_z = get_zeeman_hamiltonian(sys, PARAMS['B'], 0, 0) 
H_total = H_hf + H_z 

solver = CISSSolver(sys)
chi=0#np.pi/2
ks=kt=10
kr=10

rho0 = solver.get_initial_rho(chi,initial_nuc_state='thermal') 

# --- 3. SOLVE ---
times = np.linspace(0, T_MAX, 500)
# Unitary evolution (c_ops=[])
c_ops = solver.get_collapse_ops(chi, ks, kt, kr)

e_ops = [
            solver.sys.P_shelf_S, 
            solver.sys.P_shelf_Tp,
            solver.sys.P_shelf_T0,
            solver.sys.P_shelf_Tm, 
            solver.sys.P_shelf_R, 
            solver.P_singlet,
            solver.P_tm,
            solver.P_t0,
            solver.P_tp
        ]
result = qt.mesolve(H_total, rho0, times, c_ops, e_ops=e_ops)
singlet_fidelity = result.expect[0] #- result.expect[4] 

# --- 4. PLOTTING ---
plt.figure(figsize=(8, 6))
plt.title(f"Singlet Fidelity (A=0.5 mT, k=0)")

# Plot with black dashed line to match your previous style
plt.plot(times, singlet_fidelity, 'k--', linewidth=1.5, label='Singlet Fidelity')

plt.xlabel(r"Time ($\mu s$)")
plt.ylabel(r"$\langle S | \rho_{el} | S \rangle$")
#plt.ylim(0, 1.0)
plt.xlim(0, 1.0)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
# %%
