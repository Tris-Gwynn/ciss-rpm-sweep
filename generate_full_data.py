# generate_full_data.py
import os
# Force ALL math libraries to single-threaded mode to prevent conflicts
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import time
import pickle
import qutip as qt
import scipy.linalg as sc
import scipy.sparse as sp
import gc
from multiprocessing import Pool, cpu_count, set_start_method

# --- IMPORT FROM YOUR LIBRARY ---
from ciss_rpm.core import RPMSystem, get_hyperfine_hamiltonian, get_dipolar_hamiltonian, get_exchange_hamiltonian, get_zeeman_hamiltonian
from ciss_rpm.constants import GYRO_E, RESOLUTION, SWEEP_STEPS
from ciss_rpm.solver import CISSSolver

# --- WORKER FUNCTION ---
def compute_angle_row(args):
    itype, val, theta, phi_list, sys_p, stat_p = args
    
    # Initialize System inside the worker
    sys = RPMSystem(donor_spin=sys_p['d'], acceptor_spin=sys_p['a'])
    solver = CISSSolver(sys)
    
    # --- 1. SETUP PARAMETERS ---
    B_curr = stat_p['B']
    A_curr = stat_p['A']
    D_curr = stat_p['D']
    J_curr = stat_p['J']
    hf_phi_curr = stat_p['hf_phi']
    
    if itype == 'B': B_curr = val
    elif itype == 'Ai': A_curr = val 
    elif itype == 'Az': A_curr = val 
    elif itype == 'D': D_curr = val
    elif itype == 'J': J_curr = val
    elif itype == 'hf_phi': hf_phi_curr = val

    # --- 2. BUILD STATIC HAMILTONIANS ---
    hf_theta, hf_phi = 0, np.pi/4 
    dip_theta, dip_phi = 0, 0
    
    if itype == 'hf_phi':
        H_hf = get_hyperfine_hamiltonian(sys, -0.05, -0.05, A_curr, 0, hf_phi_curr)
    elif itype == 'Ai':
        H_hf = get_hyperfine_hamiltonian(sys, A_curr, A_curr, A_curr, 0, 0)
    elif itype == 'Az':
        H_hf = get_hyperfine_hamiltonian(sys, -0.05, -0.05, A_curr, hf_theta, hf_phi)
    else:
        H_hf = get_hyperfine_hamiltonian(sys, -0.05, -0.05, A_curr, hf_theta, hf_phi)
        
    H_dip = get_dipolar_hamiltonian(sys, D_curr*(-2/3), D_curr*(-2/3), D_curr*(4/3), dip_theta, dip_phi)
    H_ex = get_exchange_hamiltonian(sys, J_curr)
    
    H_static = H_hf + H_dip + H_ex
    
    results = []
    
    # --- 3. FIELD LOOP ---
    # CONSTANT: Using 'bdf' for everything. 
    # It is slower (implicit solver) but handles stiff matrices without crashing.
    method = 'bdf'
    
    for phi in phi_list:
        H_z = get_zeeman_hamiltonian(sys, B_curr, theta, phi)
        H_total = H_static + H_z
        
        r_min = solver.solve(H_total, chi=0, ks=stat_p['k'], kt=stat_p['k'], kr=stat_p['k'], method=method)
        r_max = solver.solve(H_total, chi=np.pi/2, ks=stat_p['k'], kt=stat_p['k'], kr=stat_p['k'], method=method)
        
        results.append((r_min, r_max))
    
    return results

def main():
    # --- STATIC PARAMETERS ---
    static_params = {'B': 0.05, 'A': 1.5, 'D': -0.4, 'J': 0, 'k': 0.1, 'hf_phi':np.pi/4}
    sys_params = {'d': 0.5, 'a': 0.0} 
    
    theta_vals = np.linspace(0, np.pi, RESOLUTION)
    phi_vals = np.linspace(0, 2*np.pi, RESOLUTION)
    
    sweeps = {
        "B": np.linspace(0, 0.1, SWEEP_STEPS),
        "Ai": np.linspace(-1, 1, SWEEP_STEPS),
        "Az": np.linspace(-1, 1, SWEEP_STEPS),
        "D": np.linspace(-1, 1, SWEEP_STEPS),
        "J": np.linspace(-1, 1, SWEEP_STEPS),
        "hf_phi": np.linspace(0, np.pi, SWEEP_STEPS)
    }

    data = {}
    
    n_cores = 25
    print(f"Starting Stable Generation (BDF) ({RESOLUTION}x{RESOLUTION}) on {n_cores} cores.")
    
    # maxtasksperchild=50: recycles workers occasionally to clear RAM
    with Pool(n_cores, maxtasksperchild=50) as pool:
        for interaction, values in sweeps.items():
            print(f"--- Processing {interaction} ---")
            
            ops_raw = ["S", "Tm", "T0", "Tp", "R", "Ps", "Ptm", "Pt0", "Ptp"]
            ops_calc = ["T", "F"]
            all_ops = ops_raw + ops_calc

            data[interaction] = {"min": {}, "max": {}}
            
            temp_store = {
                "min": {op: np.zeros((SWEEP_STEPS, RESOLUTION, RESOLUTION, 250)) for op in all_ops},
                "max": {op: np.zeros((SWEEP_STEPS, RESOLUTION, RESOLUTION, 250)) for op in all_ops}
            }

            for i, val in enumerate(values):
                start_t = time.time()
                tasks = []
                for theta in theta_vals:
                    tasks.append((interaction, val, theta, phi_vals, sys_params, static_params))
                
                row_results = pool.map(compute_angle_row, tasks)
                
                for row_idx, row_data in enumerate(row_results): 
                    for col_idx, (r_min, r_max) in enumerate(row_data):
                        # Store raw solver outputs
                        for op_idx, op_name in enumerate(ops_raw):
                            temp_store["min"][op_name][i, row_idx, col_idx, :] = r_min[op_idx]
                            temp_store["max"][op_name][i, row_idx, col_idx, :] = r_max[op_idx]
                        
                        # Calculate Aggregates
                        # MIN
                        t_min = r_min[1] + r_min[2] + r_min[3]
                        f_min = r_min[0] + t_min
                        temp_store["min"]["T"][i, row_idx, col_idx, :] = t_min
                        temp_store["min"]["F"][i, row_idx, col_idx, :] = f_min
                        
                        # MAX
                        t_max = r_max[1] + r_max[2] + r_max[3]
                        f_max = r_max[0] + t_max
                        temp_store["max"]["T"][i, row_idx, col_idx, :] = t_max
                        temp_store["max"]["F"][i, row_idx, col_idx, :] = f_max

                print(f"  Completed Value {i+1}/{SWEEP_STEPS} ({val:.3f}) in {time.time()-start_t:.2f}s")
                gc.collect()

            for bound in ["min", "max"]:
                for op in all_ops:
                    data[interaction][bound][op] = temp_store[bound][op]

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(script_dir, "Results")
    filepath = os.path.join(save_folder, "full_ciss_data.pkl")

    if not os.path.exists(save_folder): os.makedirs(save_folder)

    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    print(f"Done. Saved to '{filepath}'")

if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    main()