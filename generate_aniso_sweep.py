import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import time
import pickle
import gc
from multiprocessing import Pool, set_start_method

from ciss_rpm.core import RPMSystem, get_hyperfine_hamiltonian, get_dipolar_hamiltonian, get_exchange_hamiltonian, get_zeeman_hamiltonian
from ciss_rpm.solver import CISSSolver

RESOLUTION = 20        
SWEEP_STEPS = 31       
N_ANGLES = 5           
N_CORES = 20           
OVERWRITE = True       

def compute_angle_row(args):
    itype, val, theta, phi_list, sys_p, stat_p = args
    sys = RPMSystem(donor_spin=sys_p['d'], acceptor_spin=sys_p['a'])
    solver = CISSSolver(sys)
    
    B_curr = val if itype == 'B' else stat_p['B']
    A_curr = val if itype in ('Ai', 'Az') else stat_p['A']
    D_curr = val if itype == 'D' else stat_p['D']
    J_curr = val if itype == 'J' else stat_p['J']
    hf_phi_curr = val if itype == 'hf_phi' else stat_p['hf_phi']
    
    hf_theta = 0 
    hf_phi = stat_p['scan_angle_val'] 
    
    if itype == 'hf_phi':
        H_hf = get_hyperfine_hamiltonian(sys, -0.05, -0.05, A_curr, 0, hf_phi_curr)
    elif itype == 'Ai':
        H_hf = get_hyperfine_hamiltonian(sys, A_curr, A_curr, A_curr, 0, 0)
    else:
        H_hf = get_hyperfine_hamiltonian(sys, -0.05, -0.05, A_curr, hf_theta, hf_phi)
        
    H_dip = get_dipolar_hamiltonian(sys, D_curr*(-2/3), D_curr*(-2/3), D_curr*(4/3), 0, 0)
    H_ex = get_exchange_hamiltonian(sys, J_curr)
    H_static = H_hf + H_dip + H_ex
    
    results = []
    for phi in phi_list:
        H_z = get_zeeman_hamiltonian(sys, B_curr, theta, phi)
        H_total = H_static + H_z
        
        # FIX: Explicit chi_init and chi_recomb mapping
        r_min = solver.solve(H_total, chi_init=0, chi_recomb=0, ks=stat_p['k'], kt=stat_p['k'], kr=stat_p['k'], method='bdf')
        r_max = solver.solve(H_total, chi_init=np.pi/2, chi_recomb=np.pi/2, ks=stat_p['k'], kt=stat_p['k'], kr=stat_p['k'], method='bdf')
        
        results.append((r_min, r_max))
    
    return results

def main():
    static_params = {'B': 0.05, 'A': 0, 'D': -0.4, 'J': 0, 'k': 0.1, 'hf_phi': np.pi/4, 'scan_angle_val': 0}
    sys_params = {'d': 0.5, 'a': 0.0} 
    
    theta_vals = np.linspace(0, np.pi, RESOLUTION)
    phi_vals = np.linspace(0, 2*np.pi, RESOLUTION)
    angle_scan_values = np.linspace(0, np.pi, N_ANGLES)
    az_values = np.linspace(-1, 1, SWEEP_STEPS)

    save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results_Original")
    os.makedirs(save_folder, exist_ok=True)

    print(f"Starting Sweep ({RESOLUTION}x{RESOLUTION}) on {N_CORES} cores.")

    for angle_idx, angle_val in enumerate(angle_scan_values):
        filename = os.path.join(save_folder, f"angle_{angle_idx}.pkl")
        if os.path.exists(filename) and not OVERWRITE: continue

        print(f"\n--- Angle {angle_idx+1}/{N_ANGLES}: {angle_val:.4f} rad ---")
        current_static = static_params.copy()
        current_static['scan_angle_val'] = angle_val 
        
        ops_raw = ["S", "Tm", "T0", "Tp", "R", "Ps", "Ptm", "Pt0", "Ptp"]
        temp_store = {
            "meta": {"theta": angle_val},
            "min": {op: np.zeros((SWEEP_STEPS, RESOLUTION, RESOLUTION, 250)) for op in ops_raw + ["T", "F"]},
            "max": {op: np.zeros((SWEEP_STEPS, RESOLUTION, RESOLUTION, 250)) for op in ops_raw + ["T", "F"]}
        }

        with Pool(N_CORES, maxtasksperchild=5) as pool:
            for i, val in enumerate(az_values):
                start_t = time.time()
                tasks = [('Az', val, theta, phi_vals, sys_params, current_static) for theta in theta_vals]
                row_results = pool.map(compute_angle_row, tasks)
                
                for row_idx, row_data in enumerate(row_results): 
                    for col_idx, (r_min, r_max) in enumerate(row_data):
                        for op_idx, op_name in enumerate(ops_raw):
                            temp_store["min"][op_name][i, row_idx, col_idx, :] = r_min[op_idx]
                            temp_store["max"][op_name][i, row_idx, col_idx, :] = r_max[op_idx]
                        
                        t_min = sum(r_min[1:4])
                        temp_store["min"]["T"][i, row_idx, col_idx, :] = t_min
                        temp_store["min"]["F"][i, row_idx, col_idx, :] = r_min[0] + t_min
                        
                        t_max = sum(r_max[1:4])
                        temp_store["max"]["T"][i, row_idx, col_idx, :] = t_max
                        temp_store["max"]["F"][i, row_idx, col_idx, :] = r_max[0] + t_max
                
                print(f"  Az Step {i+1}/{SWEEP_STEPS} ({val:.2f}) - {time.time()-start_t:.1f}s")
                del row_results
                gc.collect()

        with open(filename, "wb") as f:
            pickle.dump(temp_store, f)
        print(f"Saved {filename}")

if __name__ == "__main__":
    try: set_start_method('spawn')
    except: pass
    main()