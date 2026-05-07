import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import time
import pickle
import qutip as qt
import scipy.sparse as sp
import gc
import traceback
from multiprocessing import Pool, set_start_method

from ciss_rpm.core import (
    RPMSystem, get_dipolar_hamiltonian, get_exchange_hamiltonian, 
    get_zeeman_hamiltonian, get_hyperfine_hamiltonian_2nuc
)
from ciss_rpm.constants import GYRO_E
from ciss_rpm.solver import CISSSolver

# --- CONFIGURATION ---
RESOLUTION = 18
SWEEP_STEPS = 16
HF_MULT = 1.0 

def compute_angle_row(args):
    try:
        itype, val, theta, phi_list, sys_p, stat_p, hf_mult = args
        
        sys = RPMSystem(donor_spin=sys_p['d'], acceptor_spin=sys_p['a'])
        solver = CISSSolver(sys)
        
        B_curr = stat_p['B']
        D_curr = stat_p['D']
        J_curr = stat_p['J']
        
        Axx = -0.05
        Ayy = -0.05
        Azz = stat_p['A']
        
        hf_theta1 = 0
        hf_phi1 = stat_p['hf_phi']
        hf_theta2 = 0
        hf_phi2 = stat_p['hf_phi']
        
        if itype == 'B': 
            B_curr = val
        elif itype == 'Ai': 
            Axx = val
            Ayy = val
            Azz = val 
        elif itype == 'Az': 
            Azz = val 
        elif itype == 'D': 
            D_curr = val
        elif itype == 'J': 
            J_curr = val
        elif itype == 'hf_phi': 
            hf_phi1 = val
            hf_phi2 = stat_p['hf_phi']

        H_hf = get_hyperfine_hamiltonian_2nuc(sys, Axx, Ayy, Azz, hf_theta1, hf_phi1, hf_theta2, hf_phi2, mult=hf_mult)
        H_dip = get_dipolar_hamiltonian(sys, D_curr*(-2/3), D_curr*(-2/3), D_curr*(4/3), 0, 0)
        H_ex = get_exchange_hamiltonian(sys, J_curr)
        
        H_static = H_hf + H_dip + H_ex
        results = []
        method = 'bdf'
        
        is_pole = np.isclose(theta, 0.0, atol=1e-8) or np.isclose(theta, np.pi, atol=1e-8)
        
        if is_pole:
            H_z = get_zeeman_hamiltonian(sys, B_curr, theta, 0.0)
            H_total = H_static + H_z
            r_min = solver.solve(H_total, chi_init=0, chi_recomb=0, ks=stat_p['k'], kt=stat_p['k'], kr=stat_p['k'], method=method)
            r_max = solver.solve(H_total, chi_init=np.pi/2, chi_recomb=np.pi/2, ks=stat_p['k'], kt=stat_p['k'], kr=stat_p['k'], method=method)
            results = [(r_min, r_max) for _ in phi_list]
        else:
            for phi in phi_list:
                H_z = get_zeeman_hamiltonian(sys, B_curr, theta, phi)
                H_total = H_static + H_z
                r_min = solver.solve(H_total, chi_init=0, chi_recomb=0, ks=stat_p['k'], kt=stat_p['k'], kr=stat_p['k'], method=method)
                r_max = solver.solve(H_total, chi_init=np.pi/2, chi_recomb=np.pi/2, ks=stat_p['k'], kt=stat_p['k'], kr=stat_p['k'], method=method)
                results.append((r_min, r_max))
        
        return results
    except Exception as e:
        return Exception(f"Worker Error: {str(e)}\n{traceback.format_exc()}")

def main():
    static_params = {'B': 0.05, 'A': 1.5, 'D': -0.4, 'J': 0, 'k': 0.1, 'hf_phi': np.pi/4}
    sys_params = {'d': 0.5, 'a': 0.5} 
    
    theta_vals = np.linspace(0, np.pi, RESOLUTION)
    phi_vals = np.linspace(0, 2*np.pi, RESOLUTION, endpoint=False)
    
    sweeps = {
        "B": np.linspace(0, 0.1, SWEEP_STEPS),
        "Ai": np.linspace(-1, 1, SWEEP_STEPS),
        "Az": np.linspace(-1, 1, SWEEP_STEPS),
        "D": np.linspace(-1, 1, SWEEP_STEPS),
        "J": np.linspace(-1, 1, SWEEP_STEPS),
        "hf_phi": np.linspace(0, np.pi, SWEEP_STEPS)
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(script_dir, "Results")
    filepath = os.path.join(save_folder, "2nuc_ciss_data.pkl")

    if not os.path.exists(save_folder): 
        os.makedirs(save_folder)

    # --- ADVANCED CHECKPOINT LOADING ---
    if os.path.exists(filepath):
        print(f"Found existing data file: {filepath}")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        if "_progress" not in data:
            data["_progress"] = {}
            print("Upgrading older save file to support step-by-step checkpointing.")
    else:
        data = {"_progress": {}}
    
    # Core limit capped at 18
    n_cores = min(18, os.cpu_count() or 4)
    print(f"Generating 2-Nuclei Data ({RESOLUTION}x{RESOLUTION} grid, {SWEEP_STEPS} steps) on {n_cores} cores.")
    
    with Pool(n_cores, maxtasksperchild=2) as pool:
        for interaction, values in sweeps.items():
            
            completed_steps = data["_progress"].get(interaction, 0)
            
            if completed_steps >= SWEEP_STEPS:
                print(f"--- Skipping {interaction} (Fully Complete) ---")
                continue
                
            print(f"--- Processing {interaction} (Starting at Step {completed_steps + 1}) ---")
            
            ops_raw = ["S", "Tp", "T0", "Tm", "R", "Ps", "Ptm", "Pt0", "Ptp"]
            ops_calc = ["T", "F"]
            all_ops = ops_raw + ops_calc

            if interaction not in data:
                data[interaction] = {"min": {}, "max": {}}

            for i, val in enumerate(values):
                if i < completed_steps:
                    continue
                    
                start_t = time.time()
                tasks = [(interaction, val, theta, phi_vals, sys_params, static_params, HF_MULT) for theta in theta_vals]
                
                row_results = pool.map(compute_angle_row, tasks)
                
                for r in row_results:
                    if isinstance(r, Exception):
                        raise r
                
                if not data[interaction]["min"]:
                    t_len = len(row_results[0][0][0][0])
                    for op in all_ops:
                        data[interaction]["min"][op] = np.zeros((SWEEP_STEPS, RESOLUTION, RESOLUTION, t_len), dtype=np.float32)
                        data[interaction]["max"][op] = np.zeros((SWEEP_STEPS, RESOLUTION, RESOLUTION, t_len), dtype=np.float32)
                
                for row_idx, row_data in enumerate(row_results): 
                    for col_idx, (r_min, r_max) in enumerate(row_data):
                        
                        for op_idx, op_name in enumerate(ops_raw):
                            data[interaction]["min"][op_name][i, row_idx, col_idx, :] = r_min[op_idx]
                            data[interaction]["max"][op_name][i, row_idx, col_idx, :] = r_max[op_idx]
                        
                        t_min = r_min[1] + r_min[2] + r_min[3]
                        f_min = r_min[0] + t_min
                        data[interaction]["min"]["T"][i, row_idx, col_idx, :] = t_min
                        data[interaction]["min"]["F"][i, row_idx, col_idx, :] = f_min
                        
                        t_max = r_max[1] + r_max[2] + r_max[3]
                        f_max = r_max[0] + t_max
                        data[interaction]["max"]["T"][i, row_idx, col_idx, :] = t_max
                        data[interaction]["max"]["F"][i, row_idx, col_idx, :] = f_max

                data["_progress"][interaction] = i + 1
                with open(filepath, "wb") as f:
                    pickle.dump(data, f)
                    
                print(f"  Completed & Saved Value {i+1}/{SWEEP_STEPS} ({val:.3f}) in {time.time()-start_t:.2f}s")
                gc.collect()

    print(f"Done. Final data safely stored in '{filepath}'")

if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    main()