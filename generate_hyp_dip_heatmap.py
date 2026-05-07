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

GRID_SIZE = 11      
ANGULAR_RES = 11    
SAVE_FILE = "Results/ciss_heatmap_data.pkl"

STATIC_PARAMS = {'B': 0.05, 'J': 0, 'k': 0.1}
SYS_PARAMS = {'d': 0.5, 'a': 0.0}

def compute_angle_row_2d(args):
    A_curr, D_curr, theta, phi_list = args
    
    sys = RPMSystem(donor_spin=SYS_PARAMS['d'], acceptor_spin=SYS_PARAMS['a'])
    solver = CISSSolver(sys)
    
    H_hf = get_hyperfine_hamiltonian(sys, A_curr, A_curr, A_curr, 0, 0)
    H_dip = get_dipolar_hamiltonian(sys, D_curr*(-2/3), D_curr*(-2/3), D_curr*(4/3), 0, 0)
    H_ex = get_exchange_hamiltonian(sys, STATIC_PARAMS['J'])
    
    H_static = H_hf + H_dip + H_ex
    
    results = []
    for phi in phi_list:
        H_z = get_zeeman_hamiltonian(sys, STATIC_PARAMS['B'], theta, phi)
        H_total = H_static + H_z
        
        # FIX: Explicit chi_init and chi_recomb mapping
        r_min = solver.solve(H_total, chi_init=0, chi_recomb=0, ks=STATIC_PARAMS['k'], kt=STATIC_PARAMS['k'], kr=STATIC_PARAMS['k'], method='bdf')
        r_max = solver.solve(H_total, chi_init=np.pi/2, chi_recomb=np.pi/2, ks=STATIC_PARAMS['k'], kt=STATIC_PARAMS['k'], kr=STATIC_PARAMS['k'], method='bdf')
        
        results.append((r_min[0][-1], r_max[0][-1]))
        
    return results

def main():
    A_vals = np.linspace(-1, 1, GRID_SIZE)
    D_vals = np.linspace(-1, 1, GRID_SIZE)
    theta_vals = np.linspace(0, np.pi, ANGULAR_RES)
    phi_vals = np.linspace(0, 2*np.pi, ANGULAR_RES)
    
    ani_0_map = np.zeros((GRID_SIZE, GRID_SIZE))
    ani_90_map = np.zeros((GRID_SIZE, GRID_SIZE))

    print(f"Starting 2D Sweep: {GRID_SIZE}x{GRID_SIZE} Parameters, {ANGULAR_RES}x{ANGULAR_RES} Angles")
    
    n_cores = 30  
    with Pool(n_cores) as pool:
        for i, A in enumerate(A_vals):
            start_row = time.time()
            for j, D in enumerate(D_vals):
                tasks = [(A, D, theta, phi_vals) for theta in theta_vals]
                pixel_results = pool.map(compute_angle_row_2d, tasks)
                
                yields_0 = [y0 for row in pixel_results for y0, _ in row]
                yields_90 = [y90 for row in pixel_results for _, y90 in row]
                
                ani_0_map[i, j] = np.max(yields_0) - np.min(yields_0)
                ani_90_map[i, j] = np.max(yields_90) - np.min(yields_90)
            
            print(f"  Row {i+1}/{GRID_SIZE} (A={A:.2f}) done in {time.time() - start_row:.2f}s")
            gc.collect()

    data = {
        "A_vals": A_vals,
        "D_vals": D_vals,
        "anisotropy_0": ani_0_map,
        "anisotropy_90": ani_90_map,
        "diff": ani_90_map - ani_0_map
    }
    
    os.makedirs("Results", exist_ok=True)
    with open(SAVE_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"Done. Data saved to {SAVE_FILE}")

if __name__ == "__main__":
    try: set_start_method('spawn')
    except RuntimeError: pass
    main()