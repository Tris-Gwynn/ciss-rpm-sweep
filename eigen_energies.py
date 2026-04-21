import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np
import qutip as qt
import seaborn as sns
import os
import scipy.sparse as sp

# --- IMPORT FROM YOUR LIBRARY ---
from ciss_rpm.core import RPMSystem, get_zeeman_hamiltonian, get_hyperfine_hamiltonian, get_dipolar_hamiltonian, get_exchange_hamiltonian
from ciss_rpm.solver import CISSSolver

# --- CONFIGURATION ---
RESOLUTION = 200 
STATIC_PARAMS = {
    'B': 0.05, 
    'A_iso': 0,      # Static Isotropic component
    'A_z': 1.5,      # Static Anisotropic component
    'D': -0.4, 
    'J': 0.0, 
    'k': 0.1, 
    'hf_phi': np.pi/4
}
SYS_PARAMS = {'d': 0.5, 'a': 0.0}

SNS_STYLE = "ticks" 
SNS_CONTEXT = "paper"

SWEEPS = {
    "B":  np.linspace(0, 0.1, RESOLUTION),
    "Ai": np.linspace(-1, 1, RESOLUTION),
    "Az": np.linspace(-1, 1, RESOLUTION),
    "D":  np.linspace(-1, 1, RESOLUTION),
    "J":  np.linspace(-1, 1, RESOLUTION),
    "hf_phi": np.linspace(0, np.pi, RESOLUTION)
}

# Standardized labels
LABELS = {
    "B":  r"Zeeman ($B_0$)",
    "Ai": r"Isotropic Hyperfine ($a_0$)",
    "Az": r"Anisotropic Hyperfine ($T_{zz}$)",
    "D":  r"Dipolar Coupling ($D$)",
    "J":  r"Exchange Coupling ($J$)",
    "hf_phi": r"Hyperfine Angle ($\phi_\text{hf}$)"
}

def make_flat(qobj):
    """
    Forces a Qobj to have flat dimensions [N, N] or [N, 1].
    This removes [2, 2, 2] tensor structures that cause dimension errors.
    """
    if not isinstance(qobj, qt.Qobj): return qobj
    if qobj.type == 'oper':
        return qt.Qobj(qobj.data, dims=[[qobj.shape[0]], [qobj.shape[1]]])
    elif qobj.type == 'ket':
        return qt.Qobj(qobj.data, dims=[[qobj.shape[0]], [1]])
    return qobj

def pad_vector(vec, full_dim):
    """
    Pads an active-space vector (dim 8) to full system dimension (dim 11 or 13).
    """
    vec = make_flat(vec)
    active_dim = vec.shape[0]
    
    if active_dim == full_dim:
        return vec
    
    # Calculate difference and pad
    pad_size = full_dim - active_dim
    data = vec.full().flatten()
    padded_data = np.concatenate([data, np.zeros(pad_size, dtype=complex)])
    
    # Return as flat Qobj
    return qt.Qobj(padded_data, dims=[[full_dim], [1]])

def get_projectors(sys):
    """
    Constructs robust projectors using the Solver basis.
    Ensures all projectors are Full Dimension and Flat.
    """
    solver = CISSSolver(sys)
    full_dim = sys.sys_dim
    
    # Helper to sum outer products of basis vectors
    def build_proj_from_basis_list(basis_list):
        P_total = qt.Qobj(sp.csr_matrix((full_dim, full_dim))) # Start with zero
        for vec in basis_list:
            v_p = pad_vector(vec, full_dim)
            P_total += v_p * v_p.dag()
        return make_flat(P_total)

    # 1. Electronic Projectors
    P_S  = build_proj_from_basis_list(solver.basis_S)
    P_Tp = build_proj_from_basis_list(solver.basis_Tp)
    P_T0 = build_proj_from_basis_list(solver.basis_T0)
    P_Tm = build_proj_from_basis_list(solver.basis_Tm)
    
    # 2. Nuclear Projectors (Dominant Basis Character)
    # Spin-1/2: Index 0=Up, Index 1=Down
    P_Nuc_Up = qt.Qobj(sp.csr_matrix((full_dim, full_dim)))
    P_Nuc_Dn = qt.Qobj(sp.csr_matrix((full_dim, full_dim)))
    
    if sys.n_uc_mult >= 2:
        # Up (Index 0)
        idx = 0
        vecs_up = [solver.basis_S[idx], solver.basis_T0[idx], solver.basis_Tp[idx], solver.basis_Tm[idx]]
        for v in vecs_up:
            v_p = pad_vector(v, full_dim)
            P_Nuc_Up += v_p * v_p.dag()
            
        # Down (Index 1)
        idx = 1
        vecs_dn = [solver.basis_S[idx], solver.basis_T0[idx], solver.basis_Tp[idx], solver.basis_Tm[idx]]
        for v in vecs_dn:
            v_p = pad_vector(v, full_dim)
            P_Nuc_Dn += v_p * v_p.dag()

    # Flatten
    P_Nuc_Up = make_flat(P_Nuc_Up)
    P_Nuc_Dn = make_flat(P_Nuc_Dn)

    # 3. Shelf Projector (Identity minus Active Space)
    P_Active = P_S + P_Tp + P_T0 + P_Tm
    P_shelf = qt.qeye(full_dim) - P_Active
    P_shelf = make_flat(P_shelf)
    
    return P_S, P_Tp, P_T0, P_Tm, P_Nuc_Up, P_Nuc_Dn, P_shelf

def get_full_H_flat(itype, val, sys, stat_p):
    """ Corrected Hamiltonian construction with split Ai/Az. """
    B_curr = stat_p['B']
    D_curr = stat_p['D']
    J_curr = stat_p['J']
    hf_phi_curr = stat_p['hf_phi']
    A_iso_curr = stat_p['A_iso']
    A_z_curr   = stat_p['A_z']

    if itype == 'B': B_curr = val
    elif itype == 'D': D_curr = val
    elif itype == 'J': J_curr = val
    elif itype == 'hf_phi': hf_phi_curr = val
    elif itype == 'Ai': A_iso_curr = val
    elif itype == 'Az': A_z_curr = val

    # Hyperfine Construction
    if itype == 'Ai':
        # Isotropic Sweep: Axx=Ayy=Azz=val
        H_hf = get_hyperfine_hamiltonian(sys, val, val, val, 0, 0)
    elif itype == 'Az':
        # Anisotropic Sweep: Use static Iso, sweep Axial
        H_hf = get_hyperfine_hamiltonian(sys, -0.05, -0.05, val, 0, hf_phi_curr)
    else:
        # Standard
        H_hf = get_hyperfine_hamiltonian(sys, -0.05, -0.05, A_z_curr, 0, hf_phi_curr)

    H_dip = get_dipolar_hamiltonian(sys, -2/3*D_curr, -2/3*D_curr, 4/3*D_curr, 0, 0)
    H_ex = get_exchange_hamiltonian(sys, J_curr)
    H_z = get_zeeman_hamiltonian(sys, B_curr, theta=0, phi=0)

    H_full = H_z + H_hf + H_dip + H_ex
    return make_flat(H_full)

def main():
    sys_obj = RPMSystem(donor_spin=SYS_PARAMS['d'], acceptor_spin=SYS_PARAMS['a'])
    
    # Apply global styling
    sns.set_theme(style=SNS_STYLE, context=SNS_CONTEXT, font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif' 
    plt.rcParams['mathtext.fontset'] = 'cm'

    # Get Projectors (Unpacking only what is necessary for plotting)
    projs = get_projectors(sys_obj)
    P_S = projs[0]
    P_shelf = projs[-1]
    
    save_dir = "Energy_Diagrams_Final"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    print(f"Generating Publication Grade plots in '{save_dir}'...")

    # Create the colormap once for the colorbar
    cmap = mcolors.LinearSegmentedColormap.from_list("S_character", [(0, 0, 1), (1, 0, 0)])

    for sweep_key, sweep_vals in SWEEPS.items():
        print(f"  Processing {sweep_key}...")
        energies = []
        singlet_chars = []
        
        for val in sweep_vals:
            H = get_full_H_flat(sweep_key, val, sys_obj, STATIC_PARAMS)
            evals, evecs = H.eigenstates()
            
            curr_energies = []
            curr_chars = []
            
            for i, vec in enumerate(evecs):
                vec = make_flat(vec)
                # Filter Shelf (Keep only active states)
                if np.real(qt.expect(P_shelf, vec)) < 0.1:
                    curr_energies.append(evals[i])
                    curr_chars.append(np.real(qt.expect(P_S, vec)))
            
            # Sort by energy
            if curr_energies:
                zipped = sorted(zip(curr_energies, curr_chars), key=lambda x: x[0])
                curr_energies, curr_chars = zip(*zipped)
                
            energies.append(curr_energies)
            singlet_chars.append(curr_chars)
            
        energies = np.array(energies)
        singlet_chars = np.array(singlet_chars)

        # --- PLOT ---
        fig, ax = plt.subplots(figsize=(10, 7)) 
        
        for i in range(energies.shape[1]):
            s_vals = np.clip(singlet_chars[:, i], 0, 1)
            z_ord = 10 if np.mean(s_vals) > 0.5 else 2
            
            # Create segments for continuous line plotting
            points = np.array([sweep_vals, energies[:, i]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Calculate color for each segment
            segment_s = (s_vals[:-1] + s_vals[1:]) / 2.0
            colors = np.zeros((len(segment_s), 4))
            colors[:, 0] = segment_s        # Red 
            colors[:, 2] = 1.0 - segment_s  # Blue 
            colors[:, 3] = 1.0              # Alpha
            
            # Draw the multicolor line
            lc = LineCollection(segments, colors=colors, linewidths=2.0, zorder=z_ord)
            ax.add_collection(lc)

        # --- FORMATTING & LABELS ---
        ax.set_xlim(sweep_vals.min(), sweep_vals.max())
        y_min, y_max = np.min(energies), np.max(energies)
        y_pad = (y_max - y_min) * 0.05
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ax.set_title(LABELS[sweep_key], fontsize=16, fontweight='normal', pad=15)
        ax.set_ylabel("Energy (MHz)", fontsize=14)
        
        if sweep_key == "hf_phi":
            ax.set_xlabel(r"Hyperfine Angle ($\phi_\text{hf}$) (rad)", fontsize=14)
            ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
            ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
        else:
            ax.set_xlabel("Interaction Strength (mT)", fontsize=14)
            
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.25)
        sns.despine(ax=ax)
        
        # Add the Colorbar mapped to <P_S>
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(r"Singlet Character $\langle P_S \rangle$", rotation=270, labelpad=20, fontsize=12)

        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Final_Energy_{sweep_key}.pdf"))
        plt.close(fig)
        
    print("Done. All plots generated successfully.")

if __name__ == "__main__":
    main()