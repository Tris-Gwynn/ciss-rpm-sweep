import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pickle
import os

# --- SETTINGS ---
INTERACTIONS = ["B", "Ai", "Az", "D", "J", "hf_phi"]
STATES = ["S", "T0", "Tp", "R"]  

INVERSION_MAP = {
    "S":  "S",
    "T0": "T0",
    "Tp": "Tm",  
    "R":  "R"
}

LABELS_LATEX = {
    "B":  r"$B_0$ (mT)",
    "Ai": r"$a_0$ (mT)",
    "Az": r"$T_{zz}$ (mT)",
    "D":  r"$D_0$ (mT)",
    "J":  r"$J$ (mT)",
    "hf_phi": r"$\phi_\text{hf}$ (rad)"
}

STATE_LABELS = {
    "S":  r"$S$",
    "T0": r"$T_0$",
    "Tp": r"$T_\pm$", 
    "R":  r"$R$"
}

BOUNDS = {
    "B":  (0, 0.1), 
    "Ai": (-1, 1),  
    "Az": (-1, 1),  
    "D":  (-1, 1),  
    "J":  (-1, 1),
    "hf_phi": (0, np.pi) 
}

# --- STYLING ---
SNS_STYLE = "ticks" 
SNS_CONTEXT = "paper"
C_SYM = '#4c72b0'  
C_ANTI = '#dd8452' 

def get_symmetry_amplitudes(target_data, source_data):
    target = np.max(target_data, axis=-1) if target_data.ndim == 4 else target_data
    source = np.max(source_data, axis=-1) if source_data.ndim == 4 else source_data
    
    if target.ndim == 2:
        n_sweeps, total_points = target.shape
        res = int(np.sqrt(total_points))
        target = target.reshape(n_sweeps, res, res)
        source = source.reshape(n_sweeps, res, res)
        
    n_sweeps, n_theta, n_phi = target.shape
    
    theta_grid = np.linspace(0, np.pi, n_theta)
    sin_theta = np.sin(theta_grid).reshape(1, n_theta, 1)
    norm = np.sum(sin_theta) * n_phi
    
    sph_avg_target = np.sum(target * sin_theta, axis=(1, 2), keepdims=True) / norm
    sph_avg_source = np.sum(source * sin_theta, axis=(1, 2), keepdims=True) / norm
    
    residual_target = target - sph_avg_target
    residual_source = source - sph_avg_source
    
    source_inverted = np.roll(np.flip(residual_source, axis=1), n_phi // 2, axis=2)
    
    phi_plus = 0.5 * (residual_target + source_inverted)
    phi_minus = 0.5 * (residual_target - source_inverted)
    
    amp_plus = np.max(phi_plus, axis=(1, 2)) - np.min(phi_plus, axis=(1, 2))
    amp_minus = np.max(phi_minus, axis=(1, 2)) - np.min(phi_minus, axis=(1, 2))
    
    return amp_plus, amp_minus

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "Results")
    plots_dir = os.path.join(results_dir, "Plots_Amplitudes_By_Interaction")
    if not os.path.exists(plots_dir): 
        os.makedirs(plots_dir)

    files = {
        "1 Nucleus": os.path.join(results_dir, "full_ciss_data.pkl"),
        "2 Nuclei":  os.path.join(results_dir, "2nuc_ciss_data.pkl")
    }

    loaded_data = {}
    for label, path in files.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                loaded_data[label] = pickle.load(f)
        else:
            print(f"Warning: {path} not found. Data for {label} will be omitted.")

    sns.set_theme(style=SNS_STYLE, context=SNS_CONTEXT, font_scale=1.4)
    plt.rcParams['font.family'] = 'sans-serif' 
    plt.rcParams['mathtext.fontset'] = 'cm'

    for interaction in INTERACTIONS:
        min_val, max_val = BOUNDS[interaction]

        for nuc_label in ["1 Nucleus", "2 Nuclei"]:
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 4), sharex=True, sharey=True)
            labels = ['a', 'b', 'c', 'd']
            global_max_y = 0 

            for i, state in enumerate(STATES):
                ax = axes[i] 
                source_state = INVERSION_MAP[state] 
                
                ax.text(-0.05, 1.05, labels[i], transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom')
                ax.set_title(STATE_LABELS[state], fontweight='bold', pad=8)
                
                if i == 0: 
                    ax.set_ylabel(r'$\Delta\overline{\Phi}^\pm$', labelpad=12)
                
                ax.set_xlabel(LABELS_LATEX[interaction])
                
                if interaction == "hf_phi":
                    ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
                    ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
                
                if nuc_label not in loaded_data or interaction not in loaded_data[nuc_label]:
                    ax.text(0.5, 0.5, "Data Unavailable", ha='center', va='center', transform=ax.transAxes)
                    ax.set_xlim(min_val, max_val)
                    continue

                dataset = loaded_data[nuc_label]

                try:
                    a_plus_max, a_minus_max = get_symmetry_amplitudes(
                        dataset[interaction]["max"][state],
                        dataset[interaction]["max"][source_state]
                    )
                    a_plus_min, a_minus_min = get_symmetry_amplitudes(
                        dataset[interaction]["min"][state],
                        dataset[interaction]["min"][source_state]
                    )
                except KeyError:
                    ax.text(0.5, 0.5, "Missing State Data", ha='center', va='center', transform=ax.transAxes)
                    ax.set_xlim(min_val, max_val)
                    continue
                    
                current_max = np.max([np.max(a_plus_max), np.max(a_minus_max), 
                                      np.max(a_plus_min), np.max(a_minus_min)])
                if current_max > global_max_y:
                    global_max_y = current_max

                n_sweeps = a_plus_max.shape[0]
                x_vals = np.linspace(min_val, max_val, n_sweeps)

                ax.plot(x_vals, a_plus_max, color=C_SYM, linestyle='-', lw=1.5)
                ax.plot(x_vals, a_minus_max, color=C_ANTI, linestyle='-', lw=1.5)
                ax.plot(x_vals, a_plus_min, color=C_SYM, linestyle='--', lw=1.5)
                ax.plot(x_vals, a_minus_min, color=C_ANTI, linestyle='--', lw=1.5)
                
                ax.set_xlim(min_val, max_val)
                sns.despine(ax=ax)
                ax.grid(True, linestyle=':', alpha=0.6)

            if global_max_y > 0:
                axes[0].set_ylim(0, global_max_y * 1.05) 

            legend_elements = [
                Line2D([0], [0], color=C_SYM, linestyle='-', label=r'$\Delta\overline{\Phi}^+$ (Sym, Max CISS)'),
                Line2D([0], [0], color=C_ANTI, linestyle='-', label=r'$\Delta\overline{\Phi}^-$ (Anti, Max CISS)'),
                Line2D([0], [0], color=C_SYM, linestyle='--', label=r'$\Delta\overline{\Phi}^+$ (Sym, Min CISS)'),
                Line2D([0], [0], color=C_ANTI, linestyle='--', label=r'$\Delta\overline{\Phi}^-$ (Anti, Min CISS)')
            ]
            
            fig.legend(handles=legend_elements, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05))

            plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.98], w_pad=1.0)
            
            safe_nuc_label = nuc_label.replace(" ", "_")
            outfile = os.path.join(plots_dir, f"Symmetry_Grid_{interaction}_{safe_nuc_label}.pdf")
            
            plt.savefig(outfile, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {outfile}")

if __name__ == "__main__":
    main()