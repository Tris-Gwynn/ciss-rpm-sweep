import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pickle
import os

# --- SETTINGS ---
RESULTS_FOLDER = "Results_Original"
OUTPUT_DIR = "Plots_Az_Sweep_Singlet_Comparison"
SWEEP_STEPS_DEFAULT = 31

SNS_STYLE = "ticks"
SNS_CONTEXT = "paper"

# --- COLORS ---
# Anisotropy
COLOR_MIN = "#1f77b4"  
COLOR_MAX = "#d62728"  
# Symmetry
C_SYM = '#4c72b0'   
C_ANTI = '#dd8452'  

X_LABEL = r"$T_{zz}$ (mT)"
BOUNDS = (-1, 1)

def calculate_absolute_anisotropy(data_subset, sweep_vals, product):
    metric = []
    if product not in data_subset:
        return np.full(len(sweep_vals), np.nan)
        
    for i in range(len(sweep_vals)):
        try:
            raw = np.max(data_subset[product][i], axis=-1)
            metric.append(np.max(raw) - np.min(raw))
        except (IndexError, KeyError, ValueError):
            metric.append(np.nan)
    return np.array(metric)

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

def process_combined_az_sweeps():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, RESULTS_FOLDER)
    save_dir = os.path.join(script_dir, OUTPUT_DIR)
    
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)

    print("Generating Singlet Anisotropy and Symmetry Comparisons by Angle...")
    
    sns.set_theme(style=SNS_STYLE, context=SNS_CONTEXT, font_scale=1.4)
    plt.rcParams['font.family'] = 'sans-serif' 
    plt.rcParams['mathtext.fontset'] = 'cm'

    # Set up both 1x3 figures dynamically for single column width
    fig_ani, axes_ani = plt.subplots(1, 3, figsize=(12, 4), sharey=True, sharex=True)
    fig_sym, axes_sym = plt.subplots(1, 3, figsize=(12, 4), sharey=True, sharex=True)
    
    labels = ['a', 'b', 'c']
    pi_labels = ["0", r"\pi/4", r"\pi/2"]

    for i in range(3):
        ax_ani = axes_ani[i]
        ax_sym = axes_sym[i]
        
        filename = os.path.join(results_dir, f"angle_{i}.pkl")
        title_str = r"$\phi_\text{{hf}} = {}$".format(pi_labels[i])
        
        # Consistent label injection
        ax_ani.text(-0.05, 1.05, labels[i], transform=ax_ani.transAxes, fontsize=16, fontweight='bold', va='bottom')
        ax_sym.text(-0.05, 1.05, labels[i], transform=ax_sym.transAxes, fontsize=16, fontweight='bold', va='bottom')
        
        if not os.path.exists(filename):
            ax_ani.text(0.5, 0.5, "Pending...", ha='center', transform=ax_ani.transAxes)
            ax_sym.text(0.5, 0.5, "Pending...", ha='center', transform=ax_sym.transAxes)
            continue
            
        print(f"Loading {filename}...")
        with open(filename, "rb") as f:
            data = pickle.load(f)

        n_sweeps = SWEEP_STEPS_DEFAULT
        if "min" in data and len(data["min"]) > 0:
            sample_op = next(iter(data["min"]))
            n_sweeps = len(data["min"][sample_op])

        x_vals = np.linspace(BOUNDS[0], BOUNDS[1], n_sweeps)

        # ==========================================
        # 1. ANISOTROPY PLOTTING
        # ==========================================
        ani_min = calculate_absolute_anisotropy(data["min"], x_vals, "S")
        ani_max = calculate_absolute_anisotropy(data["max"], x_vals, "S")

        # Dashed line restored for the minimum
        ax_ani.plot(x_vals, ani_max, color=COLOR_MAX, linewidth=2.0, linestyle='-')
        ax_ani.plot(x_vals, ani_min, color=COLOR_MIN, linewidth=2.0, linestyle='--')

        ax_ani.set_title(title_str, fontsize=14, fontweight='normal', pad=10)
        ax_ani.set_xlim(x_vals.min(), x_vals.max())
        ax_ani.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.25)
        sns.despine(ax=ax_ani)
        
        if i == 0:
            ax_ani.set_ylabel(r"$\Delta \Phi_S$", fontsize=14)
        ax_ani.set_xlabel(X_LABEL, fontsize=12)

        # ==========================================
        # 2. SYMMETRY PLOTTING
        # ==========================================
        try:
            a_plus_max, a_minus_max = get_symmetry_amplitudes(data["max"]["S"], data["max"]["S"])
            a_plus_min, a_minus_min = get_symmetry_amplitudes(data["min"]["S"], data["min"]["S"])
            
            ax_sym.plot(x_vals, a_plus_max, color=C_SYM, linestyle='-', lw=2.0)
            ax_sym.plot(x_vals, a_minus_max, color=C_ANTI, linestyle='-', lw=2.0)
            ax_sym.plot(x_vals, a_plus_min, color=C_SYM, linestyle='--', lw=2.0)
            ax_sym.plot(x_vals, a_minus_min, color=C_ANTI, linestyle='--', lw=2.0)
        except KeyError:
            ax_sym.text(0.5, 0.5, "Missing 'S' Data", ha='center', va='center', transform=ax_sym.transAxes)

        ax_sym.set_title(title_str, fontsize=14, fontweight='normal', pad=10)
        ax_sym.set_xlim(x_vals.min(), x_vals.max())
        ax_sym.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.25)
        sns.despine(ax=ax_sym)
        
        if i == 0:
            ax_sym.set_ylabel(r"$\Delta\overline{\Phi}_S^\pm$", fontsize=14)
        ax_sym.set_xlabel(X_LABEL, fontsize=12)

    # --- Finalize Anisotropy Figure ---
    legend_elements_ani = [
        Line2D([0], [0], color=COLOR_MAX, linestyle='-', linewidth=2, label=r'$\chi=\pi/2$'),
        Line2D([0], [0], color=COLOR_MIN, linestyle='--', linewidth=2, label=r'$\chi=0$'),
    ]
    fig_ani.legend(handles=legend_elements_ani, loc='lower center', 
                   bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False, fontsize=12)
    fig_ani.tight_layout(rect=[0, 0.05, 1, 0.98], w_pad=1.5)
    
    out_ani = os.path.join(save_dir, "Az_Sweep_Singlet_Anisotropy_By_Angle.pdf")
    fig_ani.savefig(out_ani, bbox_inches='tight')
    print(f"Saved: {out_ani}")
    plt.close(fig_ani)

    # --- Finalize Symmetry Figure ---
    legend_elements_sym = [
        Line2D([0], [0], color=C_SYM, linestyle='-', linewidth=2.0, label=r'$\Delta\overline{\Phi}^+$ $(\chi=\pi/2)$'),
        Line2D([0], [0], color=C_ANTI, linestyle='-', linewidth=2.0, label=r'$\Delta\overline{\Phi}^-$ $(\chi=\pi/2)$'),
        Line2D([0], [0], color=C_SYM, linestyle='--', linewidth=2.0, label=r'$\Delta\overline{\Phi}^+$ $(\chi=0)$'),
        Line2D([0], [0], color=C_ANTI, linestyle='--', linewidth=2.0, label=r'$\Delta\overline{\Phi}^-$ $(\chi=0)$')
    ]
    fig_sym.legend(handles=legend_elements_sym, loc='lower center', 
                   bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False, fontsize=12)
    fig_sym.tight_layout(rect=[0, 0.05, 1, 0.98], w_pad=1.5)
    
    out_sym = os.path.join(save_dir, "Az_Sweep_Singlet_Symmetry_By_Angle.pdf")
    fig_sym.savefig(out_sym, bbox_inches='tight')
    print(f"Saved: {out_sym}")
    plt.close(fig_sym)

if __name__ == "__main__":
    process_combined_az_sweeps()