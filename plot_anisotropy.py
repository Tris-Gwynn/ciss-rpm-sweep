import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pickle
import os

# --- CONFIGURATION ---
OUTPUT_DIR = "Plots_Main_Results_Combined"
SWEEP_STEPS_DEFAULT = 31

SNS_STYLE = "ticks"
SNS_CONTEXT = "paper"

INTERACTIONS = ["B", "Ai", "Az", "D", "J", "hf_phi"]

LABELS = {
    "B":  r"$B_0$ (mT)",
    "Ai": r"$a_0$ (mT)",
    "Az": r"$T_{zz}$ (mT)",
    "D":  r"$D_0$ (mT)",
    "J":  r"$J$ (mT)",
    "hf_phi": r"$\phi_\text{hf}$ (rad)"
}

BOUNDS = {
    "B":  (0, 0.1), 
    "Ai": (-1, 1),  
    "Az": (-1, 1),  
    "D":  (-1, 1),  
    "J":  (-1, 1),
    "hf_phi": (0, np.pi) 
}

# --- ANISOTROPY CONFIG (1x3 Layout) ---
ANI_OPERATORS = ["S", "T", "R", "Tp", "T0", "Tm"]
ANI_LABELS = {"S": r"$S$", "T": r"$T$", "R": r"$R$", "Tp": r"${T}_+$", "T0": r"${T}_0$", "Tm": r"${T}_-$"}
ANI_COLORS = {"S": "#e8000b", "T": "#023eff", "R": "#1ac938", "Tp": "#ff7c00", "T0": "#8b2be2", "Tm": "#9f4800"}
ANI_STYLES = {"S": "-", "T": "-", "R": "-", "Tp": "--", "T0": "--", "Tm": "--"}
ANI_WIDTHS = {"S": 1.5, "T": 1.5, "R": 1.5, "Tp": 1.2, "T0": 1.2, "Tm": 1.2}
ANI_ALPHAS = {"S": 1.0, "T": 1.0, "R": 1.0, "Tp": 0.8, "T0": 0.8, "Tm": 0.8}

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "Results")
    
    files = {
        "1_Nucleus": os.path.join(results_dir, "full_ciss_data.pkl"),
        "2_Nuclei":  os.path.join(results_dir, "2nuc_ciss_data.pkl")
    }
    
    loaded_data = {}
    for label, path in files.items():
        if os.path.exists(path):
            print(f"Loading data from {path}...")
            with open(path, "rb") as f:
                loaded_data[label] = pickle.load(f)
        else:
            print(f"Warning: Could not find {path}")
            
    return loaded_data

def calculate_absolute_anisotropy(data_subset, sweep_vals, product, dataset_key):
    metric = []
    if dataset_key not in data_subset or product not in data_subset[dataset_key]:
        return None

    for i in range(len(sweep_vals)):
        try:
            raw = np.max(data_subset[dataset_key][product][i], axis=-1)
            metric.append(np.max(raw) - np.min(raw))
        except (IndexError, KeyError, ValueError):
            metric.append(np.nan)
    return np.array(metric)

def format_axis(ax, key, x_vals):
    ax.set_xlabel(LABELS[key], fontsize=14)
    ax.set_xlim(min(x_vals), max(x_vals))
    
    if key == "hf_phi":
        ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])

def generate_anisotropy_1x3(x_vals, y_min_dict, y_max_dict, interaction_key, save_path):
    sns.set_theme(style=SNS_STYLE, context=SNS_CONTEXT, font_scale=1.4)
    plt.rcParams['font.family'] = 'sans-serif' 
    plt.rcParams['mathtext.fontset'] = 'cm'

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = ['a', 'b', 'c']
    
    for i, ax in enumerate(axes):
        ax.text(-0.05, 1.05, labels[i], transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom')
        format_axis(ax, interaction_key, x_vals)
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.25)
        sns.despine(ax=ax)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)

    for op in ANI_OPERATORS:
        if op in y_min_dict and y_min_dict[op] is not None:
            axes[0].plot(x_vals, y_min_dict[op], color=ANI_COLORS[op], linestyle=ANI_STYLES[op], 
                         linewidth=ANI_WIDTHS[op], alpha=ANI_ALPHAS[op])
        if op in y_max_dict and y_max_dict[op] is not None:
            axes[1].plot(x_vals, y_max_dict[op], color=ANI_COLORS[op], linestyle=ANI_STYLES[op], 
                         linewidth=ANI_WIDTHS[op], alpha=ANI_ALPHAS[op])
        if op in y_min_dict and op in y_max_dict and y_min_dict[op] is not None and y_max_dict[op] is not None:
            diff = np.array(y_max_dict[op]) - np.array(y_min_dict[op])
            axes[2].plot(x_vals, diff, color=ANI_COLORS[op], linestyle=ANI_STYLES[op], 
                         linewidth=ANI_WIDTHS[op], alpha=ANI_ALPHAS[op])

    axes[0].set_ylabel(r"$\Delta \Phi$"+r"$(\chi=0)$", fontsize=14)
    axes[1].set_ylabel(r"$\Delta \Phi$"+r"$(\chi=\pi/2)$", fontsize=14)
    axes[2].set_ylabel(r"$K_\Phi$", fontsize=14)

    y0_min, y0_max = axes[0].get_ylim()
    y1_min, y1_max = axes[1].get_ylim()
    shared_ymin, shared_ymax = min(y0_min, y1_min), max(y0_max, y1_max)
    axes[0].set_ylim(shared_ymin, shared_ymax)
    axes[1].set_ylim(shared_ymin, shared_ymax)

    legend_elements = [Line2D([0], [0], color=ANI_COLORS[op], linestyle=ANI_STYLES[op], 
                       linewidth=ANI_WIDTHS[op], alpha=ANI_ALPHAS[op], label=ANI_LABELS[op]) for op in ANI_OPERATORS]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=6, frameon=False, fontsize=12)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def process_all_results(data):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "Results", OUTPUT_DIR)
    
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)

    for nuc_label, dataset in data.items():
        print(f"\n--- Processing {nuc_label} ---")
        
        for key in INTERACTIONS:
            if key not in dataset:
                continue
                
            print(f"Generating plots for {key}...")

            n_sweeps = SWEEP_STEPS_DEFAULT
            if "min" in dataset[key] and len(dataset[key]["min"]) > 0:
                sample_op = next(iter(dataset[key]["min"]))
                n_sweeps = len(dataset[key]["min"][sample_op])

            x_vals = np.linspace(BOUNDS[key][0], BOUNDS[key][1], n_sweeps)
            
            ani_min_data, ani_max_data = {}, {}
            for op in ANI_OPERATORS:
                ani_min_data[op] = calculate_absolute_anisotropy(dataset[key], x_vals, op, "min")
                ani_max_data[op] = calculate_absolute_anisotropy(dataset[key], x_vals, op, "max")
            
            ani_path = os.path.join(save_dir, f"{nuc_label}_Anisotropy_1x3_{key}.pdf")
            generate_anisotropy_1x3(x_vals, ani_min_data, ani_max_data, key, ani_path)

if __name__ == "__main__":
    datasets = load_data()
    process_all_results(datasets)