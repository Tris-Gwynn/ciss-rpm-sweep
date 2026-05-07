import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os

# --- CONFIGURATION ---
INPUT_FILE = "Results/ciss_heatmap_data.pkl"
OUTPUT_FILE = "Results/Anisotropy_Heatmap.pdf"

SNS_STYLE = "ticks" 
SNS_CONTEXT = "paper"

def plot_heatmaps():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run generation script first.")
        return

    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)

    A_vals = data["A_vals"]
    D_vals = data["D_vals"]
    grid_0 = data["anisotropy_0"]
    grid_90 = data["anisotropy_90"]
    grid_diff = data["diff"]

    sns.set_theme(style=SNS_STYLE, context=SNS_CONTEXT, font_scale=1.4)
    plt.rcParams['font.family'] = 'sans-serif' 
    plt.rcParams['mathtext.fontset'] = 'cm'

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    labels = ['a', 'b', 'c']
    
    extent = [D_vals.min(), D_vals.max(), A_vals.min(), A_vals.max()]
    origin = 'lower'
    
    global_min = 0
    global_max = 0.18

    # Calculate 5 evenly spaced ticks using the min and max bounds
    x_ticks = np.linspace(D_vals.min(), D_vals.max(), 5)
    y_ticks = np.linspace(A_vals.min(), A_vals.max(), 5)
    
    for i, ax in enumerate(axes):
        ax.text(-0.05, 1.05, labels[i], transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom')
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

    im1 = axes[0].imshow(grid_0, origin=origin, extent=extent, cmap='Blues', aspect='auto', 
                         vmin=global_min, vmax=global_max)
    axes[0].set_title(r"$\chi=0$", fontsize=16, fontweight='normal', pad=10)
    axes[0].set_ylabel(r"$a_0$ (mT)", fontsize=14)
    axes[0].set_xlabel(r"$D_0$ (mT)", fontsize=14)
    fig.colorbar(im1, ax=axes[0], label="$\Delta\Phi$")

    im2 = axes[1].imshow(grid_90, origin=origin, extent=extent, cmap='Reds', aspect='auto', 
                         vmin=global_min, vmax=global_max)
    axes[1].set_title(r"$\chi=\pi/2$", fontsize=16, fontweight='normal', pad=10)
    axes[1].set_xlabel(r"$D_0$ (mT)", fontsize=14)
    fig.colorbar(im2, ax=axes[1], label=r"$\Delta\Phi$")

    max_abs = np.max(np.abs(grid_diff))
    im3 = axes[2].imshow(grid_diff, origin=origin, extent=extent, cmap='RdBu_r', aspect='auto', vmin=-max_abs, vmax=max_abs)
    axes[2].set_title(r"$\Delta = \chi_{\pi/2} - \chi_{0}$", fontsize=16, fontweight='normal', pad=10)
    axes[2].set_xlabel(r"$D_0$ (mT)", fontsize=14)
    fig.colorbar(im3, ax=axes[2], label=r"$K_\Phi$")

    plt.savefig(OUTPUT_FILE)
    print(f"Plot saved to {OUTPUT_FILE}")
    plt.close(fig)

if __name__ == "__main__":
    plot_heatmaps()