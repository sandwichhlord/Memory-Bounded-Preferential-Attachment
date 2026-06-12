import os
import glob
import re
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ==========================================
# 1. SETUP AND CONFIGURATION
# ==========================================
INPUT_DIR = "./"  
SAVE_DIR = "./ridgeline_layerwise"
os.makedirs(SAVE_DIR, exist_ok=True)

MODELS = ["MBPA", "RIGL", "SET"]
SEEDS = [42, 43, 44]
SPARSITIES = [0.80, 0.90, 0.95, 0.99, 0.995]

# The curated, readable subset of ResNet-18 layers
TARGET_LAYERS = [
    "conv1", 
    "layer1.0.conv1", 
    "layer2.0.conv1", 
    "layer3.0.conv1", 
    "layer4.0.conv1"
]

PALETTE = {"MBPA": "#2a9d8f", "RIGL": "#e76f51", "SET": "#e9c46a"}

# --- YOUR ALGORITHM CONSTANTS ---
TDST_ALPHA = 1.5
TDST_KAPPA = 4.0  

# ==========================================
# 2. FILE PARSER (WITH ERK-AWARE C_MAX)
# ==========================================
def extract_layerwise_data(filepath, sparsity):
    state_dict = torch.load(filepath, map_location='cpu')
    layer_data = []
    
    for layer_name in TARGET_LAYERS:
        weight_key = layer_name + ".weight"
        if weight_key in state_dict:
            tensor = state_dict[weight_key]
            if tensor.dim() > 1:
                # 1. Get Degrees
                flattened = tensor.view(tensor.size(0), -1)
                non_zeros = (flattened != 0).sum(dim=1).numpy()
                
                # 2. Calculate c_max dynamically based on TRUE LOCAL DENSITY (ERK-Aware)
                fan_in = flattened.size(1) 
                total_params = flattened.size(0) * flattened.size(1)
                current_active = (flattened != 0).sum().item()
                
                # EXACT MATCH to your training script (accounts for ERK density shifts)
                true_local_density = current_active / total_params
                e_k = true_local_density * fan_in
                
                # Your exact formula:
                c_max = min(fan_in, max(TDST_ALPHA * e_k, TDST_KAPPA * math.sqrt(fan_in)))
                
                layer_data.append({
                    'Layer': layer_name, 
                    'Degrees': non_zeros,
                    'C_Max': c_max
                })
    return layer_data

def load_data():
    print(f"Scanning {INPUT_DIR} for .pth files...")
    file_paths = glob.glob(os.path.join(INPUT_DIR, "*.pth"))
    
    all_data = []
    for filepath in file_paths:
        filename = os.path.basename(filepath)
        # Added a '?' after the '%' just in case there are minor naming variations
        match = re.search(r'(MBPA|RIGL|SET)_([0-9.]+)%?_Seed([0-9]+)_sparse\.pth', filename)
        if not match: continue
            
        model = match.group(1)
        sparsity = float(match.group(2)) / 100.0
        seed = int(match.group(3))
        
        try:
            layers = extract_layerwise_data(filepath, sparsity)
            for l_data in layers:
                all_data.append({
                    'Sparsity': sparsity, 'Model': model, 
                    'Seed': seed, 'Layer': l_data['Layer'], 
                    'Degrees': l_data['Degrees'],
                    'C_Max': l_data['C_Max']
                })
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            
    return pd.DataFrame(all_data)

# ==========================================
# 3. 3x3 RIDGELINE PLOTTER
# ==========================================
def plot_single_ridgeline(ax, df_subset, color):
    if df_subset.empty: return
    
    layers = list(reversed(TARGET_LAYERS))
    step = 1.0 
    
    # Lock X-axis based on the maximum degree OR max c_max in this cell
    local_max_deg = np.max([d.max() for d in df_subset['Degrees']])
    local_max_c = np.max(df_subset['C_Max'])
    global_max = max(local_max_deg, local_max_c)
    
    x_vals = np.linspace(0, global_max * 1.2, 500)
    
    for idx, layer_name in enumerate(layers):
        layer_row = df_subset[df_subset['Layer'] == layer_name]
        if layer_row.empty: continue
            
        degrees = layer_row.iloc[0]['Degrees']
        c_max = layer_row.iloc[0]['C_Max']
        
        degrees = degrees + np.random.normal(0, 1e-4, size=len(degrees))
        
        try:
            # REDUCED SMOOTHING (bw_method lowered to 0.08 for sharper spikes)
            kde = gaussian_kde(degrees, bw_method=0.08)
            y_vals = kde(x_vals)
            y_vals = (y_vals / np.max(y_vals)) * (step * 1.5) 
            
            # --- THE SMART ARTIFACT KILLER ---
            if df_subset.iloc[0]['Model'] == 'MBPA':
                actual_tensor_max = np.max(degrees)
                # If the tensor strictly obeyed the limit, sheer off the visual bleed.
                if actual_tensor_max <= c_max:
                    y_vals[x_vals > c_max] = 0
                    
        except:
            y_vals = np.zeros_like(x_vals)
            
        baseline = idx * step
        
        # Plot Mountain
        ax.fill_between(x_vals, baseline, baseline + y_vals, facecolor=color, alpha=0.85, zorder=idx)
        ax.plot(x_vals, baseline + y_vals, color='w', lw=1.2, zorder=idx+0.1)
        ax.plot(x_vals, [baseline]*len(x_vals), color='#333333', lw=1, zorder=idx+0.1)
        
        # THE KILL-SHOT: Draw the dynamic C_Max boundary line for this specific layer
        ax.plot([c_max, c_max], [baseline, baseline + (step * 1.3)], 
                color='#e63946', linestyle='--', linewidth=2.5, zorder=idx+0.2)
        
        ax.text(-global_max * 0.05, baseline + (step*0.2), layer_name, 
                ha='right', va='center', fontsize=9, fontweight='bold', color='#444')

    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(-global_max * 0.2, global_max * 1.2)

def generate_3x3_image(df, sparsity):
    print(f"Generating 3x3 Image for {sparsity*100}% Sparsity...")
    
    df_sp = df[df['Sparsity'] == sparsity]
    if df_sp.empty: return

    abs_max_degree = np.max([d.max() for d in df_sp['Degrees']])
    abs_max_c = np.max(df_sp['C_Max'])
    chart_max = max(abs_max_degree, abs_max_c)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16), sharex=False)
    fig.suptitle(f"Layer-wise Degree Distributions vs Hardware Cap ($C_{{max}}$) at {sparsity*100}% Sparsity", 
                 fontsize=22, fontweight='bold', y=0.96)

    for i, model in enumerate(MODELS):
        for j, seed in enumerate(SEEDS):
            ax = axes[i, j]
            df_cell = df_sp[(df_sp['Model'] == model) & (df_sp['Seed'] == seed)]
            
            plot_single_ridgeline(ax, df_cell, PALETTE[model])
            ax.set_xlim(-chart_max * 0.15, chart_max * 1.1)
            
            title = ""
            if i == 0: title += f"Seed {seed}\n"
            if j == 1 and i == 0: title = f"{title}" 
            
            ax.set_title(title, fontsize=15, fontweight='bold')
            
            if j == 0: 
                ax.set_ylabel(f"{model}", fontsize=18, fontweight='bold', rotation=0, labelpad=50, va='center')
            if i == 2:
                ax.set_xlabel("Neuron Degree", fontsize=14, fontweight='bold')

    # Add a custom legend just for the C_max line so reviewers know what it is
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#e63946', lw=2.5, linestyle='--')]
    fig.legend(custom_lines, ['$C_{max}$ Cache Limit'], loc='upper right', 
               bbox_to_anchor=(0.95, 0.96), fontsize=14, frameon=True)

    plt.subplots_adjust(wspace=0.1, hspace=0.25)
    
    filename = os.path.join(SAVE_DIR, f"Layerwise_Cmax_Sparsity_{sparsity*100:.1f}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='w')
    plt.close()

# ==========================================
# 4. EXECUTE
# ==========================================
df = load_data()
if not df.empty:
    for sp in df['Sparsity'].unique():
        generate_3x3_image(df, sp)
    print(f"\n✅ All Layer-wise images with C_max boundaries saved to {SAVE_DIR}!")
else:
    print("❌ No data loaded.")