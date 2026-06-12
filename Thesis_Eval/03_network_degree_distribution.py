import os
import glob
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ==========================================
# 1. SETUP AND CONFIGURATION
# ==========================================
INPUT_DIR = "./"  # Point this to where your .pth files are
SAVE_DIR = "./ridgeline_global"
os.makedirs(SAVE_DIR, exist_ok=True)

# Using upper case RIGL based on your previous code snippet
MODELS = ["MBPA", "RIGL", "SET"]
SEEDS = [42, 43, 44]
SPARSITIES = [0.80, 0.90, 0.95, 0.99, 0.995]

# Distinct academic colors
PALETTE = {"MBPA": "#2a9d8f", "RIGL": "#e76f51", "SET": "#e9c46a"}

# ==========================================
# 2. FILE PARSER (GLOBAL EXTRACTION)
# ==========================================
def extract_global_data(filepath):
    """Extracts the degree of EVERY neuron across the entire network."""
    state_dict = torch.load(filepath, map_location='cpu')
    global_degrees = []
    
    for name, tensor in state_dict.items():
        # Look at all weight matrices (Conv2d and Linear)
        if 'weight' in name and tensor.dim() > 1:
            # Flatten everything except the out_channels/out_features dimension
            flattened = tensor.view(tensor.size(0), -1)
            # Count non-zero incoming connections
            non_zeros = (flattened != 0).sum(dim=1).numpy()
            global_degrees.extend(non_zeros)
            
    return np.array(global_degrees)

def load_data():
    print(f"Scanning {INPUT_DIR} for .pth files...")
    file_paths = glob.glob(os.path.join(INPUT_DIR, "*.pth"))
    
    all_data = []
    for filepath in file_paths:
        filename = os.path.basename(filepath)
        # Regex accepts both 'RigL' and 'RIGL' just in case
        match = re.search(r'(MBPA|RigL|RIGL|SET)_([0-9.]+)%?_Seed([0-9]+)_sparse\.pth', filename, re.IGNORECASE)
        if not match: continue
            
        model = match.group(1).upper() # Normalize to uppercase
        sparsity = float(match.group(2)) / 100.0
        seed = int(match.group(3))
        
        try:
            degrees = extract_global_data(filepath)
            
            # Optional: If the network is massive, sample it to save RAM. 
            # ResNet-18 is usually small enough (~5-8k neurons) to skip sampling.
            
            # Store the entire numpy array in one row
            all_data.append({
                'Sparsity': sparsity, 
                'Model': model, 
                'Seed': seed, 
                'Degrees': degrees
            })
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            
    return pd.DataFrame(all_data)

# ==========================================
# 3. 3x3 PLOTTER (GLOBAL DISTRIBUTION)
# ==========================================
def plot_single_distribution(ax, df_subset, color):
    """Draws a single filled density plot for the whole network."""
    if df_subset.empty: return
    
    degrees = df_subset.iloc[0]['Degrees']
    
    # Add microscopic noise to prevent singular matrix errors in perfectly uniform SET layers
    degrees = degrees + np.random.normal(0, 1e-4, size=len(degrees))
    
    # Lock X-axis bounds purely on the data in this exact cell for KDE bounds
    local_max = np.max(degrees)
    x_vals = np.linspace(0, local_max * 1.2, 1000)
    
    try:
        # Using a slightly wider bandwidth (0.15) because global data has higher variance 
        # than isolated layer data, ensuring a smooth, beautiful curve.
        kde = gaussian_kde(degrees, bw_method=0.15)
        y_vals = kde(x_vals)
        # Normalize height
        y_vals = y_vals / np.max(y_vals) 
    except:
        y_vals = np.zeros_like(x_vals)
        
    # Plot the filled mountain
    ax.fill_between(x_vals, 0, y_vals, facecolor=color, alpha=0.85, zorder=1)
    # Plot the crisp white outline
    ax.plot(x_vals, y_vals, color='w', lw=1.5, zorder=2)
    # Plot the baseline
    ax.plot(x_vals, np.zeros_like(x_vals), color='#333333', lw=1.5, zorder=3)

    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def generate_3x3_global_image(df, sparsity):
    print(f"Generating Global 3x3 Image for {sparsity*100}% Sparsity...")
    
    df_sp = df[df['Sparsity'] == sparsity]
    if df_sp.empty: return

    # Lock the X-axis across the entire 3x3 grid so the scale is 100% fair
    chart_max = np.max([d.max() for d in df_sp['Degrees']])
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=False)
    fig.suptitle(f"Global Network Degree Distribution Across Seeds ({sparsity*100}% Sparsity)", 
                 fontsize=22, fontweight='bold', y=0.96)

    for i, model in enumerate(MODELS):
        for j, seed in enumerate(SEEDS):
            ax = axes[i, j]
            df_cell = df_sp[(df_sp['Model'] == model) & (df_sp['Seed'] == seed)]
            
            plot_single_distribution(ax, df_cell, PALETTE[model])
            
            # Apply the unified X-axis limit
            ax.set_xlim(-chart_max * 0.1, chart_max * 1.1)
            
            # Formatting labels
            title = ""
            if i == 0: title += f"Seed {seed}\n"
            if j == 1 and i == 0: title = f"{title}" 
            
            ax.set_title(title, fontsize=15, fontweight='bold')
            
            if j == 0: 
                ax.set_ylabel(f"{model}", fontsize=18, fontweight='bold', rotation=0, labelpad=50, va='center')
            if i == 2:
                ax.set_xlabel("Neuron Degree (Connections)", fontsize=14, fontweight='bold')

    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    # Save with precise float formatting
    filename = os.path.join(SAVE_DIR, f"Global_Distribution_Sparsity_{sparsity*100:.1f}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='w')
    plt.close()

# ==========================================
# 4. EXECUTE
# ==========================================
df = load_data()
if not df.empty:
    for sp in df['Sparsity'].unique():
        generate_3x3_global_image(df, sp)
    print(f"\n✅ All Global Distribution 3x3 images saved to {SAVE_DIR}!")
else:
    print("❌ No data loaded.")