import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from torchvision import models

# ==========================================
# 1. CONFIGURATION
# ==========================================
WEIGHTS_FILE = 'ultimate_mbpa_sparse.pth' # Drag and drop your .pth here

def get_resnet_arch():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def generate_ridgeline_plot(weights_path):
    print(f"\n--- GENERATING RIDGELINE TOPOLOGY: {weights_path} ---")
    
    if not os.path.exists(weights_path):
        print(f"ERROR: {weights_path} not found!")
        return
        
    model = get_resnet_arch()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    
    layer_names = []
    all_layer_degrees = []
    
    # Extract the data
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if isinstance(module, nn.Conv2d) and module.in_channels <= 3:
                continue # Skip the tiny input layer
                
            w = module.weight.data
            degrees = (w != 0).sum(dim=(1, 2, 3)).numpy() if isinstance(module, nn.Conv2d) else (w != 0).sum(dim=1).numpy()
            
            short_name = name.replace('layer', 'L').replace('.conv', '_C').replace('fc', 'FC')
            layer_names.append(short_name)
            all_layer_degrees.append(degrees)

    # ==========================================
    # 2. RAW JAGGED RIDGELINE GRAPHICS ENGINE
    # ==========================================
    num_layers = len(layer_names)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Sequential Colormap: Magma (Light -> Dark)
    colors = cm.magma(np.linspace(0.9, 0.2, num_layers)) 
    
    # Find global X-axis boundaries so all layers scale perfectly together
    flat_degrees = np.concatenate(all_layer_degrees)
    x_max = np.percentile(flat_degrees, 99.5) * 1.1 
    
    # Ridgeline Spacing Parameters
    overlap_factor = 1.5 
    y_offset_step = 1.0 
    
    # We iterate backwards so the first layer is plotted in the "front" (z-order)
    for i in range(num_layers - 1, -1, -1):
        degrees = all_layer_degrees[i]
        name = layer_names[i]
        color = colors[i]
        
        # Calculate the Y-baseline for this specific layer
        baseline = (num_layers - 1 - i) * y_offset_step
        z_index = num_layers - i # Ensures front layers obscure back layers
        
        variance = np.var(degrees)
        
        # --- THE HYBRID SWITCH ---
        if variance < 1e-3:
            # COMMUNIST SET TRAP: Draw a rigid spike
            val = degrees[0]
            # Create an artificial sharp spike for perfectly uniform layers
            x_vals = np.array([val - 1e-3, val, val + 1e-3])
            y_vals = np.array([0, overlap_factor, 0])
            
            ax.plot(x_vals, y_vals + baseline, color='white', linewidth=1.5, zorder=z_index)
            ax.plot(x_vals, y_vals + baseline, color=color, linewidth=1.0, zorder=z_index)
            ax.fill_between(x_vals, baseline, y_vals + baseline, facecolor=color, alpha=0.85, zorder=z_index)
        else:
            # RAW HISTOGRAM ENGINE: No KDE smoothing. 
            # We use 100 bins to keep it highly jagged but readable.
            bins = min(100, max(2, len(np.unique(degrees))))
            counts, bin_edges = np.histogram(degrees, bins=bins)
            
            # Use bin centers for the X coordinates
            x_hist = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Pad the edges with zeros so the mountain doesn't float
            step_size = x_hist[1] - x_hist[0] if len(x_hist) > 1 else 1
            x_vals = np.concatenate(([x_hist[0] - step_size], x_hist, [x_hist[-1] + step_size]))
            y_vals = np.concatenate(([0], counts, [0]))
            
            # Normalize the height so it overlaps smoothly with the layer above it
            if y_vals.max() > 0:
                y_vals = (y_vals / y_vals.max()) * overlap_factor
            
            # Plot the raw, jagged lines and fill the mountain
            ax.plot(x_vals, y_vals + baseline, color='white', linewidth=1.5, zorder=z_index)
            ax.plot(x_vals, y_vals + baseline, color=color, linewidth=1.0, zorder=z_index)
            ax.fill_between(x_vals, baseline, y_vals + baseline, facecolor=color, alpha=0.85, zorder=z_index)

        # Add the Layer Label floating just to the left of its baseline
        ax.text(-x_max * 0.02, baseline + 0.2, name, fontweight='bold', 
                fontsize=10, color=color, ha='right', va='center', zorder=z_index)

    # Clean up the aesthetics
    ax.set_xlim(-x_max * 0.15, x_max) # Give room for labels on the left
    ax.set_ylim(0, num_layers * y_offset_step + overlap_factor)
    
    # Turn off the standard Y-axis since the baselines act as our Y-axis
    ax.set_yticks([]) 
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_xlabel('In-Degree (Number of Live Connections)', fontsize=12, fontweight='bold')
    plt.title('Raw Topology Evolution: Jagged Ridgeline Plot', fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    output_name = f"{weights_path.split('.')[0]}_raw_ridgeline.png"
    plt.savefig(output_name, dpi=300, transparent=False)
    print(f"[+] Raw Ridgeline map saved to {output_name}")
    plt.show()

if __name__ == "__main__":
    generate_ridgeline_plot(WEIGHTS_FILE)