import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import models
from scipy.stats import skew 

# ==========================================
# 1. CONFIGURATION 
# ==========================================
WEIGHTS_FILE = 'rigl_cifar10_90_sparse.pth' 
DATASET = 'CIFAR10' 
# SPARSITY_DENSITY is kept here as a reference, but we will dynamically calculate it below!
GLOBAL_SPARSITY_TARGET = 0.1

def calculate_c_max(n_in, density, alpha=1.5, kappa=4.0):
    """The TDST Biological Metabolic Tax Formula"""
    e_k = density * n_in
    headroom_cap = alpha * e_k
    sqrt_cap = kappa * math.sqrt(n_in)
    return min(n_in, max(headroom_cap, sqrt_cap))

def get_resnet_arch():
    """Matches the exact architecture used in your CIFAR-10 training"""
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def run_physical_autopsy(weights_path):
    file_base = os.path.splitext(os.path.basename(weights_path))[0]
    
    if 'rigl' in file_base.lower():
        algo_label = 'RigL'
        algo_color = 'crimson'
        cmap_theme = 'magma'
    elif 'set' in file_base.lower():
        algo_label = 'SET'
        algo_color = 'darkorange'
        cmap_theme = 'plasma'
    elif 'mbpa' in file_base.lower() or 'tdst' in file_base.lower():
        algo_label = file_base.split('_')[0].upper()
        algo_color = 'forestgreen'
        cmap_theme = 'viridis'
    else:
        algo_label = file_base.split('_')[0].upper()
        algo_color = 'purple'
        cmap_theme = 'viridis'

    output_png_name = f"{file_base}_autopsy.png"

    print(f"\n--- STARTING {algo_label} PHYSICAL AUTOPSY: {weights_path} ---")
    
    model = get_resnet_arch()
    if not os.path.exists(weights_path):
        print(f"ERROR: {weights_path} not found! Check the filename.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    
    layer_names, organic_god_hubs, tdst_caps = [], [], []
    layer_input_sizes, all_layer_degrees, skewness_scores = [], [], []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            
            if isinstance(module, nn.Conv2d):
                out_c, in_c, k_h, k_w = w.shape
                if in_c <= 3 and 'conv1' in name: continue 
                in_degrees = (w != 0).sum(dim=(1, 2, 3)).cpu().numpy()
                fan_in_total = in_c * k_h * k_w
            else:
                out_c, in_c = w.shape
                in_degrees = (w != 0).sum(dim=1).cpu().numpy()
                fan_in_total = in_c
            
            all_layer_degrees.append(in_degrees)
            organic_god_hubs.append(int(in_degrees.max()))
            
            # --- THE LOCAL DENSITY FIX ---
            # Dynamically calculate the EXACT density RigL/SET assigned to this specific layer
            total_layer_params = w.numel()
            active_layer_params = int((w != 0).sum().item())
            local_density = active_layer_params / total_layer_params
            
            # Feed the accurate local density into the biological constraint formula
            tdst_caps.append(int(calculate_c_max(fan_in_total, local_density)))
            
            skewness_scores.append(skew(in_degrees))
            
            short_name = name.replace('layer', 'L').replace('.conv', '_C').replace('fc', 'FC')
            layer_names.append(short_name)
            layer_input_sizes.append(fan_in_total)

    # ---------------------------------------------------------
    # GRAPH GENERATION
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
    
    # PANEL 1: The God-Hub Bar Chart 
    x = np.arange(len(layer_names))  
    width = 0.35  
    ax1.bar(x - width/2, organic_god_hubs, width, label=f'{algo_label} Max Degree', color=algo_color)
    ax1.bar(x + width/2, tdst_caps, width, label='TDST Constraint Cap', color='dodgerblue')
    
    for i, s in enumerate(skewness_scores):
        ax1.text(i, organic_god_hubs[i] + 5, f'sk:{s:.1f}', ha='center', fontsize=8, rotation=90)

    ax1.set_ylabel('Connections per Neuron', fontsize=12, fontweight='bold')
    ax1.set_title(f'CIFAR-10 TOPOLOGY: {algo_label} Hubs vs. TDST\n(Labels: sk = Skewness)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n}\n(In:{s})" for n, s in zip(layer_names, layer_input_sizes)], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # PANEL 2: The True Step-Histograms (Filtered)
    target_layers = ['L1.0_C1', 'L2.0_C1', 'L3.0_C1', 'L4.0_C1', 'FC']
    colors = plt.get_cmap(cmap_theme)(np.linspace(0.2, 0.9, len(target_layers)))
    
    color_idx = 0
    for i, degrees in enumerate(all_layer_degrees):
        name = layer_names[i]
        
        # Only plot the target layers
        if name not in target_layers or degrees.max() == 0: 
            continue
            
        # The EXACT logic from basic_autopsy.py that worked perfectly
        ax2.hist(degrees, bins='auto', histtype='step', linewidth=2.5, 
                 alpha=0.9, color=colors[color_idx], label=name)
        color_idx += 1

    ax2.set_xlabel('In-Degree (Number of Live Weights)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Neurons', fontsize=12, fontweight='bold') # Back to raw counts!
    ax2.set_title(f'{algo_label} Evolutionary Spread: Step Histogram', fontsize=14, fontweight='bold')
    ax2.legend(title="Key Stages Sampled")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_png_name, dpi=300)
    print(f"\n[+] Physical Autopsy Complete. Result saved as: {output_png_name}")
    plt.show()

if __name__ == "__main__":
    run_physical_autopsy(WEIGHTS_FILE)