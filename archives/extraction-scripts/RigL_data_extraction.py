import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from torchvision import models
from scipy.stats import skew 

# ==========================================
# 1. CONFIGURATION (Targeting RigL Output)
# ==========================================
WEIGHTS_FILE = 'rigl_cifar10_90_sparse.pth' 
DATASET = 'CIFAR10' 
SPARSITY_DENSITY = 0.1 

def calculate_c_max(n_in, density, alpha=1.5, kappa=4.0):
    """The Biological Metabolic Tax Formula"""
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
    print(f"\n--- STARTING PHYSICAL AUTOPSY: {weights_path} ---")
    
    model = get_resnet_arch()
    if not os.path.exists(weights_path):
        print(f"ERROR: {weights_path} not found! Check the filename.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    
    layer_names = []
    organic_god_hubs = []
    tdst_caps = []
    layer_input_sizes = []
    all_layer_degrees = []
    skewness_scores = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            
            if isinstance(module, nn.Conv2d):
                out_c, in_c, k_h, k_w = w.shape
                # Skip initial layer as it usually doesn't evolve much
                if in_c <= 3 and 'conv1' in name: continue 
                in_degrees = (w != 0).sum(dim=(1, 2, 3)).cpu().numpy()
                fan_in_total = in_c * k_h * k_w
            else:
                out_c, in_c = w.shape
                in_degrees = (w != 0).sum(dim=1).cpu().numpy()
                fan_in_total = in_c
            
            all_layer_degrees.append(in_degrees)
            organic_god_hubs.append(int(in_degrees.max()))
            tdst_caps.append(int(calculate_c_max(fan_in_total, SPARSITY_DENSITY)))
            
            # Calculate Skewness: Higher = More "God-Hubby"
            skewness_scores.append(skew(in_degrees))
            
            short_name = name.replace('layer', 'L').replace('.conv', '_C').replace('fc', 'FC')
            layer_names.append(short_name)
            layer_input_sizes.append(fan_in_total)

    # ---------------------------------------------------------
    # GRAPH GENERATION (Dual Panel)
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
    
    # PANEL 1: The God-Hub Bar Chart
    x = np.arange(len(layer_names))  
    width = 0.35  
    ax1.bar(x - width/2, organic_god_hubs, width, label='RigL Max Degree', color='crimson')
    ax1.bar(x + width/2, tdst_caps, width, label='MBPA Constraint Cap', color='dodgerblue')
    
    # Annotate skewness on top of bars
    for i, s in enumerate(skewness_scores):
        ax1.text(i, organic_god_hubs[i] + 5, f'sk:{s:.1f}', ha='center', fontsize=8, rotation=90)

    ax1.set_ylabel('Connections per Neuron', fontsize=12, fontweight='bold')
    ax1.set_title(f'CIFAR-10 RigL TOPOLOGY: Evolved Hubs vs. MBPA\n(Labels: sk = Skewness)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n}\n(In:{s})" for n, s in zip(layer_names, layer_input_sizes)], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # PANEL 2: The Distribution Curves (Log-Normal/Planck Visual)
    colors = cm.magma(np.linspace(0.2, 0.9, len(layer_names)))
    for i, degrees in enumerate(all_layer_degrees):
        if degrees.max() == 0: continue
        counts, bin_edges = np.histogram(degrees, bins=40, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        lbl = layer_names[i] if i % 4 == 0 or i == len(layer_names)-1 else ""
        ax2.plot(bin_centers, counts, color=colors[i], linewidth=2.5, alpha=0.8, label=lbl)

    ax2.set_xlabel('In-Degree (Number of Live Weights)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax2.set_title('RigL Evolutionary Spread: Dense Gradients', fontsize=14, fontweight='bold')
    ax2.legend(title="Layers Sampled")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'rigl_cifar_autopsy_final.png', dpi=300)
    print(f"\n[+] Physical Autopsy Complete. Result saved as: rigl_cifar_autopsy_final.png")
    plt.show()

if __name__ == "__main__":
    run_physical_autopsy(WEIGHTS_FILE)