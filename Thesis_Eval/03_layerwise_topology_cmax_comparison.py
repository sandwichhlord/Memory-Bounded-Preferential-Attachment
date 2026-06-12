import os
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURATION 
# ==========================================
SAVE_DIR = './' 
MODES = ['set', 'rigl', 'mbpa']
SPARSITIES = [0.90, 0.95, 0.98, 0.99, 0.995]
SEEDS = [42, 43, 44]

TDST_ALPHA = 1.5  
TDST_KAPPA = 4.0  

device = torch.device("cpu")
print("\n🔍 Hardware: Forcing CPU for safe parametric extraction...")

# ==========================================
# 2. ARCHITECTURE
# ==========================================
def get_resnet():
    model = models.resnet18()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

# ==========================================
# 3. TOPOLOGY EXTRACTION ENGINE
# ==========================================
def extract_layer_stats(model, weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    
    layer_stats = []
    layer_idx = 1
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            
            if weight.dim() < 2: continue
            
            fan_in = weight.view(weight.shape[0], -1).shape[1]
            degrees = (weight != 0).view(weight.shape[0], -1).sum(dim=1).cpu().numpy()
            max_degree = degrees.max()
            
            # THE FIX: Calculate actual layer density dynamically
            actual_non_zeros = (weight != 0).sum().item()
            total_params = weight.numel()
            actual_density = actual_non_zeros / total_params if total_params > 0 else 0
            
            # Theoretical c_max based on TRUE layer density
            e_k = fan_in * actual_density
            c_max = min(fan_in, max(TDST_ALPHA * e_k, TDST_KAPPA * np.sqrt(fan_in)))
            
            layer_stats.append({
                'Layer': layer_idx,
                'Max_Degree': max_degree,
                'C_Max': c_max
            })
            layer_idx += 1
            
    return layer_stats

# ==========================================
# 4. SWEEP AND PLOT 
# ==========================================
if __name__ == '__main__':
    model = get_resnet().to(device)

    print("\n🔬 Sweeping models and rendering layer topologies...\n")

    for sparsity in SPARSITIES:
        print(f"📊 Generating 3x3 Plot for Sparsity: {sparsity*100}%")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f"Layer-wise Topology Analysis (Sparsity: {sparsity*100}%)", fontsize=22, fontweight='bold', y=0.95)
        
        for row_idx, mode in enumerate(MODES):
            for col_idx, seed in enumerate(SEEDS):
                ax1 = axes[row_idx, col_idx]
                
                base_name = f"{mode.upper()}_{sparsity*100}%_Seed{seed}"
                weights_file = os.path.join(SAVE_DIR, f"{base_name}_sparse.pth")
                
                if os.path.exists(weights_file):
                    # No longer passing target_sparsity, calculating it dynamically
                    stats = extract_layer_stats(model, weights_file)
                    
                    layers = [s['Layer'] for s in stats]
                    max_degrees = [s['Max_Degree'] for s in stats]
                    c_maxs = [s['C_Max'] for s in stats]
                    
                    ax1.bar(layers, max_degrees, color='steelblue', alpha=0.8, label='Max Degree')
                    ax1.plot(layers, c_maxs, color='red', linestyle='--', linewidth=2.5, label='Theoretical $c_{max}$')
                    ax1.set_xlabel("Layer Index", fontsize=10)
                    ax1.set_ylabel("Filter Degree (Fan-In)", fontsize=10)
                    
                    ax1.set_title(f"{mode.upper()} - Seed {seed}", fontsize=14, fontweight='bold')
                    ax1.set_xticks(layers)
                    ax1.set_xticklabels(layers, fontsize=8, rotation=90)
                    ax1.grid(True, linestyle=':', alpha=0.6)
                    
                    if row_idx == 0 and col_idx == 0:
                        ax1.legend(loc='upper left', fontsize=10)
                else:
                    ax1.text(0.5, 0.5, f"Missing File:\n{base_name}", ha='center', va='center', fontsize=12, color='red')
                    ax1.set_title(f"{mode.upper()} - Seed {seed}")
                    ax1.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) 
        
        output_filename = os.path.join(SAVE_DIR, f'Phase3_Topology_Clean_{sparsity*100}%.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_filename}")

    print("\n🏆 ALL CLEAN TOPOLOGY PLOTS GENERATED SUCESSFULLY")