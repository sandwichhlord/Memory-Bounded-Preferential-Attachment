# This script provides an analysis of how the connection density distribution of neurons looks
# after post training compression

import torch
import torchvision.models as models
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os



def calculate_c_max(n_in, density, alpha=1.5, kappa=4.0):
    """The TDST Biological Metabolic Tax Formula"""
    e_k = density * n_in
    headroom_cap = alpha * e_k
    sqrt_cap = kappa * math.sqrt(n_in)
    return min(n_in, max(headroom_cap, sqrt_cap))

def run_topological_autopsy(model, model_name, target_sparsity=0.90):
    print(f"\n--- PHASE 1A: Booting {model_name} Dual-Autopsy (Target Sparsity: {target_sparsity*100}%) ---")
    
    density = 1.0 - target_sparsity
    
    layer_names = []
    organic_god_hubs = []
    tdst_caps = []
    layer_input_sizes = []
    
    # NEW: Store raw degree data for the distribution plot
    all_layer_degrees = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            w = module.weight.data
            out_c, in_c, k_h, k_w = w.shape
            
            if in_c <= 3:
                continue 
                
            kernel_norms = torch.norm(w.view(out_c, in_c, -1), dim=2)
            total_elements = out_c * in_c
            keep_elements = int(total_elements * density)
            
            if keep_elements == 0: continue
                
            threshold = torch.kthvalue(kernel_norms.flatten(), total_elements - keep_elements).values
            mask = (kernel_norms >= threshold).float()
            
            # Extract Degrees for the current layer
            in_degrees = mask.sum(dim=1).cpu().numpy()
            
            all_layer_degrees.append(in_degrees)
            organic_god_hubs.append(int(in_degrees.max()))
            tdst_caps.append(int(calculate_c_max(in_c, density)))
            
            short_name = name.replace('layer', 'L').replace('.conv', '_C')
            layer_names.append(short_name)
            layer_input_sizes.append(in_c)

    print(f"Data extracted successfully for {model_name}. Generating Dual-Panel Graph...")
    
    # ---------------------------------------------------------
    # GRAPH GENERATION (Dual Panel)
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # PANEL 1: The God-Hub Bar Chart (Left)
    x = np.arange(len(layer_names))  
    width = 0.35  
    ax1.bar(x - width/2, organic_god_hubs, width, label='Organic God-Hub', color='crimson')
    ax1.bar(x + width/2, tdst_caps, width, label='TDST Target Cap (C_max)', color='dodgerblue')
    ax1.set_ylabel('Maximum In-Degree (Connections)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Unregulated Super-Hubs vs. TDST Tax\n({model_name} @ {target_sparsity*100}% Sparsity)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n}\n(In:{s})" for n, s in zip(layer_names, layer_input_sizes)], rotation=45, ha='right')
    ax1.legend(fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # PANEL 2: The Degree Distribution Lines (Right)
    # Generate the color gradient: Dark Red (Early) -> Saturated Bright Red (Deep)
    colors = cm.Reds(np.linspace(0.4, 1.0, len(layer_names)))
    
    for i, degrees in enumerate(all_layer_degrees):
        # Calculate a smooth frequency distribution (normalized to probability density)
        max_deg = int(degrees.max())
        if max_deg == 0: continue
        
        # We use dynamic bins to capture the shape smoothly
        counts, bin_edges = np.histogram(degrees, bins=min(40, max_deg), density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Only label the first, middle, and last layers to keep the legend clean
        label_name = layer_names[i] if i in [0, len(layer_names)//2, len(layer_names)-1] else ""
        ax2.plot(bin_centers, counts, color=colors[i], linewidth=2.5, alpha=0.85, label=label_name)

    ax2.set_xlabel('Number of Connections (In-Degree)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability Density (Filter Frequency)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Layer-Wise Degree Distributions: {model_name}\n(Dark = Shallow Layers, Bright = Deep Layers)', fontsize=14, fontweight='bold')
    ax2.legend(title="Sampled Layers", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    
    # Make filename dynamic based on the model
    safe_model_name = model_name.lower().replace("-", "")
    image_filename = f'tdst_phase1a_{safe_model_name}_autopsy.png'
    plt.savefig(image_filename, dpi=300, bbox_inches='tight')
    print(f"--- High-Resolution Thesis Graphic Saved: {os.path.abspath(image_filename)} ---")
    
    # Note: When you close the window, the loop will automatically start the next model!
    plt.show() 

if __name__ == "__main__":
    # The Baseline Model Zoo
    models_to_test = {
        "ResNet-18": models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
        "ResNet-50": models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
        "VGG-16": models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
    }
    
    for name, target_model in models_to_test.items():
        run_topological_autopsy(model=target_model, model_name=name)
        
    print("\n--- Phase 1A Control Baseline Ablation Complete! ---")