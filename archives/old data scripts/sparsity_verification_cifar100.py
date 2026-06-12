import os
import torch
import torch.nn as nn
from torchvision import models

# --- CONFIGURATION ---
# Change this to whichever model you want to autopsy
WEIGHTS_FILE = 'SET_99.5%_CIFAR100_sparse.pth' 

# ==========================================
# THE ARCHITECTURE PATCH (CRITICAL FOR CIFAR-100)
# ==========================================
def get_resnet():
    model = models.resnet18()
    
    # PATCH 1: 3x3 stem for 32x32 images (otherwise shape mismatch on load)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # PATCH 2: Remove MaxPool
    model.maxpool = nn.Identity()
    
    # PATCH 3: 100 Classes for CIFAR-100
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

def check_sparsity():
    print(f"[*] Loading model from {WEIGHTS_FILE}...")
    
    # 1. Rebuild the exact architecture using the patched builder
    model = get_resnet()
    
    # 2. Load the trained weights (map to CPU)
    try:
        model.load_state_dict(torch.load(WEIGHTS_FILE, map_location='cpu'))
    except FileNotFoundError:
        print(f"[!] Error: Could not find '{WEIGHTS_FILE}'. Check the path!")
        return
    except RuntimeError as e:
        print(f"[!] Shape Mismatch Error. Are you sure this file is a CIFAR-100 patched ResNet-18?")
        print(e)
        return

    total_targeted_params = 0
    total_active_params = 0
    
    print("-" * 65)
    print(f"{'Layer Name':<35} | {'Active / Total':<15} | {'Density'} | {'Sparsity'}")
    print("-" * 65)
    
    # 3. Autopsy the weights
    for name, param in model.named_parameters():
        # Only check weights with dim > 1 (Matches the ERK scheduler logic exactly)
        if 'weight' in name and param.dim() > 1:
            num_params = param.numel()
            # Count how many weights are exactly zero
            num_zeros = (param.data == 0.0).sum().item()
            num_active = num_params - num_zeros
            
            total_targeted_params += num_params
            total_active_params += num_active
            
            density = num_active / num_params
            sparsity = 1.0 - density
            
            print(f"{name:<35} | {num_active:<7} / {num_params:<5} | {density:^7.2%} | {sparsity:>7.2%}")
            
    print("-" * 65)
    
    # 4. Calculate Final Global Metrics
    global_density = total_active_params / total_targeted_params
    global_sparsity = 1.0 - global_density
    
    print(f"\n[+] SUMMARY:")
    print(f"    Targeted Parameters: {total_targeted_params:,}")
    print(f"    Active Parameters:   {total_active_params:,}")
    print(f"    Dead (Zero) Params:  {total_targeted_params - total_active_params:,}")
    print(f"    ====================================")
    print(f"    TRUE GLOBAL DENSITY:  {global_density:.4%}")
    print(f"    TRUE GLOBAL SPARSITY: {global_sparsity:.4%}")

if __name__ == '__main__':
    check_sparsity()