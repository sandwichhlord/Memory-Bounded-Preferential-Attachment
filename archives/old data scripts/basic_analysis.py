import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models

# ==========================================
# 1. CONFIGURATION
# ==========================================
WEIGHTS_FILE = 'ultimate_rigl_cifar10_sparse.pth' # Change to your .pth file

def get_resnet_arch():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def plot_basic_distribution(weights_path):
    print(f"Loading {weights_path}...")
    model = get_resnet_arch()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    
    plt.figure(figsize=(10, 6))
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            
            # Skip the tiny first layer
            if isinstance(module, nn.Conv2d) and module.in_channels <= 3: 
                continue
            
            w = module.weight.data
            
            # Count how many non-zero connections each neuron has
            if isinstance(module, nn.Conv2d):
                degrees = (w != 0).sum(dim=(1, 2, 3)).numpy()
            else:
                degrees = (w != 0).sum(dim=1).numpy()
            
            # Plot a standard step-histogram. 
            # bins='auto' lets the computer do the math.
            # histtype='step' draws clean outlines instead of solid blocks.
            plt.hist(degrees, bins='auto', histtype='step', linewidth=2, label=name)

    plt.title(f"Raw Connection Distribution: {weights_path}", fontsize=14)
    plt.xlabel("Number of Connections (In-Degree)")
    plt.ylabel("Number of Neurons")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show
    output_name = f"{weights_path.split('.')[0]}_basic_graph.png"
    plt.savefig(output_name, dpi=300)
    print(f"Saved raw graph to {output_name}")
    plt.show()

if __name__ == "__main__":
    plot_basic_distribution(WEIGHTS_FILE)