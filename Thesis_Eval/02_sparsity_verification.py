import os
import torch
import torch.nn as nn
from torchvision import models
import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================
SAVE_DIR = './' 

MODES = ['set', 'mbpa', 'rigl']
SPARSITIES = [0.90, 0.95, 0.98, 0.99, 0.995]
SEEDS = [42, 43, 44]

# Force CPU since we are just counting elements, no heavy math required
device = torch.device("cpu")
print("\n🔍 Hardware: Forcing CPU for safe, rapid memory inspection...")

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
# 3. VERIFICATION ENGINE
# ==========================================
def calculate_model_sparsity(model, weights_path):
    # Load weights into the model
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    
    total_params = 0
    total_zeros = 0
    
    # We only check Conv2d and Linear layers. 
    # Biases and BatchNorms are kept dense in standard sparse training.
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            
            # Count elements
            total_params += weight.numel()
            total_zeros += torch.sum(weight == 0).item()
            
    # Calculate actual percentage
    if total_params == 0: return 0
    actual_sparsity = (total_zeros / total_params)
    return actual_sparsity

# ==========================================
# 4. SWEEP AND LOG
# ==========================================
if __name__ == '__main__':
    model = get_resnet().to(device)
    results = []

    print("\n🔬 Beginning Deep-Scan Sparsity Verification...\n")

    for mode in MODES:
        for sparsity in SPARSITIES:
            for seed in SEEDS:
                base_name = f"{mode.upper()}_{sparsity*100}%_Seed{seed}"
                weights_file = os.path.join(SAVE_DIR, f"{base_name}_sparse.pth")
                
                if os.path.exists(weights_file):
                    actual_sparsity = calculate_model_sparsity(model, weights_file)
                    target_sparsity = sparsity
                    
                    # Calculate how far off the algorithm drifted
                    drift = abs(target_sparsity - actual_sparsity) * 100
                    
                    # Visual Warning for console
                    status = "✅ PERFECT" if drift < 0.1 else f"⚠️ DRIFT: {drift:.3f}%"
                    print(f"[{base_name}] Target: {target_sparsity*100:.1f}% | Actual: {actual_sparsity*100:.3f}% -> {status}")
                    
                    results.append({
                        'Mode': mode.upper(),
                        'Target_Sparsity': f"{sparsity*100}%",
                        'Seed': seed,
                        'Actual_Sparsity_Percent': round(actual_sparsity * 100, 3),
                        'Drift_Percent': round(drift, 4)
                    })
                else:
                    print(f"⚠️ [SKIPPING] Missing weights file: {weights_file}")

    # ==========================================
    # 5. AGGREGATE AND EXPORT
    # ==========================================
    results_df = pd.DataFrame(results)

    if not results_df.empty:
        # Group by Mode and Sparsity to check the average drift across seeds
        summary_df = results_df.groupby(['Mode', 'Target_Sparsity']).agg(
            Actual_Mean=('Actual_Sparsity_Percent', 'mean'),
            Drift_Max=('Drift_Percent', 'max')
        ).reset_index()
        
        # Sort logically
        summary_df['Sparsity_Float'] = summary_df['Target_Sparsity'].str.replace('%', '').astype(float)
        summary_df = summary_df.sort_values(by=['Sparsity_Float', 'Mode']).drop(columns=['Sparsity_Float'])
        
        output_file = os.path.join(SAVE_DIR, 'Phase2_Sparsity_Verification.csv')
        summary_df.to_csv(output_file, index=False)
        
        print("\n" + "="*60)
        print("🏆 SPARSITY VERIFICATION COMPLETE")
        print("="*60)
        print(summary_df)
        print(f"\n📁 Saved Audit Table to: {output_file}")