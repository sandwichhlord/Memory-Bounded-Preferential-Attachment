import os
import torch
import torch.nn as nn
import pandas as pd
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# 1. CONFIGURATION & HARDWARE DETECTION
# ==========================================
SAVE_DIR = './' 
DATASET_DIR = './data'
BATCH_SIZE = 1000 

MODES = ['set', 'mbpa', 'rigl']
SPARSITIES = [0.90, 0.95, 0.98, 0.99, 0.995]
SEEDS = [42, 43, 44]

print("\n🔍 Detecting Local Hardware...")
if torch.cuda.is_available():
    device = torch.device("cuda")
    USE_AMP = True
    print("🖥️ Hardware: NVIDIA GPU (CUDA Enabled) -> AMP Activated")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
    USE_AMP = False
    print("🖥️ Hardware: Apple Silicon (MPS Enabled) -> Standard Precision")
else:
    device = torch.device("cpu")
    USE_AMP = False
    print("🖥️ Hardware: Standard CPU -> Standard Precision")

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
# 3. EVALUATION ENGINE
# ==========================================
def evaluate_model_on_train_set(model, weights_path, loader):
    # Load the sparse weights safely
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            
            # Dynamically use AMP only if on Nvidia GPU
            if USE_AMP:
                with torch.amp.autocast('cuda'):
                    output = model(data)
            else:
                output = model(data)
                
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # THE HEARTBEAT: Prints a dot for every batch processed
            print(".", end="", flush=True)
            
    print() # Prints a clean new line when the model finishes all 50 batches
    
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    return acc, f1

# ==========================================
# 4. SWEEP AND EXPORT (WINDOWS SHIELDED)
# ==========================================
if __name__ == '__main__':
    # THE 180-IQ FIX: No RandomCrop or HorizontalFlip. Pure dataset evaluation.
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    print("\n📦 Loading Pure CIFAR-100 Training Data (No Augmentations)...")
    train_ds = datasets.CIFAR100(DATASET_DIR, train=True, download=True, transform=eval_transform)
    
    # DataLoader is now safely inside the main thread to prevent Windows spawn crashes
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = get_resnet().to(device)
    results = []

    print("\n🚀 Beginning Generalization Gap Sweep across all models...\n")

    for mode in MODES:
        for sparsity in SPARSITIES:
            for seed in SEEDS:
                base_name = f"{mode.upper()}_{sparsity*100}%_Seed{seed}"
                weights_file = os.path.join(SAVE_DIR, f"{base_name}_sparse.pth")
                hist_file = os.path.join(SAVE_DIR, f"{base_name}_history.csv")
                
                if os.path.exists(weights_file) and os.path.exists(hist_file):
                    # Added 'end=" "' so the heartbeat dots print on the same line
                    print(f"🔄 Evaluating {base_name}...", end=" ")
                    
                    # 1. Calculate True Train Metrics (Pass the loader in)
                    train_acc, train_f1 = evaluate_model_on_train_set(model, weights_file, train_loader)
                    
                    # 2. Extract Final Test Metrics from History
                    hist_df = pd.read_csv(hist_file)
                    test_acc = hist_df.iloc[-1]['Val_Accuracy']
                    test_f1 = hist_df.iloc[-1]['Val_F1']
                    
                    # 3. Calculate The Gap
                    acc_gap = train_acc - test_acc
                    f1_gap = train_f1 - test_f1
                    
                    results.append({
                        'Mode': mode.upper(),
                        'Sparsity': f"{sparsity*100}%",
                        'Seed': seed,
                        'Train_Acc': train_acc,
                        'Test_Acc': test_acc,
                        'Acc_Gap': acc_gap,
                        'Train_F1': train_f1,
                        'Test_F1': test_f1,
                        'F1_Gap': f1_gap
                    })
                else:
                    print(f"⚠️ [SKIPPING] Missing files for {base_name}")

    # ==========================================
    # 5. AGGREGATE AND EXPORT
    # ==========================================
    results_df = pd.DataFrame(results)

    if not results_df.empty:
        print("\n🧮 Calculating Statistical Means across Seeds...")
        
        summary_df = results_df.groupby(['Mode', 'Sparsity']).agg(
            Train_Acc_Mean=('Train_Acc', 'mean'),
            Test_Acc_Mean=('Test_Acc', 'mean'),
            Acc_Gap_Mean=('Acc_Gap', 'mean'),
            F1_Gap_Mean=('F1_Gap', 'mean')
        ).reset_index()
        
        # Sort logically
        summary_df['Sparsity_Float'] = summary_df['Sparsity'].str.replace('%', '').astype(float)
        summary_df = summary_df.sort_values(by=['Sparsity_Float', 'Mode']).drop(columns=['Sparsity_Float'])
        
        # Format for the thesis
        summary_df['Train_Acc_Formatted'] = (summary_df['Train_Acc_Mean']*100).round(2).astype(str) + '%'
        summary_df['Test_Acc_Formatted'] = (summary_df['Test_Acc_Mean']*100).round(2).astype(str) + '%'
        summary_df['Acc_Gap_Formatted'] = (summary_df['Acc_Gap_Mean']*100).round(2).astype(str) + '%'
        
        output_file = os.path.join(SAVE_DIR, 'Phase1_Generalization_Gap_Table.csv')
        summary_df.to_csv(output_file, index=False)
        
        print("\n" + "="*60)
        print("🏆 GENERALIZATION GAP ANALYSIS COMPLETE")
        print("="*60)
        print(summary_df[['Mode', 'Sparsity', 'Train_Acc_Formatted', 'Test_Acc_Formatted', 'Acc_Gap_Formatted']])
        print(f"\n📁 Saved Academic Table to: {output_file}")
    else:
        print("❌ ERROR: No completed runs found to evaluate. Check your folder.")