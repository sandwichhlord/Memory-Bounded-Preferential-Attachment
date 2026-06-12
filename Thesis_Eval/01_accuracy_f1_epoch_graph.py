import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_DIR = "./"  
PLOT_SAVE_DIR = "./history_visuals"
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# Exact colors from your topology graphs to keep your paper perfectly consistent
PALETTE = {"MBPA": "#2a9d8f", "RIGL": "#e76f51", "SET": "#e9c46a"}

# ==========================================
# 2. LOAD & AGGREGATE DATA
# ==========================================
print("Scanning for history CSVs...")
file_paths = glob.glob(os.path.join(INPUT_DIR, "*_history*.csv"))

all_data = []
for filepath in file_paths:
    filename = os.path.basename(filepath)
    
    # Extract Model, Sparsity, and Seed
    match = re.search(r'(MBPA|RIGL|RigL|SET)_([0-9.]+)%?_Seed([0-9]+)_history', filename, re.IGNORECASE)
    if not match:
        continue
        
    model = match.group(1).upper()
    sparsity = float(match.group(2))
    seed = int(match.group(3))
    
    try:
        df = pd.read_csv(filepath)
        df['Model'] = model
        df['Sparsity'] = sparsity
        df['Seed'] = seed
        all_data.append(df)
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")

if not all_data:
    print("❌ No valid history CSV files found. Check your file names and directory!")
    exit()

master_df = pd.concat(all_data, ignore_index=True)

# ==========================================
# 3. GENERATE SPARSITY PLOTS
# ==========================================
sns.set_theme(style="whitegrid")
unique_sparsities = sorted(master_df['Sparsity'].unique())

for sparsity in unique_sparsities:
    print(f"Generating Training Curve for {sparsity}% Sparsity...")
    
    df_sp = master_df[master_df['Sparsity'] == sparsity]
    
    # Create a 1x2 figure (Accuracy Left, F1 Right)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Model Training Dynamics across Epochs ({sparsity}% Sparsity)", fontsize=16, fontweight='bold', y=1.05)
    
    # --- PLOT 1: Validation Accuracy ---
    # sns.lineplot automatically averages the 3 seeds and draws the shaded confidence interval
    sns.lineplot(data=df_sp, x='Epoch', y='Val_Accuracy', hue='Model', 
                 palette=PALETTE, linewidth=2.5, ax=axes[0])
    axes[0].set_title("Validation Accuracy", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Accuracy", fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 1.0) # Standardize axis between 0 and 1
    axes[0].legend(loc='lower right', title="Model")

    # --- PLOT 2: Validation F1 Score ---
    sns.lineplot(data=df_sp, x='Epoch', y='Val_F1', hue='Model', 
                 palette=PALETTE, linewidth=2.5, ax=axes[1])
    axes[1].set_title("Validation F1 Score", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("F1 Score", fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 1.0) # Standardize axis between 0 and 1
    axes[1].legend(loc='lower right', title="Model")

    # Polish and Save
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.tight_layout()
    filename = os.path.join(PLOT_SAVE_DIR, f"Training_Dynamics_{sparsity}_Sparsity.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

print(f"\n✅ All training dynamics plots saved to: {PLOT_SAVE_DIR}")