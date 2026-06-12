import os
import glob
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import powerlaw
except ImportError:
    print("❌ ERROR: pip install powerlaw")
    exit()

# ==========================================
# 1. SETUP
# ==========================================
INPUT_DIR = "./"  
CSV_SAVE_FILE = "./bellcurve_vs_powerlaw_results.csv"
PLOT_SAVE_DIR = "./bellcurve_visual_fits"
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

MODELS = ["MBPA", "RIGL", "SET"]
SEEDS = [42, 43, 44]
SPARSITIES = [0.80, 0.90, 0.95, 0.99, 0.995]

PALETTE = {"MBPA": "#2a9d8f", "RIGL": "#e76f51", "SET": "#e9c46a"}

# ==========================================
# 2. FILE PARSER
# ==========================================
def extract_global_data(filepath):
    state_dict = torch.load(filepath, map_location='cpu')
    global_degrees = []
    
    for name, tensor in state_dict.items():
        if 'weight' in name and tensor.dim() > 1:
            flattened = tensor.view(tensor.size(0), -1)
            global_degrees.extend((flattened != 0).sum(dim=1).numpy())
            
    return np.array(global_degrees)

def load_all_data():
    print(f"Scanning {INPUT_DIR} for .pth files...")
    file_paths = glob.glob(os.path.join(INPUT_DIR, "*.pth"))
    
    all_data = []
    for filepath in file_paths:
        filename = os.path.basename(filepath)
        match = re.search(r'(MBPA|RIGL|RigL|SET)_([0-9.]+)%?_Seed([0-9]+)_sparse\.pth', filename, re.IGNORECASE)
        if not match: continue
            
        try:
            degrees = extract_global_data(filepath)
            degrees = degrees[degrees > 0]
            if len(degrees) >= 10:
                all_data.append({
                    'Sparsity': float(match.group(2)) / 100.0, 
                    'Model': match.group(1).upper(), 
                    'Seed': int(match.group(3)), 
                    'Degrees': degrees
                })
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            
    return pd.DataFrame(all_data)

# ==========================================
# 3. GRAPHICAL GRID GENERATOR
# ==========================================
def generate_3x3_plots(df, sparsity):
    print(f"Generating Bell Curve vs Truncated PL Grid for {sparsity*100}% Sparsity...")
    df_sp = df[df['Sparsity'] == sparsity]
    if df_sp.empty: return

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=False, sharey=False)
    fig.suptitle(f"Truncated Power Law vs. Lognormal (Bell Curve) at {sparsity*100}% Sparsity", 
                 fontsize=20, fontweight='bold', y=0.97)

    csv_records = []

    for i, model in enumerate(MODELS):
        for j, seed in enumerate(SEEDS):
            ax = axes[i, j]
            df_cell = df_sp[(df_sp['Model'] == model) & (df_sp['Seed'] == seed)]
            
            if df_cell.empty:
                ax.text(0.5, 0.5, "Missing File", ha='center', va='center')
                continue
                
            degrees = df_cell.iloc[0]['Degrees']
            color = PALETTE[model]

            try:
                # 1. Build the MLE Models
                fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
                
                # 2. Extract KS Distances (D)
                D_trunc = fit.truncated_power_law.D
                D_lognormal = fit.lognormal.D
                
                # 3. Direct LLR Competition (Positive = Truncated PL wins, Negative = Bell Curve wins)
                R, p_val = fit.distribution_compare('truncated_power_law', 'lognormal', normalized_ratio=True)
                
                winner = "Truncated Power Law" if R > 0 else "Lognormal (Bell Curve)"
                
                # --- PLOT 1: Empirical CCDF ---
                fit.plot_ccdf(ax=ax, color=color, linewidth=2.5, label="Empirical Data", zorder=2)
                
                # --- PLOT 2: Theoretical Truncated PL Curve ---
                fit.truncated_power_law.plot_ccdf(ax=ax, color='#222222', linestyle='-', 
                                                  linewidth=1.8, label="Truncated PL", zorder=3)
                
                # --- PLOT 3: Theoretical Lognormal (Bell Curve) ---
                fit.lognormal.plot_ccdf(ax=ax, color='#e63946', linestyle=':', 
                                        linewidth=2.5, label="Bell Curve", zorder=4)
                
                # Text box styling
                text_str = (f"LLR = {R:.2f} (p={p_val:.2f})\n"
                            f"$D_{{Trunc}}$ = {D_trunc:.4f}\n"
                            f"$D_{{Bell}}$ = {D_lognormal:.4f}\n"
                            f"Winner: {winner}")
                
                box_color = '#eaffea' if winner == "Truncated Power Law" else '#ffeaea'
                
                ax.text(0.05, 0.05, text_str, transform=ax.transAxes, fontsize=11, 
                        fontweight='bold', color='#222222', va='bottom', ha='left',
                        bbox=dict(facecolor=box_color, alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.4'))
                
                csv_records.append({
                    'Model': model, 'Sparsity': sparsity, 'Seed': seed,
                    'D (Trunc)': D_trunc, 'D (Bell Curve)': D_lognormal,
                    'LLR': R, 'p-val': p_val, 'Winner': winner
                })

            except Exception as e:
                print(f"  [!] Failed fit for {model} Seed {seed}: {e}")

            ax.grid(True, which="both", ls="--", alpha=0.2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            title = f"Seed {seed}\n" if i == 0 else ""
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            if j == 0: ax.set_ylabel(f"{model}\n$P(X \\geq k)$", fontsize=15, fontweight='bold', labelpad=15)
            if i == 2: ax.set_xlabel("Degree $k$", fontsize=13, fontweight='bold')
            if i == 0 and j == 0: ax.legend(loc='upper right', fontsize=10, frameon=True)

    plt.subplots_adjust(wspace=0.22, hspace=0.28)
    filename = os.path.join(PLOT_SAVE_DIR, f"BellCurve_vs_TruncPL_Sparsity_{sparsity*100:.1f}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='w')
    plt.close()
    
    return csv_records

# ==========================================
# 4. EXECUTE PIPELINE
# ==========================================
df = load_all_data()

if not df.empty:
    global_csv_data = []
    for sp in sorted(df['Sparsity'].unique()):
        records = generate_3x3_plots(df, sp)
        global_csv_data.extend(records)
        
    summary_df = pd.DataFrame(global_csv_data)
    summary_df.to_csv(CSV_SAVE_FILE, index=False)
    
    print(f"\n✅ All grids saved to: {PLOT_SAVE_DIR}")
    print(f"✅ Summary CSV saved to: {CSV_SAVE_FILE}")
else:
    print("❌ Process halted: No files loaded.")