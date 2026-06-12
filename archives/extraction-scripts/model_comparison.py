import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_final_comparison():
    plt.figure(figsize=(14, 8))
    
    files = {
        'SET (Random)': 'set_cifar10_history.csv',
        'RigL (Gradients)': 'rigl_cifar10_history.csv',
        'MBPA (Biomimetic)': 'mbpa_cifar10_history.csv'
    }
    
    colors = {'SET (Random)': 'royalblue', 'RigL (Gradients)': 'crimson', 'MBPA (Biomimetic)': 'forestgreen'}
    styles = {'SET (Random)': '--', 'RigL (Gradients)': '-.', 'MBPA (Biomimetic)': '-'}

    for label, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            plt.plot(df['Epoch'], df['Val_Accuracy'], label=label, color=colors[label], linestyle=styles[label], linewidth=2.5)
            
            # Print the final accuracy for quick reference
            final_acc = df['Val_Accuracy'].iloc[-1]
            print(f"Final Val Acc [{label}]: {final_acc:.4f}")

    plt.title('Final Thesis Comparison: Convergence Efficiency @ 90% Sparsity', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Validation Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.ylim(0.5, 0.9) # Zoom into the relevant performance range
    plt.tight_layout()
    plt.savefig('thesis_convergence_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_final_comparison()