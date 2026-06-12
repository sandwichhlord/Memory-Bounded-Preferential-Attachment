import os
import random
import csv
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# 0. GOOGLE COLAB DRIVE MOUNT
# ==========================================
try:
    from google.colab import drive
    drive.mount('/content/drive')
    SAVE_DIR = '/content/drive/MyDrive/Thesis_RigL_Run/'
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"[+] Google Drive Mounted! Saving all artifacts to: {SAVE_DIR}")
except ImportError:
    print("[-] Not running in Colab. Saving locally.")
    SAVE_DIR = './'

# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================
DATASET = 'CIFAR10'
EPOCHS = 100
BATCH_SIZE = 64
SPARSITY_DENSITY = 0.1  # 90% Sparsity
PRUNE_RATE = 0.1        # Evolve 10% per epoch
SEED = 42

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(SEED)

HISTORY_FILE = os.path.join(SAVE_DIR, f'rigl_{DATASET.lower()}_history.csv')
DEGREES_FILE = os.path.join(SAVE_DIR, f'rigl_{DATASET.lower()}_degrees.csv')
WEIGHTS_FILE = os.path.join(SAVE_DIR, f'rigl_{DATASET.lower()}_90_sparse.pth')

# Initialize CSV Logger
with open(HISTORY_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train_Loss', 'Val_Accuracy', 'Val_F1'])

# ==========================================
# 3. THE RigL ENGINE (Gradient-Based Growth)
# ==========================================
class RigLScheduler:
    def __init__(self, model, density=0.1, prune_rate=0.1):
        self.model = model
        self.density = density
        self.prune_rate = prune_rate
        self.masks = {}

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    mask = (torch.rand_like(param) < self.density).float()
                    self.masks[name] = mask
                    param.data.mul_(mask)

    def zero_gradients(self):
        pass

    def force_sparsity(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    param.data.mul_(self.masks[name])

    def evolve_topology(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                weight_magnitudes = torch.abs(param.data)

                grad_magnitudes = torch.abs(param.grad) if param.grad is not None else torch.zeros_like(param.data)

                # 1. PRUNE
                active_weights = torch.where(mask == 1, weight_magnitudes, torch.tensor(float('inf')).to(param.device))
                num_active = int(torch.sum(mask).item())
                num_to_prune = int(num_active * self.prune_rate)

                if num_to_prune > 0:
                    kth_value = torch.kthvalue(active_weights.flatten(), num_to_prune).values
                    mask[active_weights <= kth_value] = 0.0

                # 2. GROW
                inactive_grads = torch.where(mask == 0, grad_magnitudes, torch.tensor(float('-inf')).to(param.device))
                if num_to_prune > 0:
                    top_k_grads = torch.topk(inactive_grads.flatten(), num_to_prune)
                    grown_indices = top_k_grads.indices

                    flat_mask = mask.view(-1)
                    flat_mask[grown_indices] = 1.0
                    mask = flat_mask.view(mask.shape)

                self.masks[name] = mask
                param.data.mul_(mask)

# ==========================================
# 4. ARCHITECTURE & DATA LOADERS
# ==========================================
def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    # drop_last=True preserved to protect RigL's gradient sorting
    return DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True), DataLoader(test_data, batch_size=1000, shuffle=False)

def get_resnet():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

# ==========================================
# 5. THE MAIN LOOP
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing RigL on {DATASET} via: {device} | Seed: {SEED}")

    train_loader, test_loader = get_dataloaders()
    model = get_resnet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    pruner = RigLScheduler(model, density=SPARSITY_DENSITY, prune_rate=PRUNE_RATE)
    total_batches = len(train_loader)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()

        print(f"\n--- Epoch {epoch}/{EPOCHS} Started ---")

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()
            pruner.force_sparsity()
            total_loss += loss.item()

            if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == total_batches:
                elapsed_time = time.time() - epoch_start_time
                batches_completed = batch_idx + 1
                time_per_batch = elapsed_time / batches_completed
                batches_remaining = total_batches - batches_completed
                eta_mins, eta_secs = divmod(int(batches_remaining * time_per_batch), 60)
                print(f"   -> Batch {batches_completed}/{total_batches} | Loss: {loss.item():.4f} | ETA: {eta_mins}m {eta_secs}s")

        train_loss = total_loss / total_batches
        print("   -> RigL Gradient Evolution...")
        pruner.evolve_topology()

        # Telemetry Evaluation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                preds = model(data).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')

        print(f"[Epoch {epoch} Summary] Loss: {train_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")

        with open(HISTORY_FILE, mode='a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, acc, f1])

    print("\n[+] Training Complete. Saving artifacts...")
    torch.save(model.state_dict(), WEIGHTS_FILE)
    print(f"[+] Artifact saved to: {WEIGHTS_FILE}")

    print("[+] Extracting Node Degrees for 'fc.weight'...")
    final_mask = pruner.masks['fc.weight'].cpu()
    degrees = final_mask.sum(dim=1).numpy()
    df = pd.DataFrame({'Neuron_ID': range(len(degrees)), 'Degree': degrees})
    df.to_csv(DEGREES_FILE, index=False)
    print(f"[+] Topology exported to: {DEGREES_FILE}")

if __name__ == '__main__':
    main()