import os
import random
import csv
import time  # <--- Added for ETA calculation
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# 1. EXPERIMENT CONFIGURATION (Plug & Play)
# ==========================================
DATASET = 'CIFAR10'       # Options: 'MNIST' or 'CIFAR10'
EPOCHS = 100              # Use 3 for Smoke Test, 100 for Overnight Run
BATCH_SIZE = 64
SPARSITY_DENSITY = 0.1  # 90% Unstructured Sparsity
PRUNE_RATE = 0.1        # Evolve 10% of active weights per epoch
SEED = 42

# ==========================================
# 2. DETERMINISM & TELEMETRY SETUP
# ==========================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(SEED)

HISTORY_FILE = f'set_{DATASET.lower()}_history.csv'
DEGREES_FILE = f'set_{DATASET.lower()}_degrees.csv'
WEIGHTS_FILE = f'set_{DATASET.lower()}_90_sparse.pth'

# Initialize CSV Logger
with open(HISTORY_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train_Loss', 'Val_Accuracy', 'Val_F1'])

# ==========================================
# 3. THE SET ENGINE
# ==========================================
class SETScheduler:
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
        for name, param in self.model.named_parameters():
            if name in self.masks and param.grad is not None:
                param.grad.data.mul_(self.masks[name])

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
                
                active_weights = torch.where(mask == 1, weight_magnitudes, torch.tensor(float('inf')).to(param.device))
                num_active = int(torch.sum(mask).item())
                num_to_prune = int(num_active * self.prune_rate)
                
                if num_to_prune > 0:
                    kth_value = torch.kthvalue(active_weights.flatten(), num_to_prune).values
                    mask[active_weights <= kth_value] = 0.0
                
                dead_indices = torch.nonzero(mask == 0)
                if len(dead_indices) > 0 and num_to_prune > 0:
                    random_selections = torch.randperm(len(dead_indices))[:num_to_prune]
                    grown_indices = dead_indices[random_selections]
                    for idx in grown_indices: mask[tuple(idx)] = 1.0
                
                self.masks[name] = mask
                param.data.mul_(mask)

# ==========================================
# 4. ARCHITECTURE & DATA LOADERS
# ==========================================
def get_dataloaders():
    if DATASET == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    else: # CIFAR10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        
    return DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test_data, batch_size=1000, shuffle=False)

def get_resnet():
    model = models.resnet18()
    if DATASET == 'MNIST':
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

# ==========================================
# 5. THE MAIN LOOP
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing {DATASET} on: {device} | Seed: {SEED}")

    train_loader, test_loader = get_dataloaders()
    model = get_resnet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"Initializing SET Engine (Density: {SPARSITY_DENSITY})...")
    pruner = SETScheduler(model, density=SPARSITY_DENSITY, prune_rate=PRUNE_RATE)
    
    total_batches = len(train_loader)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        epoch_start_time = time.time() # <--- Start the clock for this epoch
        
        print(f"\n--- Epoch {epoch}/{EPOCHS} Started ---")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            pruner.zero_gradients() 
            optimizer.step()
            pruner.force_sparsity()
            total_loss += loss.item()
            
            # ---> THE HEARTBEAT & ETA TRACKER <---
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
                elapsed_time = time.time() - epoch_start_time
                batches_completed = batch_idx + 1
                time_per_batch = elapsed_time / batches_completed
                batches_remaining = total_batches - batches_completed
                
                eta_seconds = int(batches_remaining * time_per_batch)
                eta_mins, eta_secs = divmod(eta_seconds, 60)
                
                print(f"   -> Batch {batches_completed}/{total_batches} | Loss: {loss.item():.4f} | ETA: {eta_mins}m {eta_secs}s")
            
        train_loss = total_loss / total_batches
        print("   -> Evolving Topology...")
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
        
        # Log to CSV
        with open(HISTORY_FILE, mode='a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, acc, f1])

    # ==========================================
    # 6. ARTIFACT SAVING & TOPOLOGY EXTRACTION
    # ==========================================
    print("\n[+] Training Complete. Saving weights...")
    torch.save(model.state_dict(), WEIGHTS_FILE)
    print(f"[+] Artifact saved to: {WEIGHTS_FILE}")

    print("[+] Extracting Node Degrees for 'fc.weight'...")
    final_mask = pruner.masks['fc.weight'].cpu()
    degrees = final_mask.sum(dim=1).numpy()
    
    # Save Topology to CSV for Jupyter Analysis
    df = pd.DataFrame({'Neuron_ID': range(len(degrees)), 'Degree': degrees})
    df.to_csv(DEGREES_FILE, index=False)
    print(f"[+] Topology exported to: {DEGREES_FILE}")
    print(f"God-Hub (Max): {degrees.max()} | Loner (Min): {degrees.min()}")

if __name__ == '__main__':
    main()