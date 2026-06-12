import os
import random
import csv
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# 1. SMOKE TEST CONFIGURATION 
# ==========================================
DATASET = 'MNIST'       
EPOCHS = 2              # Just 2 epochs to verify the engine runs
BATCH_SIZE = 64
SPARSITY_DENSITY = 0.1  # 90% Sparsity
PRUNE_RATE = 0.1        
WARMUP_EPOCHS = 1       # Epoch 1: Random Growth. Epoch 2: MBPA Growth.
SEED = 42

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(SEED)

SAVE_DIR = './'
HISTORY_FILE = os.path.join(SAVE_DIR, f'mbpa_{DATASET.lower()}_history.csv')

# Initialize CSV Logger Header
with open(HISTORY_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train_Loss', 'Val_Accuracy', 'Val_F1'])

# ==========================================
# 2. THE MBPA ENGINE
# ==========================================
class MBPAScheduler:
    def __init__(self, model, density=0.1, prune_rate=0.1, warmup_epochs=15, alpha=1.5, kappa=4.0):
        self.model = model
        self.density = density       
        self.prune_rate = prune_rate 
        
        self.warmup_epochs = warmup_epochs
        self.alpha = alpha
        self.kappa = kappa
        
        self.masks = {}
        self.current_epoch = 1 
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    mask = (torch.rand_like(param) < self.density).float()
                    self.masks[name] = mask
                    param.data.mul_(mask) 

    def step_epoch(self):
        self.current_epoch += 1

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
                
                # 1. PRUNE
                active_weights = torch.where(mask == 1, weight_magnitudes, torch.tensor(float('inf')).to(param.device))
                num_active = int(torch.sum(mask).item())
                num_to_prune = int(num_active * self.prune_rate)
                
                if num_to_prune > 0:
                    kth_value = torch.kthvalue(active_weights.flatten(), num_to_prune).values
                    mask[active_weights <= kth_value] = 0.0
                
                # 2. GROW
                if num_to_prune > 0:
                    if self.current_epoch <= self.warmup_epochs:
                        # WARM-UP: Random Growth
                        dead_indices = torch.nonzero(mask.flatten() == 0).view(-1)
                        grown_indices = dead_indices[torch.randperm(len(dead_indices))[:num_to_prune]]
                    else:
                        # MBPA: Preferential Attachment bounded by C_max
                        if mask.dim() == 4:
                            out_c, in_c, k_h, k_w = mask.shape
                            fan_in = in_c * k_h * k_w
                            degrees = mask.sum(dim=(1, 2, 3)) 
                        else:
                            out_c, in_c = mask.shape
                            fan_in = in_c
                            degrees = mask.sum(dim=1)

                        e_k = self.density * fan_in
                        c_max = min(fan_in, max(self.alpha * e_k, self.kappa * math.sqrt(fan_in)))

                        scores = degrees.clone()
                        scores[scores >= c_max] = 0.0 
                        scores[scores == 0] = 1e-5    

                        if mask.dim() == 4:
                            prob_matrix = scores.view(-1, 1, 1, 1).expand_as(mask).clone()
                        else:
                            prob_matrix = scores.view(-1, 1).expand_as(mask).clone()

                        prob_matrix[mask == 1] = 0.0
                        
                        flat_probs = prob_matrix.flatten()
                        prob_sum = flat_probs.sum()

                        if prob_sum > 0:
                            flat_probs /= prob_sum 
                            grown_indices = torch.multinomial(flat_probs, num_to_prune, replacement=False)
                        else:
                            dead_indices = torch.nonzero(mask.flatten() == 0).view(-1)
                            grown_indices = dead_indices[torch.randperm(len(dead_indices))[:num_to_prune]]

                    flat_mask = mask.view(-1)
                    flat_mask[grown_indices] = 1.0
                    mask = flat_mask.view(mask.shape)
                
                self.masks[name] = mask
                param.data.mul_(mask)

# ==========================================
# 3. ARCHITECTURE & DATA LOADERS
# ==========================================
def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST Normalization
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True), DataLoader(test_data, batch_size=1000, shuffle=False)

def get_resnet():
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # 1-Channel for MNIST
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

# ==========================================
# 4. THE MAIN LOOP
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing MBPA on {DATASET} via: {device}")

    train_loader, test_loader = get_dataloaders()
    model = get_resnet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    pruner = MBPAScheduler(model, density=SPARSITY_DENSITY, prune_rate=PRUNE_RATE, warmup_epochs=WARMUP_EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        print(f"\n--- Epoch {epoch}/{EPOCHS} Started (Phase: {'Warmup' if epoch <= WARMUP_EPOCHS else 'MBPA'}) ---")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            optimizer.step()
            pruner.force_sparsity()
            total_loss += loss.item()
            
        train_loss = total_loss / len(train_loader)
        
        print(f"   -> MBPA Topology Evolution...")
        pruner.evolve_topology()
        pruner.step_epoch() 

        # Telemetry
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
        
        # FIX: The CSV Append Block
        with open(HISTORY_FILE, mode='a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, acc, f1])

    print("\n[+] MNIST Smoke Test Complete. Engine is structurally sound and data is logged.")

if __name__ == '__main__':
    main()