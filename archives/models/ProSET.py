import os
import random
import csv
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
# 0. GOOGLE COLAB / LOCAL TOGGLE
# ==========================================
IS_COLAB = 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False

if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    SAVE_DIR = '/content/drive/MyDrive/Thesis_ProSET_Run/'
    os.makedirs(SAVE_DIR, exist_ok=True)
else:
    SAVE_DIR = './'

# ==========================================
# 1. EXPERIMENT CONFIGURATION 
# ==========================================
# TOGGLE THESE FOR SMOKE TEST VS FINAL RUN
DATASET = 'CIFAR10'       # Change to 'CIFAR10' for final run
EPOCHS = 100              # Change to 100 for final run
WARMUP_EPOCHS = 0       # SET usually evolves from Epoch 1
T_END = 80              # When to stop evolving topology (Cosine Decay end)

BATCH_SIZE = 64
TOTAL_SPARSITY = 0.9    # 90% Global Sparsity
PRUNE_RATE_INIT = 0.3   # Start high for exploration
SEED = 42

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(SEED)

HISTORY_FILE = os.path.join(SAVE_DIR, f'pro_set_{DATASET.lower()}_history.csv')
WEIGHTS_FILE = os.path.join(SAVE_DIR, f'pro_set_{DATASET.lower()}_sparse.pth')

# Initialize CSV Logger
with open(HISTORY_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train_Loss', 'Val_Accuracy', 'Val_F1', 'Prune_Rate'])

# ==========================================
# 2. THE PRO SET ENGINE (ERK + Cosine + Zero-Init)
# ==========================================
class ProSETScheduler:
    def __init__(self, model, total_sparsity=0.9, prune_rate_init=0.3, T_end=80):
        self.model = model
        self.total_sparsity = total_sparsity
        self.prune_rate_init = prune_rate_init
        self.T_end = T_end
        self.masks = {}
        self.current_epoch = 1
        
        # 1. ERK Initialization logic
        self._init_erk_masks()

    def _init_erk_masks(self):
        """Calculates layer-wise density using Erdős-Rényi Kernel logic"""
        print(f"--- Initializing ERK Topology (Global Target: {self.total_sparsity:.0%}) ---")
        total_params = 0
        erk_weights = []
        layers = []

        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                n_out, n_in = param.shape[0], param.shape[1]
                k = param.shape[2] if param.dim() == 4 else 1
                # ERK formula: density proportional to (n_in + n_out + k^2) / (n_in * n_out * k^2)
                erk_weight = (n_in + n_out + k**2) / (n_in * n_out * k**2)
                
                total_params += param.numel()
                erk_weights.append(erk_weight)
                layers.append((name, param))

        # Binary search for the epsilon coefficient that hits our target sparsity
        eps = 0.5 
        for _ in range(20): 
            current_total_nonzero = 0
            for i, (name, param) in enumerate(layers):
                d_l = min(1.0, max(0.01, eps * erk_weights[i]))
                current_total_nonzero += int(d_l * param.numel())
            target_nonzero = total_params * (1 - self.total_sparsity)
            eps *= (target_nonzero / current_total_nonzero)

        # Apply the masks
        with torch.no_grad():
            for i, (name, param) in enumerate(layers):
                d_l = min(1.0, max(0.01, eps * erk_weights[i]))
                mask = (torch.rand_like(param) < d_l).float()
                self.masks[name] = mask.to(param.device)
                param.data.mul_(self.masks[name])
                print(f"   [Layer] {name:30} | Sparsity: {1-d_l:6.2%}")

    def get_prune_rate(self):
        """Cosine Annealing for the pruning rate (Mocanu 2018)"""
        if self.current_epoch >= self.T_end:
            return 0.0
        return self.prune_rate_init * 0.5 * (1 + math.cos(self.current_epoch * math.pi / self.T_end))

    def evolve_topology(self):
        p_rate = self.get_prune_rate()
        if p_rate <= 0: return 

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in self.masks or 'bias' in name: continue
                
                mask = self.masks[name]
                num_active = int(torch.sum(mask).item())
                num_to_prune = int(num_active * p_rate)
                
                if num_to_prune > 0:
                    # 1. PRUNE (Magnitude-based)
                    weights = torch.abs(param.data)
                    active_weights = torch.where(mask == 1, weights, torch.tensor(float('inf')).to(param.device))
                    kth_value = torch.kthvalue(active_weights.flatten(), num_to_prune).values
                    mask[active_weights <= kth_value] = 0.0
                    
                    # 2. GROW (Uniform Random)
                    dead_indices = torch.nonzero(mask.flatten() == 0).view(-1)
                    grown_indices = dead_indices[torch.randperm(len(dead_indices))[:num_to_prune]]
                    
                    # 3. APPLY & ZERO-INIT
                    flat_mask = mask.view(-1)
                    flat_mask[grown_indices] = 1.0
                    self.masks[name] = flat_mask.view(mask.shape)
                    
                    # Prevent weight-shock: Reset new connections to 0
                    param.data.view(-1)[grown_indices] = 0.0
                
                param.data.mul_(self.masks[name])

    def step_epoch(self):
        self.current_epoch += 1

    def force_sparsity(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    param.data.mul_(self.masks[name])

# ==========================================
# 3. ARCHITECTURE & DATA LOADERS
# ==========================================
def get_dataloaders():
    if DATASET == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    else: # CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True), \
           DataLoader(test_ds, batch_size=1000, shuffle=False)

def get_resnet():
    model = models.resnet18()
    if DATASET == 'MNIST':
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Pro SET Smoke Test: {DATASET} on {device} ---")

    train_loader, test_loader = get_dataloaders()
    model = get_resnet().to(device)
    
    # Standard Optimizer with Momentum
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    pruner = ProSETScheduler(model, total_sparsity=TOTAL_SPARSITY, 
                            prune_rate_init=PRUNE_RATE_INIT, T_end=T_END)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        p_rate = pruner.get_prune_rate()
        print(f"\n[Epoch {epoch}/{EPOCHS}] Current Prune Rate: {p_rate:.4f}")

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            pruner.force_sparsity()
            total_loss += loss.item()

        # Evolution Phase
        pruner.evolve_topology()
        pruner.step_epoch()

        # Evaluation
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
        print(f"   -> Accuracy: {acc:.4f} | F1: {f1:.4f}")

        with open(HISTORY_FILE, mode='a', newline='') as f:
            csv.writer(f).writerow([epoch, total_loss/len(train_loader), acc, f1, p_rate])

    torch.save(model.state_dict(), WEIGHTS_FILE)
    print(f"\n[+] Pro SET Smoke Test Complete. File saved to: {HISTORY_FILE}")

if __name__ == '__main__':
    main()