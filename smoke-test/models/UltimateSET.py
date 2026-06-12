import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
import csv
import time
import numpy as np
import pandas as pd
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# 0. KAGGLE / LOCAL ENVIRONMENT SETUP
# ==========================================
# Kaggle sets specific environment variables. We use this to detect it.
IS_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') != ''

if IS_KAGGLE:
    # /kaggle/working/ is the ONLY place Kaggle lets you save files 
    # that you can download after the run finishes.
    SAVE_DIR = '/kaggle/working/'
    print("[+] Kaggle Environment Detected. Saving to /kaggle/working/")
else:
    SAVE_DIR = './'
    print("[+] Local Environment Detected.")

# ==========================================
# 1. EXPERIMENT CONFIGURATION 
# ==========================================
DATASET = 'CIFAR10'       
EPOCHS = 100              
BATCH_SIZE = 64
SPARSITY = 0.9          # 90% Global Target
ALPHA = 0.3             # Initial growth fraction
LR_STEPS = [30, 70, 90] # Drop LR by 10x at these epochs
SEED = 42

def seed_everything(seed):
    random_seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(SEED)

HISTORY_FILE = os.path.join(SAVE_DIR, f'ultimate_set_{DATASET.lower()}_history.csv')
DEGREES_FILE = os.path.join(SAVE_DIR, f'ultimate_set_{DATASET.lower()}_degrees.csv')
WEIGHTS_FILE = os.path.join(SAVE_DIR, f'ultimate_set_{DATASET.lower()}_sparse.pth')

# Initialize CSV Logger
with open(HISTORY_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train_Loss', 'Val_Accuracy', 'Val_F1', 'Growth_Fraction', 'Epoch_Time_sec'])

# ==========================================
# 2. THE ULTIMATE SET SCHEDULER (Epoch-Based)
# ==========================================
class UltimateSETScheduler:
    def __init__(self, model, optimizer, sparsity=0.9, alpha=0.3, T_end=80, device='cuda'):
        self.model = model
        self.optimizer = optimizer 
        self.sparsity = sparsity
        self.alpha = alpha
        self.device = device
        self.masks = {}
        self.epochs_completed = 0
        self.T_end = T_end 
        
        self._init_erk()

    def _init_erk(self):
        print(f"--- Initializing ERK Topology (Target: {self.sparsity:.0%}) ---")
        total_params, erk_weights, layers = 0, [], []
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                n_out, n_in = param.shape[0], param.shape[1]
                k = param.shape[2] if param.dim() == 4 else 1
                erk_weight = (n_in + n_out + k**2) / (n_in * n_out * k**2)
                total_params += param.numel()
                erk_weights.append(erk_weight)
                layers.append((name, param))

        eps = 0.5 
        for _ in range(20): 
            current_nz = sum(int(min(1.0, max(0.01, eps * w)) * p.numel()) for w, (n, p) in zip(erk_weights, layers))
            eps *= ((total_params * (1 - self.sparsity)) / current_nz)

        with torch.no_grad():
            for i, (name, param) in enumerate(layers):
                d_l = min(1.0, max(0.01, eps * erk_weights[i]))
                self.masks[name] = (torch.rand_like(param) < d_l).float().to(self.device)
                param.data.mul_(self.masks[name])

    def get_growth_fraction(self):
        if self.epochs_completed >= self.T_end: return 0.0
        return self.alpha * 0.5 * (1 + math.cos(self.epochs_completed * math.pi / self.T_end))

    def on_batch_end(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks: param.data.mul_(self.masks[name])

    def on_epoch_end(self):
        if self.epochs_completed < self.T_end:
            self._evolve_topology()
        self.epochs_completed += 1

    def _evolve_topology(self):
        f_t = self.get_growth_fraction()
        if f_t <= 0: return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in self.masks or 'bias' in name: continue
                mask = self.masks[name]
                num_active = int(mask.sum().item())
                num_to_prune = int(f_t * num_active)
                
                if num_to_prune > 0:
                    # 1. Magnitude Pruning
                    weights = torch.abs(param.data)
                    active_weights = torch.where(mask == 1, weights, torch.tensor(float('inf')).to(self.device))
                    kth_val = torch.kthvalue(active_weights.flatten(), num_to_prune).values
                    mask[active_weights <= kth_val] = 0.0
                    
                    # 2. RANDOM GROWTH
                    dead_indices = (mask == 0).nonzero(as_tuple=False)
                    if len(dead_indices) > 0:
                        random_choices = torch.randperm(len(dead_indices))[:num_to_prune]
                        grown_indices = dead_indices[random_choices]
                        mask[grown_indices[:, 0], grown_indices[:, 1]] = 1.0
                        param.data[grown_indices[:, 0], grown_indices[:, 1]] = 0.0
                
                self.masks[name] = mask

                # 4. Clear stale momentum
                if param in self.optimizer.state:
                    momentum_buffer = self.optimizer.state[param].get('momentum_buffer')
                    if momentum_buffer is not None:
                        momentum_buffer.mul_(mask)

# ==========================================
# 3. ARCHITECTURE & DATALOADERS
# ==========================================
def get_dataloaders():
    if DATASET == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    else: 
        # SAME DATA AUGMENTATION AS IMPLEMENTED IN Ultimate RigL MODEL 
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
    
    return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True), DataLoader(test_ds, batch_size=1000, shuffle=False)

def get_resnet():
    model = models.resnet18()
    if DATASET == 'MNIST':
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

# ==========================================
# 4. MAIN TRAINING LOOP
# ==========================================
def main():
    global_start_time = time.time() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Ultimate SET Final Run: {DATASET} on {device} ---")
    
    train_loader, test_loader = get_dataloaders()
    model = get_resnet().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_STEPS, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    set_scheduler = UltimateSETScheduler(model, optimizer, sparsity=SPARSITY, alpha=ALPHA, T_end=80, device=device)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        epoch_start_time = time.time() 
        print(f"\n[Epoch {epoch}/{EPOCHS}] LR: {optimizer.param_groups[0]['lr']:.6f}")

        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            set_scheduler.on_batch_end()
            total_loss += loss.item()

            if (i+1) % 200 == 0 or (i+1) == len(train_loader):
                elapsed = time.time() - epoch_start_time
                eta_m, eta_s = divmod(int(((len(train_loader)-(i+1)) * (elapsed/(i+1)))), 60)
                print(f"   Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f} | ETA: {eta_m}m {eta_s}s")

        lr_scheduler.step()
        set_scheduler.on_epoch_end()
        epoch_duration = time.time() - epoch_start_time
        
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                preds = model(data).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy()); all_targets.extend(target.cpu().numpy())
        
        acc, f1 = accuracy_score(all_targets, all_preds), f1_score(all_targets, all_preds, average='weighted')
        f_t = set_scheduler.get_growth_fraction()
        print(f"   -> Val Acc: {acc:.4f} | F1: {f1:.4f} | Growth Frac: {f_t:.4f} | Time: {epoch_duration:.1f}s")

        with open(HISTORY_FILE, mode='a', newline='') as f:
            csv.writer(f).writerow([epoch, total_loss/len(train_loader), acc, f1, f_t, round(epoch_duration, 2)])

    total_train_time = time.time() - global_start_time
    tt_hrs, tt_rem = divmod(total_train_time, 3600)
    tt_mins, tt_secs = divmod(tt_rem, 60)
    
    torch.save(model.state_dict(), WEIGHTS_FILE)
    final_mask = set_scheduler.masks['fc.weight'].cpu()
    degrees = final_mask.sum(dim=1).numpy()
    pd.DataFrame({'Neuron_ID': range(len(degrees)), 'Degree': degrees}).to_csv(DEGREES_FILE, index=False)
    
    print(f"\n[+] Training Complete in {int(tt_hrs)}h {int(tt_mins)}m {int(tt_secs)}s.")
    print(f"[+] Artifacts saved to: {SAVE_DIR}")

if __name__ == '__main__':
    main()