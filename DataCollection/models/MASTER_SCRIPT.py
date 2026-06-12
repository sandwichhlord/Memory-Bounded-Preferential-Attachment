import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
import csv
import time
import random 
import numpy as np
import pandas as pd
import gc # For aggressive garbage collection
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# 0. ENVIRONMENT SETUP (GOOGLE COLAB SAFE)
# ==========================================
def is_colab():
    try: return 'google.colab' in str(get_ipython())
    except NameError: return False

IS_COLAB = is_colab()

if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    SAVE_DIR = '/content/drive/MyDrive/Thesis_Ultimate_Run/' 
    os.makedirs(SAVE_DIR, exist_ok=True)
else:
    SAVE_DIR = './'

# ==========================================
# 1. GLOBAL EXPERIMENT CONFIGURATION
# ==========================================
DATASET = 'CIFAR100'
EPOCHS = 100
BATCH_SIZE = 64
DELTA_T = 781 
DROP_RATE_INIT = 0.3
SEED = 42

# MBPA Specific Constants
TDST_ALPHA = 1.5
TDST_KAPPA = 4.0

# ==========================================
# FORT KNOX SEEDING PROTOCOL
# ==========================================
def seed_everything(seed):
    random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

seed_everything(SEED)

g = torch.Generator()
g.manual_seed(SEED)

# ==========================================
# 2. THE MASTER SPARSE SCHEDULER
# ==========================================
class MasterSparseScheduler:
    def __init__(self, mode, model, optimizer, sparsity, drop_rate_init=0.3, delta_t=781, T_end=None, device='cuda'):
        assert mode in ['set', 'rigl', 'mbpa'], "Mode must be 'set', 'rigl', or 'mbpa'"
        self.mode = mode
        self.model = model
        self.optimizer = optimizer
        self.sparsity = sparsity
        self.drop_rate_init = drop_rate_init
        self.delta_t = delta_t
        self.T_end = T_end
        self.device = device
        
        self.masks = {}
        self.iterations = 0
        self.absolute_min_weights = {} 
        
        self._init_erk()

    def _init_erk(self):
        print(f"--- Initializing ERK Topology | Mode: {self.mode.upper()} | Target: {self.sparsity:.1%} ---")
        total_params, erk_weights, layers, min_densities = 0, [], [], []
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                n_out, n_in = param.shape[0], param.shape[1]
                k = param.shape[2] if param.dim() == 4 else 1
                
                erk_weight = (n_in + n_out + k**2) / (n_in * n_out * k**2)
                total_params += param.numel()
                erk_weights.append(erk_weight)
                layers.append((name, param))
                
                abs_min = max(n_out, n_in)
                self.absolute_min_weights[name] = abs_min
                min_densities.append(abs_min / param.numel())

        eps = 0.5
        for _ in range(20):
            current_nz = sum(int(min(1.0, max(min_densities[i], eps * w)) * p.numel()) 
                             for i, (w, (n, p)) in enumerate(zip(erk_weights, layers)))
            if current_nz == 0: break 
            eps *= ((total_params * (1 - self.sparsity)) / current_nz)

        with torch.no_grad():
            for i, (name, param) in enumerate(layers):
                d_l = min(1.0, max(min_densities[i], eps * erk_weights[i]))
                self.masks[name] = (torch.rand_like(param) < d_l).float().to(self.device)
                param.data.mul_(self.masks[name])

    def get_growth_fraction(self):
        if self.iterations >= self.T_end: return 0.0
        return self.drop_rate_init * 0.5 * (1 + math.cos(self.iterations * math.pi / self.T_end))

    def on_batch_end(self):
        self.iterations += 1
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks: param.data.mul_(self.masks[name])

        if self.iterations % self.delta_t == 0 and self.iterations < self.T_end:
            self._evolve_topology()

    def _evolve_topology(self):
        f_t = self.get_growth_fraction()
        if f_t <= 0: return

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in self.masks or 'bias' in name: continue
                mask = self.masks[name]
                
                total_params = param.numel()
                current_active = int(mask.sum().item())
                
                max_droppable = max(0, current_active - self.absolute_min_weights[name])
                num_to_prune = int(f_t * current_active)
                num_to_prune = min(num_to_prune, max_droppable) 

                if num_to_prune > 0:
                    flat_weights = torch.abs(param.data.view(-1))
                    flat_mask = mask.view(-1)
                    active_indices = flat_mask.nonzero(as_tuple=True)[0]
                    active_mags = flat_weights[active_indices]
                    
                    bottom_k_relative = torch.topk(active_mags, num_to_prune, largest=False)[1]
                    indices_to_drop = active_indices[bottom_k_relative]
                    
                    flat_mask[indices_to_drop] = 0.0
                    param.data.view(-1)[indices_to_drop] = 0.0 

                    actual_grow = num_to_prune
                    indices_to_grow = None
                    
                    if self.mode == 'set':
                        zero_indices = (flat_mask == 0).nonzero(as_tuple=True)[0]
                        if len(zero_indices) > 0:
                            actual_grow = min(actual_grow, len(zero_indices))
                            grown_relative = torch.randperm(len(zero_indices), device=self.device)[:actual_grow]
                            indices_to_grow = zero_indices[grown_relative]

                    elif self.mode == 'rigl':
                        if param in self.optimizer.state and self.optimizer.state[param].get('momentum_buffer') is not None:
                            scoring_tensor = torch.abs(self.optimizer.state[param]['momentum_buffer'])
                        else:
                            scoring_tensor = torch.abs(param.grad.data)
                        
                        dead_scores = torch.where(mask == 0, scoring_tensor, -1.0)
                        indices_to_grow = torch.topk(dead_scores.flatten(), actual_grow).indices

                    elif self.mode == 'mbpa':
                        true_local_density = current_active / total_params
                        if param.dim() == 4: 
                            deg = mask.sum(dim=(1, 2, 3)) 
                            fan_in = param.shape[1] * param.shape[2] * param.shape[3]
                            expand_dims = (1, 1, 1)
                        else: 
                            deg = mask.sum(dim=1) 
                            fan_in = param.shape[1]
                            expand_dims = (1,)
                            
                        e_k = true_local_density * fan_in
                        c_max = min(fan_in, max(TDST_ALPHA * e_k, TDST_KAPPA * math.sqrt(fan_in)))
                        
                        ba_scores = deg.float().clone() + 1e-5              
                        ba_scores[deg >= c_max] = 0.0 
                        
                        grow_probs = ba_scores.view(-1, *expand_dims).expand_as(param).clone()
                        grow_probs[mask == 1] = 0.0 
                        grow_probs_flat = grow_probs.flatten()
                        
                        valid_slots = (grow_probs_flat > 0).sum().item()
                        actual_grow = min(actual_grow, valid_slots)
                        
                        if actual_grow > 0:
                            grow_probs_flat /= grow_probs_flat.sum()
                            indices_to_grow = torch.multinomial(grow_probs_flat, num_samples=actual_grow, replacement=False)

                    if indices_to_grow is not None and actual_grow > 0:
                        mask.view(-1)[indices_to_grow] = 1.0
                        param.data.view(-1)[indices_to_grow] = 0.0

                        if param in self.optimizer.state:
                            momentum_buffer = self.optimizer.state[param].get('momentum_buffer')
                            if momentum_buffer is not None:
                                momentum_buffer.flatten()[indices_to_drop] = 0.0
                                momentum_buffer.flatten()[indices_to_grow] = 0.0

                self.masks[name] = mask

# ==========================================
# 3. ARCHITECTURE & DATALOADERS
# ==========================================
def get_dataloaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    train_ds = datasets.CIFAR100('./data', train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR100('./data', train=False, download=True, transform=transform_test)
    
    # IMPLEMENTATION FIX: Added num_workers and pin_memory for GPU throughput
    return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=g), 
            DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True))

def get_resnet():
    model = models.resnet18()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

# ==========================================
# 4. EXECUTION WRAPPER
# ==========================================
def run_experiment(mode, sparsity, run_seed):
    seed_everything(run_seed)
    g.manual_seed(run_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n" + "="*60)
    print(f"🚀 LAUNCHING: Mode=[{mode.upper()}] | Sparsity=[{sparsity:.1%}] | Seed=[{run_seed}]")
    print("="*60)

    base_name = f"{mode.upper()}_{sparsity*100}%_Seed{run_seed}"
    hist_file = os.path.join(SAVE_DIR, f"{base_name}_history.csv")
    deg_file = os.path.join(SAVE_DIR, f"{base_name}_degrees.csv")
    weights_file = os.path.join(SAVE_DIR, f"{base_name}_sparse.pth")

    with open(hist_file, mode='w', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'Train_Loss', 'Val_Accuracy', 'Val_F1', 'Growth_Fraction', 'Epoch_Time_sec'])

    train_loader, test_loader = get_dataloaders()
    model = get_resnet().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    total_iters = len(train_loader) * EPOCHS
    scheduler = MasterSparseScheduler(mode=mode, model=model, optimizer=optimizer, sparsity=sparsity, 
                                      drop_rate_init=DROP_RATE_INIT, delta_t=DELTA_T, 
                                      T_end=int(total_iters * 0.75), device=device)

    global_start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        epoch_start_time = time.time() 

        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            scheduler.on_batch_end()
            total_loss += loss.item()

            if (i+1) % 200 == 0 or (i+1) == len(train_loader):
                elapsed = time.time() - epoch_start_time
                eta_m, eta_s = divmod(int(((len(train_loader)-(i+1)) * (elapsed/(i+1)))), 60)
                print(f"   [E{epoch:03d} B{i+1:03d}] Loss: {loss.item():.4f} | ETA: {eta_m}m {eta_s}s")

        lr_scheduler.step()
        epoch_duration = time.time() - epoch_start_time
        
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                preds = model(data).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy()); all_targets.extend(target.cpu().numpy())
        
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        f_t = scheduler.get_growth_fraction()
        print(f"   -> Mode: {mode.upper()} | Val Acc: {acc:.4f} | F1: {f1:.4f} | Growth: {f_t:.4f} | {epoch_duration:.1f}s")

        with open(hist_file, mode='a', newline='') as f:
            csv.writer(f).writerow([epoch, total_loss/len(train_loader), acc, f1, f_t, round(epoch_duration, 2)])

    torch.save(model.state_dict(), weights_file)
    
    # IMPLEMENTATION FIX: Extract degrees for the ENTIRE network to prove global scale-free hubs
    all_degrees = []
    for name, mask in scheduler.masks.items():
        if mask.dim() == 4:
            all_degrees.extend(mask.sum(dim=(1, 2, 3)).cpu().numpy())
        else:
            all_degrees.extend(mask.sum(dim=1).cpu().numpy())
            
    pd.DataFrame({'Global_Network_Degrees': all_degrees}).to_csv(deg_file, index=False)
    
    tt_mins, tt_secs = divmod(time.time() - global_start_time, 60)
    print(f"\n[+] {mode.upper()} Run Complete in {int(tt_mins)}m {int(tt_secs)}s.")
    
    # IMPLEMENTATION FIX: Nuke pointers and flush VRAM to prevent 3AM crashes
    del model, optimizer, lr_scheduler, scheduler, train_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# 5. THE AUTOMATED BATCH LAUNCHER
# ==========================================
if __name__ == '__main__':
    MODES_TO_TEST = ['set', 'mbpa', 'rigl']
    SPARSITIES_TO_TEST = [0.90, 0.95, 0.98, 0.99, 0.995]
    SEEDS_TO_TEST = [42, 43, 44]
    
    for current_seed in SEEDS_TO_TEST:
        for current_sparsity in SPARSITIES_TO_TEST:
            for current_mode in MODES_TO_TEST:
                run_experiment(current_mode, current_sparsity, current_seed)
                
    print("\n" + "*"*60)
    print("🏆 ALL EXPERIMENTS COMPLETE. COMMENCE DATA ANALYSIS.")
    print("*"*60)