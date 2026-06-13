"""
train.py  —  wPINN EEG dipole localization
-------------------------------------------
Run with:  python train.py

Automatically:
  1. Generates training data if not already on disk
  2. Subsamples 10,000 training samples
  3. Generates E1 validation set (nominal params, no stochasticity)
  4. Trains FiLM-MLP with all three losses backpropping:
       Lsource  — supervised dipole prediction
       Lbc      — boundary consistency (forward_torch)
       LwPINN   — weak-form PDE residual (forward_torch + autograd)
  5. Evaluates on E1, prints localization and orientation errors
  6. Saves checkpoint + results
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import FiLMMLP
from forward_torch import forward_torch
from forwardmodel import forward as forward_np, default_radii, default_sigmas, make_scalp_electrodes
import parameters as params

# -----------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------
N_TRAIN       = 15_000
N_E1          = 500
BATCH_SIZE    = 64
N_EPOCHS      = 100
LR            = 1e-3
ALPHA         = 1.0       # lowered from 10.0 — let p and r0 compete fairly
LAMBDA_BC     = 0.01      # lowered from 0.1 — prevent Lbc destabilizing early
LAMBDA_W      = 0.01
N_TEST_FUNCS  = 5
GAUSS_SIGMA   = 0.5
N_TERMS       = 50
WARMUP_EPOCHS = 10        # only Lsource for first 10 epochs
DATA_DIR      = "training_data"
RESULTS_DIR   = "results"

os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ELECTRODES_NP = make_scalp_electrodes(64, params.scalp_rad, "upper")[0]
ELECTRODES    = torch.tensor(ELECTRODES_NP, dtype=torch.float32)

_rng_tf = np.random.default_rng(0)
TEST_CENTRES = torch.tensor(
    _rng_tf.uniform(-0.4 * params.brain_rad,
                     0.4 * params.brain_rad,
                     size=(N_TEST_FUNCS, 3)),
    dtype=torch.float32
)

def grad_gaussian(x, centre, sigma=GAUSS_SIGMA):
    diff = x - centre.unsqueeze(0)
    g    = torch.exp(-0.5 * (diff**2).sum(1) / sigma**2)
    return -diff / sigma**2 * g.unsqueeze(1)

# -----------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------

def sample_radii():
    while True:
        brain = np.random.uniform(0.95*params.brain_rad,  1.05*params.brain_rad)
        csf   = np.random.uniform(0.95*params.csftop_rad, 1.05*params.csftop_rad)
        skull = np.random.uniform(0.95*params.skull_rad,  1.05*params.skull_rad)
        scalp = np.random.uniform(0.95*params.scalp_rad,  1.05*params.scalp_rad)
        if brain < csf < skull < scalp:
            return np.array([brain, csf, skull, scalp], dtype=np.float32)

def sample_conductivities():
    def lu(lo,hi): return float(np.exp(np.random.uniform(np.log(lo),np.log(hi))))
    return np.array([
        lu(0.7*params.sigma_brain,   1.3*params.sigma_brain),
        lu(0.7*params.sigma_csf,     1.3*params.sigma_csf),
        lu(0.7*params.sigma_skull20, 1.3*params.sigma_skull20),
        lu(0.7*params.sigma_scalp,   1.3*params.sigma_scalp),
    ], dtype=np.float32)

def sample_unit_vector():
    v = np.random.normal(size=3).astype(np.float32)
    return v / np.linalg.norm(v)

def sample_dipole_location():
    inner = 0.05*params.brain_rad; outer = 0.95*params.brain_rad
    u = np.random.uniform(0.0,1.0)
    r = (inner**3 + u*(outer**3-inner**3))**(1.0/3.0)
    return (r*sample_unit_vector()).astype(np.float32)

def generate_data(n_samples, nominal=False, tag="data"):
    V_all=np.zeros((n_samples,64),dtype=np.float32)
    sigma_all=np.zeros((n_samples,4),dtype=np.float32)
    r_all=np.zeros((n_samples,4),dtype=np.float32)
    p_all=np.zeros((n_samples,3),dtype=np.float32)
    r0_all=np.zeros((n_samples,3),dtype=np.float32)
    base_radii  = default_radii().astype(np.float32)
    base_sigmas = default_sigmas().astype(np.float32)
    for i in range(n_samples):
        radii  = base_radii  if nominal else sample_radii()
        sigmas = base_sigmas if nominal else sample_conductivities()
        r0     = sample_dipole_location()
        p      = sample_unit_vector()
        V      = forward_np(p=p,r0=r0,radii=radii,sigmas=sigmas,n_terms=N_TERMS).astype(np.float32)
        V_all[i]=V; sigma_all[i]=sigmas; r_all[i]=radii; p_all[i]=p; r0_all[i]=r0
        if (i+1)%1000==0: print(f"  [{tag}] {i+1}/{n_samples}")
    return V_all,sigma_all,r_all,p_all,r0_all

# -----------------------------------------------------------------------
# Step 1 — Training data
# -----------------------------------------------------------------------
train_path = os.path.join(DATA_DIR,"V_train.npy")
if os.path.exists(train_path):
    print("Loading existing training data...")
    V_all=np.load(os.path.join(DATA_DIR,"V_train.npy"))
    sigma_all=np.load(os.path.join(DATA_DIR,"sigma_train.npy"))
    r_all=np.load(os.path.join(DATA_DIR,"r_train.npy"))
    p_all=np.load(os.path.join(DATA_DIR,"p_train.npy"))
    r0_all=np.load(os.path.join(DATA_DIR,"r0_train.npy"))
    if len(V_all)>N_TRAIN:
        print(f"  Subsampling {N_TRAIN} from {len(V_all)}...")
        idx=np.random.choice(len(V_all),N_TRAIN,replace=False)
        V_all=V_all[idx]; sigma_all=sigma_all[idx]; r_all=r_all[idx]
        p_all=p_all[idx]; r0_all=r0_all[idx]
else:
    print(f"Generating {N_TRAIN} training samples...")
    V_all,sigma_all,r_all,p_all,r0_all=generate_data(N_TRAIN,tag="train")
    np.save(os.path.join(DATA_DIR,"V_train.npy"),V_all)
    np.save(os.path.join(DATA_DIR,"sigma_train.npy"),sigma_all)
    np.save(os.path.join(DATA_DIR,"r_train.npy"),r_all)
    np.save(os.path.join(DATA_DIR,"p_train.npy"),p_all)
    np.save(os.path.join(DATA_DIR,"r0_train.npy"),r0_all)
    print("Training data saved.")

# -----------------------------------------------------------------------
# Step 2 — E1
# -----------------------------------------------------------------------
print(f"Generating {N_E1} E1 samples...")
V_e1,sigma_e1,r_e1,p_e1,r0_e1=generate_data(N_E1,nominal=True,tag="E1")
print("E1 ready.")

# -----------------------------------------------------------------------
# Step 3 — Normalisation
# -----------------------------------------------------------------------
V_mean,V_std   = V_all.mean(0),    V_all.std(0)    +1e-12
s_mean,s_std   = sigma_all.mean(0),sigma_all.std(0)+1e-12
r_mean,r_std   = r_all.mean(0),    r_all.std(0)    +1e-12
r0_mean,r0_std = r0_all.mean(0),   r0_all.std(0)   +1e-12

# p are unit vectors — don't normalise by mean/std (mean ~ 0, causes instability)
# just pass raw; network sees values in [-1, 1] already
p_mean  = np.zeros(3, dtype=np.float32)
p_std   = np.ones(3,  dtype=np.float32)

p_std_t  = torch.tensor(p_std, dtype=torch.float32)
p_mean_t = torch.tensor(p_mean,dtype=torch.float32)
r0_std_t = torch.tensor(r0_std, dtype=torch.float32)
r0_mean_t= torch.tensor(r0_mean,dtype=torch.float32)

def norm(a,m,s): return ((a-m)/s).astype(np.float32)

V_n  = norm(V_all,    V_mean, V_std)
s_n  = norm(sigma_all,s_mean, s_std)
r_n  = norm(r_all,    r_mean, r_std)
p_n  = norm(p_all,    p_mean, p_std)
r0_n = norm(r0_all,   r0_mean,r0_std)

c_n = np.concatenate([s_n,r_n],axis=1)
y_n = np.concatenate([p_n,r0_n],axis=1)

dataset = TensorDataset(
    torch.from_numpy(V_n),
    torch.from_numpy(c_n),
    torch.from_numpy(y_n),
    torch.from_numpy(sigma_all),
    torch.from_numpy(r_all),
    torch.from_numpy(V_all),
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------------------------------------------------
# Step 4 — Model
# -----------------------------------------------------------------------
model     = FiLMMLP(n_electrodes=64, cond_dim=8, hidden_dim=256, n_layers=4)
optimizer = optim.Adam(model.parameters(), lr=LR)
print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Train: {N_TRAIN} | Batch: {BATCH_SIZE} | Epochs: {N_EPOCHS}\n")

# -----------------------------------------------------------------------
# Step 5 — Training loop
# -----------------------------------------------------------------------
train_losses=[]

for epoch in range(1, N_EPOCHS+1):
    model.train()
    eL=eLbc=eLw=0.0

    for V_b,c_b,y_b,s_b,r_b,Vraw_b in dataloader:
        optimizer.zero_grad()

        out      = model(V_b, c_b)
        p_pred_n = out[:,:3]
        r0_pred_n= out[:,3:]

        # Lsource
        Lsource = ((p_pred_n  - y_b[:,:3])**2).mean() \
                + ALPHA*((r0_pred_n - y_b[:,3:])**2).mean()

        # denormalise
        p_phys  = p_pred_n  * p_std_t  + p_mean_t
        r0_phys = r0_pred_n * r0_std_t + r0_mean_t

        # cast to float64 for forward_torch (float32 overflows Legendre series)
        s64   = s_b.double()
        r64   = r_b.double()
        ele64 = ELECTRODES.double()
        Vraw64= Vraw_b.double()

        # clamp r0 inside brain to prevent forward_torch nan/inf
        r0_norm = torch.linalg.norm(r0_phys, dim=1, keepdim=True).clamp(min=1e-6)
        max_r   = float(0.95 * params.brain_rad)
        r0_phys = torch.where(r0_norm > max_r, r0_phys / r0_norm * max_r, r0_phys)

        # Lbc — keep gradient path through p_phys and r0_phys
        p64_grad  = p_phys.double()
        r064_grad = r0_phys.double()
        V_recon = forward_torch(p64_grad, r064_grad, s64, r64, ele64, n_terms=N_TERMS)
        Lbc     = ((V_recon - Vraw64)**2).mean().float()

        # LwPINN — detach to prevent graph accumulation causing segfault
        p64  = p_phys.detach().double()
        r064 = r0_phys.detach().double()

        # LwPINN
        sigma_brain_b = s64[:,0]
        wpinn_res=[]
        for j in range(N_TEST_FUNCS):
            c_j    = TEST_CENTRES[j].double()
            grad_vj_r0 = grad_gaussian(r064, c_j, GAUSS_SIGMA)     # (B,3)
            rhs = (p64 * grad_vj_r0).sum(1)                         # (B,)

            c_j_ele = c_j.unsqueeze(0).requires_grad_(True)         # (1,3)
            phi_cj  = forward_torch(p64, r064, s64, r64,
                                    c_j_ele, n_terms=N_TERMS)        # (B,1)
            grad_phi = torch.autograd.grad(
                phi_cj.sum(), c_j_ele, create_graph=False, retain_graph=False
            )[0].squeeze(0).detach()                                 # (3,)

            grad_vj_cj = grad_gaussian(
                c_j.unsqueeze(0)+1e-3, c_j, GAUSS_SIGMA
            ).squeeze(0)                                             # (3,)

            lhs = sigma_brain_b * (grad_phi * grad_vj_cj).sum()     # (B,)
            R_j = lhs + rhs
            wpinn_res.append((R_j**2).mean().float())

        Lwpinn = torch.stack(wpinn_res).mean()

        loss = Lsource
        if epoch > WARMUP_EPOCHS:
            loss = loss + LAMBDA_BC*Lbc + LAMBDA_W*Lwpinn
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        del V_recon, p64_grad, r064_grad, loss

        eL+=Lsource.item(); eLbc+=Lbc.item(); eLw+=Lwpinn.item()

    nb = len(dataloader)
    total = eL/nb + LAMBDA_BC*eLbc/nb + LAMBDA_W*eLw/nb
    train_losses.append(total)

    if epoch%5==0 or epoch==1:
        print(f"Epoch {epoch:3d}/{N_EPOCHS}  "
              f"Ls={eL/nb:.5f}  Lbc={eLbc/nb:.5f}  "
              f"Lw={eLw/nb:.5f}  total={total:.5f}")

print("\nTraining complete.")

# -----------------------------------------------------------------------
# Step 6 — Save
# -----------------------------------------------------------------------
torch.save({
    "model_state":model.state_dict(),
    "V_mean":V_mean,"V_std":V_std,
    "s_mean":s_mean,"s_std":s_std,
    "r_mean":r_mean,"r_std":r_std,
    "p_mean":p_mean,"p_std":p_std,
    "r0_mean":r0_mean,"r0_std":r0_std,
}, os.path.join(RESULTS_DIR,"model.pt"))
print(f"Checkpoint saved → {RESULTS_DIR}/model.pt")

# -----------------------------------------------------------------------
# Step 7 — E1 evaluation
# -----------------------------------------------------------------------
print("\nEvaluating on E1...")
model.eval()

c_e1_n = np.concatenate([norm(sigma_e1,s_mean,s_std),
                          norm(r_e1,r_mean,r_std)],axis=1)
V_e1_t = torch.from_numpy(norm(V_e1,V_mean,V_std))
c_e1_t = torch.from_numpy(c_e1_n)

with torch.no_grad():
    out_e1       = model(V_e1_t, c_e1_t)
    p_e1_pred    = out_e1[:,:3].numpy()*p_std  + p_mean
    r0_e1_pred   = out_e1[:,3:].numpy()*r0_std + r0_mean

loc_err    = np.linalg.norm(r0_e1_pred - r0_e1, axis=1)
p_pred_u   = p_e1_pred  / (np.linalg.norm(p_e1_pred, axis=1,keepdims=True)+1e-12)
p_true_u   = p_e1       / (np.linalg.norm(p_e1,      axis=1,keepdims=True)+1e-12)
orient_err = np.degrees(np.arccos(np.clip((p_pred_u*p_true_u).sum(1),-1,1)))

print("\n========== E1 Results ==========")
print(f"  Mean   loc error  : {loc_err.mean():.4f} cm")
print(f"  Median loc error  : {np.median(loc_err):.4f} cm")
print(f"  Mean   orient err : {orient_err.mean():.2f} deg")
print(f"  Median orient err : {np.median(orient_err):.2f} deg")
print("=================================\n")

np.save(os.path.join(RESULTS_DIR,"loc_errors_e1.npy"),   loc_err)
np.save(os.path.join(RESULTS_DIR,"orient_errors_e1.npy"),orient_err)
np.save(os.path.join(RESULTS_DIR,"train_losses.npy"),    np.array(train_losses))
print(f"Results saved → {RESULTS_DIR}/")