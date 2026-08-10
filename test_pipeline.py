"""Quick smoke test: model + dataset + loss + one training step."""

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from model_span import SPANLite, count_params
from losses import CombinedLoss
from dataset import SRDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Model
model = SPANLite(upscale=2, feature_channels=32).to(device)
print(f"Params (train): {count_params(model):,}")

# Count inference-mode params
model.eval()
inf_params = 0
for m in model.modules():
    if hasattr(m, 'eval_conv'):
        inf_params += sum(p.numel() for p in [m.eval_conv.weight, m.eval_conv.bias])
    elif isinstance(m, torch.nn.Conv2d) and not hasattr(m, '_update_params'):
        parent_is_conv3xc = False
        for m2 in model.modules():
            if hasattr(m2, 'conv') and m in m2.conv:
                parent_is_conv3xc = True
                break
            if hasattr(m2, 'sk') and m is m2.sk:
                parent_is_conv3xc = True
                break
        if not parent_is_conv3xc:
            inf_params += sum(p.numel() for p in m.parameters())
print(f"Params (inference/ONNX): ~{inf_params:,}")
model.train()

# Dataset — just use DIV2K for the smoke test
ds = SRDataset(
    data_dirs=[r"C:\Users\reach\OneDrive\Documents\mamba-sr\DIV2K_train_HR\DIV2K_train_HR"],
    crop_size=256, scale=2, use_degradation=True,
)
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

# Loss (phase 1: no perceptual)
criterion_p1 = CombinedLoss(w_fft=0.01, use_perceptual=False).to(device)

# Loss (phase 2: with perceptual)
criterion_p2 = CombinedLoss(w_perceptual=0.1, w_fft=0.01, use_perceptual=True).to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scaler = GradScaler()

# One training step
lr_batch, hr_batch = next(iter(loader))
lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
print(f"\nBatch shapes: LR {lr_batch.shape}, HR {hr_batch.shape}")

# Phase 1 step
optimizer.zero_grad()
with autocast(device_type="cuda", dtype=torch.float16):
    sr = model(lr_batch).clamp(0, 1)
    loss, components = criterion_p1(sr, hr_batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
print(f"Phase 1 step: loss={loss.item():.4f} | {components}")
print(f"SR output shape: {sr.shape}")

# Phase 2 step
optimizer.zero_grad()
with autocast(device_type="cuda", dtype=torch.float16):
    sr = model(lr_batch).clamp(0, 1)
    loss, components = criterion_p2(sr, hr_batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
print(f"Phase 2 step: loss={loss.item():.4f} | {components}")

# VRAM usage
vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
print(f"\nPeak VRAM: {vram_mb:.0f} MB")
print("\nAll systems go!")
