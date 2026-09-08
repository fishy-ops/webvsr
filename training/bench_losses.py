"""What each new term costs per optimizer step, on the real model at real size.

The full-epoch smoke test could not finish in ten minutes, which says the cost
is real but not which term carries it. Timing the terms in isolation says where
the budget goes before ~10 GPU-hours are committed to a chain.
"""
import sys, time, torch
sys.path.insert(0, "training")
from model_span import SPANLite
from losses import CombinedLoss

dev = "cuda"
B, CROP, SCALE = 8, 256, 2
model = SPANLite(num_in_ch=3, feature_channels=16, upscale=SCALE).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
scaler = torch.amp.GradScaler()
lr = torch.rand(B, 3, CROP // SCALE, CROP // SCALE, device=dev)
hr = torch.rand(B, 3, CROP, CROP, device=dev)

CASES = [
    ("baseline (P2: char+fft+vgg)", {}),
    ("+ mssim 1.0",  {"w_mssim": 1.0}),
    ("+ color 1.0",  {"w_color": 1.0}),
    ("+ fdl 0.75",   {"w_fdl": 0.75}),
]
STEPS = 12
base = None
for name, kw in CASES:
    crit = CombinedLoss(use_perceptual=True, **kw).to(dev)
    for warm in range(3):
        with torch.autocast("cuda", dtype=torch.float16):
            loss, _ = crit(model(lr), hr)
        scaler.scale(loss).backward(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        with torch.autocast("cuda", dtype=torch.float16):
            loss, parts = crit(model(lr), hr)
        scaler.scale(loss).backward()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / STEPS
    if base is None:
        base = dt
    mem = torch.cuda.max_memory_allocated() / 2**20
    # 1587 batches/epoch at batch 8; 40 epochs in the planned chain
    print(f"{name:<30} {dt*1000:7.1f} ms/step  {dt/base:5.2f}x  "
          f"{mem:6.0f} MiB  -> epoch {dt*1587/60:5.1f} min, "
          f"40 epochs {dt*1587*40/3600:5.1f} h")
    del crit; torch.cuda.empty_cache()
