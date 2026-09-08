"""Can FDL be made affordable by dropping its full-resolution VGG stage?

FDL costs 5x a normal step, and stage 1 operates on a 64-channel map at the
full output resolution -- its FFT alone is ~270 MB at batch 8. If dropping it
keeps the misalignment robustness that makes FDL worth having, the loss becomes
usable in a normal-length run instead of eating a whole night.
"""
import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, "training")
from model_span import SPANLite
from losses import FDLLoss

dev = "cuda"
B, CROP = 8, 256
model = SPANLite(num_in_ch=3, feature_channels=16, upscale=2).to(dev)
lr = torch.rand(B, 3, CROP // 2, CROP // 2, device=dev)
hr = torch.rand(B, 3, CROP, CROP, device=dev)

torch.manual_seed(0)
xx = torch.linspace(0, 20, CROP, device=dev)
gt = (0.5 + 0.4 * torch.sin(xx.view(1,1,-1,1)*3) * torch.cos(xx.view(1,1,1,-1)*2)).repeat(4,3,1,1)
gt = (gt + 0.05*torch.rand_like(gt)).clamp(0,1)
k = torch.ones(3,1,5,5, device=dev)/25
blur = F.conv2d(gt, k, padding=2, groups=3)
shift = torch.roll(gt, (1,1), (2,3))

for skip in (0, 1, 2):
    crit = FDLLoss().to(dev)
    if skip:
        crit.stages = crit.stages[skip:]
        crit._STAGE_W = crit._STAGE_W[skip:]
        for i in range(skip):                       # drop the unused projections
            delattr(crit, f"proj_{i}")
        for i in range(skip, 5):
            crit.register_buffer(f"proj_{i-skip}", getattr(crit, f"proj_{i}").clone())
    with torch.no_grad():
        b, s_ = crit(blur, gt).item(), crit(shift, gt).item()
    for _ in range(3):
        crit(model(lr), hr).backward()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter(); N = 6
    for _ in range(N):
        crit(model(lr), hr).backward()
    torch.cuda.synchronize()
    dt = (time.perf_counter()-t0)/N
    print(f"  skip first {skip} stage(s): {dt*1000:6.1f} ms/step  "
          f"{torch.cuda.max_memory_allocated()/2**20:5.0f} MiB  "
          f"blur/shift ratio {b/max(s_,1e-9):5.2f}  (higher = keeps the robustness)")
    del crit; torch.cuda.empty_cache()
