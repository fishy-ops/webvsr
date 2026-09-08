"""FDL's stride, and whether the expensive setting buys anything.

The reference uses stride 1 with DINOv2, whose features are already 14x
downsampled -- so its projections are ~14px apart in image space. Applying
stride 1 to VGG stage-1 features, which are at full resolution, samples 14x
more densely than the paper ever intended and sorts 200x more elements. This
asks whether the loss's actual behaviour survives a stride that restores the
reference's spatial density.
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
blurred = F.conv2d(gt, k, padding=2, groups=3)
shifted = torch.roll(gt, (1,1), (2,3))

for stride in (1, 2, 4):
    crit = FDLLoss(stride=stride).to(dev)
    with torch.no_grad():
        b, s = crit(blurred, gt).item(), crit(shifted, gt).item()
    for _ in range(3):
        loss = crit(model(lr), hr); loss.backward()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    N = 8
    for _ in range(N):
        loss = crit(model(lr), hr); loss.backward()
    torch.cuda.synchronize()
    dt = (time.perf_counter()-t0)/N
    print(f"stride {stride}: {dt*1000:6.1f} ms/step  {torch.cuda.max_memory_allocated()/2**20:5.0f} MiB  "
          f"| blur {b:.4f} shift {s:.4f} ratio {b/s:.2f}  "
          f"| 40 epochs {dt*1587*40/3600:.1f} h")
    del crit; torch.cuda.empty_cache()
