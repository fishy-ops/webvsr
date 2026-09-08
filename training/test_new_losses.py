"""Pre-flight for the three losses imported from neosr's SPAN recipe.

Each check is a property the loss must have for the experiment to mean anything.
A loss that passes "identical -> 0" and nothing else is not evidence it works.
"""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "training")
from losses import MSSIMLoss, ColorConsistencyLoss, FDLLoss, CharbonnierLoss

dev = "cuda"
torch.manual_seed(0)
B = 4
gt = torch.rand(B, 3, 256, 256, device=dev)
# something with real structure, not just noise
xx = torch.linspace(0, 20, 256, device=dev)
gt = (0.5 + 0.4 * torch.sin(xx.view(1,1,-1,1) * 3) * torch.cos(xx.view(1,1,1,-1) * 2)
      ).repeat(B, 3, 1, 1).clamp(0, 1)
gt = (gt + 0.05 * torch.rand_like(gt)).clamp(0, 1)

k = torch.ones(3, 1, 5, 5, device=dev) / 25
blurred = F.conv2d(gt, k, padding=2, groups=3)
shifted = torch.roll(gt, shifts=(1, 1), dims=(2, 3))
hueshift = gt.clone(); hueshift[:, 0] = (hueshift[:, 0] + 0.06).clamp(0, 1)
lumashift = (gt + 0.06).clamp(0, 1)

char = CharbonnierLoss().to(dev)
print(f"reference: charbonnier(gt,blurred) = {char(blurred, gt).item():.5f}")
print(f"           charbonnier at convergence in the real runs ~ 0.021\n")

for name, L in [("MSSIM", MSSIMLoss()), ("Color", ColorConsistencyLoss()), ("FDL", FDLLoss())]:
    L = L.to(dev)
    vals = {}
    for tag, x in [("identical", gt), ("blurred", blurred), ("shift1px", shifted),
                   ("hue+0.06", hueshift), ("luma+0.06", lumashift)]:
        with torch.no_grad():
            vals[tag] = L(x, gt).item()
    print(f"{name}:")
    for t, v in vals.items():
        print(f"    {t:<10} {v:.6f}")
    # gradient check
    p = gt.clone().requires_grad_(True)
    v = L(p, blurred)
    v.backward()
    g = p.grad
    print(f"    grad: finite={torch.isfinite(g).all().item()} "
          f"absmax={g.abs().max().item():.3e} nonzero={(g!=0).float().mean().item():.2f}")
    assert vals["identical"] < 1e-4, f"{name} does not vanish on identical inputs"
    assert vals["blurred"] > vals["identical"], f"{name} indifferent to blur"
    print(f"    peak GPU {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
    torch.cuda.reset_peak_memory_stats()
    print()

# the property FDL is chosen for: shift should cost much less than blur
Lf = FDLLoss().to(dev)
with torch.no_grad():
    fb, fs = Lf(blurred, gt).item(), Lf(shifted, gt).item()
Lp = MSSIMLoss().to(dev)
with torch.no_grad():
    mb, ms = Lp(blurred, gt).item(), Lp(shifted, gt).item()
print(f"misalignment robustness (blur/shift ratio, higher = more robust):")
print(f"    FDL   blur {fb:.6f}  shift {fs:.6f}  ratio {fb/max(fs,1e-9):.2f}")
print(f"    MSSIM blur {mb:.6f}  shift {ms:.6f}  ratio {mb/max(ms,1e-9):.2f}")

# Color must see chroma error it is aimed at
Lc = ColorConsistencyLoss().to(dev)
with torch.no_grad():
    ch = Lc(hueshift, gt).item(); lu = Lc(lumashift, gt).item()
print(f"\ncolour term: hue shift {ch:.6f} vs equal-size luma shift {lu:.6f}")

# MS-SSIM must refuse a crop it cannot score
try:
    MSSIMLoss().to(dev)(gt[:, :, :64, :64], gt[:, :, :64, :64])
    print("FAIL: MSSIM accepted a 64px crop")
except ValueError as e:
    print(f"MSSIM correctly refuses 64px: {str(e)[:60]}...")
print("\nall pre-flight checks passed")
