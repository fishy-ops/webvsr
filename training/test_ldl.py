"""LDL against a realistic failure: a model that blurs.

The residual of a blur-prone model is not uniform -- it is large exactly where
detail was lost, which is the textured content. That is the case LDL has to
up-weight for it to be worth adding here.
"""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "training")
from losses import LDLLoss

torch.manual_seed(0)
ldl = LDLLoss()

gt = torch.zeros(2, 3, 96, 96)
gt[:, :, :, 48:] = torch.rand(2, 3, 96, 48)          # right half: stochastic texture
gt[:, :, 20:28, :48] = 1.0                            # left half: one clean edge

k = torch.ones(3, 1, 5, 5) / 25.0
pred = F.conv2d(F.pad(gt, (2,)*4, mode="reflect"), k, groups=3)   # a blurring model

r = (gt - pred).abs().sum(1, keepdim=True)
w = (r.var(dim=(-1,-2,-3), keepdim=True).clamp(min=1e-12) ** 0.2) * ldl._local_var(r)
w = w / w.mean().clamp(min=1e-8)

flat = float(w[:, :, 40:, :48].mean())      # left half away from the edge
edge = float(w[:, :, 18:30, :48].mean())    # the clean edge
tex  = float(w[:, :, :, 48:].mean())
print(f"weight on FLAT region   : {flat:.4f}")
print(f"weight on a CLEAN EDGE  : {edge:.4f}")
print(f"weight on TEXTURE       : {tex:.4f}")
print(f"texture / flat          : {tex/max(flat,1e-8):8.1f}x")
print(f"texture / edge          : {tex/max(edge,1e-8):8.2f}x")
print()
print("LDL up-weights stochastic texture over flat:",
      "YES" if tex > 5 * max(flat, 1e-8) else "NO -- not worth adding")
