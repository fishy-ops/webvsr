"""Verify conv_last folds into conv_cat exactly, before writing any shader.

out = conv_cat( cat[f0, mid1, mid3, conv_last(b4)] )

conv_cat is 1x1 with weight W [C, 4C]; split it into W0..W3 of [C, C]. conv_last
is 3x3 with weight L [C,C,3,3] and bias bl. The fourth term is

    W3 @ (L * b4 + bl) = (W3 @ L) * b4 + W3 @ bl

so the pair collapses into a single 3x3 kernel K[o,i] = sum_m W3[o,m] L[m,i]
plus a bias shift. One whole pass -- a full C-channel write and read -- disappears.
"""
import sys
import torch, torch.nn.functional as F
sys.path.insert(0, "training")
from model_span import SPANLite

torch.manual_seed(0)
C = 16
m = SPANLite(feature_channels=C, upscale=2).eval()
for mod in m.modules():
    if hasattr(mod, "_update_params") and hasattr(mod, "eval_conv"):
        mod._update_params()

f0, mid1, mid3, b4 = (torch.randn(1, C, 24, 24) for _ in range(4))
W = m.conv_cat.weight.detach()            # [C, 4C, 1, 1]
bc = m.conv_cat.bias.detach()
L = m.conv_last.eval_conv.weight.detach() # [C, C, 3, 3]
bl = m.conv_last.eval_conv.bias.detach()

# reference: conv_last then conv_cat
ref = F.conv2d(torch.cat([f0, mid1, mid3,
                          F.conv2d(b4, L, bl, padding=1)], 1), W, bc)

# fused: 1x1 over the first three, composed 3x3 over the fourth
W3 = W[:, 3 * C:4 * C, 0, 0]                       # [C, C]
K = torch.einsum("om,mikl->oikl", W3, L)           # [C, C, 3, 3]
bias = bc + W3 @ bl
fused = (F.conv2d(torch.cat([f0, mid1, mid3], 1), W[:, :3 * C], None)
         + F.conv2d(b4, K, bias, padding=1))

d = (ref - fused).abs().max().item()
print(f"max|ref - fused| = {d:.3e}   ({'EXACT' if d < 1e-4 else 'MISMATCH'})")
print(f"kernel K shape {tuple(K.shape)}, bias shift {tuple(bias.shape)}")
print(f"passes saved: conv_last dispatch + one {C}-channel buffer write and read")
