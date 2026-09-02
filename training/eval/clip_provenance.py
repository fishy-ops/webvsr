"""How much sensor noise does each clip's HR reference carry?

Rendered/synthetic footage has near-zero noise in flat regions; camera footage
carries grain. If the model wins only on clean synthetic content, the benchmark
is not representative of what the extension actually runs on.

Estimate: robust sigma of the Laplacian, measured only over the flattest
regions so real edges do not contaminate it.
"""
import sys, tempfile, subprocess
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, "training/eval"); sys.path.insert(0, "training")
from stratified_eval import load, grad_mag, VIDEO_EXT

dev = "cuda" if torch.cuda.is_available() else "cpu"
K = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)

def flat_noise(img):
    luma = 0.299*img[:, 0:1] + 0.587*img[:, 1:2] + 0.114*img[:, 2:3]
    lap = F.conv2d(luma, K.to(img.device), padding=1)[0, 0]
    g = grad_mag(img)[0, 0]
    thr = torch.quantile(g.flatten().float(), 0.25)   # flattest quarter
    sel = lap[g <= thr]
    if sel.numel() < 1000:
        return float("nan")
    # robust sigma: MAD scaled, so a few real edges cannot inflate it
    med = sel.median()
    mad = (sel - med).abs().median()
    return float(1.4826 * mad * 255 / 2.0)   # /2: Laplacian amplifies noise ~2x

clips = sorted(p for p in Path(sys.argv[1]).iterdir() if p.suffix.lower() in VIDEO_EXT)
rows = []
with tempfile.TemporaryDirectory() as td:
    for c in clips:
        d = Path(td)/c.stem; d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(c),
                        "-frames:v","5",str(d/"%03d.png")], check=True)
        vals = [flat_noise(load(p, dev)) for p in sorted(d.glob("*.png"))]
        rows.append((c.stem, float(np.nanmean(vals))))
print(f"{'clip':<26} {'flat-region noise sigma (8-bit levels)':>40}")
print("-"*68)
for n, v in sorted(rows, key=lambda r: r[1]):
    print(f"{n:<26} {v:40.4f}")
