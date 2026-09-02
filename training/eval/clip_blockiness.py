"""Is a clip's HR reference already compressed?

H.264 quantises on an 8x8 grid, so re-encoded material carries excess gradient
energy exactly on block boundaries relative to the interior. Pristine capture
does not. The ratio is a cheap, direct test of whether an evaluation clip's
"ground truth" has itself been through a codec -- which decides whether a
measured win against bicubic is a real win or an artifact of a soft reference.
"""
import sys, tempfile, subprocess
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "training/eval")
sys.path.insert(0, "training")
from stratified_eval import load, grad_mag, VIDEO_EXT

dev = "cuda" if torch.cuda.is_available() else "cpu"

def blockiness(img):
    g = grad_mag(img)[0, 0]
    h, w = g.shape
    ys = torch.arange(h, device=g.device)
    xs = torch.arange(w, device=g.device)
    on_y = ((ys % 8) == 0).view(-1, 1)
    on_x = ((xs % 8) == 0).view(1, -1)
    on = (on_y | on_x)
    # trim borders, which are edges for reasons unrelated to coding
    m = torch.zeros_like(on); m[4:h-4, 4:w-4] = True
    edge = g[on & m]; inner = g[(~on) & m]
    return float(edge.mean() / (inner.mean() + 1e-9))

clips = sorted(p for p in Path(sys.argv[1]).iterdir() if p.suffix.lower() in VIDEO_EXT)
print(f"{'clip':<26} {'block ratio':>12}   (>1 = energy concentrated on the 8x8 grid)")
print("-" * 66)
rows = []
with tempfile.TemporaryDirectory() as td:
    for c in clips:
        d = Path(td) / c.stem; d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(c),
                        "-frames:v", "6", str(d / "%03d.png")], check=True)
        vals = [blockiness(load(p, dev)) for p in sorted(d.glob("*.png"))]
        rows.append((c.stem, float(np.mean(vals))))
for name, v in sorted(rows, key=lambda r: -r[1]):
    print(f"{name:<26} {v:12.4f}")
