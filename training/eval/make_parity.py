"""Reference output from PyTorch for the engine-parity check.

Every quality number in RESEARCH.md is measured in PyTorch. Users get the WGSL
engine. Nothing has ever checked that the two agree end to end -- a mismatch in
preprocessing (the mean subtraction, the img_range scale), in the pixel-shuffle
indexing, or in the final colour conversion would be invisible to every
evaluation here and visible to every user.
"""
import sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, "training/eval"); sys.path.insert(0, "training")
from stratified_eval import make_pair, load, load_model, run_model

OUT = Path("/tank/webvsr/parity"); OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with tempfile.TemporaryDirectory() as td:
    hr_p, lr_p = make_pair(Path("/tank/webvsr/clips_busy/life_1080p30.mp4"),
                           Path(td) / "life", 2, 28, 2, 512)
    lr = load(lr_p[0], dev)
    # crop to a clean 256x256 LR so the transfer is small and the shapes are tidy
    lr = lr[:, :, :256, :256].contiguous()
    m = load_model("checkpoints_c16/best_phase2.pth", 16, 2, dev)
    sr = run_model(m, lr)

def save(t, name):
    a = (t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(a).save(OUT / name)
    return a.shape

print("lr ", save(lr, "lr.png"))
print("sr ", save(sr, "sr_pytorch.png"))
print("wrote", OUT)
