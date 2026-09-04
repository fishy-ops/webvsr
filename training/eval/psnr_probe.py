"""Explain the 9 dB PSNR spread in the video validation set (§23).

Overall PSNR came out 33.83 / 42.57 / 45.21 for three checkpoints the 15-clip
benchmark separates by ~0.3 dB. Either one clip dominates the mean, or something
is wrong with how PSNR is being computed here.
"""
import sys
from collections import defaultdict
import torch
sys.path.insert(0, "training"); sys.path.insert(0, "training/eval")
from video_val import VideoPairVal
from stratified_eval import load_model
import json
from pathlib import Path

dev = torch.device("cuda")
v = VideoPairVal()
man = json.loads((Path("/tank/webvsr/val_video") / "manifest.json").read_text())
clips = [m["clip"] for m in man]

models = {
    "deployed": "checkpoints_c16/deployed_2x_c16.pth",
    "shipped": "checkpoints_c16/best_phase2.pth",
    "webcodec": "/tank/webvsr/ckpt_webcodec_2x_c16/best_phase2.pth",
}
print(f"{'clip':<28} " + "".join(f"{k:>12}" for k in models))
print("-" * 68)
per = {k: defaultdict(list) for k in models}
for name, ck in models.items():
    m = load_model(ck, 16, 2, dev)
    with torch.no_grad():
        for (lp, lc, hp, hc), clip in zip(v.items, clips):
            lc, hc = lc.to(dev), hc.to(dev)
            sr = m(lc).clamp(0, 1)
            mse = torch.mean((sr - hc) ** 2)
            per[name][clip].append(float(10 * torch.log10(1.0 / mse.clamp(min=1e-12))))

for clip in dict.fromkeys(clips):
    row = f"{clip:<28} "
    for k in models:
        vals = per[k][clip]
        row += f"{sum(vals)/len(vals):12.2f}"
    print(row)

print("\nmean over all pairs (what §23 reported):")
row = " " * 29
for k in models:
    allv = [x for c in per[k] for x in per[k][c]]
    row += f"{sum(allv)/len(allv):12.2f}"
print(row)
