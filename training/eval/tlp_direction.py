"""Which direction is 'better' for tLP?

tlp = lpips(out_t, out_{t-1}) - lpips(gt_t, gt_{t-1}). A perfect reconstruction
scores exactly 0. Positive means the output changes more between frames than the
truth does (added flicker); negative means it changes less (temporal
over-smoothing). stratified_eval's header says "lower better, <=0 ideal", which
would make a blurrier output always score better -- and in the limit would rank a
constant grey frame best, since it never changes at all.

This tests that directly: score deliberately over-smoothed outputs alongside
bicubic and the model. If tLP falls as blur rises while every fidelity metric
falls too, "lower is better" rewards exactly the wrong thing.
"""
import sys, tempfile
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, "training/eval"); sys.path.insert(0, "training")
from stratified_eval import make_pair, load, Perceptual, load_model, run_model, bicubic, masked_psnr

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
perc = Perceptual(dev)
CLIPS = ["park_joy_1080p50", "old_town_cross_1080p50", "life_1080p30"]

def blur(x, sigma):
    if sigma <= 0:
        return x
    k = int(2 * round(3 * sigma) + 1)
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma * sigma)); g = g / g.sum()
    x = F.conv2d(x, g.view(1, 1, 1, k).expand(3, 1, 1, k), padding=(0, k // 2), groups=3)
    return F.conv2d(x, g.view(1, 1, k, 1).expand(3, 1, k, 1), padding=(k // 2, 0), groups=3)

model = load_model("checkpoints_c16/deployed_2x_c16.pth", 16, 2, dev)
variants = ["model", "bicubic", "blur1.0", "blur2.0", "blur4.0", "constant_grey"]
acc = {v: {"tlp": [], "psnr": []} for v in variants}

with tempfile.TemporaryDirectory() as td:
    for stem in CLIPS:
        hr_p, lr_p = make_pair(Path(f"/tank/webvsr/clips_busy/{stem}.mp4"),
                               Path(td) / stem, 2, 28, 12, 1024)
        prev = {v: None for v in variants}; prev_hr = None
        for hp, lp in zip(hr_p, lr_p):
            hr, lr = load(hp, dev), load(lp, dev)
            bc = bicubic(lr, 2)
            if bc.shape != hr.shape:
                continue
            outs = {
                "model": run_model(model, lr),
                "bicubic": bc,
                "blur1.0": blur(bc, 1.0), "blur2.0": blur(bc, 2.0), "blur4.0": blur(bc, 4.0),
                "constant_grey": torch.full_like(hr, 0.5),
            }
            for v, o in outs.items():
                if prev[v] is not None and prev_hr is not None:
                    a, b = perc.lp(o, prev[v]), perc.lp(hr, prev_hr)
                    if a is not None and b is not None:
                        acc[v]["tlp"].append(a - b)
                acc[v]["psnr"].append(masked_psnr(o, hr, torch.ones_like(hr[:, :1]).bool()))
                prev[v] = o
            prev_hr = hr
        print(f"  scored {stem}", flush=True)

print(f"\n{'variant':<16} {'tLP':>10} {'PSNR dB':>9}   reading")
print("-" * 62)
for v in variants:
    t = sum(acc[v]["tlp"]) / max(len(acc[v]["tlp"]), 1)
    p = sum(acc[v]["psnr"]) / max(len(acc[v]["psnr"]), 1)
    print(f"{v:<16} {t:10.5f} {p:9.3f}")
