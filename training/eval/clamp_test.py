"""How much of the model's advantage is just ringing suppression?

Section 13 found the model's large win on render clips is that bicubic *rings*
there -- 1.9-2.4x the ground truth's edge gradient energy, against <=1.0 on
camera footage. Ringing is bicubic's negative kernel lobes overshooting past
values that were in the source.

An anti-ringing clamp fixes that for almost nothing: an upscaled pixel cannot
legitimately fall outside the min-max of the source neighbourhood it came from,
so clamp it there. If that closes most of the gap, a large part of what the
33k-parameter network buys is available at roughly zero GPU cost, and the
content-gate strategy changes shape.
"""
import sys, tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "training/eval"); sys.path.insert(0, "training")
from stratified_eval import (make_pair, load, Perceptual, load_model, run_model,
                             bicubic, masked_psnr, masks_for, bucket_thresholds,
                             sharpness_ratio)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
perc = Perceptual(dev)

RENDER = ["bistro_30s", "chess_30s", "locomotive_30s"]
CAMERA = ["park_joy_1080p50", "life_1080p30", "blue_sky_1080p25", "crowd_run_1080p50"]

def anti_ring(sr, lr, scale):
    """Clamp each output pixel to the min-max of its source neighbourhood.

    3x3 in LR space via max-pool (and min-pool as -maxpool(-x)), then nearest-
    upsampled to HR so every output pixel sees the bounds of the source region
    that produced it. This is the standard shader anti-ringing clamp.
    """
    hi = F.max_pool2d(lr, 3, stride=1, padding=1)
    lo = -F.max_pool2d(-lr, 3, stride=1, padding=1)
    hi = F.interpolate(hi, scale_factor=scale, mode="nearest")
    lo = F.interpolate(lo, scale_factor=scale, mode="nearest")
    if hi.shape != sr.shape:
        hi = F.interpolate(hi, size=sr.shape[2:], mode="nearest")
        lo = F.interpolate(lo, size=sr.shape[2:], mode="nearest")
    return torch.max(torch.min(sr, hi), lo)

deployed = load_model("checkpoints_c16/deployed_2x_c16.pth", 16, 2, dev)
candidate = load_model("checkpoints_c16/best_phase2.pth", 16, 2, dev)

def run_group(name, stems):
    acc = {k: {"dists": [], "tex": [], "sharp": []}
           for k in ("bicubic", "bicubic+clamp", "deployed", "deployed+clamp", "candidate")}
    with tempfile.TemporaryDirectory() as td:
        allhr = []
        pairs = []
        for stem in stems:
            hp, lp = make_pair(Path(f"/tank/webvsr/clips_busy/{stem}.mp4"),
                               Path(td) / stem, 2, 28, 10, 1024)
            pairs.append((hp, lp)); allhr += hp
        lo_t, hi_t = bucket_thresholds(allhr, dev)
        for hp, lp in pairs:
            for h_, l_ in zip(hp, lp):
                hr, lr = load(h_, dev), load(l_, dev)
                bc = bicubic(lr, 2)
                if bc.shape != hr.shape:
                    continue
                m = masks_for(hr, lo_t, hi_t)
                dep = run_model(deployed, lr)
                outs = {
                    "bicubic": bc,
                    "bicubic+clamp": anti_ring(bc, lr, 2),
                    "deployed": dep,
                    "deployed+clamp": anti_ring(dep, lr, 2),
                    "candidate": run_model(candidate, lr),
                }
                for k, o in outs.items():
                    d = perc.d(o, hr)
                    if d is not None:
                        acc[k]["dists"].append(d)
                    acc[k]["tex"].append(masked_psnr(o, hr, m["texture"]))
                    acc[k]["sharp"].append(sharpness_ratio(o, hr, m["edge"]))
        print(f"  scored {name}", flush=True)
    return acc

for gname, stems in (("RENDER clips", RENDER), ("CAMERA clips", CAMERA)):
    acc = run_group(gname, stems)
    base = np.mean(acc["bicubic"]["dists"])
    print(f"\n=== {gname} ===")
    print(f"{'variant':<16} {'DISTS':>7} {'vs bic':>8} {'texPSNR':>8} {'edge sharp':>11}")
    for k, v in acc.items():
        d = np.mean(v["dists"])
        print(f"{k:<16} {d:7.4f} {(1-d/base)*100:+7.1f}% {np.mean(v['tex']):8.3f} "
              f"{np.mean(v['sharp']):11.4f}")
