"""Does tLP scale with scene motion?

tLP = lpips(out_t, out_{t-1}) - lpips(gt_t, gt_{t-1}). Both terms grow with
motion, so their difference plausibly does too -- and if it does, averaging tLP
over clips with different motion is dominated by the fast ones. That would
explain §24 (benchmark clips are high-motion nature, held-out includes static
conference video) and §27 (the shipped/webcodec flicker gap is 5% at CRF 20 and
82% at CRF 28) without either being a real property of the models.

lpips(gt_t, gt_{t-1}) is itself the motion measure, in the metric's own units.
"""
import sys, tempfile
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "training/eval"); sys.path.insert(0, "training")
from stratified_eval import make_pair, load, Perceptual, load_model, run_model, VIDEO_EXT

dev = torch.device("cuda")
perc = Perceptual(dev)
models = {
    "shipped": load_model("checkpoints_c16/best_phase2.pth", 16, 2, dev),
    "webcodec": load_model("/tank/webvsr/ckpt_webcodec_2x_c16/best_phase2.pth", 16, 2, dev),
}
CRF = int(sys.argv[1]) if len(sys.argv) > 1 else 28
clips = sorted(p for p in Path("/tank/webvsr/clips_busy").iterdir()
               if p.suffix.lower() in VIDEO_EXT)

rows = []
with tempfile.TemporaryDirectory() as td:
    for clip in clips:
        hr_p, lr_p = make_pair(clip, Path(td) / clip.stem, 2, CRF, 10, 1024)
        prev = {k: None for k in models}; prev_hr = None
        gtm, tl = {k: [] for k in models}, {k: [] for k in models}
        motion = []
        for hp, lp in zip(hr_p, lr_p):
            hr, lr = load(hp, dev), load(lp, dev)
            outs = {k: run_model(m, lr) for k, m in models.items()}
            if any(o.shape != hr.shape for o in outs.values()):
                continue
            if prev_hr is not None:
                lo_gt = perc.lp(hr, prev_hr)
                if lo_gt is not None:
                    motion.append(lo_gt)
                    for k, o in outs.items():
                        if prev[k] is not None:
                            lo = perc.lp(o, prev[k])
                            if lo is not None:
                                tl[k].append(lo - lo_gt)
            for k, o in outs.items():
                prev[k] = o
            prev_hr = hr
        if not motion:
            continue
        rows.append({"clip": clip.stem, "motion": float(np.mean(motion)),
                     **{k: float(np.mean(tl[k])) for k in models}})
        print(f"  {clip.stem}", flush=True)

rows.sort(key=lambda r: r["motion"])
print(f"\nCRF {CRF} — clips sorted by GT temporal LPIPS (motion)\n")
print(f"{'clip':<28} {'motion':>8} " + "".join(f"{k+' tLP':>16}" for k in models))
print("-" * 76)
for r in rows:
    print(f"{r['clip']:<28} {r['motion']:8.4f} " + "".join(f"{r[k]:16.5f}" for k in models))

m = np.array([r["motion"] for r in rows])
print()
for k in models:
    t = np.abs(np.array([r[k] for r in rows]))
    print(f"corr(motion, |tLP|) for {k:<10} = {np.corrcoef(m, t)[0,1]:+.3f}")
gap = np.array([abs(r["shipped"]) - abs(r["webcodec"]) for r in rows])
print(f"corr(motion, shipped-vs-webcodec |tLP| gap) = {np.corrcoef(m, gap)[0,1]:+.3f}")
