"""Pick showcase crops by where the model genuinely gets closer to the truth.

Ranking by raw |after - before| would find wherever the model changed the most,
which includes places it changed things for the worse. The score here is the
honest one: how much of the bicubic error against the original the model
actually removes, inside each candidate window.

    gain = mean|before - hr| - mean|after - hr|

A positive gain means the crop is closer to the real frame with the extension on.
Crops are also required to hold real detail (a flat sky scores a big relative
gain on nothing), so windows below a gradient floor are rejected outright.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import make_set, to_pil

CROP = 420          # source pixels; shown at 100% in a 1280x800 screenshot
STRIDE = 60


def gradient(x):
    g = x.mean(1, keepdim=True)
    gx = (g[:, :, :, 1:] - g[:, :, :, :-1]).abs().mean()
    gy = (g[:, :, 1:, :] - g[:, :, :-1, :]).abs().mean()
    return (gx + gy).item() / 2


def score_windows(s, crop=CROP, stride=STRIDE):
    hr, be, af = s["hr"], s["before"], s["after"]
    e_before = (be - hr).abs().mean(1, keepdim=True)
    e_after = (af - hr).abs().mean(1, keepdim=True)
    gain_map = e_before - e_after                      # positive = model is better
    k = torch.ones(1, 1, crop, crop) / (crop * crop)
    gain = F.conv2d(gain_map, k, stride=stride)[0, 0]
    out = []
    H, W = hr.shape[-2:]
    for i in range(gain.shape[0]):
        for j in range(gain.shape[1]):
            y, x = i * stride, j * stride
            if y + crop > H or x + crop > W:
                continue
            win = hr[:, :, y:y + crop, x:x + crop]
            det = gradient(win)
            if det < 0.02:            # flat region: a big relative gain on nothing
                continue
            out.append((gain[i, j].item(), det, y, x))
    out.sort(reverse=True)
    return out


if __name__ == "__main__":
    outdir = Path("assets/crops")
    outdir.mkdir(parents=True, exist_ok=True)
    clips = sorted(Path("src").glob("*.mp4"))
    print(f"{'clip':<26}{'gain':>10}{'detail':>9}   crop")
    print("-" * 62)
    best = {}
    for clip in clips:
        s = make_set(clip, 20)
        ranked = score_windows(s)
        if not ranked:
            print(f"{clip.stem:<26}  (no textured window)")
            continue
        g, det, y, x = ranked[0]
        best[clip.stem] = (g, y, x)
        print(f"{clip.stem:<26}{g:+10.5f}{det:9.3f}   ({x},{y})")
        for tag in ("before", "after", "hr"):
            to_pil(s[tag][:, :, y:y + CROP, x:x + CROP]).save(
                outdir / f"{clip.stem}_{tag}.png")
    print("\ncrops written to assets/crops/ at 100% zoom, no resampling")
