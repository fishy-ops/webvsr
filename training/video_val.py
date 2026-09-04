"""Validation on real video pairs, so selection can see what the benchmark sees.

Two measured problems this replaces:

  §22 -- the still-frame validation set reported a 4.8% DISTS gain on a run the
         15-clip benchmark scored as identical, and checkpoint selection runs on
         that signal.
  §11 -- flicker is the only advantage that transfers across content types, and
         a still-frame validation set cannot measure it, so no run has ever been
         able to select on it.

Clips are HELD OUT from the benchmark on purpose. Selecting on clips_busy would
be selecting on the test set, which would make every number in RESEARCH.md
self-confirming.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class VideoPairVal:
    """Consecutive (prev, curr) LR/HR pairs, preloaded — the set is ~5 MB."""

    def __init__(self, root="/tank/webvsr/val_video", device="cpu"):
        self.root = Path(root)
        self.device = device
        man = json.loads((self.root / "manifest.json").read_text())
        self.items = []
        for m in man:
            d = self.root / m["clip"]
            lp, lc, hp, hc = (
                self._load(d / sub / f"{idx:03d}.png")
                for sub, idx in (("lr", m["prev"]), ("lr", m["curr"]),
                                 ("hr", m["prev"]), ("hr", m["curr"]))
            )
            # ffmpeg derives HR and LR widths independently, so they can round
            # apart -- 456 LR against 910 HR, where 456*2 = 912. Left unhandled
            # that silently drops every pair. Crop both to the largest size
            # where HR is exactly 2x LR.
            h = min(lp.shape[2], hp.shape[2] // 2)
            w = min(lp.shape[3], hp.shape[3] // 2)
            lp, lc = lp[:, :, :h, :w], lc[:, :, :h, :w]
            hp, hc = hp[:, :, :2 * h, :2 * w], hc[:, :, :2 * h, :2 * w]
            self.items.append((lp, lc, hp, hc))
        if not self.items:
            raise ValueError(f"no usable pairs in {self.root}")

    def _load(self, p):
        a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)

    def __len__(self):
        return len(self.items)


@torch.no_grad()
def validate_video(model, vset, device, perc, depth=None):
    """Return (psnr, dists, abs_tlp) over the held-out video pairs.

    abs_tlp is |output flicker - source flicker|. §10 established that 0 is the
    target rather than minus infinity: 'lower tLP' is optimised by a constant
    grey frame, so the deviation is what selection should minimise.
    """
    model.eval()
    psnrs, dists, tlps = [], [], []
    for lr_p, lr_c, hr_p, hr_c in vset.items:
        lr_p, lr_c = lr_p.to(device), lr_c.to(device)
        hr_p, hr_c = hr_p.to(device), hr_c.to(device)
        run = (lambda x: model(x)) if depth is None else (lambda x: model(x, depth=depth))
        sr_p = run(lr_p).clamp(0, 1)
        sr_c = run(lr_c).clamp(0, 1)
        if sr_c.shape != hr_c.shape:
            # Loud: a silent skip here returns None for every metric and reads
            # exactly like "the model produced nothing", which cost a debugging
            # cycle the first time.
            raise ValueError(f"shape mismatch: model gave {tuple(sr_c.shape)}, "
                             f"ground truth is {tuple(hr_c.shape)}")
        mse = torch.mean((sr_c - hr_c) ** 2)
        psnrs.append(float(10 * torch.log10(1.0 / mse.clamp(min=1e-12))))
        d = perc.d(sr_c, hr_c)
        if d is not None:
            dists.append(d)
        lo_out, lo_gt = perc.lp(sr_c, sr_p), perc.lp(hr_c, hr_p)
        if lo_out is not None and lo_gt is not None:
            tlps.append(abs(lo_out - lo_gt))
    m = lambda xs: (sum(xs) / len(xs)) if xs else None
    return m(psnrs), m(dists), m(tlps)
