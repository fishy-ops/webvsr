"""
Visual evidence for the numeric table.

Every benchmark writes side-by-side crops of the busiest regions it can find,
because the whole point of the stratified harness is that a mean cannot show
you what texture reconstruction actually looks like. These strips live under
/tank/webvsr/evidence and are never deleted -- they are the record that a
training run did or did not earn its compute.

Crops are chosen by texture-mask density so the comparison lands on the
content the model is worst at, not on a convenient patch of sky.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

GUTTER = 4
GUTTER_RGB = (128, 128, 128)


def _to_hwc_uint8(t):
    """(1,3,H,W) or (3,H,W) float 0..1 -> (H,W,3) uint8 on CPU."""
    if t.dim() == 4:
        t = t[0]
    a = (t.detach().cpu().clamp(0, 1) * 255.0).round().to(torch.uint8)
    return a.permute(1, 2, 0).numpy()


def _pick_windows(mask, crop, top_k):
    """Top-k highest-density crop x crop windows, kept spatially distinct.

    Scans on a stride of crop//2, then after each winner suppresses every
    candidate within `crop` pixels on both axes -- otherwise the k winners are
    all neighbours of a single hotspot and show the same pixels k times.
    """
    _, _, H, W = mask.shape
    ch, cw = min(crop, H), min(crop, W)
    stride = max(1, min(ch, cw) // 2)

    ys = list(range(0, max(1, H - ch + 1), stride))
    xs = list(range(0, max(1, W - cw + 1), stride))
    if ys[-1] != H - ch:
        ys.append(H - ch)
    if xs[-1] != W - cw:
        xs.append(W - cw)

    m = mask.float()
    # Integral image makes every window sum O(1) regardless of how many we scan.
    ii = F.pad(m.cumsum(2).cumsum(3), (1, 0, 1, 0))[0, 0]

    cands = []
    for y in ys:
        for x in xs:
            s = (ii[y + ch, x + cw] - ii[y, x + cw]
                 - ii[y + ch, x] + ii[y, x]).item()
            cands.append((s, y, x))
    cands.sort(key=lambda c: -c[0])

    picked = []
    for s, y, x in cands:
        if len(picked) >= top_k:
            break
        if any(abs(y - py) < ch and abs(x - px) < cw for _, py, px in picked):
            continue
        picked.append((s, y, x))
    return [(y, x, ch, cw) for _, y, x in picked]


def _label(draw, x, y, text):
    """White text with a 1px black shadow, legible on bright and dark crops."""
    font = ImageFont.load_default()
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def save_comparison(out_dir, clip_name, frame_idx, crf, hr, outputs,
                    texture_mask, crop=128, top_k=2):
    """Write side-by-side crops: ground truth, then each model in dict order.

    hr:           (1,3,H,W) float 0..1, any device
    outputs:      {name: tensor of the same shape/device}; may be empty
    texture_mask: (1,1,H,W) bool

    Returns the list of Paths written (possibly fewer than top_k if the image
    cannot yield that many distinct windows).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [("GT", hr)] + list(outputs.items())
    windows = _pick_windows(texture_mask, crop, top_k)

    written = []
    for n, (y, x, ch, cw) in enumerate(windows):
        tiles = []
        for name, t in panels:
            a = _to_hwc_uint8(t)
            # A model output may differ in size from hr; crop what exists.
            yy, xx = min(y, a.shape[0] - 1), min(x, a.shape[1] - 1)
            tile = a[yy:yy + ch, xx:xx + cw]
            if tile.shape[0] != ch or tile.shape[1] != cw:
                pad = np.zeros((ch, cw, 3), dtype=np.uint8)
                pad[:tile.shape[0], :tile.shape[1]] = tile
                tile = pad
            tiles.append((name, tile))

        total_w = len(tiles) * cw + GUTTER * (len(tiles) - 1)
        canvas = np.zeros((ch, total_w, 3), dtype=np.uint8)
        canvas[:, :] = GUTTER_RGB
        for i, (_, tile) in enumerate(tiles):
            ox = i * (cw + GUTTER)
            canvas[:, ox:ox + cw] = tile

        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        for i, (name, _) in enumerate(tiles):
            _label(draw, i * (cw + GUTTER) + 3, 3, name)

        path = out_dir / f"{clip_name}_f{frame_idx:04d}_crf{crf}_crop{n}.png"
        img.save(path)
        written.append(path)

    return written
