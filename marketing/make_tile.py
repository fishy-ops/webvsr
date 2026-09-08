"""Promo tiles — the 440x280 asset that actually appears in store search.

The competitive grid makes the format obvious: every listing that ranks uses a
dramatic before/after with big brand type, not a subtle crop. So the "before"
here is a punishing but real case (CRF 34, the kind of bitrate a bad stream
actually hits) and the "after" is the real network plus the real Vivid preset,
which is a shipped feature rather than a photoshopped grade.

The diagonal seam is deliberate: at 440x280 a vertical split reads as two
unrelated thumbnails, while a diagonal reads as one image being transformed.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from brand3 import INK, GROUND, GREEN, DULL, font, track, track_width
from pipeline import read_frame, codec_downscale, bicubic_up, model_up, load_shipped, to_pil
from looks import grade

SS = 3                                   # supersample, then downsample for crisp type


def dramatic_pair(clip, frame, crf=34, look="vivid"):
    hr = read_frame(clip, frame)
    H, W = hr.shape[-2:]
    hr = hr[:, :, : (H // 4) * 4, : (W // 4) * 4]
    lr = codec_downscale(hr, 2, crf)
    before = bicubic_up(lr, 2)
    after = model_up(load_shipped(), lr)
    a = after[0].permute(1, 2, 0).numpy()
    a = grade(a, look)
    b = before[0].permute(1, 2, 0).numpy()
    n = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    to_img = lambda x: Image.fromarray((x[:n[0], :n[1]] * 255).round().astype(np.uint8))
    return to_img(b), to_img(a)


def compose(before, after, w, h, crop, brand_h, title_px, sub_px, out,
            label_px=None, pad=None):
    """Diagonal split, brand band along the bottom."""
    W, H = w * SS, h * SS
    x, y, cw, ch = crop
    bi = before.crop((x, y, x + cw, y + ch)).resize((W, H), Image.LANCZOS)
    ai = after.crop((x, y, x + cw, y + ch)).resize((W, H), Image.LANCZOS)

    im = bi.copy()
    # diagonal mask: after occupies the upper-right
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).polygon([(W * 0.30, 0), (W, 0), (W, H), (W * 0.62, H)], fill=255)
    im.paste(ai, (0, 0), m)
    d = ImageDraw.Draw(im, "RGBA")
    d.line([(W * 0.30, 0), (W * 0.62, H)], fill=GREEN + (255,), width=int(W * 0.010))

    lp = label_px or int(h * 0.052)
    P = pad or int(w * 0.032)
    fl = font(lp * SS, 800)
    def chip(cx, cy, text, bg, fg, right=False):
        tw = track_width(d, text, fl, 1.6 * SS)
        x0 = cx - tw - 26 * SS if right else cx
        d.rounded_rectangle([x0, cy, x0 + tw + 26 * SS, cy + lp * SS * 1.9],
                            lp * SS, fill=bg)
        track(d, (x0 + 13 * SS, cy + lp * SS * 0.42), text, fl, fg, 1.6 * SS)
    chip(P * SS, P * SS, "BEFORE", (10, 12, 11, 215), (255, 255, 255))
    chip(W - P * SS, P * SS, "AFTER", GREEN + (255,), (255, 255, 255), right=True)

    # brand band
    bh = brand_h * SS
    d.rectangle([0, H - bh, W, H], fill=(10, 16, 13, 232))
    icon = Image.open("brand/icon512.png").convert("RGBA")
    isz = int(bh * 0.52)
    icon = icon.resize((isz, isz), Image.LANCZOS)
    im.paste(icon, (P * SS, H - bh + (bh - isz) // 2), icon)
    tx = P * SS + isz + int(bh * 0.20)
    ft = font(title_px * SS, 720)
    fs = font(sub_px * SS, 520)
    d.text((tx, H - bh + bh * 0.20), "Crisp", font=ft, fill=(255, 255, 255))
    d.text((tx, H - bh + bh * 0.60), "AI video upscaler · free · on your GPU",
           font=fs, fill=(176, 190, 182))
    im.resize((w, h), Image.LANCZOS).save(out)
    print("wrote", out, f"{w}x{h}")


if __name__ == "__main__":
    Path("assets/promo").mkdir(parents=True, exist_ok=True)
    b, a = dramatic_pair("src/park_joy_1080p50.mp4", 20)
    print("source pair", b.size)
    # 440x280 small tile — the one that shows in search results
    compose(b, a, 440, 280, (300, 180, 1400, 890), brand_h=62,
            title_px=21, sub_px=11, out="assets/promo/tile_440x280.png")
    # 1400x560 marquee — what gets you considered for featuring
    compose(b, a, 1400, 560, (140, 60, 1780, 712), brand_h=104,
            title_px=38, sub_px=18, label_px=26, pad=34,
            out="assets/promo/marquee_1400x560.png")
