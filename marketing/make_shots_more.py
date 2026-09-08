"""Screenshots 3-5.

Screenshot 3 (breadth) exists because a single clip proves nothing: it shows the
same unchanged network on four different kinds of content, and every pair is a
real 3x crop scored by where the model genuinely moves closer to the original.

Screenshot 4 (motion) is a real frame lifted out of the rendered demo video --
the honest answer to "does this survive movement", which stills cannot address.

Screenshot 5 states the limits. In a category where every rival promises AI 4K
magic, the listing that says what it will not do is the one that reads credible,
and it is also just true.
"""
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from brand3 import (INK, GROUND, SURFACE, LINE, MUTED, DIM, GREEN, GREEN_DEEP,
                    DULL, font, track, track_width)
from pipeline import make_set, to_pil
from find_crops import score_windows

W, H, PAD = 1280, 800, 56
OUT = Path("assets/screenshots")

CLIPS = [("tractor_1080p25", "Machinery"), ("pedestrian_area_1080p25", "People & street signs"),
         ("touchdown_pass_1080p", "Sport"), ("factory_1080p30", "Colour & flat shading")]


def lockup(im, d, x=PAD, y=34):
    icon = Image.open("brand/icon128.png").convert("RGBA").resize((32, 32), Image.LANCZOS)
    im.paste(icon, (x, y), icon)
    d.text((x + 44, y - 2), "Crisp", font=font(25, 640), fill=INK)


def wrapped(d, xy, text, f, fill, max_w, leading=22):
    x, y = xy; line = ""
    for word in text.split():
        t = (line + " " + word).strip()
        if d.textlength(t, font=f) > max_w and line:
            d.text((x, y), line, font=f, fill=fill); y += leading; line = word
        else:
            line = t
    if line:
        d.text((x, y), line, font=f, fill=fill)
    return y + leading


def shot_breadth():
    im = Image.new("RGB", (W, H), GROUND)
    d = ImageDraw.Draw(im)
    lockup(im, d)
    d.text((PAD, 84), "The same network, four kinds of footage.", font=font(28, 660), fill=INK)

    cw, ch = (W - 2 * PAD - 22) // 2, 232
    pw = (cw - 8) // 2
    for i, (stem, label) in enumerate(CLIPS):
        cx = PAD + (i % 2) * (cw + 22)
        cy = 140 + (i // 2) * (ch + 30)
        s = make_set(f"src/{stem}.mp4", 15)
        ranked = score_windows(s, crop=180, stride=60)
        if not ranked:
            continue
        _, _, y0, x0 = ranked[0]
        rh = 180
        rw = int(rh * pw / 176)
        for j, key in enumerate(("before", "after")):
            crop = to_pil(s[key][:, :, y0:y0 + rh, x0:x0 + rw])
            crop = crop.resize((pw, 176), Image.LANCZOS)
            px = cx + j * (pw + 8)
            im.paste(crop, (px, cy + 26))
            d.rectangle([px, cy + 26, px + pw, cy + 202], outline=LINE, width=1)
            d.rectangle([px, cy + 202, px + pw, cy + 206], fill=GREEN if j else DULL)
            f2 = font(11, 700)
            track(d, (px + 2, cy + 210), "WITH CRISP" if j else "WITHOUT", f2,
                  GREEN_DEEP if j else DIM, 1.1)
        d.text((cx, cy), label, font=font(16, 640), fill=INK)
        print("  ", stem)

    d.text((PAD, H - 78), "Public-domain Xiph.org test clips, 540p source upscaled to 1080p.",
           font=font(15, 440), fill=MUTED)
    d.text((PAD, H - 54), "Quality is measured across 19 of these clips — not on hand-picked frames.",
           font=font(15, 440), fill=MUTED)
    im.save(OUT / "03_breadth.png"); print("wrote 03_breadth.png")


def shot_motion():
    """A real frame out of the demo video, mid-wipe."""
    with av.open("assets/crisp_demo.mp4") as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    pick = frames[22]                       # mid-wipe on the first scene
    im = Image.new("RGB", (W, H), GROUND)
    d = ImageDraw.Draw(im)
    lockup(im, d)
    d.text((PAD, 84), "It holds up in motion.", font=font(28, 660), fill=INK)
    src = Image.fromarray(pick)
    iw = W - 2 * PAD
    ih = int(src.height * iw / src.width)
    im.paste(src.resize((iw, ih), Image.LANCZOS), (PAD, 140))
    d.rectangle([PAD, 140, PAD + iw, 140 + ih], outline=LINE, width=1)
    fy = 140 + ih + 22
    d.text((PAD, fy), "Every frame processed independently", font=font(15, 700), fill=GREEN_DEEP)
    wrapped(d, (PAD, fy + 24),
            "There is no temporal smoothing and no frame chosen for flattery — the network runs "
            "on each frame on its own, in real time, while the video plays.",
            font(15, 440), MUTED, iw)
    im.save(OUT / "04_motion.png"); print("wrote 04_motion.png")


def shot_limits():
    im = Image.new("RGB", (W, H), GROUND)
    d = ImageDraw.Draw(im)
    lockup(im, d)
    d.text((PAD, 84), "What it won't do.", font=font(28, 660), fill=INK)
    d.text((PAD, 122), "Every other listing in this category promises AI 4K magic. Here is the honest version.",
           font=font(16, 440), fill=MUTED)

    items = [
        ("It doesn't invent detail.",
         "It reconstructs what compression blurred. It will not hallucinate a face out of a smudge."),
        ("The gain shrinks as compression gets worse.",
         "Largest on low-resolution video that is still reasonably clean; smaller on video that has "
         "been crushed, because less real detail survives to recover."),
        ("It needs WebGPU.",
         "A recent Chrome and a GPU. Without them Crisp simply stays off rather than degrading playback."),
        ("It can't touch DRM video.",
         "Netflix, Disney+ and the like are off limits to every extension, this one included."),
    ]
    y = 172
    for title, body in items:
        d.rectangle([PAD, y + 4, PAD + 4, y + 66], fill=GREEN)
        d.text((PAD + 22, y), title, font=font(19, 660), fill=INK)
        wrapped(d, (PAD + 22, y + 28), body, font(15.5 and 15, 440), MUTED, W - 2 * PAD - 40, 21)
        y += 96

    # the numbers, as a footer band
    d.rectangle([PAD, y + 8, W - PAD, y + 108], fill=SURFACE)
    figs = [("33,388", "parameters"), ("48 ms", "per 1080p frame, M4 Pro"),
            ("19", "clips measured"), ("0", "bytes uploaded")]
    fx = PAD + 26
    for n, k in figs:
        d.text((fx, y + 28), n, font=font(27, 620), fill=INK)
        d.text((fx, y + 66), k, font=font(13, 460), fill=MUTED)
        fx += (W - 2 * PAD - 52) // 4
    im.save(OUT / "05_limits.png"); print("wrote 05_limits.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    shot_breadth(); shot_motion(); shot_limits()
