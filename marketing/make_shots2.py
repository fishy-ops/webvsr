"""Store screenshots, bright rebrand.

Same discipline as before -- both panels always show the identical region of the
identical frame at the same scale, and the scale is stated on the face of the
image -- but on a light ground with the spectrum carrying the accent, because
the amber-on-near-black system was scrapped.
"""
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from brand2 import (INK, GROUND, SURFACE, LINE, MUTED, DIM, SPECTRUM, ACCENT,
                    font, track, track_width, gradient_bar)
from pipeline import make_set, to_pil

W, H, PAD = 1280, 800, 56
OUT = Path("assets/screenshots")


def load():
    c = Path("src/_otc.pt")
    if c.exists():
        return torch.load(c, weights_only=False)
    s = make_set("src/old_town_cross_1080p50.mp4", 20)
    torch.save(s, c)
    return s


def lockup(im, d, x=PAD, y=36):
    icon = Image.open("brand/icon128.png").convert("RGBA").resize((30, 30), Image.LANCZOS)
    im.paste(icon, (x, y), icon)
    d.text((x + 41, y - 1), "Crisp", font=font(24, 640), fill=INK)


def chip(d, x, y, text, bg, fg, size=14):
    f = font(size, 680)
    tw = track_width(d, text, f, 1.3)
    d.rounded_rectangle([x, y, x + tw + 24, y + 28], 14, fill=bg)
    track(d, (x + 12, y + 6), text, f, fg, 1.3)


def wrapped(d, xy, text, f, fill, max_w, leading=22):
    x, y = xy
    line = ""
    for word in text.split():
        t = (line + " " + word).strip()
        if d.textlength(t, font=f) > max_w and line:
            d.text((x, y), line, font=f, fill=fill); y += leading; line = word
        else:
            line = t
    if line:
        d.text((x, y), line, font=f, fill=fill)
    return y + leading


def pair(s, region, scale, headline, note, out, tag_l="WITHOUT", tag_r="WITH CRISP"):
    x, y, rw, rh = region
    pw = (W - 2 * PAD - 24) // 2
    ph = int(rh * scale)
    im = Image.new("RGB", (W, H), GROUND)
    d = ImageDraw.Draw(im)
    lockup(im, d)
    d.text((PAD, 84), headline, font=font(27, 640), fill=INK)

    top = 136
    for i, key in enumerate(("before", "after")):
        px = PAD + i * (pw + 24)
        src = to_pil(s[key][:, :, y:y + rh, x:x + rw])
        src = src.resize((int(rw * scale), ph), Image.NEAREST if scale > 1 else Image.LANCZOS)
        im.paste(src.crop((0, 0, pw, ph)), (px, top))
        if i:
            # spectrum rule under the "with" panel: the accent, used once
            im.paste(gradient_bar(pw, 4), (px, top + ph))
        d.rectangle([px, top, px + pw, top + ph], outline=LINE, width=1)
        chip(d, px + 13, top + 13, tag_r if i else tag_l,
             (255, 255, 255) if i else (255, 255, 255), INK if i else MUTED)

    fy = top + ph + 20
    d.text((PAD, fy), "100% — actual pixels" if scale == 1 else f"{int(scale*100)}% scale",
           font=font(15, 700), fill=ACCENT)
    wrapped(d, (PAD, fy + 24), note, font(15, 440), MUTED, W - 2 * PAD)
    im.save(OUT / out)
    print("wrote", out)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    s = load()
    pair(s, (700, 452, 596, 530), 1.0,
         "The same frame, the same instant.",
         "Public-domain test footage played at 540p — the resolution a browser actually "
         "receives — then upscaled to 1080p. Left is what your browser shows today. "
         "Right is Crisp. No zoom, no crop, no second sharpening pass.",
         "01_proof.png")
    pair(s, (1112, 715, 199, 177), 3.0,
         "Closer, on the same pixels.",
         "The identical 199x177 patch from both frames, enlarged 3x with nearest-neighbour, "
         "so you are looking at real pixels and not a smoothing filter. Letter edges tighten, "
         "the roofline separates, and the windows below stop bleeding into the wall.",
         "02_detail.png")
