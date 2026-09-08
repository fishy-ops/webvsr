"""The store screenshots.

One rule throughout: both panels always show the IDENTICAL region of the
IDENTICAL frame, at the same scale. The only thing that differs is whether the
network ran. A split-down-the-middle version was tried first and rejected --
across a seam the two sides show different parts of the scene, so there is
nothing to actually compare, and it reads as one continuous photo.

Scale is stated on every shot. At 100% a 540p->1080p difference is real but
quiet, which is the truth; the magnified shot says "3x" on its face rather than
quietly enlarging and hoping nobody asks.
"""
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from brand import GROUND, SURFACE, ACCENT, TEXT, MUTED, font, track, track_width
from pipeline import make_set, to_pil

W, H = 1280, 800
PAD = 56
OUT = Path("assets/screenshots")


def load(src="src/old_town_cross_1080p50.mp4", frame=20):
    c = Path("src/_otc.pt")
    return torch.load(c, weights_only=False) if c.exists() else make_set(src, frame)


def lockup(d, x=PAD, y=38):
    d.rounded_rectangle([x, y, x + 30, y + 30], 8, fill=ACCENT)
    d.text((x + 8, y + 4), "C", font=font(21, 800), fill=GROUND)
    d.text((x + 42, y + 2), "Crisp", font=font(25, 700), fill=TEXT)


def label(d, x, y, text, bg, fg, size=15):
    f = font(size, 700)
    tw = track_width(d, text, f, 1.4)
    d.rounded_rectangle([x, y, x + tw + 26, y + 30], 15, fill=bg)
    track(d, (x + 13, y + 7), text, f, fg, 1.4)


def wrapped(d, xy, text, f, fill, max_w, leading=22):
    x, y = xy
    line = ""
    for word in text.split():
        trial = (line + " " + word).strip()
        if d.textlength(trial, font=f) > max_w and line:
            d.text((x, y), line, font=f, fill=fill); y += leading; line = word
        else:
            line = trial
    if line:
        d.text((x, y), line, font=f, fill=fill)
    return y + leading


def pair_shot(s, region, scale, headline, note, out, tag_l="WITHOUT", tag_r="WITH CRISP"):
    """Two panels, same region, same scale."""
    x, y, rw, rh = region
    panel_w = (W - 2 * PAD - 24) // 2
    panel_h = int(rh * scale)
    crop = lambda k: to_pil(s[k][:, :, y:y + rh, x:x + rw]).resize(
        (int(rw * scale), panel_h), Image.NEAREST if scale > 1 else Image.LANCZOS)

    im = Image.new("RGB", (W, H), GROUND)
    d = ImageDraw.Draw(im)
    lockup(d)
    d.text((PAD, 88), headline, font=font(26, 600), fill=TEXT)

    top = 138
    for i, key in enumerate(("before", "after")):
        px = PAD + i * (panel_w + 24)
        im.paste(crop(key).crop((0, 0, panel_w, panel_h)), (px, top))
        d.rectangle([px, top, px + panel_w, top + panel_h],
                    outline=ACCENT if i else (52, 52, 58), width=2 if i else 1)
        label(d, px + 14, top + 14, tag_r if i else tag_l,
              ACCENT if i else (14, 14, 17), GROUND if i else TEXT)

    fy = top + panel_h + 22
    d.text((PAD, fy), f"{int(scale * 100)}% scale" if scale != 1 else "100% — actual pixels",
           font=font(15, 700), fill=ACCENT)
    wrapped(d, (PAD, fy + 24), note, font(15, 400), MUTED, W - 2 * PAD)
    im.save(OUT / out)
    print(f"wrote {out}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    s = load()
    # 100%: a wide slab of the scene, both panels identical region
    pair_shot(s, (700, 452, 596, 530), 1.0,
              "The same frame, the same instant.",
              "Public-domain test footage played at 540p — the resolution a browser "
              "actually receives — then upscaled to 1080p. Left is what your browser "
              "shows today. Right is Crisp. No zoom, no crop, no second sharpening pass.",
              "01_proof.png")
    # 3x: the signage, where the difference is easiest to read
    # Caption claims only what the image actually shows. The lettering on the
    # left is blurred, not unreadable, and saying "mush becomes letters" would be
    # a claim the picture does not support.
    pair_shot(s, (1112, 715, 199, 177), 3.0,
              "Closer, on the same pixels.",
              "The identical 199x177 patch from both frames, enlarged 3x with nearest-"
              "neighbour, so you are looking at real pixels and not a smoothing filter. "
              "Letter edges tighten, the roofline separates, and the windows below stop "
              "bleeding into the wall.",
              "02_detail.png")
