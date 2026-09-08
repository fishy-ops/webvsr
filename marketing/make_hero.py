"""Screenshot 1: the proof. One region, one instant, split down the middle.

The seam is placed through the shop signage on purpose. A split comparison only
works if the eye can cross it and land on the same object twice, and text is the
detail people read fastest -- if the letters resolve on one side and not the
other, no caption is needed.

Everything is at 100% zoom. No enlargement of either half, no sharpening pass on
top, no different crop between sides: the only difference across the seam is the
network.
"""
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from brand import GROUND, ACCENT, TEXT, MUTED, font, track, track_width
from pipeline import make_set, to_pil

W, H = 1280, 800
PAD = 50
IMG_W, IMG_H = W - 2 * PAD, 540
TOP = 132
SEAM = 596                      # window-local x of the divider


def pill(d, x, y, label, bg, fg, f):
    tw = track_width(d, label, f, 1.4)
    d.rounded_rectangle([x, y, x + tw + 26, y + 30], 15, fill=bg)
    track(d, (x + 13, y + 7), label, f, fg, 1.4)
    return tw + 26


def build(src="src/old_town_cross_1080p50.mp4", frame=20, wx=460, wy=500,
          out="assets/screenshots/01_proof.png"):
    cache = Path("src/_otc.pt")
    s = torch.load(cache, weights_only=False) if cache.exists() else make_set(src, frame)

    before = to_pil(s["before"][:, :, wy:wy + IMG_H, wx:wx + IMG_W])
    after = to_pil(s["after"][:, :, wy:wy + IMG_H, wx:wx + IMG_W])

    strip = Image.new("RGB", (IMG_W, IMG_H))
    strip.paste(before.crop((0, 0, SEAM, IMG_H)), (0, 0))
    strip.paste(after.crop((SEAM, 0, IMG_W, IMG_H)), (SEAM, 0))

    im = Image.new("RGB", (W, H), GROUND)
    im.paste(strip, (PAD, TOP))
    d = ImageDraw.Draw(im)

    # brand lockup
    d.rounded_rectangle([PAD, 40, PAD + 30, 70], 8, fill=ACCENT)
    d.text((PAD + 8, 44), "C", font=font(21, 800), fill=GROUND)
    d.text((PAD + 42, 42), "Crisp", font=font(25, 700), fill=TEXT)

    f_head = font(25, 600)
    head = "The same frame, the same instant — with and without Crisp."
    d.text((PAD, 86), head, font=f_head, fill=TEXT)

    # the divider, and the labels either side of it
    d.line([(PAD + SEAM, TOP), (PAD + SEAM, TOP + IMG_H)], fill=ACCENT, width=3)
    f_pill = font(15, 700)
    pw = track_width(d, "WITHOUT", f_pill, 1.4) + 26
    pill(d, PAD + SEAM - pw - 14, TOP + 16, "WITHOUT", (14, 14, 17), TEXT, f_pill)
    pill(d, PAD + SEAM + 14, TOP + 16, "WITH CRISP", ACCENT, GROUND, f_pill)

    d.rectangle([PAD, TOP, PAD + IMG_W, TOP + IMG_H], outline=(48, 48, 54), width=1)

    f_note = font(15, 400)
    d.text((PAD, TOP + IMG_H + 20),
           "Shown at 100% — no zoom, no crop, no enhancement pass on top. "
           "Public-domain test footage at 540p, the resolution your browser receives.",
           font=f_note, fill=MUTED)
    im.save(out)
    print(f"wrote {out}  {im.size[0]}x{im.size[1]}")


if __name__ == "__main__":
    Path("assets/screenshots").mkdir(parents=True, exist_ok=True)
    build()
