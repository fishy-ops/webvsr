"""Build every store screenshot. One place, no footnotes.

Changes from the previous set:
  * animation as the lead content -- flat colour and line art are the best case
    for an upscaler, which is why the competitors that rank target anime
  * the "after" side runs the full shipped chain (network -> sharpen 3.5 ->
    vivid look), so it shows what a user actually gets rather than the raw
    network output
  * the explanatory paragraph under each image is gone; the pictures are bigger
    instead
  * no parameter counts or frame times anywhere -- they mean nothing to someone
    deciding whether to install a video enhancer
"""
import sys
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from brand3 import INK, GROUND, SURFACE, LINE, MUTED, DIM, GREEN, GREEN_DEEP, DULL, font, track, track_width
from pipeline import make_set, to_pil

W, H, PAD = 1280, 800, 56
OUT = Path("assets/screenshots")
SETS = {}


def load_plain(stem, frame=120):
    """Ungraded, unsharpened set — used only for choosing crops."""
    k = stem + "_plain"
    if k not in SETS:
        c = Path(f"src/_plain_{stem}.pt")
        SETS[k] = (torch.load(c, weights_only=False) if c.exists()
                   else make_set(f"src/{stem}.mp4", frame, sharpen=0, look="natural"))
        torch.save(SETS[k], c)
    return SETS[k]


def load(stem, frame=120):
    if stem not in SETS:
        c = Path(f"src/_set_{stem}.pt")
        SETS[stem] = torch.load(c, weights_only=False) if c.exists() else make_set(f"src/{stem}.mp4", frame)
        torch.save(SETS[stem], c)
    return SETS[stem]


def lockup(im, d, x=PAD, y=34):
    ic = Image.open("brand/icon128.png").convert("RGBA").resize((32, 32), Image.LANCZOS)
    im.paste(ic, (x, y), ic)
    d.text((x + 44, y - 2), "Crisp", font=font(25, 640), fill=INK)


def chip(d, x, y, text, bg, fg, size=15):
    f = font(size, 720)
    tw = track_width(d, text, f, 1.3)
    d.rounded_rectangle([x, y, x + tw + 26, y + 30], 15, fill=bg)
    track(d, (x + 13, y + 7), text, f, fg, 1.3)


def pair(s, region, scale, headline, out):
    """Two panels, identical region, identical scale. Nothing underneath."""
    x, y, rw, rh = region
    pw = (W - 2 * PAD - 20) // 2
    ph = int(rh * scale)
    im = Image.new("RGB", (W, H), GROUND)
    d = ImageDraw.Draw(im)
    lockup(im, d)
    d.text((PAD, 82), headline, font=font(30, 680), fill=INK)
    top = 132
    for i, key in enumerate(("before", "after")):
        px = PAD + i * (pw + 20)
        src = to_pil(s[key][:, :, y:y + rh, x:x + rw])
        src = src.resize((int(rw * scale), ph), Image.NEAREST if scale > 1 else Image.LANCZOS)
        im.paste(src.crop((0, 0, pw, ph)), (px, top))
        d.rectangle([px, top, px + pw, top + ph], outline=LINE, width=1)
        d.rectangle([px, top + ph, px + pw, top + ph + 5], fill=GREEN if i else DULL)
        chip(d, px + 14, top + 14, "WITH CRISP" if i else "WITHOUT",
             (255, 255, 255), GREEN_DEEP if i else MUTED)
    im.save(OUT / out); print("wrote", out)


def breadth(cells, out):
    """Four clips, each cropped where the model MEASURABLY helps.

    The crop is chosen by find_crops.score_windows, not by hand. An earlier
    revision replaced those scored windows with hardcoded coordinates and the
    cells got visibly worse -- the scorer is picking regions where the model
    closes real error against the original, which is exactly what the cell is
    supposed to show.

    Scoring runs on an ungraded set: with the Vivid look applied, "after"
    differs from the original by the grade as well as the reconstruction, which
    would corrupt the ranking. Crops are found clean, then rendered graded.
    """
    from find_crops import score_windows
    im = Image.new("RGB", (W, H), GROUND)
    d = ImageDraw.Draw(im)
    lockup(im, d)
    d.text((PAD, 82), "The same network, four kinds of video.", font=font(30, 680), fill=INK)
    cw, ch = (W - 2 * PAD - 22) // 2, 306
    pw = (cw - 8) // 2
    for i, (stem, label, frame) in enumerate(cells):
        cx = PAD + (i % 2) * (cw + 22)
        cy = 140 + (i // 2) * (ch + 26)
        plain = load_plain(stem, frame)
        ranked = score_windows(plain, crop=180, stride=60)
        if not ranked:
            print(f"  !! no textured window in {stem}"); continue
        _, _, y, x = ranked[0]
        rh = 180
        rw = int(rh * pw / 238)
        s = load(stem, frame)
        for j, key in enumerate(("before", "after")):
            crop = to_pil(s[key][:, :, y:y + rh, x:x + rw]).resize((pw, 238), Image.LANCZOS)
            px = cx + j * (pw + 8)
            im.paste(crop, (px, cy + 26))
            d.rectangle([px, cy + 26, px + pw, cy + 264], outline=LINE, width=1)
            d.rectangle([px, cy + 264, px + pw, cy + 269], fill=GREEN if j else DULL)
        d.text((cx, cy), label, font=font(17, 660), fill=INK)
        print(f"  {stem:26} scored crop ({x},{y})")
    im.save(OUT / out); print("wrote", out)


def motion(out):
    with av.open("assets/crisp_demo.mp4") as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    im = Image.new("RGB", (W, H), GROUND)
    d = ImageDraw.Draw(im)
    lockup(im, d)
    d.text((PAD, 82), "It holds up in motion.", font=font(30, 680), fill=INK)
    src = Image.fromarray(frames[min(22, len(frames) - 1)])
    iw = W - 2 * PAD
    ih = int(src.height * iw / src.width)
    im.paste(src.resize((iw, ih), Image.LANCZOS), (PAD, 140))
    d.rectangle([PAD, 140, PAD + iw, 140 + ih], outline=LINE, width=1)
    im.save(OUT / out); print("wrote", out)


def limits(out):
    im = Image.new("RGB", (W, H), GROUND)
    d = ImageDraw.Draw(im)
    lockup(im, d)
    d.text((PAD, 82), "What it won't do.", font=font(30, 680), fill=INK)
    d.text((PAD, 128), "Every other listing here promises AI 4K magic. This is the honest version.",
           font=font(19, 440), fill=MUTED)
    items = [
        ("It doesn't invent detail.",
         "It rebuilds what compression blurred. It won't hallucinate a face out of a smudge."),
        ("It works best on low-resolution video.",
         "The worse the compression, the less real detail is left to recover."),
        ("It needs a recent Chrome and a GPU.",
         "Without them Crisp simply stays off instead of slowing your video down."),
        ("It can't touch Netflix or Disney+.",
         "Copy-protected video is off limits to every extension, this one included."),
    ]
    y = 208
    for title, body in items:
        d.rectangle([PAD, y + 6, PAD + 6, y + 88], fill=GREEN)
        d.text((PAD + 30, y), title, font=font(27, 700), fill=INK)
        d.text((PAD + 30, y + 40), body, font=font(19, 440), fill=MUTED)
        y += 140
    im.save(OUT / out); print("wrote", out)
