"""Graphic promo tiles — type and colour, no photography.

Following the listing that actually ranks first in this category (NexVid,
Featured, 4.5 stars), which carries no before/after imagery at all. Two further
reasons this is the right call here: our footage options are all dated or
licence-encumbered, and in a results column where every rival tile is dark or a
gradient, one flat saturated field is the only thing that isolates.

Built at 3x and downsampled so the type stays crisp at 440px.
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from brand3 import INK, GREEN, GREEN_DEEP, GROUND, DULL, font, track, track_width
from mark_final import mark_flat

SS = 3
W_ = (255, 255, 255)


def wrap_lines(d, text, f, max_w):
    lines, line = [], ""
    for word in text.split():
        t = (line + " " + word).strip()
        if d.textlength(t, font=f) > max_w and line:
            lines.append(line); line = word
        else:
            line = t
    if line:
        lines.append(line)
    return lines


def tile(w, h, headline, sub, out, head_px=34, sub_px=12,
         pad=26, mark_px=40, lead=1.06):
    """Flat field, knockout mark, headline anchored to the baseline.

    The earlier split-field version put a hard vertical rule straight through
    the headline, and the tile-form mark went green-on-green and vanished. Both
    are gone: the mark is knocked out in white, and the split device survives as
    a short two-tone rule above the headline rather than as a field division.
    """
    W, H = w * SS, h * SS
    im = Image.new("RGB", (W, H), GREEN)
    d = ImageDraw.Draw(im)
    P = pad * SS

    mk = mark_flat(mark_px * SS, fill=W_)
    im.paste(mk, (P, P), mk)

    fh = font(head_px * SS, 800)
    fs = font(sub_px * SS, 600)
    lines = wrap_lines(d, headline, fh, W - 2 * P)
    lh = head_px * SS * lead
    rule_h = max(int(head_px * SS * 0.10), 3)
    rule_gap = int(head_px * SS * 0.42)
    total = len(lines) * lh + sub_px * SS * 2.1 + rule_h + rule_gap
    y = H - P - total + head_px * SS * 0.06

    # the split, as a rule: dulled half then vivid half
    rw = int(W * 0.16)
    d.rectangle([P, y, P + rw // 2, y + rule_h], fill=(0, 122, 71))
    d.rectangle([P + rw // 2, y, P + rw, y + rule_h], fill=W_)
    y += rule_h + rule_gap

    for ln in lines:
        d.text((P, y), ln, font=fh, fill=W_)
        y += lh
    track(d, (P + 2, y + sub_px * SS * 0.22), sub, fs, (206, 242, 224), 1.25 * SS)
    im.resize((w, h), Image.LANCZOS).save(out)
    print("wrote", out, f"{w}x{h}")


if __name__ == "__main__":
    Path("assets/promo").mkdir(parents=True, exist_ok=True)
    tile(440, 280, "Make blurry video sharp.",
         "AI UPSCALING · FREE · RUNS ON YOUR GPU",
         "assets/promo/tile_440x280.png",
         head_px=36, sub_px=10, pad=24, mark_px=40)
    tile(1400, 560, "Make blurry video sharp.",
         "AI UPSCALING · FREE · RUNS ENTIRELY ON YOUR OWN GPU",
         "assets/promo/marquee_1400x560.png",
         head_px=104, sub_px=25, pad=66, mark_px=108)

