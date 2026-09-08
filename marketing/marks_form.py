"""Split as FORM, not as two colour fields.

The flat 50/50 versions were rejected: a bare colour split reads as two
rectangles, and the dull half looks unfinished rather than intentional. The
lesson from marks that work at this scale -- Vercel's triangle, Linear's
geometry -- is one shape, cut precisely, rather than a composition.

So the split becomes silhouette. Soft on one side, hard on the other, in a
single flat green form: 'crisp' expressed as geometry instead of illustrated.
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from brand3 import INK, GREEN, GREEN_DEEP, DULL, GROUND, SURFACE, font

S = 1024


def canvas(bg=(0, 0, 0, 0)):
    return Image.new("RGBA", (S, S), bg)


def asym(d, box, rl, rr, fill):
    """Rectangle with independent left and right corner radii."""
    x0, y0, x1, y1 = box
    d.rounded_rectangle([x0, y0, x0 + 2 * rl, y1], rl, fill=fill) if rl else None
    d.rounded_rectangle([x1 - 2 * rr, y0, x1, y1], rr, fill=fill) if rr else None
    d.rectangle([x0 + (rl if rl else 0), y0, x1 - (rr if rr else 0), y1], fill=fill)
    if not rl:
        d.rectangle([x0, y0, x0 + 4, y1], fill=fill)
    if not rr:
        d.rectangle([x1 - 4, y0, x1, y1], fill=fill)


def E_asym():
    """One flat form: fully round on the left, square-cornered on the right."""
    im = canvas(); d = ImageDraw.Draw(im)
    m = int(S * 0.11)
    asym(d, (m, m, S - m, S - m), rl=int((S - 2 * m) / 2), rr=0, fill=GREEN)
    return im


def F_slices():
    """Left edge separates into slices; the body stays whole. Soft resolving to solid."""
    im = canvas(); d = ImageDraw.Draw(im)
    m = int(S * 0.13)
    w = S - 2 * m
    r = int(w * 0.10)
    # solid body, right two-thirds
    bx = m + int(w * 0.36)
    d.rounded_rectangle([bx, m, S - m, S - m], r, fill=GREEN)
    d.rectangle([bx, m, bx + r, S - m], fill=GREEN)
    # three slices peeling off to the left, thinning as they go
    sw = int(w * 0.085)
    gap = int(w * 0.048)
    x = bx - gap - sw
    for i, sc in enumerate((0.90, 0.74, 0.56)):
        h = int((S - 2 * m) * sc)
        y = (S - h) // 2
        d.rounded_rectangle([x, y, x + sw, y + h], int(sw * 0.42), fill=GREEN)
        x -= gap + sw
    return im


def G_notch():
    """A precise wedge removed from one side: the 'cut' that makes it crisp."""
    im = canvas(); d = ImageDraw.Draw(im)
    m = int(S * 0.115)
    r = int((S - 2 * m) * 0.24)
    d.rounded_rectangle([m, m, S - m, S - m], r, fill=GREEN)
    cx = int(S * 0.52)
    d.polygon([(cx, m - 2), (S - m + 2, m - 2), (S - m + 2, S - m + 2)], fill=GROUND)
    d.polygon([(cx + int(S * 0.10), m - 2), (S - m + 2, m - 2),
               (S - m + 2, S * 0.40)], fill=GREEN)
    return im


def H_bars():
    """Two flat bars, unequal: wide/soft-cornered and narrow/square. Reads as a
    pair, which is what a comparison is."""
    im = canvas(); d = ImageDraw.Draw(im)
    m = int(S * 0.135)
    h = S - 2 * m
    wide = int((S - 2 * m) * 0.40)
    d.rounded_rectangle([m, m, m + wide, S - m], int(wide * 0.5), fill=DULL)
    x = m + wide + int(S * 0.085)
    d.rounded_rectangle([x, m, S - m, S - m], int(S * 0.035), fill=GREEN)
    return im


def on_tile(mark_fn, bg):
    base = Image.new("RGB", (S, S), bg)
    mk = mark_fn()
    base.paste(mk, (0, 0), mk)
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, S - 1, S - 1], int(S * 0.225), fill=255)
    out = canvas(); out.paste(base, (0, 0), m)
    return out


CANDS = [("E asym", E_asym), ("F slices", F_slices), ("G notch", G_notch), ("H pair", H_bars)]

if __name__ == "__main__":
    out = Path("brand/v8"); out.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (4 * 190 + 30, 400), (252, 252, 250))
    ImageDraw.Draw(sheet).rectangle([0, 316, sheet.width, 400], fill=(13, 26, 20))
    d = ImageDraw.Draw(sheet)
    for i, (name, fn) in enumerate(CANDS):
        x = 16 + i * 190
        d.text((x, 12), name, font=font(14, 620), fill=INK)
        # on white (how it sits on a page) and on an ink tile (how a toolbar sees it)
        onw = fn(); ont = on_tile(fn, INK)
        a = onw.resize((116, 116), Image.LANCZOS); sheet.paste(a, (x, 34), a)
        b = ont.resize((116, 116), Image.LANCZOS); sheet.paste(b, (x, 160), b)
        for j, sz in enumerate((48, 16)):
            s2 = ont.resize((sz, sz), Image.LANCZOS)
            sheet.paste(s2, (x + 128, 160 + j * 62), s2)
        z = ont.resize((16, 16), Image.LANCZOS).resize((72, 72), Image.NEAREST)
        sheet.paste(z, (x + 24, 322), z)
        onw.resize((512, 512), Image.LANCZOS).save(out / f"{name.replace(' ', '_')}.png")
    sheet.save(out / "_v8_sheet.png")
    print("wrote brand/v8/_v8_sheet.png — mark on white, on ink tile, 48/16, then 16 magnified")
