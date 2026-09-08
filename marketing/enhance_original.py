"""Enhancements of the ORIGINAL extension icon, not replacements for it.

The original is a white double-headed diagonal arrow on a flat cyan rounded
square. What it gets right: one clear gesture, legible at 16px, and existing
users already recognise it. What it gets wrong: the double-headed diagonal is
the universal *fullscreen* glyph, so it says "resize" rather than "quality"; and
flat cyan disappears in a category where nearly every rival icon is blue.

Every variant below keeps the diagonal gesture and the rounded-square tile, so
it still reads as the same product, and changes only what is weak.
"""
import sys, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from brand2 import SPECTRUM, INK, ramp, font
from marks_v4 import grad, tile, S

W = (255, 255, 255)
CYAN = (0, 160, 200)


def arrow(d, cx=S/2, cy=S/2, half=S*0.255, t=S*0.082, colour=W, heads=(True, True),
          head=S*0.135):
    """A double-headed diagonal.

    Heads are filled triangles, not two thick strokes meeting at a right angle:
    at this weight the right-angle construction fuses into a square blob and the
    arrow stops reading as an arrow at all.
    """
    a = math.radians(45)
    dx, dy = math.cos(a), -math.sin(a)
    px_, py_ = -dy, dx                      # unit normal to the shaft
    x0, y0 = cx - dx * half, cy - dy * half
    x1, y1 = cx + dx * half, cy + dy * half

    # shaft stops short of each head so the triangle sits on its tip
    inset = head * 0.72
    sx0, sy0 = (x0 + dx * inset, y0 + dy * inset) if heads[0] else (x0, y0)
    sx1, sy1 = (x1 - dx * inset, y1 - dy * inset) if heads[1] else (x1, y1)
    d.line([(sx0, sy0), (sx1, sy1)], fill=colour, width=int(t))

    def head_at(tipx, tipy, sgn):
        bx, by = tipx - sgn * dx * head, tipy - sgn * dy * head
        w = head * 0.78
        d.polygon([(tipx, tipy), (bx + px_ * w, by + py_ * w),
                   (bx - px_ * w, by - py_ * w)], fill=colour)

    if heads[1]:
        head_at(x1, y1, 1)
    if heads[0]:
        head_at(x0, y0, -1)


def V1_spectrum():
    """Faithful: identical gesture, spectrum tile instead of flat cyan."""
    base = grad(S, 40).convert("RGB")
    arrow(ImageDraw.Draw(base))
    return tile(base)


def V2_refined():
    """Heavier stroke, tighter heads, optically centred — the same idea, drawn better."""
    base = grad(S, 40).convert("RGB")
    arrow(ImageDraw.Draw(base), half=S * 0.275, t=S * 0.092, head=S * 0.150)
    return tile(base)


def V3_resolve():
    """One arm soft, one arm crisp: the arrow now says quality, not just size."""
    base = grad(S, 40).convert("RGB")
    soft = Image.new("L", (S, S), 0)
    ds = ImageDraw.Draw(soft)
    arrow(ds, colour=255, heads=(True, False))
    soft = soft.filter(ImageFilter.GaussianBlur(S * 0.028))
    base.paste(Image.new("RGB", (S, S), W), (0, 0), soft)
    crisp = Image.new("L", (S, S), 0)
    arrow(ImageDraw.Draw(crisp), colour=255, heads=(False, True))
    base.paste(Image.new("RGB", (S, S), W), (0, 0), crisp)
    return tile(base)


def V4_invert():
    """White tile, spectrum arrow. Brightest in a light store grid."""
    base = Image.new("RGB", (S, S), (252, 252, 254))
    m = Image.new("L", (S, S), 0)
    arrow(ImageDraw.Draw(m), colour=255, t=S * 0.092, half=S * 0.275, head=S * 0.150)
    base.paste(grad(S, 40), (0, 0), m)
    return tile(base)


CANDS = [("original", None), ("1 spectrum", V1_spectrum), ("2 refined", V2_refined),
         ("3 resolve", V3_resolve)]

if __name__ == "__main__":
    out = Path("brand/v6"); out.mkdir(parents=True, exist_ok=True)
    orig = Image.open("brand/original/icon128.png").convert("RGBA")
    sheet = Image.new("RGB", (4 * 176 + 30, 320), (252, 252, 254))
    ImageDraw.Draw(sheet).rectangle([0, 236, sheet.width, 320], fill=(17, 17, 22))
    d = ImageDraw.Draw(sheet)
    for i, (name, fn) in enumerate(CANDS):
        big = orig if fn is None else fn()
        if fn is not None:
            big.resize((512, 512), Image.LANCZOS).save(out / f"{name.replace(' ', '_')}.png")
        x = 16 + i * 176
        d.text((x, 12), name, font=font(14, 620), fill=(20, 20, 28))
        b = big.resize((124, 124), Image.LANCZOS); sheet.paste(b, (x, 36), b)
        for j, sz in enumerate((48, 16)):
            s2 = big.resize((sz, sz), Image.LANCZOS)
            sheet.paste(s2, (x + j * 64, 172 + (0 if sz == 48 else 16)), s2)
        z = big.resize((16, 16), Image.LANCZOS).resize((72, 72), Image.NEAREST)
        sheet.paste(z, (x + 26, 242), z)
    sheet.save(out / "_v6_sheet.png")
    print("wrote brand/v6/_v6_sheet.png — original first, then four enhancements")
