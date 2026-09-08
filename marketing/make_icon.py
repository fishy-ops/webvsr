"""Icon candidates. Every one is judged at 16px, because that is the size the
browser toolbar actually renders and a mark that only works at 128 is a mark
that fails where users see it most.

All are drawn at 8x and downsampled with LANCZOS rather than drawn small, so the
curves are clean.
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from brand import GROUND, ACCENT, TEXT

S = 1024                     # supersample size
R = int(S * 0.22)            # corner radius, squircle-ish


def tile(bg):
    im = Image.new("RGB", (S, S), GROUND)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, S - 1, S - 1], R, fill=bg)
    return im, d


def ring_c(d, cx, cy, outer, thick, colour, gap_deg=62):
    """A geometric C: a thick arc with a wedge open to the right."""
    d.ellipse([cx - outer, cy - outer, cx + outer, cy + outer], fill=colour)
    inner = outer - thick
    return inner


def var_a():
    """Bold C, dark on amber. Colour does the standing out."""
    im, d = tile(ACCENT)
    cx = cy = S // 2
    outer, thick = int(S * 0.33), int(S * 0.115)
    d.ellipse([cx - outer, cy - outer, cx + outer, cy + outer], fill=GROUND)
    inner = outer - thick
    d.ellipse([cx - inner, cy - inner, cx + inner, cy + inner], fill=ACCENT)
    d.pieslice([cx - outer, cy - outer, cx + outer, cy + outer], -31, 31, fill=ACCENT)
    return im


def var_b():
    """Bold C, amber on dark."""
    im, d = tile(GROUND)
    cx = cy = S // 2
    outer, thick = int(S * 0.33), int(S * 0.115)
    d.ellipse([cx - outer, cy - outer, cx + outer, cy + outer], fill=ACCENT)
    inner = outer - thick
    d.ellipse([cx - inner, cy - inner, cx + inner, cy + inner], fill=GROUND)
    d.pieslice([cx - outer, cy - outer, cx + outer, cy + outer], -31, 31, fill=GROUND)
    return im


def var_c():
    """Focus brackets round a solid centre — 'lock on and sharpen'."""
    im, d = tile(GROUND)
    m, t, L = int(S * 0.20), int(S * 0.075), int(S * 0.21)
    for (x, y, dx, dy) in ((m, m, 1, 1), (S - m, m, -1, 1),
                           (m, S - m, 1, -1), (S - m, S - m, -1, -1)):
        d.rectangle([min(x, x + dx * L), y - t // 2,
                     max(x, x + dx * L), y + t // 2], fill=ACCENT)
        d.rectangle([x - t // 2, min(y, y + dy * L),
                     x + t // 2, max(y, y + dy * L)], fill=ACCENT)
    d.ellipse([S // 2 - int(S * 0.13), S // 2 - int(S * 0.13),
               S // 2 + int(S * 0.13), S // 2 + int(S * 0.13)], fill=ACCENT)
    return im


def var_d():
    """The product itself: soft blocks on the left resolving to a clean edge."""
    im, d = tile(GROUND)
    m = int(S * 0.20)
    w = S - 2 * m
    # left: coarse blocks (what compressed video looks like)
    n = 4
    step = w // (2 * n)
    for i in range(n):
        for j in range(n):
            v = 0.30 + 0.62 * ((i * 3 + j * 5) % 4) / 3
            c = tuple(int(ch * v) for ch in ACCENT)
            d.rectangle([m + j * step, m + i * (w // n),
                         m + (j + 1) * step, m + (i + 1) * (w // n)], fill=c)
    # right: one clean amber field
    d.rectangle([m + w // 2, m, m + w, m + w], fill=ACCENT)
    return im


if __name__ == "__main__":
    out = Path("brand"); out.mkdir(exist_ok=True)
    sheet = Image.new("RGB", (4 * 150 + 50, 260), (30, 30, 34))
    for i, (name, fn) in enumerate([("a_dark_on_amber", var_a), ("b_amber_on_dark", var_b),
                                    ("c_focus", var_c), ("d_blocks", var_d)]):
        big = fn()
        big.resize((512, 512), Image.LANCZOS).save(out / f"cand_{name}.png")
        sheet.paste(big.resize((128, 128), Image.LANCZOS), (20 + i * 150, 20))
        sheet.paste(big.resize((48, 48), Image.LANCZOS), (20 + i * 150, 165))
        sheet.paste(big.resize((16, 16), Image.LANCZOS), (20 + i * 150 + 60, 180))
        sheet.paste(big.resize((16, 16), Image.LANCZOS).resize((64, 64), Image.NEAREST),
                    (20 + i * 150 + 80, 172))
    sheet.save(out / "_candidates.png")
    print("wrote brand/_candidates.png  (128px, 48px, true 16px, 16px magnified)")
