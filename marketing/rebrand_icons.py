"""Icon candidates, take two. Bright and multi-hue.

The first attempt (amber glyph on near-black) was scrapped: that exact
combination reads as a well-known adult site and is unusable, however well it
scored on grid contrast.

Spectrum is not an arbitrary replacement. Splitting light into its colours is
literally what resolution and clarity look like as a picture, so a spectrum mark
says "detail" rather than just "bright". Curated 4-5 stop gradients are used
rather than a full ROYGBIV rainbow, which reads as clip art.
"""
import sys, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))

S = 1024
RADIUS = 0.225

# curated spectra -- bright, saturated, and none of them orange-on-black
SPECTRA = {
    "aurora":  [(124, 58, 237), (37, 99, 235), (6, 182, 212), (16, 185, 129)],
    "vivid":   [(255, 61, 138), (168, 60, 255), (56, 132, 255), (0, 214, 214)],
    "prism":   [(255, 45, 120), (255, 190, 40), (60, 220, 130), (40, 150, 255), (150, 70, 245)],
    "candy":   [(0, 224, 255), (90, 120, 255), (200, 80, 255), (255, 70, 160)],
}


def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def ramp(stops, n):
    """n colours interpolated across the stop list."""
    out = []
    seg = len(stops) - 1
    for i in range(n):
        t = i / max(n - 1, 1) * seg
        k = min(int(t), seg - 1)
        out.append(lerp(stops[k], stops[k + 1], t - k))
    return out


def grad_image(stops, size=S, angle=45):
    """Linear multi-stop gradient, drawn big then rotated for a clean diagonal."""
    d = int(size * 1.5)
    strip = Image.new("RGB", (d, 1))
    px = strip.load()
    cols = ramp(stops, d)
    for x in range(d):
        px[x, 0] = cols[x]
    g = strip.resize((d, d), Image.NEAREST).rotate(angle, resample=Image.BICUBIC)
    off = (d - size) // 2
    return g.crop((off, off, off + size, off + size))


def rounded_mask(size=S, r=None):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size - 1, size - 1],
                                        int(size * RADIUS) if r is None else r, fill=255)
    return m


def var_bars(spec):
    """Left bars soft and wide, right bars tight and sharp: the product itself."""
    base = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    g = grad_image(SPECTRA[spec], S, 15)
    layer = Image.new("RGB", (S, S), (255, 255, 255))
    d = ImageDraw.Draw(layer)
    m = int(S * 0.17)
    w = S - 2 * m
    # five bars, thinning left to right
    n = 5
    x = m
    for i in range(n):
        bw = int(w * (0.26 - 0.030 * i))
        d.rectangle([x, m, x + bw, S - m], fill=(0, 0, 0))
        x += bw + int(w * (0.020 + 0.014 * i))
    blur = Image.new("L", (S, S), 0)
    bd = ImageDraw.Draw(blur)
    x = m
    for i in range(n):
        bw = int(w * (0.26 - 0.030 * i))
        bd.rectangle([x, m, x + bw, S - m], fill=255)
        x += bw + int(w * (0.020 + 0.014 * i))
    # blur only the left of the mask, so bars resolve left->right
    left = blur.crop((0, 0, S // 2, S)).filter(ImageFilter.GaussianBlur(S * 0.022))
    blur.paste(left, (0, 0))
    tile = Image.new("RGB", (S, S), (255, 255, 255))
    tile.paste(g, (0, 0), blur)
    base.paste(tile, (0, 0), rounded_mask())
    return base


def var_ring(spec):
    """A spectrum aperture on white -- optics, not letters."""
    base = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    tile = Image.new("RGB", (S, S), (255, 255, 255))
    g = grad_image(SPECTRA[spec], S, 45)
    ring = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(ring)
    c, o, t = S / 2, S * 0.335, S * 0.125
    d.ellipse([c - o, c - o, c + o, c + o], fill=255)
    d.ellipse([c - (o - t), c - (o - t), c + (o - t), c + (o - t)], fill=0)
    tile.paste(g, (0, 0), ring)
    d2 = ImageDraw.Draw(tile)
    inner = int(S * 0.115)
    tile.paste(g, (0, 0), _dot_mask(c, inner))
    base.paste(tile, (0, 0), rounded_mask())
    return base


def _dot_mask(c, r):
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).ellipse([c - r, c - r, c + r, c + r], fill=255)
    return m


def var_full(spec):
    """Full-bleed spectrum tile with a white play triangle punched out."""
    base = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    tile = grad_image(SPECTRA[spec], S, 45).convert("RGB")
    d = ImageDraw.Draw(tile)
    cx, cy, r = S * 0.52, S * 0.5, S * 0.20
    d.polygon([(cx - r * 0.72, cy - r), (cx - r * 0.72, cy + r), (cx + r * 0.86, cy)],
              fill=(255, 255, 255))
    base.paste(tile, (0, 0), rounded_mask())
    return base


def var_chev(spec):
    """Three chevrons tightening to a point: convergence on detail."""
    base = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    tile = Image.new("RGB", (S, S), (18, 18, 26))
    g = grad_image(SPECTRA[spec], S, 90)
    mask = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(mask)
    t = int(S * 0.085)
    for i, sc in enumerate((0.30, 0.20, 0.10)):
        x0, x1 = S * (0.30 + i * 0.10), S * (0.30 + i * 0.10) + S * 0.22
        d.line([(x0, S * 0.5 - S * sc), (x1, S * 0.5), (x0, S * 0.5 + S * sc)],
               fill=255, width=t, joint="curve")
    tile.paste(g, (0, 0), mask)
    base.paste(tile, (0, 0), rounded_mask())
    return base


if __name__ == "__main__":
    out = Path("brand/v2"); out.mkdir(parents=True, exist_ok=True)
    cands = [("bars", var_bars, "vivid"), ("ring", var_ring, "aurora"),
             ("play", var_full, "prism"), ("chev", var_chev, "candy")]
    sheet = Image.new("RGB", (4 * 165 + 40, 300), (247, 247, 250))
    d = ImageDraw.Draw(sheet)
    d.rectangle([0, 215, sheet.width, 300], fill=(16, 16, 20))
    for i, (name, fn, spec) in enumerate(cands):
        big = fn(spec)
        big.resize((512, 512), Image.LANCZOS).save(out / f"cand_{name}.png")
        x = 20 + i * 165
        sheet.paste(big.resize((128, 128), Image.LANCZOS), (x, 20), big.resize((128, 128), Image.LANCZOS))
        for j, sz in enumerate((48, 32, 16)):
            s2 = big.resize((sz, sz), Image.LANCZOS)
            sheet.paste(s2, (x + j * 52, 165), s2)
        s16 = big.resize((16, 16), Image.LANCZOS).resize((72, 72), Image.NEAREST)
        sheet.paste(s16, (x + 30, 228))
    sheet.save(out / "_v2_candidates.png")
    print("wrote brand/v2/_v2_candidates.png — 128px, then 48/32/16, then 16px magnified on dark")
