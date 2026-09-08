"""Crisp brand, v2 — bright and multi-hue.

v1 was a single amber accent on near-black. It was scrapped outright: that
combination reads as a well-known adult site, which no amount of grid-contrast
argument survives.

v2 inverts everything. Light ground, dark ink, and colour carried by a four-stop
spectrum rather than one hue. The spectrum is not decoration -- it runs pink to
cyan in the same direction the icon's bars go from soft to sharp, so the gradient
and the mark say the same thing.
"""
from PIL import Image, ImageDraw, ImageFont

# ── ink and ground ──────────────────────────────────────────────
INK      = (20, 20, 28)        # #14141C  text
GROUND   = (255, 255, 255)     # #FFFFFF  page
SURFACE  = (246, 246, 250)     # #F6F6FA  panels
LINE     = (226, 226, 236)     # #E2E2EC  hairlines
MUTED    = (107, 107, 123)     # #6B6B7B  secondary text
DIM      = (150, 150, 165)     # #9696A5  tertiary

# ── the spectrum ────────────────────────────────────────────────
SPECTRUM = [(255, 61, 138), (168, 60, 255), (56, 132, 255), (0, 214, 214)]
ACCENT   = (168, 60, 255)      # #A83CFF  the single-hue stand-in, mid-spectrum
ACCENT_INK = (255, 255, 255)   # text on accent

SF = "/System/Library/Fonts/SFNS.ttf"


def font(size, weight=400, width=100, grad=0):
    """SF Pro at a real weight.

    SFNS.ttf exposes FOUR variation axes in this order -- Width, Optical Size,
    GRAD, Weight -- so passing a single value to set_variation_by_axes sets the
    WIDTH axis, not the weight. Every earlier asset built here was therefore
    wide rather than bold. All four axes are now set explicitly, with optical
    size tracking the point size the way the face intends.
    """
    f = ImageFont.truetype(SF, size)
    try:
        opsz = max(17, min(96, size))
        f.set_variation_by_axes([width, opsz, 400 + grad, weight])
    except Exception:
        pass
    return f


def track(draw, xy, text, fnt, fill, spacing=0):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += draw.textlength(ch, font=fnt) + spacing
    return x - xy[0]


def track_width(draw, text, fnt, spacing=0):
    return sum(draw.textlength(c, font=fnt) for c in text) + spacing * (len(text) - 1)


def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def ramp(stops, n):
    out, seg = [], len(stops) - 1
    for i in range(n):
        t = i / max(n - 1, 1) * seg
        k = min(int(t), seg - 1)
        out.append(lerp(stops[k], stops[k + 1], t - k))
    return out


def gradient_bar(w, h, stops=None, horizontal=True):
    stops = stops or SPECTRUM
    cols = ramp(stops, w if horizontal else h)
    im = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(im)
    for i, c in enumerate(cols):
        if horizontal:
            d.rectangle([i, 0, i, h], fill=c)
        else:
            d.rectangle([0, i, w, i], fill=c)
    return im


def css_spectrum(angle="100deg"):
    s = ", ".join(f"rgb({r},{g},{b})" for r, g, b in SPECTRUM)
    return f"linear-gradient({angle}, {s})"
