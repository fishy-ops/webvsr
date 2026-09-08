"""Crisp identity v3 — the split, in signal green. Flat, no gradients.

Strategy, not decoration:

  * Ehrenberg-Bass: growth comes from DISTINCTIVE assets, not from differentiating
    claims. So the mark stops trying to explain super-resolution (bars resolving,
    a prism splitting light, an arrow sharpening -- all rejected) and instead
    becomes one shape that is simply, unmistakably ours.

  * The split IS that asset. A hard vertical divider with a duller side and a
    vivid side is what this product does, and it repeats across the icon, the
    screenshots, the video, the page and the in-product toggle. Recognisable
    with the name removed, which is the actual test.

  * Green because the category has not claimed it. Every rival video enhancer is
    blue, purple, or black-and-red; blue in particular is now so crowded it has
    stopped working. The isolation effect does the rest.

  * Flat because gradients -- violet-to-blue especially -- have become the visual
    signature of AI-generated identity work.
"""
from PIL import ImageFont

# ── core ────────────────────────────────────────────────────────
INK      = (13, 26, 20)        # #0D1A14  deep green-black
GROUND   = (255, 255, 255)
SURFACE  = (242, 245, 241)     # #F2F5F1  neutral with a deliberate green bias
LINE     = (223, 229, 221)     # #DFE5DD
MUTED    = (95, 106, 99)       # #5F6A63
DIM      = (142, 152, 145)     # #8E9891

# ── the accent ──────────────────────────────────────────────────
# Pulled away from Spotify (#1DB954) and Cash App (#00D632) on purpose: near
# neighbours of a famous green borrow its associations instead of building ours.
GREEN      = (0, 179, 104)     # #00B368  signal green
GREEN_DEEP = (0, 138, 80)      # #008A50  pressed / hover
GREEN_TINT = (230, 248, 239)   # #E6F8EF  wash

# the dulled side of the split -- the "before"
DULL = (154, 166, 158)         # #9AA69E

SF = "/System/Library/Fonts/SFNS.ttf"


def font(size, weight=400, width=100, grad=0):
    """SF Pro at a real weight.

    SFNS.ttf exposes four variation axes in this order -- Width, Optical Size,
    GRAD, Weight -- so passing a single value sets WIDTH, not weight. All four
    are set explicitly here.
    """
    f = ImageFont.truetype(SF, size)
    try:
        f.set_variation_by_axes([width, max(17, min(96, size)), 400 + grad, weight])
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


def hexs(c):
    return "#%02X%02X%02X" % c
