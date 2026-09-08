"""The Crisp brand system, in one place so every asset agrees.

Amber on near-black: chosen because the category it competes in is wall-to-wall
purple-blue AI gradients and black-red video marks, so a warm mark is the only
one on the screen. The amber is pushed toward gold rather than orange so it does
not read as a warning state.
"""
from PIL import ImageFont

GROUND = (14, 14, 17)          # #0E0E11
SURFACE = (24, 24, 28)         # one step up, for panels
ACCENT = (255, 176, 32)        # #FFB020
ACCENT_DIM = (140, 96, 18)
TEXT = (245, 245, 247)         # #F5F5F7
MUTED = (138, 138, 148)        # #8A8A94

SF = "/System/Library/Fonts/SFNS.ttf"
SF_ITALIC = "/System/Library/Fonts/SFNSItalic.ttf"


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
    """Letter-spaced text. PIL has no tracking, and uppercase labels need it."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += draw.textlength(ch, font=fnt) + spacing
    return x - xy[0]


def track_width(draw, text, fnt, spacing=0):
    return sum(draw.textlength(c, font=fnt) for c in text) + spacing * (len(text) - 1)
