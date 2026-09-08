"""The Crisp mark.

A dark geometric C on a full-bleed amber squircle. The whole tile carries the
accent rather than a glyph floating on dark, because in a store grid and a
browser toolbar the thing that registers first is a block of colour, and every
competitor's block is blue or purple. Judged at 16px first; the counter and the
aperture gap are sized so both survive the downsample.
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from brand import GROUND, ACCENT

S = 1024
RADIUS = 0.225          # of side
OUTER = 0.345           # C outer radius, of side
STROKE = 0.108          # ring thickness, of side
GAP = 30                # half-angle of the aperture opening, degrees


def mark(size=S, bg=ACCENT, fg=GROUND):
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, S - 1, S - 1], int(S * RADIUS), fill=bg)
    c, o, t = S / 2, S * OUTER, S * STROKE
    d.ellipse([c - o, c - o, c + o, c + o], fill=fg)
    i = o - t
    d.ellipse([c - i, c - i, c + i, c + i], fill=bg)
    d.pieslice([c - o, c - o, c + o, c + o], -GAP, GAP, fill=bg)
    return im.resize((size, size), Image.LANCZOS) if size != S else im


if __name__ == "__main__":
    out = Path("brand"); out.mkdir(exist_ok=True)
    ext = Path("assets/icons"); ext.mkdir(parents=True, exist_ok=True)
    for n in (16, 32, 48, 128, 256, 512):
        m = mark(n)
        m.save(out / f"icon{n}.png")
        if n in (16, 48, 128):
            m.save(ext / f"icon{n}.png")
    # proof sheet: how it actually lands, on both store themes
    sheet = Image.new("RGB", (620, 200), (245, 245, 247))
    d = ImageDraw.Draw(sheet)
    d.rectangle([310, 0, 620, 200], fill=(18, 18, 22))
    for col, x0 in ((0, 0), (1, 310)):
        sheet.paste(mark(128), (x0 + 30, 36), mark(128))
        sheet.paste(mark(48), (x0 + 186, 76), mark(48))
        sheet.paste(mark(16), (x0 + 250, 92), mark(16))
        sheet.paste(mark(16).resize((64, 64), Image.NEAREST), (x0 + 276, 68))
    sheet.save(out / "_icon_final.png")
    print("wrote brand/icon{16,32,48,128,256,512}.png and assets/icons/ for the extension")
