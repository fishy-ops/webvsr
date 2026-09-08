"""The Crisp mark, v2: white bars on a full-bleed spectrum tile.

The bars widen and blur toward the left and tighten and sharpen toward the
right, so the shape is the product: compressed on one side, reconstructed on the
other. Judged at 16px first, because that is where a browser toolbar renders it.
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from brand2 import SPECTRUM, ramp

S = 1024
RADIUS = 0.225
WIDTHS = [0.255, 0.225, 0.195, 0.165, 0.135]
GAPS = [0.022, 0.026, 0.030, 0.034]


def _grad(size, angle=40):
    d = int(size * 1.6)
    strip = Image.new("RGB", (d, 1))
    px = strip.load()
    for x, c in enumerate(ramp(SPECTRUM, d)):
        px[x, 0] = c
    g = strip.resize((d, d), Image.NEAREST).rotate(angle, resample=Image.BICUBIC)
    o = (d - size) // 2
    return g.crop((o, o, o + size, o + size))


def _bars(size):
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    mm = int(size * 0.175)
    w = size - 2 * mm
    x = mm
    for i, wf in enumerate(WIDTHS):
        bw = int(w * wf)
        d.rectangle([x, mm, x + bw, size - mm], fill=255)
        if i < len(GAPS):
            x += bw + int(w * GAPS[i])
    left = m.crop((0, 0, int(size * 0.46), size)).filter(
        ImageFilter.GaussianBlur(size * 0.020))
    m.paste(left, (0, 0))
    return m


def mark(size=S):
    tile = _grad(S, 40).convert("RGB")
    tile.paste(Image.new("RGB", (S, S), (255, 255, 255)), (0, 0), _bars(S))
    rm = Image.new("L", (S, S), 0)
    ImageDraw.Draw(rm).rounded_rectangle([0, 0, S - 1, S - 1], int(S * RADIUS), fill=255)
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(tile, (0, 0), rm)
    return out.resize((size, size), Image.LANCZOS) if size != S else out


if __name__ == "__main__":
    for d in ("brand", "assets/icons", "extension/icons"):
        Path(d).mkdir(parents=True, exist_ok=True)
    for n in (16, 32, 48, 128, 256, 512):
        m = mark(n)
        m.save(f"brand/icon{n}.png")
        if n in (16, 48, 128):
            m.save(f"assets/icons/icon{n}.png")
            m.save(f"extension/icons/icon{n}.png")
    sheet = Image.new("RGB", (640, 210), (250, 250, 252))
    ImageDraw.Draw(sheet).rectangle([320, 0, 640, 210], fill=(17, 17, 22))
    for x0 in (0, 320):
        for (sz, pos) in ((128, (x0 + 26, 40)), (48, (x0 + 176, 80)), (16, (x0 + 240, 96))):
            im = mark(sz); sheet.paste(im, pos, im)
        z = mark(16).resize((64, 64), Image.NEAREST); sheet.paste(z, (x0 + 266, 72), z)
    sheet.save("brand/_icon_v2_final.png")
    print("wrote icons to brand/, assets/icons/, extension/icons/")
