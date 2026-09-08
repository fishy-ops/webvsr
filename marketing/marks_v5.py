"""The three non-letter finalists, refined.

Prism is rebuilt: thicker beam, wider fan, light tile. Its idea is the best of
the six -- white light entering and a spectrum leaving is literally what
resolving detail looks like -- but the first cut lost the beam at 16px.
"""
import sys, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from brand2 import SPECTRUM, INK, ramp, font
from marks_v4 import grad, tile, A_bars, S

W = (255, 255, 255)


def prism(light_tile=True):
    bg = (252, 252, 254) if light_tile else INK
    beam = INK if light_tile else W
    base = Image.new("RGB", (S, S), bg)
    d = ImageDraw.Draw(base)
    cx, cy = S * 0.40, S * 0.50
    # the fan: wide wedges so each hue survives the downsample
    for i, col in enumerate(SPECTRUM):
        a0 = -26 + i * 13.5
        d.pieslice([cx - S * 0.70, cy - S * 0.70, cx + S * 0.70, cy + S * 0.70],
                   a0, a0 + 13.5, fill=col)
    # incoming beam, thick enough to read at 16px
    d.line([(S * 0.10, cy), (cx + S * 0.01, cy)], fill=beam, width=int(S * 0.085))
    return tile(base)


def meter():
    """Three ascending bars, spectrum-filled, on a light tile."""
    base = Image.new("RGB", (S, S), (252, 252, 254))
    g = grad(S, 90)
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    bw, gap = int(S * 0.155), int(S * 0.075)
    x = (S - (bw * 3 + gap * 2)) // 2
    for hf in (0.26, 0.42, 0.60):
        h = int(S * hf)
        d.rounded_rectangle([x, S * 0.78 - h, x + bw, S * 0.78], int(bw * 0.35), fill=255)
        x += bw + gap
    base.paste(g, (0, 0), m)
    return tile(base)


CANDS = [("A  bars", A_bars), ("E  prism", prism), ("D  meter", meter)]

if __name__ == "__main__":
    out = Path("brand/v5"); out.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (3 * 210 + 40, 330), (252, 252, 254))
    ImageDraw.Draw(sheet).rectangle([0, 244, sheet.width, 330], fill=(17, 17, 22))
    d = ImageDraw.Draw(sheet)
    for i, (name, fn) in enumerate(CANDS):
        big = fn()
        big.resize((512, 512), Image.LANCZOS).save(out / f"{name.split()[0]}.png")
        x = 20 + i * 210
        d.text((x, 12), name, font=font(14, 620), fill=(20, 20, 28))
        b = big.resize((128, 128), Image.LANCZOS); sheet.paste(b, (x, 36), b)
        for j, sz in enumerate((48, 32, 16)):
            s2 = big.resize((sz, sz), Image.LANCZOS)
            sheet.paste(s2, (x + j * 56, 176 + (0 if sz == 48 else (8 if sz == 32 else 16))), s2)
        z = big.resize((16, 16), Image.LANCZOS).resize((72, 72), Image.NEAREST)
        sheet.paste(z, (x + 30, 250), z)
    sheet.save(out / "_v5_sheet.png")
    print("wrote brand/v5/_v5_sheet.png")
