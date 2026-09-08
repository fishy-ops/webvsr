"""Non-letter marks. No C, no gapped ring, nothing that rhymes with v1.

Hard constraints this round:
  * not a letterform of any kind
  * not a circle-with-a-gap (that is what the rejected mark was)
  * bright / multi-hue
  * legible at 16px, which is where a toolbar renders it
  * few elements -- the Apple read comes from restraint, not detail

The wordmark stays the primary logo; this is only the square tile that Chrome
demands and that a wordmark cannot fill.
"""
import sys, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from brand2 import SPECTRUM, INK, ramp, font

S = 1024
R = 0.225
W = (255, 255, 255)


def grad(size=S, angle=40, stops=None):
    d = int(size * 1.7)
    strip = Image.new("RGB", (d, 1)); px = strip.load()
    for x, c in enumerate(ramp(stops or SPECTRUM, d)):
        px[x, 0] = c
    g = strip.resize((d, d), Image.NEAREST).rotate(angle, resample=Image.BICUBIC)
    o = (d - size) // 2
    return g.crop((o, o, o + size, o + size))


def tile(img):
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, S - 1, S - 1], int(S * R), fill=255)
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(img, (0, 0), m)
    return out


def A_bars():
    """Five bars: wide and soft on the left, tight and sharp on the right."""
    base = grad(S, 35).convert("RGB")
    mask = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(mask)
    m = int(S * 0.175); w = S - 2 * m; x = m
    for i, wf in enumerate([0.255, 0.225, 0.195, 0.165, 0.135]):
        bw = int(w * wf)
        d.rectangle([x, m, x + bw, S - m], fill=255)
        x += bw + int(w * (0.022 + 0.004 * i))
    left = mask.crop((0, 0, int(S * 0.46), S)).filter(ImageFilter.GaussianBlur(S * 0.020))
    mask.paste(left, (0, 0))
    base.paste(Image.new("RGB", (S, S), W), (0, 0), mask)
    return tile(base)


def B_shard():
    """One diagonal: blurred spectrum on one side, a crisp white wedge on the other."""
    base = grad(S, 55).convert("RGB")
    soft = base.filter(ImageFilter.GaussianBlur(S * 0.055))
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).polygon([(0, 0), (S, 0), (0, S)], fill=255)
    base.paste(soft, (0, 0), m)
    d = ImageDraw.Draw(base)
    t = int(S * 0.075)
    d.line([(S * 0.20, S * 0.80), (S * 0.80, S * 0.20)], fill=W, width=t)
    return tile(base)


def C_iris():
    """A straight-edged aperture: six blades, not a ring."""
    base = grad(S, 40).convert("RGB")
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    c, rad = S / 2, S * 0.30
    pts = [(c + rad * math.cos(math.radians(a - 90)),
            c + rad * math.sin(math.radians(a - 90))) for a in range(0, 360, 60)]
    d.polygon(pts, fill=255)
    base.paste(Image.new("RGB", (S, S), W), (0, 0), m)
    d2 = ImageDraw.Draw(base)
    for i in range(6):
        a = math.radians(i * 60 - 90)
        d2.line([(c, c), (c + rad * 1.02 * math.cos(a), c + rad * 1.02 * math.sin(a))],
                fill=tuple(int(v * .0) for v in (0, 0, 0)) if False else (20, 20, 28),
                width=int(S * 0.022))
    return tile(base)


def D_meter():
    """Three ascending bars — signal, quality, level. Reads instantly."""
    base = Image.new("RGB", (S, S), INK)
    g = grad(S, 90)
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    bw, gap = int(S * 0.155), int(S * 0.075)
    total = bw * 3 + gap * 2
    x = (S - total) // 2
    for i, hf in enumerate((0.26, 0.42, 0.60)):
        h = int(S * hf)
        d.rounded_rectangle([x, S * 0.78 - h, x + bw, S * 0.78], int(bw * 0.35), fill=255)
        x += bw + gap
    base.paste(g, (0, 0), m)
    return tile(base)


def E_prism():
    """White light in, spectrum out. What resolution looks like as a picture."""
    base = Image.new("RGB", (S, S), INK)
    d = ImageDraw.Draw(base)
    # incoming beam
    d.line([(S * 0.06, S * 0.40), (S * 0.42, S * 0.40)], fill=W, width=int(S * 0.055))
    # the fanned spectrum, thick wedges so it survives 16px
    cx, cy = S * 0.44, S * 0.42
    n = len(SPECTRUM)
    for i, col in enumerate(SPECTRUM):
        a0 = -4 + i * 15
        a1 = a0 + 15
        d.pieslice([cx - S * 0.62, cy - S * 0.62, cx + S * 0.62, cy + S * 0.62],
                   a0, a1, fill=col)
    d.polygon([(cx, cy), (cx, cy + S * 0.02), (cx + S * 0.02, cy)], fill=INK)
    return tile(base)


def F_grid():
    """A 3x3 field resolving left to right: blocks become pixels become detail."""
    base = grad(S, 20).convert("RGB")
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    mm = int(S * 0.19); w = S - 2 * mm
    cell = w / 3
    for r in range(3):
        for c in range(3):
            inset = cell * (0.06 + 0.10 * (2 - c))
            d.rounded_rectangle([mm + c * cell + inset, mm + r * cell + inset,
                                 mm + (c + 1) * cell - inset, mm + (r + 1) * cell - inset],
                                int(cell * 0.16), fill=255)
    base.paste(Image.new("RGB", (S, S), W), (0, 0), m)
    return tile(base)


CANDS = [("A bars", A_bars), ("B shard", B_shard), ("C iris", C_iris),
         ("D meter", D_meter), ("E prism", E_prism), ("F grid", F_grid)]

if __name__ == "__main__":
    out = Path("brand/v4"); out.mkdir(parents=True, exist_ok=True)
    cols = len(CANDS)
    sheet = Image.new("RGB", (cols * 158 + 30, 320), (252, 252, 254))
    ImageDraw.Draw(sheet).rectangle([0, 236, sheet.width, 320], fill=(17, 17, 22))
    d = ImageDraw.Draw(sheet)
    for i, (name, fn) in enumerate(CANDS):
        big = fn()
        big.resize((512, 512), Image.LANCZOS).save(out / f"{name.split()[0]}_{name.split()[1]}.png")
        x = 16 + i * 158
        d.text((x, 12), name, font=font(13, 620), fill=(20, 20, 28))
        b = big.resize((112, 112), Image.LANCZOS); sheet.paste(b, (x, 34), b)
        for j, sz in enumerate((48, 16)):
            s2 = big.resize((sz, sz), Image.LANCZOS)
            sheet.paste(s2, (x + j * 60, 158 + (0 if sz == 48 else 16)), s2)
        z = big.resize((16, 16), Image.LANCZOS).resize((68, 68), Image.NEAREST)
        sheet.paste(z, (x + 22, 244), z)
    sheet.save(out / "_v4_sheet.png")
    print("wrote brand/v4/_v4_sheet.png — 112px, 48/16, then 16px magnified on dark")
