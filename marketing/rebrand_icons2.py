"""Refined bright candidates.

Kept from round one: a full-bleed spectrum tile is the brightest thing possible
in a store grid, and the resolving-bars motif is the only mark that says what the
product does rather than just "video". These four combine those two ideas.

Dropped: anything on a dark tile, and anything using a single warm hue.
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from rebrand_icons import grad_image, rounded_mask, SPECTRA, S

WHITE = (255, 255, 255)


def bars_mask(size=S, blur_left=True, n=5, m=0.175):
    """Bars that widen and soften to the left, tighten and sharpen to the right."""
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    mm = int(size * m)
    w = size - 2 * mm
    x = mm
    widths = [0.255, 0.225, 0.195, 0.165, 0.135]
    gaps = [0.022, 0.026, 0.030, 0.034]
    for i in range(n):
        bw = int(w * widths[i])
        d.rectangle([x, mm, x + bw, size - mm], fill=255)
        if i < n - 1:
            x += bw + int(w * gaps[i])
    if blur_left:
        half = mask.crop((0, 0, int(size * 0.46), size))
        half = half.filter(ImageFilter.GaussianBlur(size * 0.020))
        mask.paste(half, (0, 0))
    return mask


def A_spectrum_bars(spec="vivid"):
    """Full-bleed spectrum, white bars punched through it."""
    tile = grad_image(SPECTRA[spec], S, 40).convert("RGB")
    white = Image.new("RGB", (S, S), WHITE)
    tile.paste(white, (0, 0), bars_mask())
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(tile, (0, 0), rounded_mask())
    return out


def B_spectrum_play(spec="prism"):
    """Full-bleed spectrum, white play triangle. Instantly 'video'."""
    tile = grad_image(SPECTRA[spec], S, 40).convert("RGB")
    d = ImageDraw.Draw(tile)
    cx, cy, r = S * 0.525, S * 0.5, S * 0.205
    d.polygon([(cx - r * 0.70, cy - r), (cx - r * 0.70, cy + r), (cx + r * 0.88, cy)], fill=WHITE)
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(tile, (0, 0), rounded_mask())
    return out


def C_white_bars(spec="candy"):
    """White tile, spectrum bars. Lighter, calmer, still multi-hue."""
    tile = Image.new("RGB", (S, S), WHITE)
    tile.paste(grad_image(SPECTRA[spec], S, 12).convert("RGB"), (0, 0), bars_mask())
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(tile, (0, 0), rounded_mask())
    return out


def D_spectrum_lens(spec="aurora"):
    """Full-bleed spectrum, white aperture ring with a clear centre."""
    tile = grad_image(SPECTRA[spec], S, 40).convert("RGB")
    ring = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(ring)
    c, o, t = S / 2, S * 0.315, S * 0.105
    d.ellipse([c - o, c - o, c + o, c + o], fill=255)
    d.ellipse([c - (o - t), c - (o - t), c + (o - t), c + (o - t)], fill=0)
    tile.paste(Image.new("RGB", (S, S), WHITE), (0, 0), ring)
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(tile, (0, 0), rounded_mask())
    return out


CANDS = [("A_spectrum_bars", A_spectrum_bars), ("B_spectrum_play", B_spectrum_play),
         ("C_white_bars", C_white_bars), ("D_spectrum_lens", D_spectrum_lens)]

if __name__ == "__main__":
    out = Path("brand/v2"); out.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (4 * 168 + 40, 310), (247, 247, 250))
    ImageDraw.Draw(sheet).rectangle([0, 222, sheet.width, 310], fill=(16, 16, 20))
    for i, (name, fn) in enumerate(CANDS):
        big = fn()
        big.resize((512, 512), Image.LANCZOS).save(out / f"{name}.png")
        x = 20 + i * 168
        b128 = big.resize((128, 128), Image.LANCZOS)
        sheet.paste(b128, (x, 20), b128)
        for j, sz in enumerate((48, 32, 16)):
            s2 = big.resize((sz, sz), Image.LANCZOS)
            sheet.paste(s2, (x + j * 54, 168), s2)
        s16 = big.resize((16, 16), Image.LANCZOS).resize((76, 76), Image.NEAREST)
        sheet.paste(s16, (x + 28, 232))
    sheet.save(out / "_v2_refined.png")
    print("wrote brand/v2/_v2_refined.png   A B C D — 128px, 48/32/16, 16px magnified on dark")
