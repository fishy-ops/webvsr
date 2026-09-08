"""The split mark, three constructions. Flat colour only.

Each is the same idea -- a hard vertical division between a dull side and a
vivid side -- built differently, and each is judged at 16px because that is
where a toolbar renders it.
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from brand3 import INK, GREEN, DULL, GROUND, SURFACE, font

S = 1024
R = 0.225


def tile(img, r=R):
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, S - 1, S - 1], int(S * r), fill=255)
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(img, (0, 0), m)
    return out


def A_halves():
    """The plainest statement: dull left, signal green right, hard seam."""
    im = Image.new("RGB", (S, S), DULL)
    d = ImageDraw.Draw(im)
    d.rectangle([S // 2, 0, S, S], fill=GREEN)
    d.rectangle([S // 2 - 6, 0, S // 2 + 5, S], fill=GROUND)
    return tile(im)


def B_blocks():
    """Left side breaks into coarse blocks -- compression -- right side is whole."""
    im = Image.new("RGB", (S, S), GREEN)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, S // 2, S], fill=DULL)
    n = 4
    cell = (S // 2) / n
    for r in range(n):
        for c in range(n):
            if (r + c) % 2:
                continue
            d.rectangle([c * cell, r * cell * (S / (S / 1)) / 1 + r * (S / n - cell),
                         (c + 1) * cell, r * (S / n) + S / n], fill=(178, 188, 181))
    d.rectangle([0, 0, S // 2, S], outline=None)
    # redraw cleanly: a checker of two dull tones on the left half only
    d.rectangle([0, 0, S // 2, S], fill=DULL)
    cw, ch = (S / 2) / n, S / n
    for r in range(n):
        for c in range(n):
            if (r + c) % 2 == 0:
                d.rectangle([c * cw, r * ch, (c + 1) * cw, (r + 1) * ch], fill=(176, 187, 180))
    d.rectangle([S // 2 - 6, 0, S // 2 + 5, S], fill=GROUND)
    return tile(im)


def C_offset():
    """A green field with a narrow dull band -- the divider off-centre, so the
    shape is asymmetric and therefore more memorable than a bisected square."""
    im = Image.new("RGB", (S, S), GREEN)
    d = ImageDraw.Draw(im)
    x = int(S * 0.34)
    d.rectangle([0, 0, x, S], fill=DULL)
    d.rectangle([x - 6, 0, x + 5, S], fill=GROUND)
    return tile(im)


def D_bar():
    """Green tile, one white split bar, left side knocked back. Reads as a
    single object rather than two halves."""
    im = Image.new("RGB", (S, S), GREEN)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, int(S * 0.46), S], fill=(0, 138, 80))
    d.rectangle([int(S * 0.46) - 7, 0, int(S * 0.46) + 6, S], fill=GROUND)
    return tile(im)


CANDS = [("A halves", A_halves), ("B blocks", B_blocks),
         ("C offset", C_offset), ("D tone", D_bar)]

if __name__ == "__main__":
    out = Path("brand/v7"); out.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (4 * 176 + 30, 316), (250, 250, 248))
    ImageDraw.Draw(sheet).rectangle([0, 232, sheet.width, 316], fill=(15, 15, 18))
    d = ImageDraw.Draw(sheet)
    for i, (name, fn) in enumerate(CANDS):
        big = fn()
        big.resize((512, 512), Image.LANCZOS).save(out / f"{name.replace(' ', '_')}.png")
        x = 16 + i * 176
        d.text((x, 12), name, font=font(14, 620), fill=INK)
        b = big.resize((122, 122), Image.LANCZOS); sheet.paste(b, (x, 34), b)
        for j, sz in enumerate((48, 16)):
            s2 = big.resize((sz, sz), Image.LANCZOS)
            sheet.paste(s2, (x + j * 64, 170 + (0 if sz == 48 else 16)), s2)
        z = big.resize((16, 16), Image.LANCZOS).resize((70, 70), Image.NEAREST)
        sheet.paste(z, (x + 24, 238), z)
    sheet.save(out / "_v7_sheet.png")
    print("wrote brand/v7/_v7_sheet.png")
