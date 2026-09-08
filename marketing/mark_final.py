"""The Crisp mark: slices resolving into a solid body. Flat signal green.

Refined from the v8 'F slices' study. Changes made after looking at it at 16px:
slices are thicker and more widely spaced (they merged into a smear before), and
there are three, not four, because the fourth added nothing the eye could see at
toolbar size.

Why this shape and not the others tried:
  * not a letterform -- 'E asym' resolved into a D
  * not a colour-blocked split -- flat 50/50 halves read as two rectangles
  * not an illustration of super-resolution -- per Ehrenberg-Bass, a distinctive
    asset does not need to explain the product, it needs to be unmistakable
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from brand3 import INK, GREEN, GROUND, font

S = 1024
SLICES = 3


def _draw(d, fill, m=0.145):
    mm = int(S * m)
    w = S - 2 * mm
    h = S - 2 * mm
    r = int(w * 0.115)
    bx = mm + int(w * 0.40)                    # solid body starts here
    d.rounded_rectangle([bx, mm, S - mm, S - mm], r, fill=fill)
    d.rectangle([bx, mm, bx + r, S - mm], fill=fill)
    sw = int(w * 0.105)                        # thick enough to survive 16px
    gap = int(w * 0.058)
    x = bx - gap - sw
    for sc in (0.86, 0.66):                    # two slices, stepping down
        hh = int(h * sc)
        d.rounded_rectangle([x, (S - hh) // 2, x + sw, (S + hh) // 2],
                            int(sw * 0.44), fill=fill)
        x -= gap + sw


def mark_on_green(size=S):
    """White slices on a green tile — the app icon."""
    im = Image.new("RGB", (S, S), GREEN)
    _draw(ImageDraw.Draw(im), GROUND)
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, S - 1, S - 1], int(S * 0.225), fill=255)
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(im, (0, 0), m)
    return out.resize((size, size), Image.LANCZOS) if size != S else out


def mark_flat(size=S, fill=GREEN):
    """The bare mark, no tile — for the wordmark lockup and the page."""
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    _draw(ImageDraw.Draw(im), fill)
    return im.resize((size, size), Image.LANCZOS) if size != S else im


def wordmark(h=140, fg=INK, mark_fill=GREEN, sub=True):
    """Mark + 'Crisp' set tight, Apple-style: weight and spacing carry it."""
    from PIL import ImageDraw as _D
    pad = int(h * 0.30)
    f = font(int(h * 0.80), 640)
    tmp = _D.Draw(Image.new("RGB", (10, 10)))
    tr = -int(h * 0.020)
    tw = int(track_width_local(tmp, "Crisp", f, tr))
    ms = int(h * 0.92)
    fs = font(max(int(h * 0.125), 8), 620)
    sp = h * 0.050
    subw = int(track_width_local(tmp, "AI VIDEO QUALITY ENHANCER", fs, sp)) if sub else 0
    gap = int(h * 0.26)
    W = pad * 2 + ms + gap + max(tw, subw)
    H = int(h * (1.42 if sub else 1.24))
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = _D.Draw(im)
    mk = mark_flat(ms, mark_fill)
    im.paste(mk, (pad, (H - ms) // 2), mk)
    tx = pad + ms + gap
    ty = int(H / 2 - h * 0.50) if sub else int(H / 2 - h * 0.42)
    track_local(d, (tx, ty), "Crisp", f, fg, tr)
    if sub:
        track_local(d, (tx + 2, ty + int(h * 0.86)), "AI VIDEO QUALITY ENHANCER", fs,
                    (146, 156, 149), sp)
    return im


def track_local(draw, xy, text, fnt, fill, spacing=0):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += draw.textlength(ch, font=fnt) + spacing


def track_width_local(draw, text, fnt, spacing=0):
    return sum(draw.textlength(c, font=fnt) for c in text) + spacing * (len(text) - 1)


if __name__ == "__main__":
    for d_ in ("brand", "assets/icons", "extension/icons", "brand/v9"):
        Path(d_).mkdir(parents=True, exist_ok=True)
    for n in (16, 32, 48, 128, 256, 512):
        im = mark_on_green(n)
        im.save(f"brand/icon{n}.png")
        if n in (16, 48, 128):
            im.save(f"assets/icons/icon{n}.png")
            im.save(f"extension/icons/icon{n}.png")
    wordmark(150).save("brand/v9/wordmark.png")
    wordmark(150, fg=(255, 255, 255), mark_fill=GREEN).save("brand/v9/wordmark_dark.png")
    wordmark(150, sub=False).save("brand/v9/wordmark_plain.png")
    mark_flat(512).save("brand/v9/mark.png")

    sheet = Image.new("RGB", (860, 400), (252, 252, 250))
    ImageDraw.Draw(sheet).rectangle([0, 250, 860, 400], fill=INK)
    wm = wordmark(96); sheet.paste(wm, (30, 26), wm)
    big = mark_on_green(112); sheet.paste(big, (520, 22), big)
    for j, sz in enumerate((48, 32, 16)):
        s = mark_on_green(sz); sheet.paste(s, (664 + j * 56, 40), s)
    z = mark_on_green(16).resize((80, 80), Image.NEAREST); sheet.paste(z, (664, 100), z)
    wmd = wordmark(96, fg=(255, 255, 255)); sheet.paste(wmd, (30, 274), wmd)
    for j, sz in enumerate((48, 32, 16)):
        s = mark_on_green(sz); sheet.paste(s, (664 + j * 56, 296), s)
    big2 = mark_on_green(112); sheet.paste(big2, (520, 278), big2)
    sheet.save("brand/v9/_system.png")
    print("wrote brand/v9/_system.png and shipped icons")
