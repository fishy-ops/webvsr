"""Logo v3: a typographic wordmark, with the colour moved to the page.

The brief changed to an Apple-style approach -- a quiet mark that earns trust by
restraint, and a product page that does the showing off. So the spectrum does not
disappear, it relocates: nothing on the logo, everything on the landing page.

Three icon treatments are drawn against the same wordmark, because Chrome still
demands a square tile and a wordmark cannot be one. Each is judged at 16px.
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from brand2 import INK, GROUND, MUTED, SPECTRUM, ramp, font, track, track_width

S = 1024
RADIUS = 0.225


def grad(size, angle=40, stops=None):
    d = int(size * 1.6)
    strip = Image.new("RGB", (d, 1)); px = strip.load()
    for x, c in enumerate(ramp(stops or SPECTRUM, d)):
        px[x, 0] = c
    g = strip.resize((d, d), Image.NEAREST).rotate(angle, resample=Image.BICUBIC)
    o = (d - size) // 2
    return g.crop((o, o, o + size, o + size))


def rounded(size, r=RADIUS):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size - 1, size - 1], int(size * r), fill=255)
    return m


def letterform(size, fg, bg, weight=760):
    """A single 'C' set in the wordmark's own face, optically centred.

    Uses the typeface rather than a drawn ring so the tile and the wordmark are
    demonstrably the same letter -- which is the whole point of a type logo.
    """
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    f = font(int(size * 0.66), weight)
    box = d.textbbox((0, 0), "C", font=f)
    w, h = box[2] - box[0], box[3] - box[1]
    d.text((size / 2 - w / 2 - box[0], size / 2 - h / 2 - box[1]), "C", font=f, fill=fg)
    return im


def icon_light(size=S):
    """Ink letter on a near-white tile. The quietest option."""
    im = letterform(S, INK, (250, 250, 252))
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0)); out.paste(im, (0, 0), rounded(S))
    return out.resize((size, size), Image.LANCZOS)


def icon_ink(size=S):
    """White letter on an ink tile. Reads as a system app."""
    im = letterform(S, (255, 255, 255), INK)
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0)); out.paste(im, (0, 0), rounded(S))
    return out.resize((size, size), Image.LANCZOS)


def icon_grad_letter(size=S):
    """Ink tile, letter filled with the spectrum -- one restrained nod to colour."""
    im = Image.new("RGB", (S, S), INK)
    mask = letterform(S, (255, 255, 255), (0, 0, 0)).convert("L")
    im.paste(grad(S, 30), (0, 0), mask)
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0)); out.paste(im, (0, 0), rounded(S))
    return out.resize((size, size), Image.LANCZOS)


def wordmark(h=120, fg=INK, bg=None, sub=True):
    """'Crisp' set tight. Tracking is negative because at display sizes the
    default spacing in a UI face looks loose and unbranded."""
    pad = int(h * 0.35)
    f = font(int(h * 0.86), 620)
    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    tr = -int(h * 0.022)
    w = int(track_width(tmp, "Crisp", f, tr))
    # The descriptor is a caption, not a second headline: ~13% of the wordmark
    # height, letter-spaced wide so it reads as a label at any size.
    fs = font(max(int(h * 0.132), 8), 620)
    sp = h * 0.052
    subw = int(track_width(tmp, "AI VIDEO QUALITY ENHANCER", fs, sp)) if sub else 0
    W = max(w, subw) + pad * 2
    H = int(h * (1.52 if sub else 1.28))
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0) if bg is None else bg)
    d = ImageDraw.Draw(im)
    track(d, (pad, int(h * 0.14)), "Crisp", f, fg, tr)
    if sub:
        track(d, (pad + 2, int(h * 1.10)), "AI VIDEO QUALITY ENHANCER", fs, MUTED, sp)
    return im


if __name__ == "__main__":
    out = Path("brand/v3"); out.mkdir(parents=True, exist_ok=True)
    wordmark(160).save(out / "wordmark_ink.png")
    wordmark(160, fg=(255, 255, 255)).save(out / "wordmark_white.png")

    sheet = Image.new("RGB", (980, 470), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    wm = wordmark(112)
    sheet.paste(wm, (60, 34), wm)
    d.line([(60, 190), (920, 190)], fill=(226, 226, 236), width=1)
    d.text((60, 208), "icon treatments — 128 / 48 / 16, then 16 magnified",
           font=font(15, 500), fill=MUTED)
    for i, (name, fn) in enumerate([("light", icon_light), ("ink", icon_ink),
                                    ("spectrum letter", icon_grad_letter)]):
        x = 60 + i * 300
        d.text((x, 240), name, font=font(14, 600), fill=INK)
        big = fn(112); sheet.paste(big, (x, 266), big)
        for j, sz in enumerate((48, 16)):
            s2 = fn(sz); sheet.paste(s2, (x + 130 + j * 60, 266 + (0 if sz == 48 else 16)), s2)
        z = fn(16).resize((64, 64), Image.NEAREST); sheet.paste(z, (x + 130, 330), z)
    for i, (name, fn) in enumerate([("light", icon_light), ("ink", icon_ink),
                                    ("spectrum letter", icon_grad_letter)]):
        pass
    sheet.save(out / "_v3_sheet.png")
    print("wrote brand/v3/_v3_sheet.png")
