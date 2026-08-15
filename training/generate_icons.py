"""Generate extension icons for WebVSR.

Mark: two arrows pointing outward (diagonal "expand" / upscale) in white on a
cyan rounded square. Supersampled + LANCZOS-downsampled for clean anti-aliasing.
"""
from PIL import Image, ImageDraw
import os

ICON_DIR = r"D:\webvsr\extension\icons"
os.makedirs(ICON_DIR, exist_ok=True)

SIZES = [16, 48, 128]
BG = (0, 165, 200, 255)   # brand cyan
FG = (255, 255, 255, 255)

# Line segments in a 24-unit viewBox: top-right arrow + bottom-left arrow, each a
# corner bracket plus a diagonal shaft toward the centre (arrows pointing out).
SEGMENTS = [
    [(15, 3), (21, 3)], [(21, 3), (21, 9)], [(21, 3), (14, 10)],   # top-right
    [(9, 21), (3, 21)], [(3, 21), (3, 15)], [(3, 21), (10, 14)],   # bottom-left
]
VERTS = [(15, 3), (21, 3), (21, 9), (14, 10), (9, 21), (3, 21), (3, 15), (10, 14)]


def draw_icon(size):
    ss = 4                      # supersample factor
    S = size * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    margin = max(1, S // 16)
    radius = max(2, S // 5)
    d.rounded_rectangle([margin, margin, S - margin - 1, S - margin - 1],
                        radius=radius, fill=BG)

    m = S * 0.16
    sc = (S - 2 * m) / 24.0
    def P(pt): return (m + pt[0] * sc, m + pt[1] * sc)

    w = max(2, int(2.6 * sc))   # stroke width
    for a, b in SEGMENTS:
        d.line([P(a), P(b)], fill=FG, width=w)
    r = w / 2.0                 # round caps/joints
    for v in VERTS:
        x, y = P(v)
        d.ellipse([x - r, y - r, x + r, y + r], fill=FG)

    img = img.resize((size, size), Image.LANCZOS)
    img.save(os.path.join(ICON_DIR, f"icon{size}.png"))
    print(f"Saved icon{size}.png")


for s in SIZES:
    draw_icon(s)
