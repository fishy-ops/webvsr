"""Generate extension icons for WebVSR."""
from PIL import Image, ImageDraw, ImageFont
import os

ICON_DIR = r"D:\webvsr\extension\icons"
os.makedirs(ICON_DIR, exist_ok=True)

SIZES = [16, 48, 128]

BG_COLOR = (0, 165, 200)
ACCENT = (255, 255, 255)

for size in SIZES:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = max(1, size // 16)
    r = max(2, size // 6)
    draw.rounded_rectangle(
        [margin, margin, size - margin - 1, size - margin - 1],
        radius=r,
        fill=BG_COLOR,
    )

    cx, cy = size // 2, size // 2

    if size >= 48:
        # Draw a stylized upward arrow / upscale symbol
        arrow_h = int(size * 0.4)
        arrow_w = int(size * 0.28)
        shaft_w = max(2, int(size * 0.1))

        # Arrow shaft
        draw.rectangle(
            [cx - shaft_w // 2, cy - arrow_h // 2 + arrow_w // 2,
             cx + shaft_w // 2, cy + arrow_h // 2],
            fill=ACCENT,
        )
        # Arrow head
        draw.polygon(
            [
                (cx, cy - arrow_h // 2),
                (cx - arrow_w // 2, cy - arrow_h // 2 + arrow_w // 2),
                (cx + arrow_w // 2, cy - arrow_h // 2 + arrow_w // 2),
            ],
            fill=ACCENT,
        )

        # Small "2x" text at bottom
        font_size = max(8, size // 8)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), "2x", font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (cx - tw // 2, cy + arrow_h // 2 - th + 2),
            "2x",
            fill=ACCENT,
            font=font,
        )
    else:
        # 16px: simple up-arrow
        draw.polygon(
            [(cx, 3), (cx - 4, 8), (cx + 4, 8)],
            fill=ACCENT,
        )
        draw.rectangle([cx - 1, 7, cx + 1, 12], fill=ACCENT)

    img.save(os.path.join(ICON_DIR, f"icon{size}.png"))
    print(f"Saved icon{size}.png")
