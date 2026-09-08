"""Ship the enhanced original: the same diagonal gesture, on a spectrum tile.

Chosen over the soft-arm "resolve" variant, whose idea was better: at 16px a
deliberately blurred arm cannot be told apart from a blurry icon, and a
sharpness product whose logo looks smudged is a worse failure than a slightly
generic one. The resolve variant is kept in brand/v6 if that trade ever changes.
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from enhance_original import V2_refined, V3_resolve

for d in ("brand", "assets/icons", "extension/icons"):
    Path(d).mkdir(parents=True, exist_ok=True)

big = V2_refined()
for n in (16, 32, 48, 128, 256, 512):
    im = big.resize((n, n), Image.LANCZOS)
    im.save(f"brand/icon{n}.png")
    if n in (16, 48, 128):
        im.save(f"assets/icons/icon{n}.png")
        im.save(f"extension/icons/icon{n}.png")
V3_resolve().resize((512, 512), Image.LANCZOS).save("brand/v6/alt_resolve.png")

sheet = Image.new("RGB", (600, 210), (252, 252, 254))
ImageDraw.Draw(sheet).rectangle([300, 0, 600, 210], fill=(17, 17, 22))
for x0 in (0, 300):
    b = big.resize((128, 128), Image.LANCZOS); sheet.paste(b, (x0 + 24, 30), b)
    for j, sz in enumerate((48, 32, 16)):
        s = big.resize((sz, sz), Image.LANCZOS)
        sheet.paste(s, (x0 + 176, 34 + j * 52), s)
    z = big.resize((16, 16), Image.LANCZOS).resize((64, 64), Image.NEAREST)
    sheet.paste(z, (x0 + 224, 40), z)
sheet.save("brand/_icon_shipped.png")
print("shipped enhanced-original icon to brand/, assets/icons/, extension/icons/")
