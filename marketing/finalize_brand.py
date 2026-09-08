"""Ship the v3 mark: a spectrum-filled 'C' on an ink tile, plus the wordmark.

The synthesis of two rounds of feedback -- a simple typographic logo, with the
colour that was asked for present but restrained. The saturated work happens on
the landing page, not on a 16px tile.
"""
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from wordmark import icon_grad_letter, icon_ink, icon_light, wordmark

for d in ("brand", "assets/icons", "extension/icons", "brand/v3"):
    Path(d).mkdir(parents=True, exist_ok=True)

for n in (16, 32, 48, 128, 256, 512):
    im = icon_grad_letter(n)
    im.save(f"brand/icon{n}.png")
    if n in (16, 48, 128):
        im.save(f"assets/icons/icon{n}.png")
        im.save(f"extension/icons/icon{n}.png")

# monochrome fallbacks, for places colour is wrong (favicons, print, watermark)
icon_ink(512).save("brand/v3/icon_mono_ink.png")
icon_light(512).save("brand/v3/icon_mono_light.png")
wordmark(160).save("brand/v3/wordmark_ink.png")
wordmark(160, fg=(255, 255, 255)).save("brand/v3/wordmark_white.png")
wordmark(160, sub=False).save("brand/v3/wordmark_plain.png")
print("v3 mark shipped to brand/, assets/icons/, extension/icons/")
