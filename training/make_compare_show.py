"""Showcase comparison at MAX settings: native-resolution 16ch 2x model plus a
strong sharpen (what you get on GPU load = Max, Sharpness cranked up).
Layout: Plain upscale | WebVSR (enhanced) | Original, larger and labelled."""
import torch, os, glob
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from model_span import SPANLite
from dataset import degrade_second_order

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VAL = r"C:\Users\reach\OneDrive\Documents\mamba-sr\DIV2K_valid_HR\DIV2K_valid_HR"
OUT = r"D:\webvsr\results\compare_showcase.png"
SHARP = 1.3          # strong sharpen (extension "High" is 1.4)
CROP = 190           # half-size of the center crop -> 380px tiles
INDICES = [0, 1, 2]  # which sorted validation images to use

m = SPANLite(feature_channels=16, upscale=2).to(DEV).eval()
ck = torch.load(r"D:\webvsr\checkpoints_c16\best_phase1.pth", map_location="cpu", weights_only=False)
m.load_state_dict(ck.get("model", ck))
with torch.no_grad():
    for mod in m.modules():
        if hasattr(mod, "_update_params") and hasattr(mod, "eval_conv"):
            mod._update_params()

def to_t(im): return torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255).unsqueeze(0)
def to_im(a): return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
def np3(t): return t.squeeze(0).permute(1, 2, 0).cpu().numpy()
def rcas(img, s):
    c = img
    l = np.pad(img, ((0, 0), (1, 0), (0, 0)), mode="edge")[:, :-1]
    r = np.pad(img, ((0, 0), (0, 1), (0, 0)), mode="edge")[:, 1:]
    t = np.pad(img, ((1, 0), (0, 0), (0, 0)), mode="edge")[:-1]
    b = np.pad(img, ((0, 1), (0, 0), (0, 0)), mode="edge")[1:]
    sh = c + s * (4 * c - l - r - t - b)
    mn = np.minimum.reduce([c, l, r, t, b]); mx = np.maximum.reduce([c, l, r, t, b])
    return np.clip(sh, mn, mx)

torch.manual_seed(0)
files = sorted(glob.glob(os.path.join(VAL, "*.png")))
sel = [files[i] for i in INDICES]

GAP, HDR, BG = 10, 40, (16, 16, 20)
rows = []
for f in sel:
    hr = Image.open(f).convert("RGB")
    W, H = hr.size
    hr = hr.crop((W // 2 - CROP, H // 2 - CROP, W // 2 + CROP, H // 2 + CROP))
    hr_t = to_t(hr).to(DEV)
    lr = degrade_second_order(hr_t.squeeze(0).cpu(), scale=2).unsqueeze(0).to(DEV).clamp(0, 1)
    bic = F.interpolate(lr, size=hr_t.shape[2:], mode="bicubic", align_corners=False).clamp(0, 1)
    with torch.no_grad():
        sr = m(lr).clamp(0, 1)
        if sr.shape[2:] != hr_t.shape[2:]:
            sr = F.interpolate(sr, size=hr_t.shape[2:], mode="bicubic", align_corners=False).clamp(0, 1)
    cols = [to_im(np3(bic)), to_im(rcas(np3(sr), SHARP)), hr]
    w, h = cols[0].size
    row = Image.new("RGB", (w * 3 + GAP * 2, h), BG)
    for i, im in enumerate(cols):
        row.paste(im, (i * (w + GAP), 0))
    rows.append(row)

CW = rows[0].width
canvas = Image.new("RGB", (CW, HDR + sum(r.height for r in rows) + GAP * len(rows)), BG)
d = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arialbd.ttf", 18)
except OSError:
    font = ImageFont.load_default()

w = 2 * CROP
labels = [("Plain upscale", (170, 170, 176)), ("WebVSR (enhanced)", (90, 205, 245)), ("Original", (170, 170, 176))]
for i, (txt, col) in enumerate(labels):
    cx = i * (w + GAP) + w // 2
    d.text((cx, HDR // 2), txt, font=font, fill=col, anchor="mm")

y = HDR
for r in rows:
    canvas.paste(r, (0, y)); y += r.height + GAP

os.makedirs(os.path.dirname(OUT), exist_ok=True)
canvas.save(OUT)
print("saved", OUT, canvas.size)
