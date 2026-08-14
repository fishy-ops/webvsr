"""Visual quality check for the 16-channel model: Bicubic | WebVSR(16ch) | GT."""
import torch, os, glob
import torch.nn.functional as F
from PIL import Image
import numpy as np
from model_span import SPANLite

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VAL = r"C:\Users\reach\OneDrive\Documents\mamba-sr\DIV2K_valid_HR\DIV2K_valid_HR"
OUT = r"D:\webvsr\results\compare_c16.png"

m = SPANLite(feature_channels=16, upscale=2).to(DEV).eval()
ck = torch.load(r"D:\webvsr\checkpoints_c16\best_phase1.pth", map_location="cpu", weights_only=False)
m.load_state_dict(ck.get("model", ck))
with torch.no_grad():
    for mod in m.modules():
        if hasattr(mod, "_update_params") and hasattr(mod, "eval_conv"):
            mod._update_params()

def to_t(im): return torch.from_numpy(np.array(im)).permute(2,0,1).float().div(255).unsqueeze(0)
def to_im(t): return Image.fromarray((t.squeeze(0).clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))

files = sorted(glob.glob(os.path.join(VAL, "*.png")))[:3]
CROP = 160  # HR crop size (shows a zoomed region)
rows = []
for f in files:
    hr = Image.open(f).convert("RGB")
    W, H = hr.size
    cx, cy = W//2, H//2
    hr = hr.crop((cx-CROP, cy-CROP, cx+CROP, cy+CROP))          # 320x320 GT
    hr_t = to_t(hr).to(DEV)
    lr_t = F.interpolate(hr_t, scale_factor=0.5, mode="bicubic", align_corners=False).clamp(0,1)
    bic = F.interpolate(lr_t, scale_factor=2, mode="bicubic", align_corners=False).clamp(0,1)
    with torch.no_grad():
        sr = m(lr_t).clamp(0,1)
    trip = [to_im(bic), to_im(sr), hr]
    w, h = trip[0].size
    row = Image.new("RGB", (w*3+16, h), (12,12,14))
    for i, im in enumerate(trip): row.paste(im, (i*(w+8), 0))
    rows.append(row)

W = rows[0].width
canvas = Image.new("RGB", (W, sum(r.height for r in rows)+len(rows)*8+30), (12,12,14))
from PIL import ImageDraw
d = ImageDraw.Draw(canvas)
d.text((8, 6), "Bicubic            |            WebVSR 16ch            |            Ground Truth", fill=(210,210,210))
y = 26
for r in rows: canvas.paste(r, (0, y)); y += r.height+8
os.makedirs(os.path.dirname(OUT), exist_ok=True)
canvas.save(OUT)
print("saved", OUT, canvas.size)
