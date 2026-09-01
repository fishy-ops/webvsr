"""4-way: Bicubic | WebVSR 16ch | WebVSR 16ch + sharpen | Ground Truth."""
import torch, os, glob
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np
from model_span import SPANLite

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VAL = r"/tank/webvsr/datasets/DIV2K_valid_HR"
OUT = r"results/compare_sharp.png"

m = SPANLite(feature_channels=16, upscale=2).to(DEV).eval()
ck = torch.load(r"checkpoints_c16/best_phase1.pth", map_location="cpu", weights_only=False)
m.load_state_dict(ck.get("model", ck))
with torch.no_grad():
    for mod in m.modules():
        if hasattr(mod, "_update_params") and hasattr(mod, "eval_conv"):
            mod._update_params()

def to_t(im): return torch.from_numpy(np.array(im)).permute(2,0,1).float().div(255).unsqueeze(0)
def to_im(a): return Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8))

def rcas(img, s=0.45):  # img HxWx3 float; clamped unsharp (matches the WGSL pass)
    c = img
    l = np.pad(img, ((0,0),(1,0),(0,0)), mode="edge")[:, :-1]
    r = np.pad(img, ((0,0),(0,1),(0,0)), mode="edge")[:, 1:]
    t = np.pad(img, ((1,0),(0,0),(0,0)), mode="edge")[:-1]
    b = np.pad(img, ((0,1),(0,0),(0,0)), mode="edge")[1:]
    sharp = c + s*(4*c - l - r - t - b)
    mn = np.minimum.reduce([c,l,r,t,b]); mx = np.maximum.reduce([c,l,r,t,b])
    return np.clip(sharp, mn, mx)

files = sorted(glob.glob(os.path.join(VAL, "*.png")))[:3]
CROP = 160
rows = []
for f in files:
    hr = Image.open(f).convert("RGB")
    W, H = hr.size
    hr = hr.crop((W//2-CROP, H//2-CROP, W//2+CROP, H//2+CROP))
    hr_t = to_t(hr).to(DEV)
    lr = F.interpolate(hr_t, scale_factor=0.5, mode="bicubic", align_corners=False).clamp(0,1)
    bic = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False).clamp(0,1)
    with torch.no_grad():
        sr = m(lr).clamp(0,1)
    sr_np = sr.squeeze(0).permute(1,2,0).cpu().numpy()
    cols = [to_im(bic.squeeze(0).permute(1,2,0).cpu().numpy()),
            to_im(sr_np), to_im(rcas(sr_np)), hr]
    w, h = cols[0].size
    row = Image.new("RGB", (w*4+24, h), (12,12,14))
    for i, im in enumerate(cols): row.paste(im, (i*(w+8), 0))
    rows.append(row)

canvas = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)+len(rows)*8+26), (12,12,14))
ImageDraw.Draw(canvas).text((8,6), "Bicubic   |   WebVSR 16ch   |   16ch + sharpen   |   Ground Truth", fill=(210,210,210))
y = 24
for r in rows: canvas.paste(r, (0, y)); y += r.height+8
os.makedirs(os.path.dirname(OUT), exist_ok=True)
canvas.save(OUT); print("saved", OUT, canvas.size)
