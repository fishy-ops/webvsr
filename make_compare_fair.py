"""FAIR test: realistic compressed/blurred LR (what real low-res video is like).
Bicubic | WebVSR 16ch (+sharpen) | Ground Truth."""
import torch, os, glob
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np
from model_span import SPANLite
from dataset import degrade_second_order

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VAL = r"C:\Users\reach\OneDrive\Documents\mamba-sr\DIV2K_valid_HR\DIV2K_valid_HR"
OUT = r"D:\webvsr\results\compare_fair.png"

m = SPANLite(feature_channels=16, upscale=2).to(DEV).eval()
ck = torch.load(r"D:\webvsr\checkpoints_c16\best_phase1.pth", map_location="cpu", weights_only=False)
m.load_state_dict(ck.get("model", ck))
with torch.no_grad():
    for mod in m.modules():
        if hasattr(mod, "_update_params") and hasattr(mod, "eval_conv"):
            mod._update_params()

def to_t(im): return torch.from_numpy(np.array(im)).permute(2,0,1).float().div(255).unsqueeze(0)
def to_im(a): return Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8))
def np3(t): return t.squeeze(0).permute(1,2,0).cpu().numpy()
def rcas(img, s=0.35):
    c=img
    l=np.pad(img,((0,0),(1,0),(0,0)),mode="edge")[:, :-1]; r=np.pad(img,((0,0),(0,1),(0,0)),mode="edge")[:, 1:]
    t=np.pad(img,((1,0),(0,0),(0,0)),mode="edge")[:-1]; b=np.pad(img,((0,1),(0,0),(0,0)),mode="edge")[1:]
    sh=c+s*(4*c-l-r-t-b); mn=np.minimum.reduce([c,l,r,t,b]); mx=np.maximum.reduce([c,l,r,t,b])
    return np.clip(sh,mn,mx)

torch.manual_seed(0)
files = sorted(glob.glob(os.path.join(VAL, "*.png")))[:3]
CROP = 160
rows = []
for f in files:
    hr = Image.open(f).convert("RGB")
    W,H = hr.size
    hr = hr.crop((W//2-CROP, H//2-CROP, W//2+CROP, H//2+CROP))
    hr_t = to_t(hr).to(DEV)
    lr = degrade_second_order(hr_t.squeeze(0).cpu(), scale=2).unsqueeze(0).to(DEV).clamp(0,1)  # realistic LR
    bic = F.interpolate(lr, size=hr_t.shape[2:], mode="bicubic", align_corners=False).clamp(0,1)
    with torch.no_grad():
        sr = m(lr).clamp(0,1)
        if sr.shape[2:] != hr_t.shape[2:]:
            sr = F.interpolate(sr, size=hr_t.shape[2:], mode="bicubic", align_corners=False).clamp(0,1)
    cols = [to_im(np3(bic)), to_im(rcas(np3(sr))), hr]
    w,h = cols[0].size
    row = Image.new("RGB", (w*3+16, h), (12,12,14))
    for i,im in enumerate(cols): row.paste(im, (i*(w+8),0))
    rows.append(row)

canvas = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)+len(rows)*8+26), (12,12,14))
ImageDraw.Draw(canvas).text((8,6), "Bicubic (of compressed LR)   |   WebVSR 16ch + sharpen   |   Ground Truth", fill=(210,210,210))
y=24
for r in rows: canvas.paste(r,(0,y)); y+=r.height+8
os.makedirs(os.path.dirname(OUT), exist_ok=True)
canvas.save(OUT); print("saved", OUT, canvas.size)
