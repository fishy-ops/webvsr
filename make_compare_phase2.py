"""Judge perceptual (phase2) vs PSNR (phase1) on realistic compressed input.
Bicubic | 16ch phase1 (PSNR) | 16ch phase2 (perceptual) | Ground Truth.
Raw model output (no sharpen) so any fabricated detail is visible."""
import torch, os, glob
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np
from model_span import SPANLite
from dataset import degrade_second_order

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VAL = r"C:\Users\reach\OneDrive\Documents\mamba-sr\DIV2K_valid_HR\DIV2K_valid_HR"
OUT = r"D:\webvsr\results\compare_phase2.png"

def load(ckpt):
    m = SPANLite(feature_channels=16, upscale=2).to(DEV).eval()
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(c.get("model", c))
    with torch.no_grad():
        for mod in m.modules():
            if hasattr(mod, "_update_params") and hasattr(mod, "eval_conv"):
                mod._update_params()
    return m

m1 = load(r"D:\webvsr\checkpoints_c16\best_phase1.pth")
m2 = load(r"D:\webvsr\checkpoints_c16\best_phase2.pth")

def to_t(im): return torch.from_numpy(np.array(im)).permute(2,0,1).float().div(255).unsqueeze(0)
def to_im(t): return Image.fromarray((t.squeeze(0).clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))

# Pick images with fine/regular structure (good hallucination stress test).
files = sorted(glob.glob(os.path.join(VAL, "*.png")))
picks = [files[i] for i in [3, 12, 24, 41] if i < len(files)]
CROP = 170
torch.manual_seed(0)
rows = []
for f in picks:
    hr = Image.open(f).convert("RGB"); W,H = hr.size
    hr = hr.crop((W//2-CROP, H//2-CROP, W//2+CROP, H//2+CROP))
    hr_t = to_t(hr).to(DEV)
    lr = degrade_second_order(hr_t.squeeze(0).cpu(), scale=2).unsqueeze(0).to(DEV).clamp(0,1)
    bic = F.interpolate(lr, size=hr_t.shape[2:], mode="bicubic", align_corners=False).clamp(0,1)
    with torch.no_grad():
        s1 = m1(lr).clamp(0,1); s2 = m2(lr).clamp(0,1)
        for s in (s1, s2): pass
    def fit(t): return t if t.shape[2:]==hr_t.shape[2:] else F.interpolate(t,size=hr_t.shape[2:],mode="bicubic",align_corners=False).clamp(0,1)
    cols = [to_im(bic), to_im(fit(s1)), to_im(fit(s2)), hr]
    w,h = cols[0].size
    row = Image.new("RGB",(w*4+24,h),(12,12,14))
    for i,im in enumerate(cols): row.paste(im,(i*(w+8),0))
    rows.append(row)

canvas = Image.new("RGB",(rows[0].width, sum(r.height for r in rows)+len(rows)*8+26),(12,12,14))
ImageDraw.Draw(canvas).text((8,6),"Bicubic  |  phase1 (PSNR)  |  phase2 (perceptual)  |  Ground Truth",fill=(210,210,210))
y=24
for r in rows: canvas.paste(r,(0,y)); y+=r.height+8
os.makedirs(os.path.dirname(OUT), exist_ok=True)
canvas.save(OUT); print("saved", OUT, canvas.size)
