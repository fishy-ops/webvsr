"""Generate zoomed-in comparison crops: Bicubic | SR | SR+FXAA+RCAS | Ground Truth"""
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from model_span import SPANLite
from fxaa import fxaa_pass
from rcas import rcas_sharpen

IMG_DIR = r'/tank/webvsr/datasets/DIV2K_valid_HR'
OUT_DIR = r'comparisons_zoomed'
CKPT = r'checkpoints/best_phase2.pth'

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

CROPS = [
    ('0830.png', 800, 200, 160, 'Castle windows + stone'),
    ('0898.png', 800, 600, 160, 'Blossoms vs blue sky'),
    ('0832.png', 1600, 350, 160, 'Tram signage + stripes'),
    ('0812.png', 600, 200, 160, 'Dome ornamentation'),
    ('0823.png', 1300, 1000, 160, 'Cliff rock texture'),
    ('0849.png', 1050, 280, 160, 'Vine bark + leaves'),
]

ZOOM = 3

@torch.no_grad()
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cuda')

    model = SPANLite(num_in_ch=3, num_out_ch=3, feature_channels=32, upscale=2)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model', ckpt.get('model_state_dict', ckpt)))
    model.to(device).eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')}, PSNR {ckpt.get('best_psnr', 0):.2f})")

    for fname, cx, cy, size, label in CROPS:
        path = os.path.join(IMG_DIR, fname)
        hr_pil = Image.open(path).convert('RGB')
        w, h = hr_pil.size

        cx = min(cx, w - size)
        cy = min(cy, h - size)
        size = size - size % 2

        hr_crop = hr_pil.crop((cx, cy, cx + size, cy + size))
        hr_t = to_tensor(hr_crop).unsqueeze(0).to(device)

        lr_t = F.interpolate(hr_t, scale_factor=0.5, mode='bicubic', align_corners=False).clamp(0, 1)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            sr_t = model(lr_t).float().clamp(0, 1)

        bic_t = F.interpolate(lr_t, scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1)

        sr_fxaa_t = fxaa_pass(sr_t)
        sr_final_t = rcas_sharpen(sr_fxaa_t, sharpness=0.25)

        bic_pil = to_pil(bic_t.squeeze(0).cpu())
        sr_pil = to_pil(sr_t.squeeze(0).cpu())
        sr_final_pil = to_pil(sr_final_t.squeeze(0).cpu())
        hr_pil_out = to_pil(hr_t.squeeze(0).cpu())

        zoom_size = size * ZOOM
        bic_zoom = bic_pil.resize((zoom_size, zoom_size), Image.NEAREST)
        sr_zoom = sr_pil.resize((zoom_size, zoom_size), Image.NEAREST)
        sr_final_zoom = sr_final_pil.resize((zoom_size, zoom_size), Image.NEAREST)
        hr_zoom = hr_pil_out.resize((zoom_size, zoom_size), Image.NEAREST)

        gap = 4
        label_h = 28
        total_w = zoom_size * 4 + gap * 3
        total_h = zoom_size + label_h
        montage = Image.new('RGB', (total_w, total_h), (20, 20, 20))

        montage.paste(bic_zoom, (0, label_h))
        montage.paste(sr_zoom, (zoom_size + gap, label_h))
        montage.paste(sr_final_zoom, ((zoom_size + gap) * 2, label_h))
        montage.paste(hr_zoom, ((zoom_size + gap) * 3, label_h))

        draw = ImageDraw.Draw(montage)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()

        labels = ['Bicubic 2x', 'SR (raw)', 'SR+FXAA+RCAS', 'Ground Truth']
        for i, lbl in enumerate(labels):
            x = (zoom_size + gap) * i
            draw.text((x + 4, 4), lbl, fill=(200, 200, 200), font=font)

        name = os.path.splitext(fname)[0]
        out_path = os.path.join(OUT_DIR, f'zoom_{name}.png')
        montage.save(out_path)
        print(f"  {label}: {out_path}")

    print(f"\nDone. Zoomed comparisons in {OUT_DIR}")

if __name__ == '__main__':
    main()
