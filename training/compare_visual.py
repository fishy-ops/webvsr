"""
Generate side-by-side comparison montages:
  Bicubic | SR | SR+FXAA+RCAS | Ground Truth

Usage:
    python compare_visual.py --checkpoint best_phase2.pth --images 0801.png 0802.png
    python compare_visual.py --checkpoint best_phase2.pth --count 5
"""
import argparse
import os
import random

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from model_span import SPANLite
from fxaa import fxaa_pass
from rcas import rcas_sharpen


def load_model(checkpoint_path, device):
    model = SPANLite(num_in_ch=3, num_out_ch=3, feature_channels=32, upscale=2)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state)
    model.to(device).eval()
    return model


IMG_DIR = r'/tank/webvsr/datasets/DIV2K_valid_HR'
OUT_DIR = r'comparisons'
CROP_SIZE = 256

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()


@torch.no_grad()
def make_comparison(model, img_path, device, crop_size=CROP_SIZE):
    hr_pil = Image.open(img_path).convert('RGB')
    w, h = hr_pil.size

    cx = random.randint(0, max(0, w - crop_size))
    cy = random.randint(0, max(0, h - crop_size))
    crop_w = min(crop_size, w)
    crop_h = min(crop_size, h)
    hr_crop = hr_pil.crop((cx, cy, cx + crop_w, cy + crop_h))

    crop_w = crop_w - crop_w % 2
    crop_h = crop_h - crop_h % 2
    hr_crop = hr_crop.resize((crop_w, crop_h), Image.LANCZOS)

    hr_t = to_tensor(hr_crop).unsqueeze(0).to(device)
    lr_t = F.interpolate(hr_t, scale_factor=0.5, mode='bicubic', align_corners=False).clamp(0, 1)

    with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
        sr_t = model(lr_t)
    sr_t = sr_t.float().clamp(0, 1)

    bic_t = F.interpolate(lr_t, scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1)

    # Post-processing pipeline: SR → FXAA → RCAS
    sr_fxaa_t = fxaa_pass(sr_t)
    sr_final_t = rcas_sharpen(sr_fxaa_t, sharpness=0.25)

    bic_pil = to_pil(bic_t.squeeze(0).cpu())
    sr_pil = to_pil(sr_t.squeeze(0).cpu())
    sr_final_pil = to_pil(sr_final_t.squeeze(0).cpu())
    hr_pil_out = to_pil(hr_t.squeeze(0).cpu())

    gap = 4
    num_cols = 4
    total_w = crop_w * num_cols + gap * (num_cols - 1)
    label_h = 24
    total_h = crop_h + label_h
    montage = Image.new('RGB', (total_w, total_h), (30, 30, 30))

    montage.paste(bic_pil, (0, label_h))
    montage.paste(sr_pil, (crop_w + gap, label_h))
    montage.paste(sr_final_pil, ((crop_w + gap) * 2, label_h))
    montage.paste(hr_pil_out, ((crop_w + gap) * 3, label_h))

    draw = ImageDraw.Draw(montage)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    labels = ['Bicubic 2x', 'SR (raw)', 'SR+FXAA+RCAS', 'Ground Truth']
    for i, label in enumerate(labels):
        x = (crop_w + gap) * i
        draw.text((x + 4, 4), label, fill=(200, 200, 200), font=font)

    return montage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best_phase2.pth')
    parser.add_argument('--images', nargs='*', help='Specific image filenames')
    parser.add_argument('--count', type=int, default=5, help='Random images to compare')
    parser.add_argument('--crop', type=int, default=256)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device)
    print(f"Model loaded, device: {device}")

    if args.images:
        img_files = [os.path.join(IMG_DIR, f) for f in args.images]
    else:
        all_imgs = sorted(
            f for f in os.listdir(IMG_DIR)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        )
        img_files = [os.path.join(IMG_DIR, f) for f in random.sample(all_imgs, min(args.count, len(all_imgs)))]

    for img_path in img_files:
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"Processing {name}...")
        montage = make_comparison(model, img_path, device, args.crop)
        out_path = os.path.join(OUT_DIR, f"compare_{name}.png")
        montage.save(out_path, quality=95)
        print(f"  Saved: {out_path}")

    print(f"\nDone. {len(img_files)} comparisons saved to {OUT_DIR}")


if __name__ == '__main__':
    main()
