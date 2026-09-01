"""
Generate an HTML comparison page with slider-based before/after views.
Run after training to create a visual quality report.

Usage:
    python generate_comparison_page.py --checkpoint best_phase2.pth --count 8
"""
import argparse
import os
import base64
import io
import random

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model_span import SPANLite

IMG_DIR = r'/tank/webvsr/datasets/DIV2K_valid_HR'
CROP_SIZE = 384
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()


def img_to_data_uri(pil_img, fmt='JPEG', quality=90):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = 'image/jpeg' if fmt == 'JPEG' else 'image/png'
    return f'data:{mime};base64,{b64}'


@torch.no_grad()
def process_image(model, img_path, device, crop_size=CROP_SIZE):
    hr_pil = Image.open(img_path).convert('RGB')
    w, h = hr_pil.size

    cx = random.randint(0, max(0, w - crop_size))
    cy = random.randint(0, max(0, h - crop_size))
    cw = min(crop_size, w)
    ch = min(crop_size, h)
    hr_crop = hr_pil.crop((cx, cy, cx + cw, cy + ch))
    cw = cw - cw % 2
    ch = ch - ch % 2
    hr_crop = hr_crop.resize((cw, ch), Image.LANCZOS)

    hr_t = to_tensor(hr_crop).unsqueeze(0).to(device)
    lr_t = F.interpolate(hr_t, scale_factor=0.5, mode='bicubic', align_corners=False).clamp(0, 1)

    with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
        sr_t = model(lr_t)
    sr_t = sr_t.float().clamp(0, 1)

    bic_t = F.interpolate(lr_t, scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1)

    bic_pil = to_pil(bic_t.squeeze(0).cpu())
    sr_pil = to_pil(sr_t.squeeze(0).cpu())

    return bic_pil, sr_pil


def generate_html(comparisons, output_path):
    cards_html = []
    for i, (name, bic_uri, sr_uri) in enumerate(comparisons):
        cards_html.append(f'''
        <div class="card">
          <div class="card-title">{name}</div>
          <div class="slider-container" id="sc{i}">
            <img class="img-left" src="{sr_uri}" alt="SR">
            <img class="img-right" src="{bic_uri}" alt="Bicubic">
            <div class="slider-line" id="sl{i}"></div>
            <input type="range" min="0" max="100" value="50" class="slider"
                   oninput="updateSlider({i}, this.value)">
            <span class="label-left">SPAN-Lite 2x</span>
            <span class="label-right">Bicubic 2x</span>
          </div>
        </div>''')

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>WebVSR Quality Comparison</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Segoe UI", sans-serif; background: #0e1117; color: #c9d1d9; padding: 32px; }}
  h1 {{ font-size: 20px; margin-bottom: 8px; }}
  .subtitle {{ color: #6b7a8d; font-size: 13px; margin-bottom: 28px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 20px; }}
  .card {{ background: #161b22; border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; overflow: hidden; }}
  .card-title {{ padding: 10px 14px; font-size: 12px; color: #8b949e; border-bottom: 1px solid rgba(255,255,255,0.04); }}
  .slider-container {{ position: relative; width: 100%; aspect-ratio: 1; overflow: hidden; }}
  .slider-container img {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; }}
  .img-right {{ clip-path: inset(0 0 0 50%); }}
  .slider-line {{ position: absolute; top: 0; left: 50%; width: 2px; height: 100%; background: #fff; z-index: 2; pointer-events: none; }}
  .slider {{ position: absolute; bottom: 0; left: 0; width: 100%; z-index: 3; opacity: 0; cursor: col-resize; height: 100%; }}
  .label-left, .label-right {{ position: absolute; bottom: 8px; z-index: 2; font-size: 10px; padding: 2px 6px; background: rgba(0,0,0,0.7); border-radius: 3px; color: #fff; }}
  .label-left {{ left: 8px; }}
  .label-right {{ right: 8px; }}
</style>
</head>
<body>
<h1>WebVSR Super-Resolution Comparison</h1>
<p class="subtitle">Drag the slider to compare SPAN-Lite 2x SR vs bicubic upscale. Crops from DIV2K validation.</p>
<div class="grid">
{"".join(cards_html)}
</div>
<script>
function updateSlider(i, val) {{
  const pct = val + '%';
  document.getElementById('sc'+i).querySelector('.img-right').style.clipPath = 'inset(0 0 0 '+pct+')';
  document.getElementById('sl'+i).style.left = pct;
}}
</script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Comparison page saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best_phase1.pth')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--output', default='comparison_report.html')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SPANLite(num_in_ch=3, num_out_ch=3, feature_channels=32, upscale=2)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state)
    model.to(device).eval()

    all_imgs = sorted(f for f in os.listdir(IMG_DIR) if f.endswith('.png'))
    selected = random.sample(all_imgs, min(args.count, len(all_imgs)))

    comparisons = []
    for fname in selected:
        path = os.path.join(IMG_DIR, fname)
        print(f"Processing {fname}...")
        bic, sr = process_image(model, path, device)
        bic_uri = img_to_data_uri(bic)
        sr_uri = img_to_data_uri(sr)
        comparisons.append((fname, bic_uri, sr_uri))

    generate_html(comparisons, args.output)


if __name__ == '__main__':
    main()
