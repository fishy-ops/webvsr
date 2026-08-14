"""
Evaluate SPAN-Lite on standard SR benchmarks.

Usage:
    python evaluate.py --checkpoint best_phase2.pth
    python evaluate.py --checkpoint best_phase2.pth --dataset Set5 --save-images
"""
import argparse
import os
import math
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from model_span import SPANLite


def calc_psnr(sr, hr, border=2):
    sr_np = sr.cpu().numpy().astype(np.float64)
    hr_np = hr.cpu().numpy().astype(np.float64)
    if border > 0:
        sr_np = sr_np[:, :, border:-border, border:-border]
        hr_np = hr_np[:, :, border:-border, border:-border]
    mse = np.mean((sr_np - hr_np) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


def calc_ssim_channel(img1, img2):
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    mu1 = F.avg_pool2d(img1, 11, 1, 5)
    mu2 = F.avg_pool2d(img2, 11, 1, 5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 ** 2, 11, 1, 5) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, 11, 1, 5) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 11, 1, 5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def calc_ssim(sr, hr, border=2):
    if border > 0:
        sr = sr[:, :, border:-border, border:-border]
        hr = hr[:, :, border:-border, border:-border]
    ssim_vals = []
    for c in range(sr.shape[1]):
        ssim_vals.append(calc_ssim_channel(
            sr[:, c:c+1, :, :], hr[:, c:c+1, :, :]
        ))
    return sum(ssim_vals) / len(ssim_vals)


def load_model(checkpoint_path, device):
    model = SPANLite(num_in_ch=3, num_out_ch=3, feature_channels=32, upscale=2)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state)
    model.to(device).eval()
    return model


DATASET_PATHS = {
    'DIV2K_val': r'C:\Users\reach\OneDrive\Documents\mamba-sr\DIV2K_valid_HR\DIV2K_valid_HR',
}

to_tensor = transforms.ToTensor()


def get_test_images(dataset_name):
    base = DATASET_PATHS.get(dataset_name)
    if not base or not os.path.isdir(base):
        print(f"Dataset '{dataset_name}' not found at {base}")
        return []
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    images = sorted(
        p for p in Path(base).iterdir()
        if p.suffix.lower() in exts
    )
    return images


@torch.no_grad()
def evaluate(model, dataset_name, device, save_dir=None):
    images = get_test_images(dataset_name)
    if not images:
        return None

    results = []
    for img_path in images:
        hr_pil = Image.open(img_path).convert('RGB')
        hr = to_tensor(hr_pil).unsqueeze(0).to(device)

        h, w = hr.shape[2], hr.shape[3]
        h_new = h - h % 2
        w_new = w - w % 2
        hr = hr[:, :, :h_new, :w_new]

        lr = F.interpolate(hr, scale_factor=0.5, mode='bicubic', align_corners=False)
        lr = lr.clamp(0, 1)

        with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            sr = model(lr)
        sr = sr.float().clamp(0, 1)

        psnr = calc_psnr(sr, hr)
        ssim = calc_ssim(sr, hr)
        results.append({
            'image': img_path.name,
            'psnr': psnr,
            'ssim': ssim,
        })
        print(f"  {img_path.name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            sr_img = transforms.ToPILImage()(sr.squeeze(0).cpu())
            sr_img.save(os.path.join(save_dir, f"sr_{img_path.name}"))

            lr_up = F.interpolate(lr, scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1)
            bic_img = transforms.ToPILImage()(lr_up.squeeze(0).cpu())
            bic_img.save(os.path.join(save_dir, f"bicubic_{img_path.name}"))

    avg_psnr = sum(r['psnr'] for r in results) / len(results)
    avg_ssim = sum(r['ssim'] for r in results) / len(results)
    return {
        'dataset': dataset_name,
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'per_image': results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='D:/webvsr/checkpoints/best_phase2.pth')
    parser.add_argument('--dataset', default='all', help='DIV2K_val or all')
    parser.add_argument('--save-images', action='store_true')
    parser.add_argument('--output', default='D:/webvsr/eval_results.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    print(f"Model loaded from {args.checkpoint}")

    datasets = list(DATASET_PATHS.keys()) if args.dataset == 'all' else [args.dataset]
    all_results = []

    for ds in datasets:
        print(f"\n=== {ds} ===")
        save_dir = os.path.join('D:/webvsr/eval_output', ds) if args.save_images else None
        result = evaluate(model, ds, device, save_dir)
        if result:
            all_results.append(result)
            print(f"  Average: PSNR={result['avg_psnr']:.2f} dB, SSIM={result['avg_ssim']:.4f}")

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
