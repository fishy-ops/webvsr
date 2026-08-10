"""
RCAS — Robust Contrast-Adaptive Sharpening (PyTorch)

Based on AMD FidelityFX RCAS. Edge-aware sharpening that enhances
texture detail without halos. Meant as a post-process after SR.

Usage:
    from rcas import rcas_sharpen
    sharpened = rcas_sharpen(sr_output, sharpness=0.25)
"""
import torch
import torch.nn.functional as F


def rcas_sharpen(img, sharpness=0.25):
    """
    Apply RCAS sharpening to a [B, 3, H, W] tensor in [0, 1].

    Args:
        img: Input tensor [B, C, H, W] in [0, 1]
        sharpness: Sharpening strength, 0.0 = max sharpening, 1.0 = no sharpening.
                   Default 0.25 is a good balance for SR output.
    Returns:
        Sharpened tensor, same shape.
    """
    # Pad for border pixels
    padded = F.pad(img, (1, 1, 1, 1), mode='reflect')

    # 5-tap: center + cardinal neighbors
    c = img
    n = padded[:, :, :-2, 1:-1]
    s = padded[:, :, 2:, 1:-1]
    w = padded[:, :, 1:-1, :-2]
    e = padded[:, :, 1:-1, 2:]

    # Local min/max
    mn = torch.min(torch.min(n, s), torch.min(w, e))
    mx = torch.max(torch.max(n, s), torch.max(w, e))
    mn2 = torch.min(mn, c)
    mx2 = torch.max(mx, c)

    # Contrast-adaptive weight
    rng = mx2 - mn2
    peak = torch.where(
        rng > 1e-3,
        (1.0 - torch.min(mn2, 1.0 - mx2) / rng),
        torch.zeros_like(rng),
    )
    wt = torch.sqrt(torch.clamp(peak, 0, 1)) * (-1.0 / (4.0 * sharpness + 4.0))

    # Apply sharpening filter
    result = (c + (n + s + w + e) * wt) / (1.0 + 4.0 * wt)

    # Clamp to local range to prevent ringing
    result = torch.clamp(result, mn2, mx2)

    return result


if __name__ == '__main__':
    import os
    import argparse
    from PIL import Image
    from torchvision import transforms

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input image path')
    parser.add_argument('--sharpness', type=float, default=0.25)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    img = transforms.ToTensor()(Image.open(args.input).convert('RGB')).unsqueeze(0)
    result = rcas_sharpen(img, args.sharpness)
    out_img = transforms.ToPILImage()(result.squeeze(0))

    out_path = args.output or args.input.replace('.', '_rcas.')
    out_img.save(out_path)
    print(f"Saved: {out_path}")
