"""
FXAA — Fast Approximate Anti-Aliasing (PyTorch)

Based on FXAA 3.11 by Timothy Lottes (NVIDIA). Detects edges via
luminance contrast and blends along them to remove jaggies.

Pipeline: SR output → FXAA (smooth edges) → RCAS (sharpen textures)

Usage:
    from fxaa import fxaa_pass
    smoothed = fxaa_pass(sr_output, edge_threshold=0.125)
"""
import torch
import torch.nn.functional as F


LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


def _luminance(img):
    return (img * LUMA_WEIGHTS.to(img.device, img.dtype)).sum(dim=1, keepdim=True)


def fxaa_pass(img, edge_threshold=0.125, edge_threshold_min=0.0312, subpix_quality=0.75):
    """
    Apply FXAA to a [B, 3, H, W] tensor in [0, 1].

    Args:
        img: Input tensor [B, 3, H, W]
        edge_threshold: Minimum contrast to detect an edge (lower = more AA)
        edge_threshold_min: Absolute minimum luma difference
        subpix_quality: Sub-pixel AA strength, 0.0-1.0
    Returns:
        Anti-aliased tensor, same shape.
    """
    B, C, H, W = img.shape
    luma = _luminance(img)

    padded_luma = F.pad(luma, (1, 1, 1, 1), mode='reflect')
    padded_img = F.pad(img, (1, 1, 1, 1), mode='reflect')

    # Sample 3x3 neighborhood of luma
    luma_n = padded_luma[:, :, :-2, 1:-1]
    luma_s = padded_luma[:, :, 2:, 1:-1]
    luma_w = padded_luma[:, :, 1:-1, :-2]
    luma_e = padded_luma[:, :, 1:-1, 2:]
    luma_c = luma

    luma_nw = padded_luma[:, :, :-2, :-2]
    luma_ne = padded_luma[:, :, :-2, 2:]
    luma_sw = padded_luma[:, :, 2:, :-2]
    luma_se = padded_luma[:, :, 2:, 2:]

    # Local contrast
    range_max = torch.max(torch.max(torch.max(luma_n, luma_s), torch.max(luma_w, luma_e)), luma_c)
    range_min = torch.min(torch.min(torch.min(luma_n, luma_s), torch.min(luma_w, luma_e)), luma_c)
    local_range = range_max - range_min

    # Skip pixels below threshold
    threshold = torch.max(
        torch.full_like(local_range, edge_threshold_min),
        range_max * edge_threshold,
    )
    is_edge = local_range > threshold

    # Edge direction: horizontal vs vertical
    edge_h = (
        abs(luma_n + luma_s - 2.0 * luma_c) * 2.0 +
        abs(luma_ne + luma_se - 2.0 * luma_e) +
        abs(luma_nw + luma_sw - 2.0 * luma_w)
    )
    edge_v = (
        abs(luma_w + luma_e - 2.0 * luma_c) * 2.0 +
        abs(luma_nw + luma_ne - 2.0 * luma_n) +
        abs(luma_sw + luma_se - 2.0 * luma_s)
    )
    is_horizontal = edge_h >= edge_v

    # Sub-pixel aliasing: blend factor based on local contrast vs neighborhood avg
    luma_avg = (luma_n + luma_s + luma_w + luma_e) * 0.25
    sub_pix = torch.abs(luma_avg - luma_c) / (local_range + 1e-8)
    sub_pix = torch.clamp(sub_pix, 0.0, 1.0)
    sub_pix = (-2.0 * sub_pix + 3.0) * sub_pix * sub_pix
    sub_pix = sub_pix * sub_pix * subpix_quality

    # Blend along the edge direction
    # For horizontal edges: blend with N/S neighbors
    # For vertical edges: blend with E/W neighbors
    blend_n = padded_img[:, :, :-2, 1:-1]
    blend_s = padded_img[:, :, 2:, 1:-1]
    blend_w = padded_img[:, :, 1:-1, :-2]
    blend_e = padded_img[:, :, 1:-1, 2:]

    # Determine which neighbor to blend toward (positive or negative side)
    luma_pos_h = luma_s
    luma_neg_h = luma_n
    grad_pos_h = torch.abs(luma_pos_h - luma_c)
    grad_neg_h = torch.abs(luma_neg_h - luma_c)

    luma_pos_v = luma_e
    luma_neg_v = luma_w
    grad_pos_v = torch.abs(luma_pos_v - luma_c)
    grad_neg_v = torch.abs(luma_neg_v - luma_c)

    # For horizontal edges, blend toward the stronger gradient (N or S)
    blend_h = torch.where(grad_neg_h > grad_pos_h, blend_n, blend_s)
    # For vertical edges, blend toward the stronger gradient (W or E)
    blend_v = torch.where(grad_neg_v > grad_pos_v, blend_w, blend_e)

    # Pick blend direction
    blended = torch.where(is_horizontal, blend_h, blend_v)

    # Compute final blend factor
    blend_factor = sub_pix.expand_as(img)

    # Apply: mix original with blended neighbor, only on edge pixels
    result = torch.where(
        is_edge.expand_as(img),
        img * (1.0 - blend_factor) + blended * blend_factor,
        img,
    )

    return result.clamp(0, 1)


if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256).clamp(0, 1)
    y = fxaa_pass(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Changed pixels: {(x != y).any(dim=1).sum().item()}")
