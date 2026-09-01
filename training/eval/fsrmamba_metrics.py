"""Quality metrics for upscaler comparison.

Three numbers matter for this project, and they are not interchangeable:

* **PSNR** -- raw fidelity to ground truth. Easy to compute, easy to game, and
  weakly correlated with what the eye notices. Report it, don't trust it alone.
* **SSIM** -- structural similarity. Better aligned with perception than PSNR,
  still a single-frame measure.
* **Temporal instability** -- how much the output flickers between frames once
  motion is compensated for. This is the metric upscalers actually live or die
  on, and it is the one a single-frame model cannot optimise for. Lower is
  better.

The temporal metric is the interesting one here: a learned accumulator that
wins on PSNR but loses on temporal instability has not beaten FSR, it has
overfit to still frames.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["psnr", "ssim", "ssim_map_mean", "gradient_l1", "temporal_instability"]


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB. Inputs (H, W, 3) in [0, max_val]."""
    mse = F.mse_loss(pred.clamp(0, max_val), target.clamp(0, max_val)).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(max_val**2 / mse)).item()


def _gaussian_window(size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.outer(g)


def ssim_map_mean(
    pred: torch.Tensor, target: torch.Tensor, window: int = 11, sigma: float = 1.5
) -> torch.Tensor:
    """Mean SSIM as a differentiable tensor (for use in a loss). Inputs (H,W,3).

    Same computation as ``ssim`` but keeps the graph -- ``1 - ssim_map_mean`` is a
    valid loss term that directly optimises the SSIM the eval reports.
    """
    x = pred.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    y = target.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    c = x.shape[1]
    w = _gaussian_window(window, sigma, x.device).expand(c, 1, window, window)

    mu_x = F.conv2d(x, w, padding=window // 2, groups=c)
    mu_y = F.conv2d(y, w, padding=window // 2, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y

    sig_x = F.conv2d(x * x, w, padding=window // 2, groups=c) - mu_x2
    sig_y = F.conv2d(y * y, w, padding=window // 2, groups=c) - mu_y2
    sig_xy = F.conv2d(x * y, w, padding=window // 2, groups=c) - mu_xy

    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2))
    return s.mean()


def ssim(pred: torch.Tensor, target: torch.Tensor, window: int = 11, sigma: float = 1.5) -> float:
    """Mean SSIM over the image, averaged across colour channels."""
    return ssim_map_mean(pred, target, window, sigma).item()


def gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 distance between image gradients -- penalises blur and edge softness.

    L1/PSNR training tends to soft, structurally-weak output (high PSNR, low
    SSIM). Matching finite-difference gradients pushes the model to reproduce
    edges and high-frequency detail, which is where a hand-tuned sharpener (FSR's
    RCAS) otherwise wins. Inputs (H,W,3).
    """
    dx_p = pred[:, 1:] - pred[:, :-1]
    dx_t = target[:, 1:] - target[:, :-1]
    dy_p = pred[1:, :] - pred[:-1, :]
    dy_t = target[1:, :] - target[:-1, :]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)


_VGG = None


def perceptual_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """VGG16 feature-space L1 -- the standard cure for L1/PSNR blur.

    Pixel losses (L1, even + gradient/freq) are minimised by the conditional mean,
    which is smooth: that is the whole reason for our SSIM ceiling. A perceptual
    loss compares deep features instead of pixels, so producing *plausibly
    structured* detail (matching the target's textures/edges in feature space) is
    rewarded even when it is not pixel-exact -- pushing the net off the smooth mean.
    Features taken at relu1_2 / relu2_2 / relu3_3. Inputs (H,W,3) in ~[0,1] RGB.
    """
    global _VGG
    if _VGG is None:
        from torchvision.models import VGG16_Weights, vgg16
        v = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval().to(pred.device)
        for p in v.parameters():
            p.requires_grad_(False)
        _VGG = v
    mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)

    def prep(x):
        x = x.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
        return (x - mean) / std

    xp, xt = prep(pred), prep(target)
    taps = {3, 8, 15}
    loss = pred.new_zeros(())
    for i, layer in enumerate(_VGG):
        xp = layer(xp)
        xt = layer(xt)
        if i in taps:
            loss = loss + F.l1_loss(xp, xt)
    return loss


def freq_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 on the log-magnitude 2-D FFT spectrum -- a direct blur penalty.

    L1/gradient losses are local; a smoothed output can still track them while
    missing the mid/high-frequency *energy* that carries perceived detail (the
    exact failure behind our SSIM ceiling: the accumulator regresses to a smooth
    conditional mean). Matching |FFT| forces the output to carry the same
    frequency content as the target, so missing high-freq is penalised globally
    regardless of where the loss of detail happened. log1p compresses the huge
    DC/low-freq terms so the gradient isn't dominated by them. Inputs (H,W,3).
    """
    p = pred.permute(2, 0, 1)
    t = target.permute(2, 0, 1)
    fp = torch.log1p(torch.abs(torch.fft.rfft2(p, norm="ortho")))
    ft = torch.log1p(torch.abs(torch.fft.rfft2(t, norm="ortho")))
    return F.l1_loss(fp, ft)


def temporal_instability(
    curr: torch.Tensor, prev: torch.Tensor, mv: torch.Tensor
) -> float:
    """Mean absolute motion-compensated frame difference. Lower is more stable.

    Warps `prev` forward along `mv` (FSR convention: reprojected_uv = uv + mv)
    and measures what is left. On static content this goes to zero; under motion
    it captures shimmer, ghosting, and flicker that PSNR-per-frame misses.

    Pixels whose reprojection lands off-screen are excluded rather than counted
    as instability -- they are disocclusions, which are a different problem.
    """
    h, w, _ = curr.shape
    ys = (torch.arange(h, device=curr.device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=curr.device, dtype=torch.float32) + 0.5) / w
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    uv = torch.stack((gx, gy), dim=-1) + mv

    valid = (uv[..., 0] >= 0) & (uv[..., 0] < 1) & (uv[..., 1] >= 0) & (uv[..., 1] < 1)
    if not bool(valid.any()):
        return 0.0

    warped = F.grid_sample(
        prev.permute(2, 0, 1).unsqueeze(0),
        (uv * 2 - 1).unsqueeze(0),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0].permute(1, 2, 0)

    diff = (curr - warped).abs().mean(dim=-1)
    return diff[valid].mean().item()


def edge_gradient_l1(pred: torch.Tensor, target: torch.Tensor, power: float = 1.0) -> torch.Tensor:
    """Gradient L1 **weighted by ground-truth edge strength**. Inputs (H,W,3).

    `gradient_l1` spreads its penalty over the whole frame, so flat regions -- which are
    the majority of pixels and already correct -- dominate the term. Direct measurement
    says the deficit is not global softness but *edges specifically*: on GT edge pixels
    our edge contrast is 0.55-0.69 of ground truth while FSR reaches 0.82, and the same
    models match or beat FSR on textured non-edge regions. Weighting the gradient penalty
    by |grad(GT)| concentrates it exactly where the measured gap is, instead of asking the
    network to sharpen areas that are already right.

    The weight uses the *target's* gradient, not the prediction's, so the model cannot
    reduce the loss by simply flattening its own output.
    """
    dx_p = pred[:, 1:] - pred[:, :-1]
    dx_t = target[:, 1:] - target[:, :-1]
    dy_p = pred[1:, :] - pred[:-1, :]
    dy_t = target[1:, :] - target[:-1, :]

    wx = dx_t.abs().mean(dim=-1, keepdim=True).detach()
    wy = dy_t.abs().mean(dim=-1, keepdim=True).detach()
    if power != 1.0:
        wx, wy = wx.pow(power), wy.pow(power)
    # Normalise so the term's scale does not depend on how edgy the crop happens to be.
    wx = wx / (wx.mean() + 1e-6)
    wy = wy / (wy.mean() + 1e-6)
    return ((dx_p - dx_t).abs() * wx).mean() + ((dy_p - dy_t).abs() * wy).mean()
