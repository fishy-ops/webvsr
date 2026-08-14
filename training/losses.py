import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (smooth L1 variant), more robust than L1 at zero."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class PerceptualLoss(nn.Module):
    """VGG-19 feature matching loss at conv3_4 (layer index 16).
    No adversarial component — sharpens via feature similarity, not a GAN."""

    def __init__(self, layer_idx=16):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:layer_idx + 1]
        for p in vgg.parameters():
            p.requires_grad = False
        self.features = vgg.eval()
        self.register_buffer(
            "vgg_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "vgg_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    def _normalize(self, x):
        return (x - self.vgg_mean) / self.vgg_std

    def forward(self, pred, target):
        pred_f = self.features(self._normalize(pred))
        with torch.no_grad():
            target_f = self.features(self._normalize(target))
        return F.l1_loss(pred_f, target_f)


class FFTLoss(nn.Module):
    """Frequency-domain L1 loss. Directly penalizes missing high-frequency
    content (edges, fine detail) that spatial L1 under-weights."""

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
        target_fft = torch.fft.rfft2(target.float(), norm="ortho")
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


class CombinedLoss(nn.Module):
    """L_total = 1.0 * L_charbonnier + w_perceptual * L_perceptual + w_fft * L_fft"""

    def __init__(self, w_perceptual=0.1, w_fft=0.01, use_perceptual=True):
        super().__init__()
        self.l1 = CharbonnierLoss()
        self.perceptual = PerceptualLoss() if use_perceptual else None
        self.fft = FFTLoss()
        self.w_perceptual = w_perceptual
        self.w_fft = w_fft
        self.use_perceptual = use_perceptual

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_fft = self.fft(pred, target)
        total = loss_l1 + self.w_fft * loss_fft

        if self.use_perceptual and self.perceptual is not None:
            loss_perc = self.perceptual(pred, target)
            total = total + self.w_perceptual * loss_perc
            return total, {
                "l1": loss_l1.item(),
                "perceptual": loss_perc.item(),
                "fft": loss_fft.item(),
            }

        return total, {"l1": loss_l1.item(), "fft": loss_fft.item()}
