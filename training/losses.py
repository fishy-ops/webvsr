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


class DISTSLoss(nn.Module):
    """DISTS as a training term, not only a selection metric.

    Checkpoints here are chosen on DISTS but nothing in the loss ever pointed at
    it, so training optimised one thing and selection rewarded another. DISTS is
    differentiable and compares texture *statistics* rather than pixels, which is
    the property L1 lacks -- it does not collapse toward the conditional mean, so
    unlike a heavier pixel term it should not push the model toward blur.
    """

    def __init__(self):
        super().__init__()
        from DISTS_pytorch import DISTS
        self.d = DISTS()
        for p_ in self.d.parameters():
            p_.requires_grad_(False)

    def forward(self, pred, target):
        return self.d(pred.clamp(0, 1), target.clamp(0, 1)).mean()


class LDLLoss(nn.Module):
    """Locally Discriminative Learning: weight the pixel loss by where the
    residual is locally erratic.

    From "Details or Artifacts" (CVPR 2022, github.com/csjliang/LDL). Not a
    discriminator -- it only ever compares output to ground truth, so it does not
    break the no-GAN rule. The artifact map is the local variance of the residual
    times a patch-level scale, which is large exactly in stochastic texture:
    foliage, crowds, water, smoke. §21 measured that as the one content type
    still beating this model, so the loss is pointed at the remaining failure.

    Variance is computed by avg_pool (E[x^2] - E[x]^2) rather than unfold. The
    reference implementation unfolds a 7x7 window, which for a batch of 8 at
    256px is 49x the tensor -- about 100 MB of activations for a term that costs
    two pooling passes this way.
    """

    def __init__(self, ksize=7):
        super().__init__()
        self.ksize = ksize

    def _local_var(self, r):
        pad = self.ksize // 2
        m = F.avg_pool2d(r, self.ksize, 1, pad, count_include_pad=False)
        m2 = F.avg_pool2d(r * r, self.ksize, 1, pad, count_include_pad=False)
        return (m2 - m * m).clamp(min=0)

    def forward(self, pred, target):
        residual = (target - pred).abs().sum(1, keepdim=True)
        patch_w = residual.var(dim=(-1, -2, -3), keepdim=True).clamp(min=1e-12) ** 0.2
        w = (patch_w * self._local_var(residual)).detach()
        # Normalise so the term's scale does not drift with content -- otherwise
        # the effective weight changes between a still frame and a busy one.
        w = w / w.mean().clamp(min=1e-8)
        return (w * (pred - target).abs()).mean()


class CombinedLoss(nn.Module):
    """L_total = L_charbonnier + w_perceptual*L_perc + w_fft*L_fft + w_dists*L_dists"""

    def __init__(self, w_perceptual=0.1, w_fft=0.01, use_perceptual=True,
                 w_dists=0.0, w_ldl=0.0):
        super().__init__()
        self.l1 = CharbonnierLoss()
        self.perceptual = PerceptualLoss() if use_perceptual else None
        self.fft = FFTLoss()
        self.w_perceptual = w_perceptual
        self.w_fft = w_fft
        self.use_perceptual = use_perceptual
        self.w_dists = w_dists
        self.dists = DISTSLoss() if w_dists > 0 else None
        self.w_ldl = w_ldl
        self.ldl = LDLLoss() if w_ldl > 0 else None

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_fft = self.fft(pred, target)
        total = loss_l1 + self.w_fft * loss_fft
        parts = {"l1": loss_l1.item(), "fft": loss_fft.item()}

        if self.use_perceptual and self.perceptual is not None:
            loss_perc = self.perceptual(pred, target)
            total = total + self.w_perceptual * loss_perc
            parts["perceptual"] = loss_perc.item()

        if self.dists is not None:
            loss_dists = self.dists(pred, target)
            total = total + self.w_dists * loss_dists
            parts["dists"] = loss_dists.item()

        if self.ldl is not None:
            loss_ldl = self.ldl(pred, target)
            total = total + self.w_ldl * loss_ldl
            parts["ldl"] = loss_ldl.item()

        return total, parts
