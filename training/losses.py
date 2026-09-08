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


def _fp32_forward(fn):
    """Run a loss term in fp32 even inside an autocast region.

    The training loop wraps the step in autocast, so a term written against
    float tensors still receives fp16 and, worse, keeps computing in fp16
    internally. MS-SSIM divides by variances near zero, the Oklab conversion
    takes cube roots, and FDL takes an FFT -- in fp16 that last one produces
    ComplexHalf, which torch itself flags as experimental. All three want full
    precision, and at these tensor sizes it costs almost nothing.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(self, pred, target):
        with torch.autocast(device_type=pred.device.type, enabled=False):
            return fn(self, pred.float(), target.float())

    return wrapper


class MSSIMLoss(nn.Module):
    """1 - MS-SSIM. The structural loss this project has never had.

    Charbonnier is pointwise and VGG is feature-space; neither compares local
    *statistics*. SSIM does: it scores luminance, contrast and covariance inside
    an 11px Gaussian window, so it is sensitive to the loss of local contrast
    that pixel losses are indifferent to. That is exactly the failure mode PSNR
    selection rewards. neosr's SPAN recipe uses this as its ONLY pixel loss at
    weight 1.0 (RESEARCH.md 33); here it is added alongside Charbonnier rather
    than replacing it, because this is a fine-tune of a converged checkpoint and
    swapping the pixel term outright is a different experiment.

    Five scales, so the input must be at least 11 * 2^4 = 176px. Asserted rather
    than silently degraded -- a run at crop 64 would otherwise pool down to a
    1px map and score noise.
    """

    _SCALE_W = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

    def __init__(self, window_size=11, sigma=1.5, channels=3, K1=0.01, K2=0.03):
        super().__init__()
        self.C1 = K1 ** 2
        self.C2 = K2 ** 2
        self.pad = window_size // 2
        x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        w = torch.exp(-0.5 * x ** 2 / sigma ** 2)
        w = w / w.sum()
        k = torch.outer(w, w).view(1, 1, window_size, window_size)
        self.register_buffer("kernel", k.repeat(channels, 1, 1, 1))
        self.min_size = window_size * (2 ** (len(self._SCALE_W) - 1))

    def _blur(self, x):
        return F.conv2d(x, self.kernel, padding=self.pad, groups=x.shape[1])

    def _ssim(self, x, y):
        mu_x, mu_y = self._blur(x), self._blur(y)
        s2_x = self._blur(x * x) - mu_x * mu_x
        s2_y = self._blur(y * y) - mu_y * mu_y
        s_xy = self._blur(x * y) - mu_x * mu_y
        lum = (2 * mu_x * mu_y + self.C1) / (mu_x ** 2 + mu_y ** 2 + self.C1)
        cs = (2 * s_xy + self.C2) / (s2_x + s2_y + self.C2)
        return (lum * cs).mean(), cs.mean()

    @_fp32_forward
    def forward(self, pred, target):
        h, w = pred.shape[-2:]
        if min(h, w) < self.min_size:
            raise ValueError(
                f"MS-SSIM needs at least {self.min_size}px, got {h}x{w}. Five "
                f"scales pool the map to nothing below that; use a larger crop "
                f"or drop --w-mssim."
            )
        x = pred.float()
        y = target.float()
        out = 1.0
        for i, sw in enumerate(self._SCALE_W):
            ssim, cs = self._ssim(x, y)
            if i == len(self._SCALE_W) - 1:
                out = out * ssim.clamp(min=1e-6) ** sw
            else:
                out = out * cs.clamp(min=1e-6) ** sw
                pad = [s % 2 for s in x.shape[2:]]
                x = F.avg_pool2d(x, 2, 2, padding=pad)
                y = F.avg_pool2d(y, 2, 2, padding=pad)
        return 1.0 - out


class ColorConsistencyLoss(nn.Module):
    """Oklab chroma + CIE L* luma agreement.

    The degradation chain deliberately includes 4:2:0 chroma subsampling -- every
    codec in codec_degrade.py encodes yuv420p -- so colour error is a degradation
    this model is asked to undo, and nothing in the loss has ever mentioned
    colour. Charbonnier in linear-light-free sRGB weights a hue shift by its RGB
    distance, which is not how the error is seen.

    Oklab is perceptually uniform, so an L1 in its (a,b) plane is close to an L1
    in perceived hue/saturation error. Luma is compared separately in CIE L*
    after a Gaussian blur, which restricts that half of the term to low-frequency
    brightness drift and leaves detail to the other losses -- otherwise this
    would just be a second, weaker pixel loss.

    Technique from neosr's consistency_loss (RESEARCH.md 33); written here rather
    than copied. Two things are deliberately different: their L* branch squares
    the linear value in the low-luma leg (`img * (img * 24389/27)`), and their
    cosine-similarity term is only added when it is already below 1e-3, so it
    contributes nothing. Neither is reproduced.
    """

    def __init__(self, blur_sigma=3.0, blur_size=21):
        super().__init__()
        x = torch.arange(blur_size, dtype=torch.float32) - blur_size // 2
        w = torch.exp(-0.5 * x ** 2 / blur_sigma ** 2)
        w = w / w.sum()
        k = torch.outer(w, w).view(1, 1, blur_size, blur_size)
        self.register_buffer("blur_k", k.repeat(3, 1, 1, 1))
        self.blur_pad = blur_size // 2
        self.register_buffer(
            "luma_w", torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        )

    @staticmethod
    def _linearize(img):
        return torch.where(
            img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055).clamp(min=0) ** 2.4
        )

    def _blur(self, x):
        return F.conv2d(x, self.blur_k, padding=self.blur_pad, groups=3)

    def _oklab_ab(self, img):
        lin = self._linearize(img)
        r, g, b = lin[:, 0], lin[:, 1], lin[:, 2]
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
        # cbrt via sign*|x|^(1/3): the inputs are non-negative here, but the
        # clamp keeps the gradient finite at exactly zero.
        l_ = l.clamp(min=1e-8) ** (1 / 3)
        m_ = m.clamp(min=1e-8) ** (1 / 3)
        s_ = s.clamp(min=1e-8) ** (1 / 3)
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
        return torch.stack([a, bb], dim=1)

    def _l_star(self, img):
        y = (self._linearize(img) * self.luma_w).sum(1, keepdim=True)
        # CIE L*: 903.3*Y below the knee, 116*Y^(1/3)-16 above. Normalised to
        # [0,1] so it sits on the same scale as the chroma term.
        low = y * (24389 / 27)
        high = 116 * y.clamp(min=1e-8) ** (1 / 3) - 16
        return torch.where(y <= (216 / 24389), low, high) / 100

    @_fp32_forward
    def forward(self, pred, target):
        p = pred.clamp(0, 1).float()
        t = target.clamp(0, 1).float()
        chroma = F.l1_loss(self._oklab_ab(p), self._oklab_ab(t))
        luma = F.l1_loss(self._l_star(self._blur(p)), self._l_star(self._blur(t)))
        return chroma + luma


class FDLLoss(nn.Module):
    """Frequency Distribution Loss (Ni et al., arXiv:2402.18192).

    Every feature loss here so far compares features *position by position*, so a
    texture reproduced correctly but shifted a pixel scores as an error and the
    cheapest way to satisfy it is to blur. FDL compares the feature FFT's
    magnitude and phase as *distributions*: random 4x4 projections of each, sorted
    independently, then L1 between the sorted vectors -- a sliced 1D Wasserstein
    distance. Sorting throws away position, so it asks for the right statistics
    without demanding they land in the right place, which is the property this
    project wants for stochastic texture (RESEARCH.md 21).

    Uses VGG19 features rather than the paper's and neosr's DINOv2: the VGG
    weights are already downloaded for PerceptualLoss and a ViT forward pass per
    step does not fit alongside training on 8 GB. The 0.01 rescale is the one the
    reference applies to its non-DINO backbones.

    Features are taken one stage at a time and freed before the next, because a
    complex64 FFT of the stage-1 map at 256px and batch 8 is ~270 MB on its own.
    """

    _STAGES = ((0, 4), (4, 9), (9, 18), (18, 27), (27, 36))
    _CHNS = (64, 128, 256, 512, 512)
    _STAGE_W = (0.5, 0.5, 1.0, 1.0, 1.0)

    def __init__(self, patch_size=4, num_proj=24, stride=1, phase_weight=1.0):
        super().__init__()
        feats = vgg19(weights=VGG19_Weights.DEFAULT).features
        for p in feats.parameters():
            p.requires_grad = False
        self.stages = nn.ModuleList(
            [nn.Sequential(*[feats[i] for i in range(a, b)]) for a, b in self._STAGES]
        ).eval()
        self.register_buffer(
            "vgg_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "vgg_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.stride = stride
        self.phase_weight = phase_weight
        g = torch.Generator().manual_seed(0)  # fixed projections: the loss must
        for i, c in enumerate(self._CHNS):    # not change between runs
            r = torch.randn(num_proj, c, patch_size, patch_size, generator=g)
            r = r / r.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
            self.register_buffer(f"proj_{i}", r)

    def _sliced_w1(self, x, y, idx):
        proj = getattr(self, f"proj_{idx}")
        px = F.conv2d(x, proj, stride=self.stride).flatten(2)
        py = F.conv2d(y, proj, stride=self.stride).flatten(2)
        px, _ = torch.sort(px, dim=-1)
        py, _ = torch.sort(py, dim=-1)
        return (px - py).abs().mean()

    @_fp32_forward
    def forward(self, pred, target):
        x = (pred.float().clamp(0, 1) - self.vgg_mean) / self.vgg_std
        with torch.no_grad():
            y = (target.float().clamp(0, 1) - self.vgg_mean) / self.vgg_std
        score = 0.0
        for i, stage in enumerate(self.stages):
            x = stage(x)
            with torch.no_grad():
                y = stage(y)
            fx = torch.fft.fftn(x, dim=(-2, -1))
            fy = torch.fft.fftn(y, dim=(-2, -1))
            s = self._sliced_w1(fx.abs(), fy.abs(), i)
            s = s + self.phase_weight * self._sliced_w1(
                torch.angle(fx), torch.angle(fy), i
            )
            score = score + self._STAGE_W[i] * s
            del fx, fy
        return score * 0.01


class CombinedLoss(nn.Module):
    """L_total = Charbonnier + w_fft*FFT + w_perc*VGG + optional extra terms.

    The optional terms all default to 0, so a run that does not ask for one pays
    nothing -- not even the module's construction, which matters for FDL and
    DISTS because each pulls a frozen backbone onto the GPU.
    """

    def __init__(self, w_perceptual=0.1, w_fft=0.01, use_perceptual=True,
                 w_dists=0.0, w_ldl=0.0, w_mssim=0.0, w_color=0.0, w_fdl=0.0):
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
        self.w_mssim = w_mssim
        self.mssim = MSSIMLoss() if w_mssim > 0 else None
        self.w_color = w_color
        self.color = ColorConsistencyLoss() if w_color > 0 else None
        self.w_fdl = w_fdl
        self.fdl = FDLLoss() if w_fdl > 0 else None

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

        if self.mssim is not None:
            loss_mssim = self.mssim(pred, target)
            total = total + self.w_mssim * loss_mssim
            parts["mssim"] = loss_mssim.item()

        if self.color is not None:
            loss_color = self.color(pred, target)
            total = total + self.w_color * loss_color
            parts["color"] = loss_color.item()

        if self.fdl is not None:
            loss_fdl = self.fdl(pred, target)
            total = total + self.w_fdl * loss_fdl
            parts["fdl"] = loss_fdl.item()

        return total, parts
