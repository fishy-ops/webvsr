import os
import random
import math
from pathlib import Path
from PIL import Image, ImageFile
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
import io

ImageFile.LOAD_TRUNCATED_IMAGES = True


def random_crop(img, crop_size):
    """Random crop from a PIL Image, returns PIL Image.

    Refuses a crop larger than the source. Upscaling the frame to satisfy the
    crop would make the HR *target* a bicubic upsample of itself, which trains
    the model to reproduce blur -- silently, since the loss still falls. That
    is what invalidated the 4x crop-512 run (see RESEARCH.md 6a); Vimeo-90K is
    448x256, so 256 is the ceiling.
    """
    w, h = img.size
    if w < crop_size or h < crop_size:
        raise ValueError(
            f"crop_size {crop_size} exceeds source frame {w}x{h}; max usable "
            f"crop is {min(w, h)}. Upscaling here would fabricate HR detail "
            f"and teach the model bicubic blur. Use a larger-resolution "
            f"dataset instead of a larger crop."
        )
    left = random.randint(0, w - crop_size)
    top = random.randint(0, h - crop_size)
    return img.crop((left, top, left + crop_size, top + crop_size))


def augment_pair(hr, lr):
    """Random flip + 90-degree rotation on both HR and LR tensors consistently."""
    if random.random() < 0.5:
        hr = torch.flip(hr, [2])
        lr = torch.flip(lr, [2])
    if random.random() < 0.5:
        hr = torch.flip(hr, [1])
        lr = torch.flip(lr, [1])
    if random.random() < 0.5:
        hr = hr.permute(0, 2, 1)
        lr = lr.permute(0, 2, 1)
    return hr, lr


# ── Real-ESRGAN-style second-order degradation ─────────────────────────

def gaussian_blur_tensor(img, kernel_size, sigma):
    """Apply Gaussian blur to a CHW tensor."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    half = kernel_size // 2
    ax = torch.arange(-half, half + 1, dtype=torch.float32, device=img.device)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(img.shape[0], -1, -1, -1)
    return F.conv2d(
        img.unsqueeze(0),
        kernel,
        padding=half,
        groups=img.shape[0],
    ).squeeze(0)


def add_gaussian_noise(img, sigma_range=(1, 30)):
    sigma = random.uniform(*sigma_range) / 255.0
    noise = torch.randn_like(img) * sigma
    return (img + noise).clamp(0, 1)


def add_poisson_noise(img, scale_range=(0.5, 4.0)):
    scale = random.uniform(*scale_range)
    vals = img * scale * 255
    noisy = torch.poisson(vals.clamp(0)) / (scale * 255)
    return noisy.clamp(0, 1)


def jpeg_compress_tensor(img, quality_range=(30, 95)):
    """JPEG compress and decompress a CHW float tensor."""
    quality = random.randint(*quality_range)
    pil = transforms.ToPILImage()(img.cpu().clamp(0, 1))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    pil_out = Image.open(buf).convert("RGB")
    return transforms.ToTensor()(pil_out).to(img.device)


def degrade_second_order(hr_tensor, scale=2):
    """Real-ESRGAN-style second-order degradation pipeline.

    First order:  blur → resize → noise → JPEG
    Second order: blur → resize-to-target → JPEG

    Produces realistic video-like degradation (compression, noise, blur stacking).
    """
    img = hr_tensor.clone()
    _, h, w = img.shape
    target_h, target_w = h // scale, w // scale

    # ── First degradation ──
    # Blur
    if random.random() < 0.8:
        ks = random.choice([3, 5, 7])
        sigma = random.uniform(0.1, 2.0)
        img = gaussian_blur_tensor(img, ks, sigma)

    # Resize (intermediate, not final) — random between 0.5x and 1.0x of original
    inter_scale = random.uniform(0.5, 1.0)
    inter_h = max(int(h * inter_scale), target_h)
    inter_w = max(int(w * inter_scale), target_w)
    mode = random.choice(["bilinear", "bicubic", "area"])
    img = F.interpolate(
        img.unsqueeze(0), size=(inter_h, inter_w), mode=mode,
        align_corners=False if mode != "area" else None,
    ).squeeze(0).clamp(0, 1)

    # Noise
    if random.random() < 0.5:
        img = add_gaussian_noise(img, (1, 25))
    if random.random() < 0.3:
        img = add_poisson_noise(img, (0.5, 3.0))

    # JPEG
    if random.random() < 0.7:
        img = jpeg_compress_tensor(img, (30, 95))

    # ── Second degradation ──
    # Blur
    if random.random() < 0.5:
        ks = random.choice([3, 5])
        sigma = random.uniform(0.1, 1.5)
        img = gaussian_blur_tensor(img, ks, sigma)

    # Resize to final target
    mode = random.choice(["bilinear", "bicubic", "area"])
    img = F.interpolate(
        img.unsqueeze(0), size=(target_h, target_w), mode=mode,
        align_corners=False if mode != "area" else None,
    ).squeeze(0).clamp(0, 1)

    # Final JPEG
    if random.random() < 0.5:
        img = jpeg_compress_tensor(img, (50, 95))

    return img


# ── Dataset ─────────────────────────────────────────────────────────────

class SRDataset(Dataset):
    """Multi-source SR dataset with on-the-fly degradation.

    data_dirs: list of directories containing HR images
    crop_size: HR patch size
    scale: downscale factor
    use_degradation: if True, uses second-order degradation; else simple bicubic
    """

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, data_dirs, crop_size=256, scale=2,
                 use_degradation=True, min_size=256, clean_ratio=0.3,
                 degrade_fn=None):
        self.crop_size = crop_size
        self.scale = scale
        self.use_degradation = use_degradation
        self.min_size = min_size
        self.clean_ratio = clean_ratio
        self.to_tensor = transforms.ToTensor()
        # Swappable degradation. Default is the JPEG-on-stills chain; pass
        # codec_degrade.second_order_codec to train against real encoder
        # artifacts instead. Signature: fn(hr_tensor, scale) -> lr_tensor.
        self.degrade_fn = degrade_fn or degrade_second_order

        self.paths = []
        for d in data_dirs:
            d = Path(d)
            if not d.exists():
                print(f"Warning: {d} does not exist, skipping")
                continue
            for f in sorted(d.rglob("*")):
                if f.suffix.lower() in self.EXTENSIONS:
                    self.paths.append(str(f))
        print(f"SRDataset: {len(self.paths)} images from {len(data_dirs)} sources")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.crop_size, self.crop_size), (0, 0, 0))

        if min(img.size) < self.min_size:
            img = img.resize(
                (max(img.size[0], self.min_size), max(img.size[1], self.min_size)),
                Image.BICUBIC,
            )

        hr_pil = random_crop(img, self.crop_size)
        hr = self.to_tensor(hr_pil)

        if self.use_degradation and random.random() > self.clean_ratio:
            lr = self.degrade_fn(hr, self.scale)
        else:
            h, w = hr.shape[1], hr.shape[2]
            lr = F.interpolate(
                hr.unsqueeze(0),
                size=(h // self.scale, w // self.scale),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).clamp(0, 1)

        hr, lr = augment_pair(hr, lr)
        return lr, hr


class ValidationDataset(Dataset):
    """Validation set, degraded the same way the training set is.

    A validation set that only downscales bicubically measures a different
    task from the one the model is deployed on, and selection then optimises
    toward the wrong distribution: a 200-epoch run improved bicubic-val DISTS
    0.0656 -> 0.0587 while its DISTS on codec-degraded video got *worse*,
    0.0988 -> 0.1023. Pass `degrade_fn` to keep validation in the deployment
    domain.

    The degradation is randomised, so it is seeded per index: image i draws
    the same codec, quality and resize mode on every epoch. Without that,
    val DISTS jitters with the draw and "best checkpoint" starts to mean
    "easiest sample of encoders", which is not a property of the model. The
    global RNG state is saved and restored so seeding here cannot perturb
    the training stream.
    """

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, data_dir, scale=2, degrade_fn=None, seed=20260901):
        self.scale = scale
        self.degrade_fn = degrade_fn
        self.seed = seed
        self.to_tensor = transforms.ToTensor()
        data_dir = Path(data_dir)
        self.paths = sorted(
            str(f) for f in data_dir.rglob("*")
            if f.suffix.lower() in self.EXTENSIONS
        )
        how = "codec-degraded (seeded)" if degrade_fn else "bicubic only"
        print(f"ValidationDataset: {len(self.paths)} images from {data_dir} "
              f"[{how}]")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        hr = self.to_tensor(img)
        _, h, w = hr.shape
        h = h - h % self.scale
        w = w - w % self.scale
        hr = hr[:, :h, :w]

        if self.degrade_fn is not None:
            state = random.getstate()
            random.seed(self.seed + idx)
            try:
                lr = self.degrade_fn(hr, self.scale)
            finally:
                random.setstate(state)
            # A codec stage can return an off-by-one size on odd dimensions.
            if lr.shape[1:] != (h // self.scale, w // self.scale):
                lr = F.interpolate(
                    lr.unsqueeze(0),
                    size=(h // self.scale, w // self.scale),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0)
            return lr.clamp(0, 1), hr

        lr = F.interpolate(
            hr.unsqueeze(0),
            size=(h // self.scale, w // self.scale),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).clamp(0, 1)
        return lr, hr
