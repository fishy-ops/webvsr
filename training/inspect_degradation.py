import argparse
from pathlib import Path

import numpy as np
import random
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import codec_degrade

CELL = 256
GUTTER = 4

try:
    _BICUBIC = Image.Resampling.BICUBIC
    _NEAREST = Image.Resampling.NEAREST
except AttributeError:
    _BICUBIC = Image.BICUBIC
    _NEAREST = Image.NEAREST


def _load_hr_tensor(path):
    img = Image.open(path).convert('RGB')

    if img.size[0] < CELL or img.size[1] < CELL:
        w, h = img.size
        factor = max(CELL / max(w, 1), CELL / max(h, 1))
        new_size = (
            max(CELL, int(round(w * factor))),
            max(CELL, int(round(h * factor))),
        )
        img = img.resize(new_size, _BICUBIC)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _random_crop(tensor, rng):
    h, w = tensor.shape[1], tensor.shape[2]
    top = rng.randint(0, h - CELL) if h > CELL else 0
    left = rng.randint(0, w - CELL) if w > CELL else 0
    return tensor[:, top:top + CELL, left:left + CELL].contiguous()


def _as_rgb_01(tensor):
    if isinstance(tensor, (list, tuple)):
        tensor = tensor[0]

    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    elif not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)

    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    elif not torch.is_floating_point(tensor):
        tensor = tensor.float()
        if tensor.numel() and tensor.max() > 1.0:
            tensor = tensor / 255.0
    else:
        tensor = tensor.float()
        if tensor.numel() and tensor.max() > 1.5 and tensor.min() >= -0.01:
            tensor = tensor / 255.0

    tensor = tensor.detach().cpu()

    if tensor.dim() == 4:
        tensor = tensor[0]

    if tensor.dim() != 3:
        raise ValueError(f"Expected a CHW or HWC image tensor, got shape {tuple(tensor.shape)}")

    if tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1)

    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] > 3:
        tensor = tensor[:3]

    return tensor.clamp(0.0, 1.0).contiguous()


def _bicubic_roundtrip(tensor, scale):
    small = (max(1, CELL // scale), max(1, CELL // scale))
    down = F.interpolate(tensor.unsqueeze(0), size=small, mode='bicubic', align_corners=False)
    up = F.interpolate(down, size=(CELL, CELL), mode='bicubic', align_corners=False)
    return up.squeeze(0).clamp(0.0, 1.0).contiguous()


def _codec_roundtrip(tensor, scale):
    lr = codec_degrade.second_order_codec(tensor.contiguous().clone(), scale)
    lr = _as_rgb_01(lr)

    if tuple(lr.shape[-2:]) != (CELL, CELL):
        lr = F.interpolate(lr.unsqueeze(0), size=(CELL, CELL), mode='nearest').squeeze(0)

    return lr.clamp(0.0, 1.0).contiguous()


def _tensor_to_pil(tensor):
    tensor = _as_rgb_01(tensor)

    if tuple(tensor.shape[-2:]) != (CELL, CELL):
        tensor = F.interpolate(tensor.unsqueeze(0), size=(CELL, CELL), mode='nearest').squeeze(0)

    arr = (tensor.permute(1, 2, 0).numpy() * 255.0)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, 'RGB')


def _text_bbox(draw, text, font):
    try:
        return draw.textbbox((0, 0), text, font=font)
    except AttributeError:
        w, h = draw.textsize(text, font=font)
        return (0, 0, w, h)


def _centered_text(draw, center, text, font):
    bbox = _text_bbox(draw, text, font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = center[0] - w / 2.0 - bbox[0]
    y = center[1] - h / 2.0 - bbox[1]
    draw.text((x, y), text, font=font, fill='black')


def _header_height(font):
    dummy = Image.new('RGB', (16, 16), 'white')
    draw = ImageDraw.Draw(dummy)
    bbox = _text_bbox(draw, 'CODEC', font)
    return max(24, (bbox[3] - bbox[1]) + 16)


def main():
    parser = argparse.ArgumentParser(
        description='Render HR, bicubic and codec-degraded crops for visual inspection.'
    )
    parser.add_argument('--hr-dir', type=Path, default=Path('/tank/webvsr/train_hr'),
                        help='Directory of HR PNG images.')
    parser.add_argument('--out', type=Path, default=Path('/tank/webvsr/degradation_check.png'),
                        help='Output PNG path.')
    parser.add_argument('--n', type=int, default=6, help='Number of sample images.')
    parser.add_argument('--scale', type=int, default=2, help='Downscale factor.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    args = parser.parse_args()

    scale = max(1, int(args.scale))
    seed = args.seed % (2 ** 32)

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    if not args.hr_dir.is_dir():
        raise SystemExit(f'HR directory not found: {args.hr_dir}')

    files = sorted(
        p for p in args.hr_dir.iterdir()
        if p.is_file() and p.suffix.lower() == '.png'
    )

    if not files:
        raise SystemExit(f'No PNG images found in {args.hr_dir}')

    if args.n <= 0:
        selected = []
    elif len(files) >= args.n:
        selected = rng.sample(files, args.n)
    else:
        selected = rng.choices(files, k=args.n)

    rows = []

    with torch.no_grad():
        for idx, path in enumerate(selected):
            tensor = _load_hr_tensor(path)
            hr_crop = _random_crop(tensor, rng)

            bicubic = _bicubic_roundtrip(hr_crop, scale)
            codec = _codec_roundtrip(hr_crop, scale)

            mad = torch.mean(torch.abs(hr_crop - codec)).item()
            print(f'[{idx + 1}/{len(selected)}] {path.name}: HR-vs-codec MAD={mad:.6f}')

            rows.append((hr_crop, bicubic, codec))

    font = ImageFont.load_default()
    header_h = _header_height(font)

    width = 3 * CELL + 2 * GUTTER
    if rows:
        height = header_h + GUTTER + len(rows) * CELL + (len(rows) - 1) * GUTTER
    else:
        height = header_h

    canvas = Image.new('RGB', (width, max(1, height)), 'white')
    draw = ImageDraw.Draw(canvas)

    labels = ('HR', 'BICUBIC', 'CODEC')
    for col, label in enumerate(labels):
        x = col * (CELL + GUTTER) + CELL // 2
        y = header_h // 2
        _centered_text(draw, (x, y), label, font)

    for row_idx, tensors in enumerate(rows):
        y = header_h + GUTTER + row_idx * (CELL + GUTTER)
        for col, tensor in enumerate(tensors):
            x = col * (CELL + GUTTER)
            cell = _tensor_to_pil(tensor)
            if cell.size != (CELL, CELL):
                cell = cell.resize((CELL, CELL), _NEAREST)
            canvas.paste(cell, (x, y))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(args.out)


if __name__ == '__main__':
    main()
