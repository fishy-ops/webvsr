"""
Train RIFE-Lite frame interpolation model.

Uses Vimeo-90K septuplet data: for each 7-frame clip, we train on
(frame1, frame3) -> predict frame2, and (frame5, frame7) -> predict frame6.

Since we only extracted center frames (im4.png) for SR training, this script
needs the triplet data. We'll use the center frames we have and create
synthetic training pairs using temporal augmentation.

For proper training, we'd need Vimeo-90K triplet or septuplet data with
multiple frames per clip. As a practical alternative, we train on pairs of
random crops from our existing HR dataset with synthetic motion.
"""
import argparse
import os
import random
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.amp import autocast, GradScaler
from PIL import Image

from model_rife_lite import RIFELite


class SyntheticFlowDataset(Dataset):
    """
    Creates synthetic frame interpolation training data from still images.

    For each sample:
    1. Load a random HR image and take a random crop
    2. Apply a random affine transform to create "frame 0"
    3. Apply the inverse/different transform for "frame 1"
    4. The original crop is the ground truth intermediate frame
    """
    def __init__(self, img_dirs, crop_size=256, max_flow=16):
        self.crop_size = crop_size
        self.max_flow = max_flow
        self.images = []
        exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        for d in img_dirs:
            if not os.path.isdir(d):
                continue
            for f in Path(d).rglob('*'):
                if f.suffix.lower() in exts:
                    self.images.append(str(f))
        print(f"SyntheticFlowDataset: {len(self.images)} images")
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images) * 4

    def __getitem__(self, idx):
        img_idx = idx % len(self.images)
        img = Image.open(self.images[img_idx]).convert('RGB')
        w, h = img.size

        cs = self.crop_size + self.max_flow * 2
        if w < cs or h < cs:
            img = img.resize((max(cs, w), max(cs, h)), Image.LANCZOS)
            w, h = img.size

        x = random.randint(0, w - cs)
        y = random.randint(0, h - cs)
        crop = img.crop((x, y, x + cs, y + cs))
        crop_t = self.to_tensor(crop)

        # Random flow vectors
        dx = random.uniform(-self.max_flow, self.max_flow)
        dy = random.uniform(-self.max_flow, self.max_flow)

        m = self.max_flow
        gt = crop_t[:, m:m + self.crop_size, m:m + self.crop_size]

        # Frame 0: shift by (-dx/2, -dy/2) from center
        ox0 = int(m - dx / 2)
        oy0 = int(m - dy / 2)
        img0 = crop_t[:, oy0:oy0 + self.crop_size, ox0:ox0 + self.crop_size]

        # Frame 1: shift by (+dx/2, +dy/2) from center
        ox1 = int(m + dx / 2)
        oy1 = int(m + dy / 2)
        img1 = crop_t[:, oy1:oy1 + self.crop_size, ox1:ox1 + self.crop_size]

        # Random augmentation
        if random.random() < 0.5:
            img0 = img0.flip(2)
            img1 = img1.flip(2)
            gt = gt.flip(2)
        if random.random() < 0.5:
            img0 = img0.flip(1)
            img1 = img1.flip(1)
            gt = gt.flip(1)

        return img0, img1, gt


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = RIFELite().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}")

    img_dirs = [
        r'/tank/webvsr/datasets/DIV2K_train_HR',
        r'/tank/webvsr/datasets/Flickr2K',
        r'/tank/webvsr/train_hr',
    ]

    dataset = SyntheticFlowDataset(img_dirs, crop_size=256, max_flow=16)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    log = []
    best_loss = float('inf')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        t0 = time.time()

        for i, (img0, img1, gt) in enumerate(loader):
            img0 = img0.to(device, non_blocking=True)
            img1 = img1.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            with autocast(device_type='cuda'):
                pred = model(img0, img1, t=0.5)
                loss = F.l1_loss(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0

        entry = {
            'epoch': epoch,
            'loss': avg_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'time_s': elapsed,
        }
        log.append(entry)
        print(f"Epoch {epoch:3d} | Loss {avg_loss:.6f} | LR {entry['lr']:.6f} | {elapsed:.0f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
            }, os.path.join(args.ckpt_dir, 'rife_best.pth'))

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
            }, os.path.join(args.ckpt_dir, 'rife_latest.pth'))

        with open(os.path.join(args.ckpt_dir, 'rife_log.json'), 'w') as f:
            json.dump(log, f, indent=2)

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--ckpt-dir', default='checkpoints')
    args = parser.parse_args()
    train(args)
