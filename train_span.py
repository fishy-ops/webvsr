"""
Train SPAN-Lite for 2x super-resolution.

Phase 1 (warm-start):  L1 + FFT only, 100 epochs
Phase 2 (full loss):   L1 + perceptual + FFT, 400 epochs

Usage:
  python train_span.py                     # start from scratch
  python train_span.py --resume            # resume from latest checkpoint
  python train_span.py --phase2            # jump to phase 2 with best phase1 weights
"""

import argparse
import os
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from model_span import SPANLite, count_params
from losses import CombinedLoss
from dataset import SRDataset, ValidationDataset


# ── Config ──────────────────────────────────────────────────────────────

CONFIG = {
    "scale": 2,
    "feature_channels": 32,

    # Training
    "crop_size": 256,
    "batch_size": 16,
    "accumulation_steps": 2,      # effective batch = 32
    "lr": 5e-4,
    "lr_min": 1e-6,
    "weight_decay": 1e-4,
    "phase1_epochs": 100,
    "phase2_epochs": 400,
    "total_epochs": 500,

    # Loss weights (phase 2)
    "w_perceptual": 0.1,
    "w_fft": 0.01,

    # Data — paths on YOUR machine
    "train_dirs": [
        r"C:\Users\reach\OneDrive\Documents\mamba-sr\DIV2K_train_HR\DIV2K_train_HR",
        r"C:\Users\reach\OneDrive\Documents\mamba-sr\Flickr2K",
        # Add LSDIR path here when downloaded:
        # r"D:\webvsr\data\LSDIR\train",
        r"D:\webvsr\data\vimeo_frames",
    ],
    "val_dir": r"C:\Users\reach\OneDrive\Documents\mamba-sr\DIV2K_valid_HR\DIV2K_valid_HR",

    # Output
    "checkpoint_dir": r"D:\webvsr\checkpoints",
    "log_file": r"D:\webvsr\training_log.json",

    # Workers
    "num_workers": 6,
}


def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse.item())


def validate(model, val_loader, device):
    model.eval()
    total_psnr = 0
    count = 0
    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            sr = sr.clamp(0, 1)
            if sr.shape != hr.shape:
                sr = F.interpolate(sr, size=hr.shape[2:], mode="bicubic",
                                   align_corners=False).clamp(0, 1)
            for i in range(sr.shape[0]):
                total_psnr += psnr(sr[i], hr[i])
                count += 1
    model.train()
    return total_psnr / max(count, 1)


def build_scheduler(optimizer, total_epochs, warmup_epochs=5):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_psnr,
                    phase, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_psnr": best_psnr,
        "phase": phase,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0), ckpt.get("best_psnr", 0), ckpt.get("phase", 1)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--phase2", action="store_true",
                        help="Start phase 2 from best phase 1 checkpoint")
    parser.add_argument("--channels", type=int, default=CONFIG["feature_channels"])
    parser.add_argument("--scale", type=int, default=CONFIG["scale"])
    parser.add_argument("--ckpt-dir", default=CONFIG["checkpoint_dir"])
    parser.add_argument("--total-epochs", type=int, default=CONFIG["total_epochs"])
    parser.add_argument("--phase1-epochs", type=int, default=CONFIG["phase1_epochs"])
    args = parser.parse_args()

    cfg = dict(CONFIG)  # copy so per-run overrides don't mutate the shared config
    cfg["feature_channels"] = args.channels
    cfg["scale"] = args.scale
    cfg["checkpoint_dir"] = args.ckpt_dir
    cfg["total_epochs"] = args.total_epochs
    cfg["phase1_epochs"] = args.phase1_epochs
    cfg["log_file"] = os.path.join(args.ckpt_dir, "training_log.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)


    # ── Model ───────────────────────────────────────────────────────
    model = SPANLite(
        upscale=cfg["scale"],
        feature_channels=cfg["feature_channels"],
    ).to(device)
    print(f"SPAN-Lite: {count_params(model):,} trainable parameters")

    # ── Data ────────────────────────────────────────────────────────
    train_dataset = SRDataset(
        data_dirs=cfg["train_dirs"],
        crop_size=cfg["crop_size"],
        scale=cfg["scale"],
        use_degradation=True,
    )
    val_dataset = ValidationDataset(
        data_dir=cfg["val_dir"],
        scale=cfg["scale"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if cfg["num_workers"] > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print(f"Train: {len(train_dataset)} images, {len(train_loader)} batches/epoch")
    print(f"Val:   {len(val_dataset)} images")

    # ── Optimizer & scheduler ───────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.99),
        weight_decay=cfg["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, cfg["total_epochs"])
    scaler = GradScaler()

    # ── Resume / phase logic ────────────────────────────────────────
    start_epoch = 0
    best_psnr = 0
    current_phase = 1
    log_entries = []

    latest_ckpt = os.path.join(cfg["checkpoint_dir"], "latest.pth")
    best_p1_ckpt = os.path.join(cfg["checkpoint_dir"], "best_phase1.pth")
    best_p2_ckpt = os.path.join(cfg["checkpoint_dir"], "best_phase2.pth")

    if args.phase2:
        if os.path.exists(best_p1_ckpt):
            print(f"Loading best phase 1 weights from {best_p1_ckpt}")
            start_epoch, best_psnr, _ = load_checkpoint(
                best_p1_ckpt, model, optimizer, scheduler, scaler
            )
            current_phase = 2
            start_epoch = cfg["phase1_epochs"]
            best_psnr = 0
        else:
            print("No phase 1 checkpoint found, starting from scratch")
    elif args.resume and os.path.exists(latest_ckpt):
        print(f"Resuming from {latest_ckpt}")
        start_epoch, best_psnr, current_phase = load_checkpoint(
            latest_ckpt, model, optimizer, scheduler, scaler
        )
        start_epoch += 1

    if os.path.exists(cfg["log_file"]):
        with open(cfg["log_file"], "r") as f:
            log_entries = json.load(f)

    # ── Training loop ───────────────────────────────────────────────
    print(f"\nStarting training from epoch {start_epoch}, phase {current_phase}")
    print(f"Effective batch size: {cfg['batch_size'] * cfg['accumulation_steps']}")

    for epoch in range(start_epoch, cfg["total_epochs"]):
        # Phase transition
        if epoch == cfg["phase1_epochs"] and current_phase == 1:
            current_phase = 2
            best_psnr = 0
            print("\n" + "=" * 60)
            print("PHASE 2: Adding perceptual + FFT loss")
            print("=" * 60)

        # Build loss for current phase
        use_perceptual = (current_phase == 2)
        criterion = CombinedLoss(
            w_perceptual=cfg["w_perceptual"],
            w_fft=cfg["w_fft"],
            use_perceptual=use_perceptual,
        ).to(device)

        model.train()
        epoch_loss = 0
        loss_components = {}
        batch_count = 0
        t0 = time.time()

        optimizer.zero_grad()

        for batch_idx, (lr, hr) in enumerate(train_loader):
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.float16):
                sr = model(lr)
                sr = sr.clamp(0, 1)
                if sr.shape != hr.shape:
                    sr = F.interpolate(sr, size=hr.shape[2:], mode="bicubic",
                                       align_corners=False).clamp(0, 1)
                loss, components = criterion(sr, hr)
                loss = loss / cfg["accumulation_steps"]

            scaler.scale(loss).backward()

            if (batch_idx + 1) % cfg["accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * cfg["accumulation_steps"]
            for k, v in components.items():
                loss_components[k] = loss_components.get(k, 0) + v
            batch_count += 1

        scheduler.step()

        avg_loss = epoch_loss / max(batch_count, 1)
        avg_components = {k: v / max(batch_count, 1) for k, v in loss_components.items()}
        elapsed = time.time() - t0

        # Validate every 5 epochs or last epoch of each phase
        val_psnr = 0
        if epoch % 5 == 0 or epoch == cfg["phase1_epochs"] - 1 or epoch == cfg["total_epochs"] - 1:
            val_psnr = validate(model, val_loader, device)

        # Logging
        lr_now = optimizer.param_groups[0]["lr"]
        comp_str = " | ".join(f"{k}: {v:.4f}" for k, v in avg_components.items())
        print(
            f"[P{current_phase}] Epoch {epoch:03d}/{cfg['total_epochs']} | "
            f"loss: {avg_loss:.4f} | {comp_str} | "
            f"PSNR: {val_psnr:.2f} | lr: {lr_now:.2e} | {elapsed:.0f}s"
        )

        entry = {
            "epoch": epoch, "phase": current_phase, "loss": avg_loss,
            "components": avg_components, "psnr": val_psnr, "lr": lr_now,
            "time_s": elapsed,
        }
        log_entries.append(entry)
        with open(cfg["log_file"], "w") as f:
            json.dump(log_entries, f, indent=2)

        # Save checkpoints
        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch, best_psnr,
            current_phase, latest_ckpt
        )

        if val_psnr > best_psnr and val_psnr > 0:
            best_psnr = val_psnr
            best_path = best_p1_ckpt if current_phase == 1 else best_p2_ckpt
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_psnr,
                current_phase, best_path
            )
            print(f"  >> New best PSNR: {best_psnr:.2f} dB (saved)")

        if epoch % 50 == 0 and epoch > 0:
            milestone = os.path.join(cfg["checkpoint_dir"], f"epoch_{epoch:03d}.pth")
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_psnr,
                current_phase, milestone
            )

    print(f"\nTraining complete. Best PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    train()
