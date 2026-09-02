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

    # Data. The 9500/500 split under /tank is what every recent run uses;
    # override with --train-dirs / --val-dir for anything else.
    "train_dirs": [
        r"/tank/webvsr/train_hr",
    ],
    "val_dir": r"/tank/webvsr/val_hr",

    # Output
    "checkpoint_dir": r"checkpoints",
    "log_file": r"training_log.json",

    # Workers
    "num_workers": 6,
}


def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse.item())


def _make_dists(device):
    """DISTS scorer, or None if unavailable -- never fatal to a training run."""
    try:
        from DISTS_pytorch import DISTS
        return DISTS().to(device).eval()
    except Exception as e:
        print(f"DISTS unavailable ({type(e).__name__}); selection falls back to PSNR")
        return None


def validate(model, val_loader, device, dists_fn=None):
    model.eval()
    total_psnr = 0
    total_dists = 0.0
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
            if dists_fn is not None:
                total_dists += dists_fn(sr.clamp(0, 1).float(),
                                        hr.clamp(0, 1).float()).mean().item() * sr.shape[0]
    model.train()
    n = max(count, 1)
    return total_psnr / n, (total_dists / n if dists_fn is not None else None)


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
    # The committed CONFIG paths are Windows paths from the original machine and
    # no longer resolve; these let a run specify its own data without editing it.
    parser.add_argument("--train-dirs", nargs="+", default=None,
                        help="override CONFIG['train_dirs']")
    parser.add_argument("--val-dir", default=None,
                        help="override CONFIG['val_dir']")
    parser.add_argument("--log-file", default=None,
                        help="override CONFIG['log_file']")
    parser.add_argument("--codec-degrade", action="store_true",
                        help="degrade with real video encoders (H.264/H.265/MPEG-4) "
                             "instead of JPEG -- matches the deployment domain")
    parser.add_argument("--num-workers", type=int, default=None)
    # crop_size is HR; the LR crop the model actually sees is crop_size/scale.
    # At scale 4 the shipped 256 gives a 64px LR crop while the harness feeds
    # 256px, so the model is judged on four times the context it trained on.
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accum-steps", type=int, default=None)
    # w_fft ships at 0.01, which is ~0.6% of the loss -- effectively off. It is
    # the term that penalises missing high frequency, i.e. the direct lever
    # against the over-smoothing that PSNR selection rewards.
    parser.add_argument("--w-fft", type=float, default=None)
    parser.add_argument("--w-perceptual", type=float, default=None)
    # "unshuffle" runs the trunk at half resolution via PixelUnshuffle(2),
    # ~4x fewer spatial positions for the same channel count.
    parser.add_argument("--arch", choices=["span", "unshuffle", "spanv2"],
                        default="span")
    parser.add_argument("--init-from", default=None,
                        help="load model weights from this checkpoint and start "
                             "fresh (no optimizer or epoch state) -- for adapting "
                             "an existing model to a new degradation")
    args = parser.parse_args()

    cfg = dict(CONFIG)  # copy so per-run overrides don't mutate the shared config
    cfg["feature_channels"] = args.channels
    cfg["scale"] = args.scale
    cfg["checkpoint_dir"] = args.ckpt_dir
    cfg["total_epochs"] = args.total_epochs
    cfg["phase1_epochs"] = args.phase1_epochs
    if args.train_dirs:
        cfg["train_dirs"] = args.train_dirs
    if args.val_dir:
        cfg["val_dir"] = args.val_dir
    if args.log_file:
        cfg["log_file"] = args.log_file
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.crop_size is not None:
        cfg["crop_size"] = args.crop_size
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.accum_steps is not None:
        cfg["accumulation_steps"] = args.accum_steps
    if args.w_fft is not None:
        cfg["w_fft"] = args.w_fft
    if args.w_perceptual is not None:
        cfg["w_perceptual"] = args.w_perceptual

    degrade_fn = None
    if args.codec_degrade:
        from codec_degrade import second_order_codec
        degrade_fn = second_order_codec
        print("degradation: real video codecs (H.264/H.265/MPEG-4)")
    else:
        print("degradation: JPEG (legacy)")

    cfg["log_file"] = os.path.join(args.ckpt_dir, "training_log.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dists_fn = _make_dists(device)
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)


    # ── Model ───────────────────────────────────────────────────────
    if args.arch == "unshuffle":
        from model_span_unshuffle import SPANLiteUnshuffle as _Arch
    elif args.arch == "spanv2":
        from model_span_v2 import SPANLiteV2 as _Arch
    else:
        _Arch = SPANLite
    model = _Arch(
        upscale=cfg["scale"],
        feature_channels=cfg["feature_channels"],
    ).to(device)
    print(f"SPAN-Lite: {count_params(model):,} trainable parameters")

    # ── Data ────────────────────────────────────────────────────────
    train_dataset = SRDataset(
        data_dirs=cfg["train_dirs"],
        degrade_fn=degrade_fn,
        crop_size=cfg["crop_size"],
        scale=cfg["scale"],
        use_degradation=True,
    )
    val_dataset = ValidationDataset(
        data_dir=cfg["val_dir"],
        scale=cfg["scale"],
        # Validate in the same domain we train and deploy in, or selection
        # optimises toward bicubic while the product runs on codec artifacts.
        degrade_fn=degrade_fn,
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
    best_dists = float('inf')
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
    if args.init_from:
        blob = torch.load(args.init_from, map_location="cpu", weights_only=False)
        state = blob.get("model", blob.get("model_state_dict", blob))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"init-from {args.init_from}: "
              f"{len(missing)} missing, {len(unexpected)} unexpected tensors")
        start_epoch, best_psnr, current_phase = 0, 0, 1

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
        val_dists = None
        if epoch % 5 == 0 or epoch == cfg["phase1_epochs"] - 1 or epoch == cfg["total_epochs"] - 1:
            val_psnr, val_dists = validate(model, val_loader, device, dists_fn)

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
            "dists": val_dists,
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

        # Selection: DISTS when available (lower is better), PSNR otherwise.
        # Run 1 showed PSNR selection ships the blurriest good model.
        if val_dists is not None:
            improved = val_dists < best_dists
        else:
            improved = val_psnr > best_psnr and val_psnr > 0
        if improved:
            if val_dists is not None:
                best_dists = val_dists
            best_psnr = max(best_psnr, val_psnr)
            best_path = best_p1_ckpt if current_phase == 1 else best_p2_ckpt
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_psnr,
                current_phase, best_path
            )
            if val_dists is not None:
                print(f"  >> New best DISTS: {best_dists:.4f} "
                      f"(PSNR {val_psnr:.2f} dB) (saved)")
            else:
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
