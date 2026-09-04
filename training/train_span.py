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

    # Data. Vimeo-90K frames are 448x256, which caps the HR crop at 256 and is
    # why the 4x crop-512 experiment could not run (RESEARCH.md 6a). DIV2K has a
    # median short side of 1356px, so it lifts that ceiling and supplies clean
    # HR targets -- the Vimeo frames are themselves compressed web video, so
    # they teach the model to reproduce artifacts already present in the target.
    #
    # A source may be written "path:repeat". Sampling is per-image, so without a
    # repeat a 2040x1356 DIV2K photo yields one crop per epoch exactly like a
    # 448x256 Vimeo frame despite holding ~27x the pixels. x4 puts DIV2K at
    # ~25% of the mix; a 2K image holds ~42 non-overlapping 256px crops, so this
    # is nowhere near re-using the same pixels.
    #
    # NOTE: the shipped checkpoints predate this mix. Reproducing them exactly
    # needs --train-dirs /tank/webvsr/train_hr on its own.
    "train_dirs": [
        r"/tank/webvsr/train_hr",
        r"/tank/webvsr/datasets/DIV2K_train_HR:4",
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


class EMA:
    """Exponential moving average of weights.

    Every top NTIRE 2026 efficient-SR team used this, the winner on the same
    SPAN family at decay 0.999. It costs one extra weight buffer and an
    in-place lerp per step, and it is evaluated and saved in place of the raw
    weights -- the running average is the model, the live weights are only how
    it gets there.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                    alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, model):
        """Install the average, returning the live weights for restoration."""
        sd = model.state_dict()
        backup = {k: sd[k].detach().clone() for k in self.shadow}
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        sd = model.state_dict()
        for k, v in backup.items():
            sd[k].copy_(v)


def validate(model, val_loader, device, dists_fn=None, depth=None):
    """Validate one exit. `depth` selects it on a multi-exit model.

    Without this the loop scores `model(lr)`, which defaults to the deepest
    exit -- so on a multi-exit run the early heads are invisible to checkpoint
    selection. A run initialised from a converged checkpoint then picks epoch 0
    as "best", because the deep exit is already perfect there while the early
    heads are still at random init, and the saved checkpoint contains a garbage
    early exit (measured: 8.6 dB against the deep exit's 37.4).
    """
    model.eval()
    total_psnr = 0
    total_dists = 0.0
    count = 0
    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr) if depth is None else model(lr, depth=depth)
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
    parser.add_argument("--arch", choices=["span", "unshuffle", "spanv2", "spanme"],
                        default="span")
    # spanme: shared trunk, one small head per exit depth. Trains every exit
    # jointly in a single trunk pass, so the deep exit is not degraded by the
    # shallow one competing for the same head.
    parser.add_argument("--exit-depths", default="2,4",
                        help="spanme only: comma-separated block depths to train exits at")
    parser.add_argument("--aux-weight", type=float, default=0.5,
                        help="spanme only: weight on each non-deepest exit's loss")
    # The early-exit heads start random while the trunk is already converged,
    # so joint training from epoch 0 lets their gradients degrade the deep
    # exit -- measured as 37.37 -> 35.91 dB across one epoch. Train the new
    # heads against a frozen trunk first, then unfreeze.
    # Flicker is the only advantage that transfers across content types
    # (RESEARCH.md 11), and nothing in this loss function targets it. This does:
    # two independent degradations of one HR crop stand in for two frames of
    # near-identical content, and the model is penalised for answering them
    # differently. Costs a second forward pass per step.
    parser.add_argument("--twin-consistency", type=float, default=0.0,
                        help="weight on |f(lr) - f(lr')| for two degradations "
                             "of the same crop; 0 disables and costs nothing")
    # The unmasked version of this loss failed (RESEARCH.md 12): penalising
    # disagreement everywhere has a trivial minimum -- make the network less
    # responsive -- and the run found it, converging onto bicubic's sharpness
    # and bicubic's tLP. Restricting the penalty to pixels where the two
    # degradations ALREADY agree removes that escape: the model cannot buy
    # agreement by blurring, because in those regions its two inputs are nearly
    # identical and a correct model would already agree. Where they genuinely
    # differ, no consistency is demanded. Same principle as Anime4K's line gate
    # and the masked temporal losses in the video-restoration literature.
    parser.add_argument("--twin-mask-q", type=float, default=0.0,
                        help="quantile of |lr-lr'| below which the consistency "
                             "penalty applies; 0.5 = the half of pixels the two "
                             "degradations agree on most. 0 disables masking")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="EMA decay for weights; 0 disables. 0.999 is what "
                             "the NTIRE 2026 winner used on this architecture")
    parser.add_argument("--w-dists", type=float, default=0.0,
                        help="weight on DISTS as a training term. Checkpoints "
                             "are selected on DISTS but nothing in the loss "
                             "pointed at it; 0 keeps the old behaviour")
    parser.add_argument("--lr", type=float, default=None,
                        help="override CONFIG['lr']; the 5e-4 default is a "
                             "from-scratch rate and will damage a converged "
                             "checkpoint loaded via --init-from")
    parser.add_argument("--freeze-trunk-epochs", type=int, default=0,
                        help="spanme only: epochs to train early-exit heads with "
                             "the trunk and deepest head frozen")
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
    if args.lr is not None:
        cfg["lr"] = args.lr
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
    elif args.arch == "spanme":
        from model_span_me import SPANLiteME as _Arch
    else:
        _Arch = SPANLite
    _kw = {}
    if args.arch == "spanme":
        _kw["exit_depths"] = tuple(int(d) for d in args.exit_depths.split(","))
    model = _Arch(
        upscale=cfg["scale"],
        feature_channels=cfg["feature_channels"],
        **_kw,
    ).to(device)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    if ema:
        print(f"EMA enabled, decay {args.ema_decay}")
    multi_exit = args.arch == "spanme"
    if multi_exit:
        print(f"multi-exit depths {model.exit_depths}, aux weight {args.aux_weight}")
    print(f"SPAN-Lite: {count_params(model):,} trainable parameters")

    # ── Data ────────────────────────────────────────────────────────
    twin = args.twin_consistency > 0
    if twin:
        print(f"twin consistency: weight {args.twin_consistency}")
    train_dataset = SRDataset(
        data_dirs=cfg["train_dirs"],
        degrade_fn=degrade_fn,
        crop_size=cfg["crop_size"],
        scale=cfg["scale"],
        use_degradation=True,
        twin=twin,
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
        if multi_exit and any(k.startswith("block_") for k in state):
            # A plain SPANLite checkpoint: key names differ, so remap rather
            # than silently dropping the whole trunk via strict=False.
            loaded, n_missing = model.load_span_lite(state)
            print(f"init-from {args.init_from}: remapped {loaded} SPANLite tensors, "
                  f"{n_missing} left at init (early-exit heads)")
        else:
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
            w_dists=args.w_dists,
            w_perceptual=cfg["w_perceptual"],
            w_fft=cfg["w_fft"],
            use_perceptual=use_perceptual,
        ).to(device)

        model.train()
        epoch_loss = 0
        loss_components = {}
        batch_count = 0
        t0 = time.time()

        if multi_exit and args.freeze_trunk_epochs > 0:
            frozen = epoch < args.freeze_trunk_epochs
            full_key = model._key(model.max_depth)
            for name, prm in model.named_parameters():
                is_early_head = (
                    name.startswith(("conv_cat.", "conv_last.", "upsampler."))
                    and f".{full_key}." not in name
                )
                prm.requires_grad_(is_early_head if frozen else True)
            if epoch in (0, args.freeze_trunk_epochs):
                n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  [multi-exit] trunk {'FROZEN' if frozen else 'unfrozen'}: "
                      f"{n_train:,} trainable params")

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            if twin:
                lr, lr2, hr = batch
                lr2 = lr2.to(device, non_blocking=True)
            else:
                lr, hr = batch
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.float16):
                if multi_exit:
                    outs = model.forward_all_exits(lr)
                    loss = 0.0
                    for d, sr_d in outs.items():
                        sr_d = sr_d.clamp(0, 1)
                        if sr_d.shape != hr.shape:
                            sr_d = F.interpolate(sr_d, size=hr.shape[2:], mode="bicubic",
                                                 align_corners=False).clamp(0, 1)
                        l_d, comp_d = criterion(sr_d, hr)
                        w = 1.0 if d == model.max_depth else args.aux_weight
                        loss = loss + w * l_d
                        if d == model.max_depth:
                            sr, components = sr_d, comp_d
                else:
                    sr = model(lr)
                    sr = sr.clamp(0, 1)
                    if sr.shape != hr.shape:
                        sr = F.interpolate(sr, size=hr.shape[2:], mode="bicubic",
                                           align_corners=False).clamp(0, 1)
                    loss, components = criterion(sr, hr)

                if twin:
                    # Same HR, so any disagreement between the two outputs is
                    # error by construction -- no ground truth needed for this
                    # term. L1 rather than L2: flicker shows up as a few pixels
                    # moving a lot, which a squared penalty would let dominate.
                    sr2 = model(lr2)
                    if multi_exit:
                        sr2 = sr2.clamp(0, 1)
                    else:
                        sr2 = sr2.clamp(0, 1)
                    if sr2.shape != sr.shape:
                        sr2 = F.interpolate(sr2, size=sr.shape[2:], mode="bicubic",
                                            align_corners=False).clamp(0, 1)
                    d_out = (sr - sr2).abs()
                    if args.twin_mask_q > 0:
                        # Per-pixel LR disagreement, upsampled to output size.
                        d_in = (lr - lr2).abs().mean(1, keepdim=True)
                        d_in = F.interpolate(d_in, size=sr.shape[2:],
                                             mode="nearest")
                        thr = torch.quantile(
                            d_in.flatten(1).float(), args.twin_mask_q, dim=1
                        ).view(-1, 1, 1, 1)
                        mask = (d_in <= thr).to(d_out.dtype)
                        # mask is single-channel and d_out is not, so the
                        # denominator has to count channels too -- otherwise the
                        # term comes out C times larger than the unmasked one and
                        # --twin-consistency means something different in each mode.
                        denom = (mask.sum() * d_out.shape[1]).clamp(min=1.0)
                        tc = (d_out * mask).sum() / denom
                    else:
                        tc = d_out.mean()
                    loss = loss + args.twin_consistency * tc
                    components = dict(components)
                    components["twin"] = float(tc.detach())

                loss = loss / cfg["accumulation_steps"]

            scaler.scale(loss).backward()

            if (batch_idx + 1) % cfg["accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

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
        # From here to the end of the epoch the model carries the averaged
        # weights: validating one set and saving another would make the
        # selection metric describe a checkpoint that was never written.
        ema_backup = ema.swap_in(model) if ema else None

        if epoch % 5 == 0 or epoch == cfg["phase1_epochs"] - 1 or epoch == cfg["total_epochs"] - 1:
            if multi_exit:
                # Score every exit and select on the WORST of them. A multi-exit
                # checkpoint is only as useful as its weakest head, and selecting
                # on the deepest one alone saves epoch 0 with a random early exit.
                per_exit = {}
                for d_ in model.exit_depths:
                    per_exit[d_] = validate(model, val_loader, device, dists_fn, depth=d_)
                val_psnr = min(v[0] for v in per_exit.values())
                val_dists = max(v[1] for v in per_exit.values())
                print("      exits: " + "  ".join(
                    f"d{d_}: {v[0]:.2f}dB/{v[1]:.4f}" for d_, v in sorted(per_exit.items())))
            else:
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

        if ema_backup is not None:
            ema.restore(model, ema_backup)

    print(f"\nTraining complete. Best PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    train()
