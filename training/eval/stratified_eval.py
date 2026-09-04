"""
Stratified evaluation for WebVSR — phase 00.

Answers the question a whole-image PSNR mean cannot: *where* does the model
fail? Every pixel is bucketed by local gradient energy into flat / edge /
texture and metrics are reported per bucket, so a model that mushes dense
detail while acing text can no longer hide behind a good average.

Two deliberate departures from evaluate.py:

  1. Ground truth comes from real video, not stills. An HR clip is downscaled
     and re-encoded with ffmpeg to produce the LR input, so the degradation
     contains genuine inter-frame codec artifacts (motion-compensation
     blocking, temporal ringing) rather than per-image JPEG. Training and
     eval on bicubic/JPEG pairs is the mismatch this whole exercise exists
     to close.

  2. Temporal flicker is measured. tLP compares LPIPS between consecutive
     *output* frames against LPIPS between consecutive *GT* frames. Flicker
     is invisible on stills and is the main way training for perceptual
     quality can make video worse while every still-image metric improves.

     **0 is ideal, not minus infinity.** A perfect reconstruction scores
     exactly 0. Positive means the output changes more between frames than the
     truth does -- added flicker. Negative means it changes *less*, which is
     temporal over-smoothing, and it falls monotonically with blur: measured on
     three clips, bicubic -0.0236, blurred bicubic -0.0315 / -0.0472 / -0.0658
     at sigma 1/2/4, and a constant grey frame -0.1257 at 10.4 dB. An earlier
     version of this header said "lower better", which would rank that grey
     frame first. Read |tLP| as the deviation; among outputs that are all
     negative, closer to zero is better.

Selection guidance: choose on DISTS with a PSNR floor and a bound on |tLP|.
Never select on PSNR alone — it is minimised by the conditional mean, which
is the blurry answer.

Usage
-----
  python stratified_eval.py \
      --clips /path/to/hr_clips \
      --model c16=../../checkpoints_c16/best_phase1.pth:16:2 \
      --model c32=../../checkpoints/best_phase1.pth:32:2 \
      --scale 2 --crf 28 --frames 48

Requires: torch, torchvision, numpy, Pillow, ffmpeg on PATH.
Optional:  pip install DISTS-pytorch lpips   (DISTS + LPIPS/tLP columns)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# model_span.py lives one level up, in training/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_span import SPANLite  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_dump import save_comparison  # noqa: E402

VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}


# ── degradation: real video, real codec ──────────────────────────────

def make_pair(clip, workdir, scale, crf, frames, height):
    """HR frames + codec-degraded LR frames from one video file.

    The LR pass is a genuine encode, not a resize: x264 at a punishing CRF
    reproduces the artifacts the extension actually meets in the wild.
    """
    hr_dir = workdir / "hr"
    lr_dir = workdir / "lr"
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    # HR reference: scale to an even multiple of `scale` so SR output aligns.
    h = (height // (2 * scale)) * (2 * scale)

    # Refuse to enlarge. Scaling a 1038-tall source up to 1080 makes the HR
    # *reference* an interpolation, which holds no detail the model has to
    # recover and so inflates any model's apparent win: measured at +1.4 dB for
    # the three 1038-tall clips against <=+0.46 for every native-1080 clip.
    # Same failure as RESEARCH.md 6a, on the evaluation side.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=height", "-of", "csv=p=0", str(clip)],
        capture_output=True, text=True,
    )
    try:
        src_h = int(probe.stdout.strip().split(",")[0])
    except (ValueError, IndexError):
        src_h = None
    if src_h is not None and h > src_h:
        raise ValueError(
            f"{Path(clip).name} is {src_h}px tall but --height asks for {h}. "
            f"Upscaling the HR reference makes it an interpolation and inflates "
            f"the measured gain. Pass --height {(src_h // (2 * scale)) * (2 * scale)} "
            f"or lower."
        )

    vf_hr = f"scale=-2:{h}:flags=lanczos"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip),
         "-vf", vf_hr, "-frames:v", str(frames), "-start_number", "0",
         str(hr_dir / "%04d.png")],
        check=True,
    )

    # LR: downscale then encode. Encoding *after* the downscale is what puts
    # the codec artifacts at the resolution the network will actually see.
    lr_mp4 = workdir / "lr.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip),
         "-vf", f"scale=-2:{h // scale}:flags=bicubic",
         "-frames:v", str(frames), "-c:v", "libx264", "-crf", str(crf),
         "-preset", "veryfast", "-pix_fmt", "yuv420p", str(lr_mp4)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(lr_mp4),
         "-start_number", "0", str(lr_dir / "%04d.png")],
        check=True,
    )

    hr = sorted(hr_dir.glob("*.png"))
    lr = sorted(lr_dir.glob("*.png"))
    n = min(len(hr), len(lr))
    return hr[:n], lr[:n]


def load(path, device):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


# ── complexity stratification ────────────────────────────────────────

_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
_SOBEL_Y = _SOBEL_X.transpose(2, 3).clone()


def grad_mag(img):
    """Sobel gradient magnitude of luma. (1,3,H,W) -> (1,1,H,W)."""
    luma = (0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3])
    kx = _SOBEL_X.to(img.device, img.dtype)
    ky = _SOBEL_Y.to(img.device, img.dtype)
    gx = F.conv2d(luma, kx, padding=1)
    gy = F.conv2d(luma, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def bucket_thresholds(hr_paths, device, sample_cap=2_000_000):
    """Global 33rd/66th percentiles of GT gradient magnitude.

    Global rather than per-frame: absolute thresholds are what let you say
    "the model is worse on textured content", instead of only "worse on the
    busiest third of each frame whatever that happens to be".
    """
    vals = []
    for p in hr_paths:
        g = grad_mag(load(p, device)).flatten().cpu().numpy()
        if g.size > 40_000:
            g = np.random.choice(g, 40_000, replace=False)
        vals.append(g)
    allv = np.concatenate(vals)
    if allv.size > sample_cap:
        allv = np.random.choice(allv, sample_cap, replace=False)
    return float(np.percentile(allv, 33)), float(np.percentile(allv, 66))


def masks_for(hr, lo, hi):
    g = grad_mag(hr)
    return {"flat": g < lo, "edge": (g >= lo) & (g < hi), "texture": g >= hi}


# ── metrics ──────────────────────────────────────────────────────────

def masked_psnr(sr, hr, mask):
    """PSNR over masked pixels only. mask is (1,1,H,W) bool."""
    m = mask.expand_as(hr)
    n = int(m.sum().item())
    if n == 0:
        return float("nan")
    mse = (((sr - hr) ** 2) * m).sum().item() / n
    if mse <= 1e-12:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def sharpness_ratio(sr, hr, mask, floor=0.01):
    """Mean SR gradient / mean GT gradient inside the mask.

    1.0 matches ground truth. Below 1 is over-smoothed (the L1 failure);
    above 1 is over-sharpened, which is how staircasing and halos read
    numerically. This is the number that tells you what a sharpness
    setting is actually doing, per content type.

    Restricted to pixels where the GT has real gradient to compare against
    (>= floor). Without that guard a bucket of near-flat pixels divides a
    real numerator by ~0 and reports ratios in the thousands -- a
    divide-by-zero artifact, not over-sharpening. Rendered content with
    large perfectly-flat areas triggers this constantly.
    """
    gs, gh = grad_mag(sr), grad_mag(hr)
    m = mask & (gh >= floor)
    if int(m.sum().item()) < 64:
        return float("nan")
    return ((gs * m).sum() / (gh * m).sum().clamp_min(1e-8)).item()


class Perceptual:
    """DISTS / LPIPS if installed, else the columns are dropped."""

    def __init__(self, device):
        self.device = device
        self.dists = None
        self.lpips = None
        try:
            from DISTS_pytorch import DISTS
            self.dists = DISTS().to(device).eval()
        except Exception as e:
            print(f"  [note] DISTS unavailable ({e.__class__.__name__}); "
                  f"pip install DISTS-pytorch", file=sys.stderr)
        try:
            import lpips
            self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except Exception as e:
            print(f"  [note] LPIPS unavailable ({e.__class__.__name__}); "
                  f"pip install lpips  (tLP will be skipped)", file=sys.stderr)

    @torch.no_grad()
    def d(self, a, b):
        return None if self.dists is None else self.dists(a, b).mean().item()

    @torch.no_grad()
    def lp(self, a, b):
        if self.lpips is None:
            return None
        # lpips expects [-1, 1]
        return self.lpips(a * 2 - 1, b * 2 - 1).mean().item()


# ── models ───────────────────────────────────────────────────────────

def freeze_reparam(model):
    """Conv3XC re-fuses its branches on *every* eval forward, which dominates
    runtime over a long clip. Fuse once, then call the fused conv directly."""
    for m in model.modules():
        if hasattr(m, "_update_params") and hasattr(m, "eval_conv"):
            m._update_params()
            m.forward = m.eval_conv.forward
    return model


def load_model(ckpt, channels, scale, device, arch="span", depth=None):
    """Build and load a model. A multi-exit checkpoint is detected from its
    keys, so `--model name=ckpt:ch:scale:depth` works without also passing
    --arch, and `depth` selects which exit to evaluate."""
    blob_peek = torch.load(ckpt, map_location="cpu", weights_only=False)
    peek = blob_peek.get("model", blob_peek.get("model_state_dict", blob_peek))
    # SPANLite names its blocks "block_1..4"; SPANLiteME uses an nn.ModuleList,
    # so "blocks.0..3" is present in one and absent in the other.
    if any(k.startswith("blocks.") for k in peek):
        from model_span_me import SPANLiteME
        n_blocks = 1 + max(int(k.split(".")[1]) for k in peek if k.startswith("blocks."))
        depths = set()
        for k in peek:
            if k.startswith("conv_cat."):
                tag = k.split(".")[1]
                depths.add(n_blocks if tag == "full" else int(tag.lstrip("d")))
        model = SPANLiteME(feature_channels=channels, upscale=scale,
                           exit_depths=tuple(sorted(depths)), num_blocks=n_blocks)
        model.load_state_dict(peek)
        model.eval().to(device)
        model._eval_depth = depth if depth is not None else model.max_depth
        if model._eval_depth not in model.exit_depths:
            raise SystemExit(f"{ckpt}: depth {model._eval_depth} is not a trained "
                             f"exit; have {model.exit_depths}")
        return freeze_reparam(model)

    if arch == "unshuffle":
        from model_span_unshuffle import SPANLiteUnshuffle
        model = SPANLiteUnshuffle(feature_channels=channels, upscale=scale)
    elif arch == "spanv2":
        from model_span_v2 import SPANLiteV2
        model = SPANLiteV2(feature_channels=channels, upscale=scale)
    else:
        model = SPANLite(feature_channels=channels, upscale=scale)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = blob.get("model", blob.get("model_state_dict", blob))
    model.load_state_dict(state)
    model.eval().to(device)
    return freeze_reparam(model)


@torch.no_grad()
def run_model(model, lr):
    d = getattr(model, "_eval_depth", None)
    out = model(lr) if d is None else model(lr, depth=d)
    return out.clamp(0, 1)


def bicubic(lr, scale):
    return F.interpolate(lr, scale_factor=scale, mode="bicubic",
                         align_corners=False).clamp(0, 1)


# ── main ─────────────────────────────────────────────────────────────

def evaluate(models, hr_paths, lr_paths, lo, hi, perc, scale, device,
             evidence_dir=None, clip_name="clip", crf=0, every=8):
    buckets = ("flat", "edge", "texture")
    acc = {name: {"psnr": {b: [] for b in buckets},
                  "sharp": {b: [] for b in buckets},
                  "psnr_all": [], "dists": [], "tlp": []}
           for name in models}

    prev = {name: None for name in models}
    prev_hr = None

    for i, (hp, lp) in enumerate(zip(hr_paths, lr_paths)):
        hr = load(hp, device)
        lr = load(lp, device)
        m = masks_for(hr, lo, hi)
        frame_out = {}

        for name, fn in models.items():
            sr = fn(lr)
            if sr.shape[-2:] != hr.shape[-2:]:
                sr = F.interpolate(sr, size=hr.shape[-2:], mode="bicubic",
                                   align_corners=False).clamp(0, 1)
            a = acc[name]
            for b in buckets:
                a["psnr"][b].append(masked_psnr(sr, hr, m[b]))
                a["sharp"][b].append(sharpness_ratio(sr, hr, m[b]))
            a["psnr_all"].append(masked_psnr(sr, hr, torch.ones_like(m["flat"])))
            d = perc.d(sr, hr)
            if d is not None:
                a["dists"].append(d)
            # tLP: how much more does the output flicker than the source did?
            if prev[name] is not None and prev_hr is not None:
                lo_out = perc.lp(sr, prev[name])
                lo_gt = perc.lp(hr, prev_hr)
                if lo_out is not None and lo_gt is not None:
                    a["tlp"].append(lo_out - lo_gt)
            prev[name] = sr
            frame_out[name] = sr
        # Save evidence periodically rather than every frame: enough to see
        # the trend across a clip without writing thousands of PNGs.
        if evidence_dir is not None and i % every == 0:
            try:
                save_comparison(evidence_dir, clip_name, i, crf, hr,
                                frame_out, m["texture"])
            except Exception as e:
                print(f"    [evidence] {type(e).__name__}: {e}", file=sys.stderr)
        prev_hr = hr
        print(f"    frame {i + 1}/{len(hr_paths)}", end="\r", file=sys.stderr)
    print(file=sys.stderr)
    return acc


def mean(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    return float(np.mean(xs)) if xs else float("nan")


def report(acc, indent=""):
    buckets = ("flat", "edge", "texture")
    names = list(acc)
    w = max(len(n) for n in names) + 2

    p = lambda t="": print(indent + t)
    print("\nPSNR by content complexity (dB) — texture is the column that matters")
    p(f"{'model':<{w}}" + "".join(f"{b:>12}" for b in buckets) + f"{'all':>12}")
    for n in names:
        row = "".join(f"{mean(acc[n]['psnr'][b]):>12.2f}" for b in buckets)
        p(f"{n:<{w}}" + row + f"{mean(acc[n]['psnr_all']):>12.2f}")

    print("\nSharpness ratio vs ground truth (1.00 = matches; >1 over-sharpened)")
    print(f"{'model':<{w}}" + "".join(f"{b:>12}" for b in buckets))
    for n in names:
        print(f"{n:<{w}}" + "".join(
            f"{mean(acc[n]['sharp'][b]):>12.3f}" for b in buckets))

    if any(acc[n]["dists"] for n in names):
        print("\nDISTS (lower better) — use this to select checkpoints")
        for n in names:
            print(f"{n:<{w}}{mean(acc[n]['dists']):>12.4f}")

    if any(acc[n]["tlp"] for n in names):
        print("\ntLP (output flicker minus source flicker; 0 = matches truth, "
              "|tLP| is the deviation)")
        for n in names:
            print(f"{n:<{w}}{mean(acc[n]['tlp']):>12.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True,
                    help="directory of HR video files")
    ap.add_argument("--model", action="append", default=[], metavar="NAME=CKPT:CH:SCALE",
                    help="repeatable, e.g. c16=../../checkpoints_c16/best_phase1.pth:16:2")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--height", type=int, default=1080,
                    help="HR height to evaluate at")
    ap.add_argument("--crf", type=int, default=28,
                    help="x264 CRF for the LR pass; higher = more compressed")
    ap.add_argument("--frames", type=int, default=48, help="frames per clip")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", help="write raw results here")
    # Timestamped by default: filenames encode clip/frame/crf but not which
    # models were compared, so two evals of different model sets collide and
    # the earlier images are lost. These are the record; they never overwrite.
    ap.add_argument("--evidence",
                    default="/tank/webvsr/evidence/" + __import__("time").strftime("%m%d_%H%M%S"),
                    help="directory for side-by-side comparison crops; these are "
                         "the visual record behind the numbers and are never "
                         "deleted. Pass an empty string to disable.")
    args = ap.parse_args()

    if not args.evidence:
        args.evidence = None
    device = torch.device(args.device)
    clips = [p for p in sorted(Path(args.clips).iterdir())
             if p.suffix.lower() in VIDEO_EXT]
    if not clips:
        sys.exit(f"no video files in {args.clips}")

    models = {"bicubic": lambda lr: bicubic(lr, args.scale)}
    for spec in args.model:
        name, rest = spec.split("=", 1)
        # NAME=CKPT:CH:SCALE[:ARCH], or [:DEPTH] to pick a multi-exit exit.
        parts = rest.rsplit(":", 3)
        depth = None
        if len(parts) == 4 and not parts[3].isdigit():
            ckpt, ch, sc, arch = parts
        elif len(parts) == 4 and parts[3].isdigit() and parts[2].isdigit():
            ckpt, ch, sc, arch = parts[0], parts[1], parts[2], "span"
            depth = int(parts[3])
        else:
            ckpt, ch, sc = rest.rsplit(":", 2); arch = "span"
        m = load_model(ckpt, int(ch), int(sc), device, arch, depth=depth)
        models[name] = (lambda mm: (lambda lr: run_model(mm, lr)))(m)

    perc = Perceptual(device)
    combined = None

    for clip in clips:
        print(f"\n{clip.name}")
        with tempfile.TemporaryDirectory() as td:
            hr, lr = make_pair(clip, Path(td), args.scale, args.crf,
                               args.frames, args.height)
            if not hr:
                print("  no frames extracted, skipping")
                continue
            lo, hi = bucket_thresholds(hr, device)
            print(f"  gradient thresholds: flat<{lo:.4f} "
                  f"edge<{hi:.4f} texture>=")
            acc = evaluate(models, hr, lr, lo, hi, perc, args.scale, device,
                           evidence_dir=args.evidence, clip_name=clip.stem,
                           crf=args.crf)
        print(f"  -- {clip.name} --")
        report(acc, indent="  ")
        if combined is None:
            combined = acc
        else:
            for n in acc:
                for k in ("psnr", "sharp"):
                    for b in acc[n][k]:
                        combined[n][k][b] += acc[n][k][b]
                for k in ("psnr_all", "dists", "tlp"):
                    combined[n][k] += acc[n][k]

    if combined is None:
        sys.exit("nothing evaluated")

    print("\n" + "=" * 64)
    print(f"ALL CLIPS  (scale {args.scale}x, HR {args.height}p, LR crf {args.crf})")
    print("=" * 64)
    report(combined)

    if args.json:
        out = {n: {"psnr": {b: mean(v) for b, v in d["psnr"].items()},
                   "sharp": {b: mean(v) for b, v in d["sharp"].items()},
                   "psnr_all": mean(d["psnr_all"]),
                   "dists": mean(d["dists"]), "tlp": mean(d["tlp"])}
               for n, d in combined.items()}
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
