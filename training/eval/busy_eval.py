"""Does the model fall apart when the whole scene is busy?

`stratified_eval` buckets *pixels* by gradient energy, which answers "is the
model worse on textured pixels". That is not the reported failure. The report
is that busy *scenes* collapse -- text and simple shots look good, then a shot
with a lot going on degrades badly enough to turn the extension off.

Those are different axes. A frame can be 5% texture (a sign against sky) or 60%
texture (foliage, a crowd, confetti), and a per-pixel average over the whole
clip hides that difference completely: the texture bucket of a calm frame and
the texture bucket of a chaotic frame are pooled into one number.

So here the *frame* is the unit. Each frame gets a busyness score -- the
fraction of its pixels above the global texture threshold -- frames are binned
by that score, and the model's gain over bicubic is reported per bin. If the
complaint is real, the gain falls as busyness rises, and may go negative.

Thresholds are the same global 33rd/66th percentiles stratified_eval uses, so
"texture" means the same thing in both tools.

    python training/eval/busy_eval.py --clips /tank/webvsr/clips \
        --model shipped=checkpoints_c16/best_phase2.pth:16:2 --crf 28
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stratified_eval import (  # noqa: E402
    VIDEO_EXT, make_pair, load, grad_mag, bucket_thresholds, masks_for,
    masked_psnr, sharpness_ratio, load_model, run_model, bicubic,
)


def busyness(hr, hi):
    """Fraction of frame pixels at or above the global texture threshold."""
    return float((grad_mag(hr) >= hi).float().mean())


def profile(clips, args, device):
    """Per-clip busyness, using the same global thresholds as the full eval."""
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        pairs = []
        for clip in clips:
            hr_paths, _ = make_pair(clip, Path(td) / clip.stem, args.scale,
                                    args.crf, args.frames, args.height)
            pairs.append((clip.stem, hr_paths))
        all_hr = [p for _, hrs in pairs for p in hrs]
        lo, hi = bucket_thresholds(all_hr, device)
        print(f"\ngradient thresholds: flat<{lo:.4f} edge<{hi:.4f} texture>={hi:.4f}\n")
        print(f"{'clip':<28} {'n':>3} {'busy min':>9} {'busy max':>9} {'mean':>8}")
        print("-" * 60)
        stats = []
        for stem, hr_paths in pairs:
            b = [busyness(load(p, device), hi) for p in hr_paths]
            stats.append((stem, b))
            print(f"{stem:<28} {len(b):>3} {min(b):9.3f} {max(b):9.3f} {np.mean(b):8.3f}")
        allb = np.concatenate([b for _, b in stats])
        print(f"\noverall range {allb.min():.3f} - {allb.max():.3f}; "
              f"deciles: {np.round(np.quantile(allb, np.linspace(0,1,11)), 3).tolist()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--model", action="append", default=[],
                    metavar="NAME=CKPT:CH:SCALE")
    ap.add_argument("--arch", default="span")
    ap.add_argument("--scale", type=int, default=2)
    # 1024, matching every other eval in this project: it is at or below the
    # native height of all current clips, so no HR reference is ever enlarged.
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--crf", type=int, default=28)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json")
    # Answering "does this clip set span busyness?" needs no model, and asking
    # it first is what keeps a confounded set (RESEARCH.md 7) from being
    # discovered only after a full evaluation has been paid for.
    ap.add_argument("--profile", action="store_true",
                    help="report per-clip busyness only; runs no model")
    args = ap.parse_args()

    device = torch.device(args.device)
    clips = [p for p in sorted(Path(args.clips).iterdir())
             if p.suffix.lower() in VIDEO_EXT]
    if not clips:
        sys.exit(f"no video files in {args.clips}")

    if args.profile:
        profile(clips, args, device)
        return

    models = {}
    for spec in args.model:
        name, rest = spec.split("=", 1)
        parts = rest.rsplit(":", 3)
        # NAME=CKPT:CH:SCALE, or NAME=CKPT:CH:SCALE:DEPTH for a multi-exit model.
        if len(parts) == 4 and parts[3].isdigit() and parts[2].isdigit():
            ckpt, ch, sc, dep = parts[0], parts[1], parts[2], int(parts[3])
        else:
            ckpt, ch, sc = rest.rsplit(":", 2)
            dep = None
        models[name] = load_model(ckpt, int(ch), int(sc), device,
                                  arch=args.arch, depth=dep)
    if not models:
        sys.exit("pass at least one --model")

    rows = []
    with tempfile.TemporaryDirectory() as td:
        pairs = []
        for clip in clips:
            wd = Path(td) / clip.stem
            hr_paths, lr_paths = make_pair(
                clip, wd, args.scale, args.crf, args.frames, args.height)
            pairs.append((clip.stem, hr_paths, lr_paths))
            print(f"  prepared {clip.stem}: {len(hr_paths)} frames", flush=True)

        all_hr = [p for _, hrs, _ in pairs for p in hrs]
        lo, hi = bucket_thresholds(all_hr, device)
        print(f"  gradient thresholds: flat<{lo:.4f} edge<{hi:.4f} texture>={hi:.4f}\n",
              flush=True)

        for stem, hr_paths, lr_paths in pairs:
            for hp, lp in zip(hr_paths, lr_paths):
                hr = load(hp, device)
                lr = load(lp, device)
                m = masks_for(hr, lo, hi)
                bc = bicubic(lr, args.scale)
                if bc.shape != hr.shape:
                    continue
                row = {
                    "clip": stem,
                    "frame": hp.stem,
                    "busy": busyness(hr, hi),
                    "bicubic_tex": masked_psnr(bc, hr, m["texture"]),
                    "bicubic_sharp": sharpness_ratio(bc, hr, m["texture"]),
                }
                for name, model in models.items():
                    sr = run_model(model, lr)
                    if sr.shape != hr.shape:
                        continue
                    row[f"{name}_tex"] = masked_psnr(sr, hr, m["texture"])
                    row[f"{name}_sharp"] = sharpness_ratio(sr, hr, m["texture"])
                rows.append(row)
            print(f"  scored {stem}", flush=True)

    rows = [r for r in rows if all(f"{n}_tex" in r for n in models)]
    busy = np.array([r["busy"] for r in rows])
    edges = np.quantile(busy, np.linspace(0, 1, args.bins + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(busy, edges[1:-1]), 0, args.bins - 1)

    print(f"\n{len(rows)} frames from {len(clips)} clips, CRF {args.crf}, "
          f"binned by scene busyness (texture pixel fraction)\n")
    name = list(models)[0]
    hdr = (f"{'bin':>4}  {'busyness':>15}  {'n':>4}  {'bicubic':>8}  "
           f"{name:>8}  {'gain':>7}  {'sharp bic':>9}  {'sharp mdl':>9}")
    print(hdr)
    print("-" * len(hdr))
    out_bins = []
    for b in range(args.bins):
        sel = [r for r, i in zip(rows, idx) if i == b]
        if not sel:
            continue
        bl = np.mean([r["bicubic_tex"] for r in sel])
        ml = np.mean([r[f"{name}_tex"] for r in sel])
        bs = np.mean([r["bicubic_sharp"] for r in sel])
        ms = np.mean([r[f"{name}_sharp"] for r in sel])
        lo_b, hi_b = edges[b], edges[b + 1]
        print(f"{b:>4}  {lo_b:6.3f}-{hi_b:<8.3f}  {len(sel):>4}  {bl:8.3f}  "
              f"{ml:8.3f}  {ml - bl:+7.3f}  {bs:9.4f}  {ms:9.4f}")
        out_bins.append({"bin": b, "busy_lo": lo_b, "busy_hi": hi_b, "n": len(sel),
                         "bicubic_tex": bl, "model_tex": ml, "gain": ml - bl,
                         "bicubic_sharp": bs, "model_sharp": ms})

    print("\nworst 8 frames by gain:")
    worst = sorted(rows, key=lambda r: r[f"{name}_tex"] - r["bicubic_tex"])[:8]
    for r in worst:
        print(f"  {r['clip']}/{r['frame']}  busy={r['busy']:.3f}  "
              f"gain={r[f'{name}_tex'] - r['bicubic_tex']:+.3f} dB")

    if args.json:
        import json
        with open(args.json, "w") as f:
            json.dump({"bins": out_bins, "frames": rows,
                       "thresholds": {"lo": lo, "hi": hi}}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
