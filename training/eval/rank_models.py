"""Rank every candidate checkpoint on the metrics that decide what ships.

Four separate evaluations produced four JSON files this session and the ranking
had to be assembled by eye each time. This runs one evaluation with every model
at once -- frame preparation and the bicubic reference are shared, so N models
cost far less than N runs -- and prints the comparison already split the way
§9 showed matters.

Selection order, from what this project has established:
  1. DISTS on the 12 REAL-CAMERA clips. §14: the render clips inflate every
     margin, and §11: PSNR does not transfer.
  2. |tLP| on the same clips. §11: the only advantage that transfers across
     content types. §19: worth protecting, a codec retrain traded it away.
  3. Clips won, as a robustness check against one clip carrying an average.

    python training/eval/rank_models.py --crf 28 \\
        shipped=checkpoints_c16/best_phase2.pth \\
        webcodec=/tank/webvsr/ckpt_webcodec_2x_c16/best_phase2.pth
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RENDERS = {"bistro_30s.mp4", "chess_30s.mp4", "locomotive_30s.mp4"}


def per_clip(txt, model):
    blocks = re.split(r"^(\S+\.mp4)\s*$", txt, flags=re.M)
    out = {}
    for i in range(1, len(blocks), 2):
        clip, body = blocks[i], blocks[i + 1]
        vals = {}
        for key, header in (("dists", "DISTS (lower better)"),
                            ("tlp", "tLP (output flicker")):
            m = re.search(re.escape(header) + r".*?\n(.*?)(?:\n\s*\n|\Z)", body, re.S)
            if not m:
                continue
            for line in m.group(1).splitlines():
                p = line.split()
                if p and p[0] == model:
                    vals[key] = float(p[-1])
        if "dists" in vals:
            out[clip] = vals
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", metavar="NAME=CKPT")
    ap.add_argument("--crf", type=int, default=28)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--clips", default="/tank/webvsr/clips_busy")
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or f"/tank/webvsr/rank_crf{args.crf}.txt")
    cmd = [sys.executable, "stratified_eval.py", "--clips", args.clips,
           "--scale", str(args.scale), "--height", "1024",
           "--crf", str(args.crf), "--frames", str(args.frames)]
    names = []
    for spec in args.models:
        name, ckpt = spec.split("=", 1)
        p = Path(ckpt)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            sys.exit(f"missing checkpoint: {p}")
        names.append(name)
        cmd += ["--model", f"{name}={p}:{args.channels}:{args.scale}"]

    print(f"evaluating {len(names)} models in one pass at CRF {args.crf}...", flush=True)
    with open(out, "w") as f:
        rc = subprocess.run(cmd, cwd=ROOT / "training" / "eval",
                            stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        sys.exit(f"eval failed (rc={rc}); see {out}")

    txt = out.read_text(errors="replace")
    base = per_clip(txt, "bicubic")
    rows = []
    for n in names:
        got = per_clip(txt, n)
        cam = [c for c in got if c not in RENDERS and c in base]
        ren = [c for c in got if c in RENDERS and c in base]
        if not cam:
            continue
        g = [(1 - got[c]["dists"] / base[c]["dists"]) * 100 for c in cam]
        t = [abs(got[c].get("tlp", 0)) - abs(base[c].get("tlp", 0)) for c in cam]
        gr = [(1 - got[c]["dists"] / base[c]["dists"]) * 100 for c in ren]
        rows.append({"name": n, "cam_dists": np.mean(g), "cam_tlp": np.mean(t),
                     "wins": sum(1 for x in g if x > 0), "n": len(g),
                     "ren_dists": np.mean(gr) if gr else float("nan")})

    rows.sort(key=lambda r: -r["cam_dists"])
    print(f"\nCRF {args.crf} — ranked on DISTS over the {rows[0]['n']} real-camera clips")
    print(f"{'model':<14} {'cam DISTS':>10} {'wins':>7} {'|tLP| vs bic':>13} {'render DISTS':>13}")
    print("-" * 62)
    for r in rows:
        print(f"{r['name']:<14} {r['cam_dists']:+9.1f}% {r['wins']:>4}/{r['n']:<2} "
              f"{r['cam_tlp']:+13.5f} {r['ren_dists']:+12.1f}%")
    print("\n|tLP| vs bic: negative is better (closer to matching the truth's "
          "temporal behaviour; see §10 on why 'lower tLP' is not the rule)")

    # PAIRED comparison against the top model. §28: between-clip tLP variance is
    # seven times the between-model difference, so a difference of clip-set means
    # sits at ~0.3 SEM and cannot rank anything. Pairing on the clip cancels that
    # -- the same data went from unresolvable to t = -2.30. The columns above are
    # kept because they are readable; these are the numbers to decide on.
    ref = rows[0]["name"]
    ref_clips = per_clip(txt, ref)
    print(f"\nPaired against {ref}, per clip (positive favours {ref}):")
    print(f"{'model':<14} {'ΔDISTS':>12} {'t':>7} {'Δ|tLP|':>12} {'t':>7} {'clips':>7}")
    print("-" * 64)
    for r in rows[1:]:
        got = per_clip(txt, r["name"])
        cam = [c for c in got if c not in RENDERS and c in base and c in ref_clips]
        dd = np.array([got[c]["dists"] - ref_clips[c]["dists"] for c in cam])
        dt = np.array([abs(got[c].get("tlp", 0)) - abs(ref_clips[c].get("tlp", 0))
                       for c in cam])
        def tstat(v):
            sd = np.std(v, ddof=1)
            return v.mean() / (sd / np.sqrt(len(v))) if sd > 0 else float("nan")
        print(f"{r['name']:<14} {dd.mean():+12.5f} {tstat(dd):+7.2f} "
              f"{dt.mean():+12.5f} {tstat(dt):+7.2f} "
              f"{int((dd > 0).sum()):>4}/{len(cam):<2}")
    print("|t| > 2.2 is significant at 11 df. A large mean with small |t| means "
          "one clip is carrying it.")
    print(f"raw output: {out}")


if __name__ == "__main__":
    main()
