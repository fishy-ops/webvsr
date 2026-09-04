"""Can a no-reference signal predict where the network is worth running?

Section 15 proposed gating the network on content type, using section 13's
mechanism: bicubic RINGS on band-limited content (edge sharpness ratio 1.9-2.4)
and merely softens camera footage (<=1.0). That ratio needs ground truth, so it
cannot be used at runtime -- but bicubic's own edge overshoot can be measured
against the LR input alone.

This asks whether that signal actually separates winners from losers. If it only
separates renders from camera clips it is useless, because the renders were never
the problem: the question is whether it can pick out the camera clips where the
model loses.
"""
import re, sys
import numpy as np

txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
MODEL = sys.argv[2]
RENDERS = {"bistro_30s.mp4", "chess_30s.mp4", "locomotive_30s.mp4"}
blocks = re.split(r"^(\S+\.mp4)\s*$", txt, flags=re.M)

def grab(body, header, model, col=-1):
    m = re.search(re.escape(header) + r".*?\n(.*?)(?:\n\s*\n|\Z)", body, re.S)
    if not m: return None
    for line in m.group(1).splitlines():
        p = line.split()
        if p and p[0] == model:
            try: return float(p[col])
            except (ValueError, IndexError): return None
    return None

rows = []
for i in range(1, len(blocks), 2):
    clip, body = blocks[i], blocks[i + 1]
    bd = grab(body, "DISTS (lower better)", "bicubic")
    md = grab(body, "DISTS (lower better)", MODEL)
    # sharpness table columns are flat / edge / texture
    bs = grab(body, "Sharpness ratio vs ground truth", "bicubic", col=2)
    if None in (bd, md, bs): continue
    rows.append({"clip": clip, "render": clip in RENDERS,
                 "gain": (1 - md / bd) * 100, "bic_edge_sharp": bs})

cam = [r for r in rows if not r["render"]]
print(f"model: {MODEL}   (camera clips only, sorted by measured gain)\n")
print(f"{'clip':<28} {'DISTS gain':>10} {'bicubic edge sharpness':>23}")
print("-" * 64)
for r in sorted(cam, key=lambda x: x["gain"]):
    flag = "  <-- model LOSES" if r["gain"] <= 0 else ""
    print(f"{r['clip']:<28} {r['gain']:+10.1f}% {r['bic_edge_sharp']:23.3f}{flag}")

losers = [r for r in cam if r["gain"] <= 0]
winners = [r for r in cam if r["gain"] > 0]
if losers and winners:
    lo = np.mean([r["bic_edge_sharp"] for r in losers])
    wi = np.mean([r["bic_edge_sharp"] for r in winners])
    print(f"\nlosers  n={len(losers):<2} mean bicubic edge sharpness {lo:.3f}")
    print(f"winners n={len(winners):<2} mean bicubic edge sharpness {wi:.3f}")
    b = np.array([r["bic_edge_sharp"] for r in cam])
    g = np.array([r["gain"] for r in cam])
    print(f"corr(bicubic edge sharpness, gain) over camera clips = {np.corrcoef(b, g)[0,1]:+.3f}")
    sep = min(r["bic_edge_sharp"] for r in winners) > max(r["bic_edge_sharp"] for r in losers) \
       or max(r["bic_edge_sharp"] for r in winners) < min(r["bic_edge_sharp"] for r in losers)
    print(f"cleanly separable by a single threshold: {sep}")
else:
    print(f"\nno losers among camera clips for {MODEL} -- nothing for a gate to catch")
