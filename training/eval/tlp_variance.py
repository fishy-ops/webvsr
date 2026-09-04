"""How repeatable is tLP between two evaluations of the same model on the same clips?

Section 19 measured shipped vs webcodec tLP as +0.0022 against +0.0012 at CRF 20.
A fresh evaluation of the identical checkpoints on the identical clips came back
-0.00156 against -0.00148 -- same information, opposite sign convention, but a 5%
gap where the first run showed 45%. If tLP swings that much between runs, it
cannot rank models, and section 24's cross-set disagreement needs no exotic
explanation.
"""
import re, sys
import numpy as np

RENDERS = {"bistro_30s.mp4", "chess_30s.mp4", "locomotive_30s.mp4"}

def tlp_by_clip(path, model):
    txt = open(path, encoding="utf-8", errors="replace").read()
    blocks = re.split(r"^(\S+\.mp4)\s*$", txt, flags=re.M)
    out = {}
    for i in range(1, len(blocks), 2):
        clip, body = blocks[i], blocks[i + 1]
        m = re.search(r"tLP \(output flicker.*?\n(.*?)(?:\n\s*\n|\Z)", body, re.S)
        if not m:
            continue
        for line in m.group(1).splitlines():
            p = line.split()
            if p and p[0] == model:
                out[clip] = float(p[-1])
    return out

A, B = "/tank/webvsr/eval15wc_2x_crf20.txt", "/tank/webvsr/rank_crf20.txt"
pairs = [("bicubic", "bicubic"), ("candidate", "shipped"), ("webcodec", "webcodec")]

print(f"{'clip':<28} " + "".join(f"{n:>22}" for _, n in pairs))
print(f"{'':<28} " + "".join(f"{'run A':>10}{'run B':>7}{'Δ':>5}" for _ in pairs))
print("-" * 96)
deltas = {n: [] for _, n in pairs}
clips = None
for a_name, b_name in pairs:
    ta, tb = tlp_by_clip(A, a_name), tlp_by_clip(B, b_name)
    common = [c for c in ta if c in tb and c not in RENDERS]
    clips = clips or common
    for c in common:
        deltas[b_name].append((c, ta[c], tb[c]))

for c in clips:
    row = f"{c:<28} "
    for _, n in pairs:
        e = next((x for x in deltas[n] if x[0] == c), None)
        row += f"{e[1]:10.5f}{e[2]:7.5f}{e[2]-e[1]:+5.0f}" if e is None else \
               f"{e[1]:10.5f}{e[2]:7.5f}{(e[2]-e[1])*1000:+5.1f}"
    print(row)

print("\nrun-to-run change in tLP (x1000), camera clips:")
for _, n in pairs:
    d = np.array([b - a for _, a, b in deltas[n]]) * 1000
    print(f"  {n:<10} mean {d.mean():+6.2f}  sd {d.std():5.2f}  max|Δ| {np.abs(d).max():5.2f}")
sd = np.std([b - a for _, a, b in deltas["shipped"]])
print(f"\nshipped-vs-webcodec tLP gap being used to rank them: ~0.00008")
print(f"per-clip run-to-run standard deviation of tLP        : {sd:.5f}")
print(f"=> the gap is {sd/0.00008:.0f}x SMALLER than the noise" if sd > 0.00008
      else "=> the gap exceeds the noise")
