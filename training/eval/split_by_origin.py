import re, sys
import numpy as np
txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
MODEL = sys.argv[2]
blocks = re.split(r"^(\S+\.mp4)\s*$", txt, flags=re.M)
RENDERS = {"bistro_30s.mp4", "chess_30s.mp4", "locomotive_30s.mp4"}
def grab(body, header, model):
    m = re.search(re.escape(header) + r".*?\n(.*?)(?:\n\s*\n|\Z)", body, re.S)
    if not m: return None
    for line in m.group(1).splitlines():
        p = line.split()
        if p and p[0] == model: return float(p[-1])
    return None
rows = []
for i in range(1, len(blocks), 2):
    clip, body = blocks[i], blocks[i+1]
    bd, md = grab(body, "DISTS (lower better)", "bicubic"), grab(body, "DISTS (lower better)", MODEL)
    bt, mt = grab(body, "tLP (output flicker", "bicubic"), grab(body, "tLP (output flicker", MODEL)
    if None in (bd, md): continue
    rows.append({"clip": clip, "render": clip in RENDERS,
                 "d": (1 - md/bd)*100, "t": (mt - bt) if None not in (bt, mt) else float("nan")})
print(f"model: {MODEL}")
for lbl, sel in (("3 renders", [r for r in rows if r["render"]]),
                 ("12 real-camera", [r for r in rows if not r["render"]]),
                 ("all 15", rows)):
    if not sel: continue
    wins = sum(1 for r in sel if r["d"] > 0)
    print(f"  {lbl:<16} n={len(sel):<3} DISTS {np.mean([r['d'] for r in sel]):+6.1f}%   "
          f"tLP {np.mean([r['t'] for r in sel]):+.4f}   wins {wins}/{len(sel)}")
