"""Where does the model's advantage sit, per clip?

Sections 9 and 11 established that three render clips gain far more than the
twelve real-camera clips, and rejected busyness, harness enlargement and prior
codec compression as the cause. This asks a different question: not how much the
model gains, but in WHICH content it gains it. Flat, edge and texture are already
separated per clip by the stratified eval; if the renders gain somewhere the real
clips do not, that names the mechanism.
"""
import re, sys
import numpy as np

txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
blocks = re.split(r"^(\S+\.mp4)\s*$", txt, flags=re.M)
RENDERS = {"bistro_30s.mp4", "chess_30s.mp4", "locomotive_30s.mp4"}

def rows_of(body, header, model):
    m = re.search(re.escape(header) + r".*?\n(.*?)(?:\n\s*\n|\Z)", body, re.S)
    if not m:
        return None
    for line in m.group(1).splitlines():
        p = line.split()
        if p and p[0] == model:
            return [float(x) for x in p[1:]]
    return None

out = []
for i in range(1, len(blocks), 2):
    clip, body = blocks[i], blocks[i + 1]
    b = rows_of(body, "PSNR by content complexity", "bicubic")
    s = rows_of(body, "PSNR by content complexity", "shipped")
    bs = rows_of(body, "Sharpness ratio vs ground truth", "bicubic")
    ss = rows_of(body, "Sharpness ratio vs ground truth", "shipped")
    if not b or not s or len(b) < 4:
        continue
    out.append({
        "clip": clip, "render": clip in RENDERS,
        "d_flat": s[0] - b[0], "d_edge": s[1] - b[1], "d_tex": s[2] - b[2],
        "bic_flat": b[0], "bic_edge": b[1], "bic_tex": b[2],
        "sharp_edge_bic": bs[1] if bs else float("nan"),
        "sharp_edge_mdl": ss[1] if ss else float("nan"),
    })

print(f"{'clip':<26} {'dFLAT':>7} {'dEDGE':>7} {'dTEX':>7} | {'bic edge PSNR':>13} {'edge sharp b>m':>15}")
print("-" * 84)
for r in sorted(out, key=lambda x: -x["d_edge"]):
    tag = " [R]" if r["render"] else ""
    print(f"{r['clip']:<26} {r['d_flat']:+7.2f} {r['d_edge']:+7.2f} {r['d_tex']:+7.2f} | "
          f"{r['bic_edge']:13.2f} {r['sharp_edge_bic']:6.3f}>{r['sharp_edge_mdl']:.3f}{tag}")

for label, sel in (("3 renders", [r for r in out if r["render"]]),
                   ("12 real", [r for r in out if not r["render"]])):
    if not sel:
        continue
    print(f"\n{label:<12} dFLAT {np.mean([r['d_flat'] for r in sel]):+6.2f}   "
          f"dEDGE {np.mean([r['d_edge'] for r in sel]):+6.2f}   "
          f"dTEX {np.mean([r['d_tex'] for r in sel]):+6.2f}   "
          f"| bicubic edge PSNR {np.mean([r['bic_edge'] for r in sel]):6.2f}")
