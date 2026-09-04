"""Two conventions for summarising tLP, and only one is right.

split2.py -- used for §19, §22 and §24 -- reports mean(model_tlp - bicubic_tlp).
rank_models.py reports mean(|model_tlp| - |bicubic_tlp|).

§10 established that 0 is the target: positive tLP is added flicker, negative is
temporal over-smoothing, and 'lower is better' is optimised by a constant grey
frame. So the deviation |tLP| is what matters, and the raw difference credits a
model unboundedly for flickering LESS than the truth.
"""
import re
import numpy as np

RENDERS = {"bistro_30s.mp4", "chess_30s.mp4", "locomotive_30s.mp4"}

def by_clip(path, model):
    txt = open(path, encoding="utf-8", errors="replace").read()
    blocks = re.split(r"^(\S+\.mp4)\s*$", txt, flags=re.M)
    out = {}
    for i in range(1, len(blocks), 2):
        clip, body = blocks[i], blocks[i+1]
        m = re.search(r"tLP \(output flicker.*?\n(.*?)(?:\n\s*\n|\Z)", body, re.S)
        if not m: continue
        for line in m.group(1).splitlines():
            p = line.split()
            if p and p[0] == model: out[clip] = float(p[-1])
    return out

F = "/tank/webvsr/eval15wc_2x_crf20.txt"
bic = by_clip(F, "bicubic")
for name, label in (("candidate", "shipped"), ("webcodec", "webcodec")):
    mv = by_clip(F, name)
    cam = [c for c in mv if c in bic and c not in RENDERS]
    raw = np.mean([mv[c] - bic[c] for c in cam])
    dev = np.mean([abs(mv[c]) - abs(bic[c]) for c in cam])
    print(f"{label:<10} raw diff {raw:+.5f}   |deviation| change {dev:+.5f}")

print("\nWhere the two disagree — clips whose tLP is POSITIVE (output flickers")
print("MORE than truth), where sign handling matters:")
mv = by_clip(F, "candidate")
for c in sorted(mv, key=lambda x: -mv[x])[:3]:
    if c in RENDERS: continue
    print(f"  {c:<28} bicubic {bic[c]:+.5f}  model {mv[c]:+.5f}")
    print(f"      raw says {mv[c]-bic[c]:+.5f} (better)   "
          f"deviation says {abs(mv[c])-abs(bic[c]):+.5f}")
print("\nvsr_test_video is the case: bicubic over-smooths at -0.0071 and the model")
print("adds flicker at +0.0044. Raw credits the model +0.0115 as if it were a large")
print("gain; by deviation it went 0.0071 -> 0.0044, a real but moderate improvement.")
