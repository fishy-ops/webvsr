"""Paired re-analysis of the CRF 28 ranking already on disk (§28's rule)."""
import re, sys
import numpy as np
sys.path.insert(0, "training/eval")
from rank_models import per_clip, RENDERS

txt = open("/tank/webvsr/rank_crf28.txt", encoding="utf-8", errors="replace").read()
base = per_clip(txt, "bicubic")
ref = "shipped"
rc = per_clip(txt, ref)

print(f"CRF 28, paired against the currently shipped model, 12 camera clips")
print(f"(negative ΔDISTS = better perceptually; negative Δ|tLP| = better temporally)\n")
print(f"{'model':<14} {'ΔDISTS':>11} {'t':>7} {'sig':>5} {'Δ|tLP|':>11} {'t':>7} {'sig':>5} {'wins':>7}")
print("-" * 74)
for name in ("webcodec", "ema_dists", "masked_twin"):
    got = per_clip(txt, name)
    cam = [c for c in got if c not in RENDERS and c in base and c in rc]
    dd = np.array([got[c]["dists"] - rc[c]["dists"] for c in cam])
    dt = np.array([abs(got[c].get("tlp", 0)) - abs(rc[c].get("tlp", 0)) for c in cam])
    def t(v):
        sd = np.std(v, ddof=1)
        return v.mean() / (sd / np.sqrt(len(v))) if sd > 0 else float("nan")
    td, tt = t(dd), t(dt)
    print(f"{name:<14} {dd.mean():+11.5f} {td:+7.2f} {'YES' if abs(td)>2.2 else 'no':>5} "
          f"{dt.mean():+11.5f} {tt:+7.2f} {'YES' if abs(tt)>2.2 else 'no':>5} "
          f"{int((dd<0).sum()):>4}/{len(cam):<2}")
print("\n|t| > 2.2 significant at 11 df")
