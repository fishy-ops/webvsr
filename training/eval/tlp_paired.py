"""Is the shipped-vs-webcodec flicker difference statistically real?

Per-clip tLP spans -0.045 to +0.013 across the benchmark while the reported
model difference is ~0.0015. If between-clip variance swamps between-model
variance, a difference of clip-set MEANS cannot rank models -- but a PAIRED
comparison (same clip, two models) cancels the clip effect and can.

§19, §24 and §27 all compared means. This checks whether that was ever valid.
"""
import numpy as np

# per-clip tLP at CRF 28, from tlp_motion.py
DATA = [
    ("controlled_burn", -0.01328, -0.01345), ("life",          0.00742,  0.00774),
    ("chess",            0.00156,  0.00067), ("vsr_test",      0.00357, -0.00047),
    ("ducks_take_off",  -0.01499, -0.01849), ("old_town_cross",-0.02886, -0.03096),
    ("in_to_tree",      -0.03429, -0.03588), ("crowd_run",     -0.00829, -0.00867),
    ("dinner",          -0.01097, -0.01337), ("blue_sky",       0.01323,  0.01336),
    ("locomotive",      -0.00550, -0.00511), ("aspen",         -0.03386, -0.03757),
    ("factory",         -0.00285, -0.00524), ("bistro",        -0.00447, -0.00605),
    ("park_joy",        -0.04502, -0.04613),
]
RENDERS = {"chess", "locomotive", "bistro"}
cam = [(c, a, b) for c, a, b in DATA if c not in RENDERS]

sh = np.array([abs(a) for _, a, _ in cam])
wc = np.array([abs(b) for _, _, b in cam])
n = len(cam)

print(f"{n} camera clips, |tLP| deviation (lower is better)\n")
print(f"between-clip spread   : sd {np.std(np.concatenate([sh, wc])):.5f}")
print(f"difference of means   : {sh.mean() - wc.mean():+.5f}")
print(f"SEM of a clip-set mean: {np.std(sh, ddof=1)/np.sqrt(n):.5f}")
print(f"  -> the model difference is {abs(sh.mean()-wc.mean())/(np.std(sh, ddof=1)/np.sqrt(n)):.2f} SEM "
      f"— comparing MEANS cannot resolve it\n")

d = sh - wc              # paired, per clip: cancels the clip effect
sem = np.std(d, ddof=1) / np.sqrt(n)
t = d.mean() / sem
print(f"PAIRED difference (shipped - webcodec), per clip:")
print(f"  mean {d.mean():+.5f}   sd {np.std(d, ddof=1):.5f}   SEM {sem:.5f}")
print(f"  t = {t:+.2f} on {n-1} df")
wins = int((d < 0).sum())
print(f"  shipped has smaller |tLP| on {wins}/{n} clips")
print()
if abs(t) > 2.2:
    print(f"=> the paired difference IS significant: webcodec is genuinely worse")
    print(f"   on flicker, by {abs(d.mean()):.5f} on average, even though the")
    print(f"   difference of means could never have shown it.")
else:
    print(f"=> not significant at n={n}. The flicker gap blocking the swap is")
    print(f"   within noise, and §27's trade-off may not be a real trade at all.")
