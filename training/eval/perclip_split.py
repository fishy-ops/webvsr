import re, sys, numpy as np
txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
blocks = re.split(r"^(\S+\.mp4)\s*$", txt, flags=re.M)
RENDERS = {"bistro_30s.mp4", "chess_30s.mp4", "locomotive_30s.mp4"}
rows = []
for i in range(1, len(blocks), 2):
    clip, body = blocks[i], blocks[i + 1]
    def grab(header, model):
        m = re.search(re.escape(header) + r".*?\n(.*?)(?:\n\s*\n|\Z)", body, re.S)
        if not m: return None
        for line in m.group(1).splitlines():
            p = line.split()
            if p and p[0] == model:
                return float(p[-1])
        return None
    def psnr_tex(model):
        m = re.search(r"PSNR by content complexity.*?\n(.*?)(?:\n\s*\n|\Z)", body, re.S)
        if not m: return None
        for line in m.group(1).splitlines():
            p = line.split()
            if p and p[0] == model and len(p) >= 5:
                return float(p[3])
        return None
    r = {"clip": clip, "render": clip in RENDERS}
    for mdl in ("bicubic", "shipped"):
        r[mdl + "_dists"] = grab("DISTS (lower better)", mdl)
        r[mdl + "_tlp"] = grab("tLP (output flicker", mdl)
        r[mdl + "_tex"] = psnr_tex(mdl)
    if None not in (r["bicubic_dists"], r["shipped_dists"]):
        rows.append(r)

print(f"{'clip':<26} {'DISTS bic':>9} {'DISTS shp':>9} {'DISTS %':>8} {'tex dB':>8} {'tLP d':>8}")
print("-" * 74)
for r in sorted(rows, key=lambda x: -(1 - x["shipped_dists"] / x["bicubic_dists"])):
    d = (1 - r["shipped_dists"] / r["bicubic_dists"]) * 100
    tex = (r["shipped_tex"] - r["bicubic_tex"]) if None not in (r["shipped_tex"], r["bicubic_tex"]) else float("nan")
    tl = r["shipped_tlp"] - r["bicubic_tlp"]
    tag = " [render]" if r["render"] else ""
    print(f"{r['clip']:<26} {r['bicubic_dists']:9.4f} {r['shipped_dists']:9.4f} {d:+7.1f}% {tex:+8.3f} {tl:+8.4f}{tag}")

for label, sel in (("ALL 15", rows),
                   ("3 render clips", [r for r in rows if r["render"]]),
                   ("12 real-camera clips", [r for r in rows if not r["render"]])):
    if not sel: continue
    d = np.mean([(1 - r["shipped_dists"] / r["bicubic_dists"]) * 100 for r in sel])
    tex = np.mean([r["shipped_tex"] - r["bicubic_tex"] for r in sel])
    tl = np.mean([r["shipped_tlp"] - r["bicubic_tlp"] for r in sel])
    wins = sum(1 for r in sel if r["shipped_dists"] < r["bicubic_dists"])
    print(f"\n{label:<22} n={len(sel):<3} DISTS {d:+.1f}%  texPSNR {tex:+.3f} dB  tLP {tl:+.4f}  DISTS wins {wins}/{len(sel)}")
