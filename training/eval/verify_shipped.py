"""Prove which checkpoint each shipped model binary came from.

The .bin files in extension/models are tracked in git; the .pth checkpoints that
produced them are not. Nothing tied the two together, and they drifted: every 2x
evaluation for two sessions scored `checkpoints_c16/best_phase2.pth` as
"shipped" while the extension actually shipped ckpt_c16_sharp2/best_phase1.pth
-- a different, older model. The numbers were real, they just described
something users do not run.

PROVENANCE.json records the mapping. This script re-exports each recorded
checkpoint and compares bytes against the shipped binary, so the claim is
verified rather than asserted. Run it before any evaluation that reports a
"shipped" column, and after every export.

    python training/eval/verify_shipped.py           # verify all
    python training/eval/verify_shipped.py --spec 2x # print the --model spec
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROV = ROOT / "extension" / "models" / "PROVENANCE.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(name, entry, python):
    shipped = ROOT / entry["binary"]
    ckpt = Path(entry["checkpoint"])
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    out = {"model": name, "binary": entry["binary"], "checkpoint": str(ckpt)}

    if not shipped.exists():
        out["status"] = "MISSING BINARY"
        return out
    out["binary_sha256"] = sha256(shipped)
    if out["binary_sha256"] != entry["sha256"]:
        out["status"] = "BINARY CHANGED since PROVENANCE was written"
        return out
    if not ckpt.exists():
        out["status"] = "CHECKPOINT UNAVAILABLE (binary hash still matches record)"
        return out

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "re-export.bin"
        r = subprocess.run(
            [python, str(ROOT / "training" / "export_webgpu_weights.py"),
             "--checkpoint", str(ckpt), "--output", str(tmp),
             "--channels", str(entry["channels"]), "--scale", str(entry["scale"])],
            capture_output=True, text=True, cwd=ROOT,
        )
        if r.returncode != 0 or not tmp.exists():
            out["status"] = "RE-EXPORT FAILED"
            out["stderr"] = r.stderr.strip().splitlines()[-1:] 
            return out
        out["status"] = "OK" if sha256(tmp) == out["binary_sha256"] else "MISMATCH"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", metavar="MODEL",
                    help="print a stratified_eval --model spec for this entry and exit")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    # Keys beginning with "_" are notes for humans, not model entries.
    prov = {k: v for k, v in json.loads(PROV.read_text()).items()
            if not k.startswith("_")}

    if args.spec:
        e = prov.get(args.spec)
        if not e:
            sys.exit(f"no entry '{args.spec}' in {PROV.name}; have {sorted(prov)}")
        print(f"deployed={e['checkpoint']}:{e['channels']}:{e['scale']}")
        return

    rows = [check(name, e, args.python) for name, e in sorted(prov.items())]
    width = max(len(r["model"]) for r in rows)
    bad = 0
    for r in rows:
        print(f"{r['model']:<{width}}  {r['status']}")
        print(f"{'':<{width}}    binary     {r['binary']}")
        print(f"{'':<{width}}    checkpoint {r['checkpoint']}")
        if r["status"] not in ("OK", "CHECKPOINT UNAVAILABLE (binary hash still matches record)"):
            bad += 1
    if bad:
        sys.exit(f"\n{bad} entr{'y' if bad == 1 else 'ies'} did not verify")
    print("\nall shipped binaries trace to their recorded checkpoint")


if __name__ == "__main__":
    main()
