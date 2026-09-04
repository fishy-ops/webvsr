"""Build a VIDEO validation set: consecutive frame pairs from held-out clips.

Two problems with the current validation set, both measured:

  §22 -- it is Vimeo still frames, and it reported a 4.8% DISTS gain on a run the
         clip benchmark scored as identical. Checkpoint selection runs on it.
  §11 -- flicker is the advantage that transfers across content types, and it
         cannot be measured on stills at all, so nothing has ever been able to
         select on it.

Consecutive pairs fix both: the content matches the benchmark's domain (real
clips through a real encode) and tLP becomes computable during training.

Held out on purpose -- none of these clips are in clips_busy. Selecting
checkpoints on the benchmark would be selecting on the test set.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "training/eval")
sys.path.insert(0, "training")
from stratified_eval import make_pair, VIDEO_EXT

SRC = Path("/tank/webvsr/clips_val")
OUT = Path("/tank/webvsr/val_video")
PAIRS_PER_CLIP = 6
CRF = 28
HEIGHT = 512          # smaller than the benchmark's 1024: validation runs every
                      # 5 epochs and must not dominate training time

def main():
    if OUT.exists():
        import shutil; shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    manifest = []
    clips = sorted(p for p in SRC.iterdir() if p.suffix.lower() in VIDEO_EXT)
    with tempfile.TemporaryDirectory() as td:
        for clip in clips:
            hr, lr = make_pair(clip, Path(td) / clip.stem, 2, CRF,
                               PAIRS_PER_CLIP + 1, HEIGHT)
            n = min(len(hr), len(lr))
            d = OUT / clip.stem
            (d / "hr").mkdir(parents=True); (d / "lr").mkdir(parents=True)
            for i in range(n):
                for sub, src in (("hr", hr[i]), ("lr", lr[i])):
                    (d / sub / f"{i:03d}.png").write_bytes(src.read_bytes())
            # consecutive indices -- the pair is what makes tLP computable
            for i in range(n - 1):
                manifest.append({"clip": clip.stem, "prev": i, "curr": i + 1})
            print(f"  {clip.stem}: {n} frames -> {max(n-1,0)} pairs", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n{len(manifest)} consecutive pairs from {len(clips)} held-out clips -> {OUT}")

if __name__ == "__main__":
    main()
