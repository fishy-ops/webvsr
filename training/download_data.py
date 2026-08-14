"""
Data download helper for WebVSR training.

Storage layout:
  C:\...\mamba-sr\DIV2K_train_HR\  (already exists, ~3.4 GB)
  C:\...\mamba-sr\DIV2K_valid_HR\  (already exists, ~430 MB)
  C:\...\mamba-sr\Flickr2K\        (already exists, ~11 GB)
  D:\webvsr\data\vimeo_frames\     (to download, ~2 GB sampled)
  D:\webvsr\data\LSDIR\            (to download, ~50-100 GB)

Run this script to:
  1. Download Vimeo-90K septuplet dataset
  2. Sample center frames from Vimeo clips
  3. Print instructions for LSDIR (manual download required)
"""

import os
import sys
import subprocess
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

VIMEO_URL = "http://data.csail.mit.edu/toflow/vimeo_septuplet.zip"
DATA_DIR = Path(r"D:\webvsr\data")
VIMEO_DIR = DATA_DIR / "vimeo_septuplet"
VIMEO_FRAMES_DIR = DATA_DIR / "vimeo_frames"


def download_vimeo():
    """Download Vimeo-90K septuplet dataset."""
    zip_path = DATA_DIR / "vimeo_septuplet.zip"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if VIMEO_DIR.exists() and any(VIMEO_DIR.rglob("*.png")):
        print(f"Vimeo-90K already exists at {VIMEO_DIR}")
        return

    if not zip_path.exists():
        print(f"Downloading Vimeo-90K septuplet (~20 GB)...")
        print(f"URL: {VIMEO_URL}")
        print(f"Destination: {zip_path}")
        print()
        print("This is a large download. You can also download manually:")
        print(f"  1. Download from {VIMEO_URL}")
        print(f"  2. Place the zip at {zip_path}")
        print(f"  3. Re-run this script")
        print()

        try:
            subprocess.run(
                ["curl", "-L", "-o", str(zip_path), "--progress-bar", VIMEO_URL],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("curl failed. Trying with Python urllib...")
            import urllib.request
            urllib.request.urlretrieve(VIMEO_URL, str(zip_path))

    print(f"Extracting {zip_path}...")
    import zipfile
    with zipfile.ZipFile(str(zip_path), "r") as z:
        z.extractall(str(DATA_DIR))
    print("Extraction complete.")


def sample_center_frames(num_samples=10000):
    """Sample center frames (frame 4 of 7) from Vimeo-90K clips.
    These add natural video characteristics (motion blur, compression) to training."""

    VIMEO_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    sequences_dir = VIMEO_DIR / "sequences"
    if not sequences_dir.exists():
        print(f"Error: {sequences_dir} not found. Download Vimeo-90K first.")
        return

    # Find all clip directories
    clips = []
    for group in sorted(sequences_dir.iterdir()):
        if group.is_dir():
            for clip in sorted(group.iterdir()):
                if clip.is_dir():
                    center = clip / "im4.png"
                    if center.exists():
                        clips.append(center)

    print(f"Found {len(clips)} Vimeo clips")

    if len(clips) > num_samples:
        random.seed(42)
        clips = random.sample(clips, num_samples)

    print(f"Sampling {len(clips)} center frames to {VIMEO_FRAMES_DIR}")

    def copy_frame(args):
        idx, src = args
        dst = VIMEO_FRAMES_DIR / f"vimeo_{idx:05d}.png"
        if dst.exists():
            return
        try:
            img = Image.open(src).convert("RGB")
            img.save(dst)
        except Exception as e:
            print(f"  Skip {src}: {e}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(copy_frame, enumerate(clips)))

    actual = len(list(VIMEO_FRAMES_DIR.glob("*.png")))
    print(f"Done: {actual} frames saved to {VIMEO_FRAMES_DIR}")


def print_lsdir_instructions():
    print()
    print("=" * 60)
    print("LSDIR Dataset (manual download required)")
    print("=" * 60)
    print()
    print("LSDIR (Large Scale Diverse Image Restoration) has ~85K images")
    print("and is the key dataset for training a universal model.")
    print()
    print("Download from: https://data.vision.ee.ethz.ch/yawli/index.html")
    print("  or: https://github.com/ofsoundof/LSDIR")
    print()
    print(f"Place the training images at: D:\\webvsr\\data\\LSDIR\\train\\")
    print()
    print("After downloading, uncomment the LSDIR path in train_span.py CONFIG:")
    print('  r"D:\\webvsr\\data\\LSDIR\\train",')
    print()
    print("If LSDIR is too large, you can start training with just")
    print("DIV2K + Flickr2K (~3,450 images) and add LSDIR later.")
    print()


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "all"

    if action in ("all", "vimeo"):
        download_vimeo()

    if action in ("all", "sample"):
        sample_center_frames()

    if action in ("all", "lsdir", "info"):
        print_lsdir_instructions()

    if action == "all":
        print()
        print("=" * 60)
        print("SUMMARY: Data status")
        print("=" * 60)
        datasets = {
            "DIV2K train": Path(r"C:\Users\reach\OneDrive\Documents\mamba-sr\DIV2K_train_HR\DIV2K_train_HR"),
            "DIV2K valid": Path(r"C:\Users\reach\OneDrive\Documents\mamba-sr\DIV2K_valid_HR\DIV2K_valid_HR"),
            "Flickr2K": Path(r"C:\Users\reach\OneDrive\Documents\mamba-sr\Flickr2K"),
            "Vimeo frames": VIMEO_FRAMES_DIR,
            "LSDIR train": Path(r"D:\webvsr\data\LSDIR\train"),
        }
        for name, p in datasets.items():
            if p.exists():
                count = sum(1 for f in p.rglob("*") if f.suffix.lower() in {".png", ".jpg", ".jpeg"})
                print(f"  {name:15s}: {count:,} images")
            else:
                print(f"  {name:15s}: NOT FOUND")
