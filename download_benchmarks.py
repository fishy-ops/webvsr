"""Download standard SR benchmark datasets (Set5, Set14)."""
import os
import zipfile
import urllib.request

BENCHMARKS = {
    'Set5': [
        'https://github.com/xinntao/BasicSR/releases/download/data/Set5.zip',
        'https://github.com/ChaofWang/Awesome-Super-Resolution/raw/master/dataset/benchmark/Set5.zip',
    ],
    'Set14': [
        'https://github.com/xinntao/BasicSR/releases/download/data/Set14.zip',
        'https://github.com/ChaofWang/Awesome-Super-Resolution/raw/master/dataset/benchmark/Set14.zip',
    ],
}

OUT_DIR = r'D:\webvsr\data\benchmarks'
os.makedirs(OUT_DIR, exist_ok=True)

for name, urls in BENCHMARKS.items():
    dest = os.path.join(OUT_DIR, name)
    if os.path.isdir(dest) and len(os.listdir(dest)) > 0:
        print(f"{name} already exists, skipping")
        continue

    zip_path = os.path.join(OUT_DIR, f"{name}.zip")
    downloaded = False
    for url in urls:
        print(f"Downloading {name} from {url}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            downloaded = True
            break
        except Exception as e:
            print(f"  Failed: {e}")
    if not downloaded:
        print(f"  All URLs failed for {name}")
        continue

    print(f"Extracting {name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(OUT_DIR)
    os.remove(zip_path)

    if os.path.isdir(dest):
        count = len([f for f in os.listdir(dest) if not f.startswith('.')])
        print(f"  {name}: {count} images")
    else:
        print(f"  Warning: Expected directory {dest} not found after extraction")
        extracted = os.listdir(OUT_DIR)
        print(f"  Contents of {OUT_DIR}: {extracted}")

print("Done.")
