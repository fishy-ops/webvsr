"""Auto-restart wrapper for training. Re-launches train_span.py --resume on crash."""
import subprocess
import sys
import time

PYTHON = r"C:\Users\reach\AppData\Local\Programs\Python\Python311\python.exe"
SCRIPT = r"D:\webvsr\train_span.py"
MAX_RESTARTS = 50
RESTART_DELAY = 10

for attempt in range(MAX_RESTARTS):
    print(f"\n{'='*60}")
    print(f"Training attempt {attempt + 1}/{MAX_RESTARTS}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [PYTHON, SCRIPT, "--resume"],
        cwd=r"D:\webvsr",
    )

    if result.returncode == 0:
        print("\nTraining completed successfully!")
        break

    print(f"\nProcess exited with code {result.returncode}")
    print(f"Restarting in {RESTART_DELAY}s...")
    time.sleep(RESTART_DELAY)
else:
    print(f"Exceeded {MAX_RESTARTS} restarts")
