"""Quick training status checker. Run anytime to see progress."""
import json
import os
from datetime import datetime

LOG_PATH = r'training_log.json'
CKPT_DIR = r'checkpoints'

if not os.path.exists(LOG_PATH):
    print("No training log found.")
    exit()

with open(LOG_PATH) as f:
    log = json.load(f)

if not log:
    print("Training log is empty.")
    exit()

last = log[-1]
total_epochs = 500

print(f"=== WebVSR Training Status ===")
print(f"Current epoch: {last['epoch']} / {total_epochs}")
print(f"Phase: {last['phase']}")
print(f"Loss: {last['loss']:.4f}")
print(f"LR: {last['lr']:.6f}")
print(f"Last epoch time: {last['time_s']:.0f}s")

# Progress
pct = last['epoch'] / total_epochs * 100
remaining = total_epochs - last['epoch']
avg_time = sum(e['time_s'] for e in log[-20:]) / min(20, len(log))
eta_hours = (remaining * avg_time) / 3600
print(f"\nProgress: {pct:.0f}% ({remaining} epochs remaining)")
print(f"ETA: ~{eta_hours:.1f} hours ({avg_time:.0f}s/epoch avg)")

# Best PSNR per phase
for phase in [1, 2]:
    entries = [e for e in log if e['phase'] == phase and e['psnr'] > 0]
    if entries:
        best = max(entries, key=lambda x: x['psnr'])
        print(f"\nPhase {phase} best PSNR: {best['psnr']:.2f} dB (epoch {best['epoch']})")

# Recent PSNR trend
eval_entries = [e for e in log if e['psnr'] > 0]
if len(eval_entries) >= 2:
    recent = eval_entries[-5:]
    print(f"\nRecent validation PSNR:")
    for e in recent:
        print(f"  Epoch {e['epoch']:3d}: {e['psnr']:.2f} dB")

# Loss trend
print(f"\nLoss trend (last 10 epochs):")
for e in log[-10:]:
    components = e.get('components', {})
    parts = []
    for k, v in components.items():
        parts.append(f"{k}={v:.4f}")
    comp_str = f" ({', '.join(parts)})" if parts else ""
    print(f"  Epoch {e['epoch']:3d}: {e['loss']:.4f}{comp_str}")

# Checkpoints
print(f"\nCheckpoints:")
if os.path.isdir(CKPT_DIR):
    for f in sorted(os.listdir(CKPT_DIR)):
        path = os.path.join(CKPT_DIR, f)
        size = os.path.getsize(path) / 1024 / 1024
        mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%H:%M:%S')
        print(f"  {f}: {size:.1f} MB (modified {mtime})")

# Check if training process is running
import subprocess
result = subprocess.run(
    ['powershell', '-c', "Get-Process python* | Select-Object Id,CPU,WorkingSet64 | Format-Table"],
    capture_output=True, text=True
)
if result.stdout.strip():
    print(f"\nPython processes:")
    print(result.stdout.strip())
