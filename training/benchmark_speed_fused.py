"""Benchmark with reparameterized (inference-mode) model."""
import time
import torch
from model_span import SPANLite

RESOLUTIONS = [
    ('360p', 360, 640),
    ('480p', 480, 854),
    ('720p', 720, 1280),
    ('1080p', 1080, 1920),
]

device = torch.device('cuda')
print(f"GPU: {torch.cuda.get_device_name()}")

model = SPANLite(num_in_ch=3, num_out_ch=3, feature_channels=32, upscale=2)
ckpt = torch.load('checkpoints/best_phase2.pth', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.to(device).eval()

# Reparameterize: fuse Conv3XC to single 3x3
with torch.no_grad():
    for module in model.modules():
        if hasattr(module, '_update_params') and hasattr(module, 'eval_conv'):
            module._update_params()

# Count inference params
inf_params = sum(p.numel() for p in model.parameters())
print(f"Inference params: {inf_params:,}\n")

print(f"{'Resolution':<12} {'Avg ms':>8} {'Min ms':>8} {'FPS':>8}  {'60fps?'}")
print('-' * 55)

for name, h, w in RESOLUTIONS:
    dummy = torch.randn(1, 3, h, w, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(100):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    avg = sum(times) / len(times)
    mn = min(times)
    fps = 1000 / avg
    ok = 'YES' if avg < 16.6 else ('CLOSE' if avg < 25 else 'NO')
    print(f"{name:<12} {avg:>8.1f} {mn:>8.1f} {fps:>8.1f}  {ok}")

# Also test with FP16
print(f"\n--- FP16 (half precision) ---")
model_fp16 = model.half()
for name, h, w in RESOLUTIONS:
    dummy = torch.randn(1, 3, h, w, device=device, dtype=torch.float16)
    with torch.no_grad():
        for _ in range(20):
            _ = model_fp16(dummy)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(100):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model_fp16(dummy)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    avg = sum(times) / len(times)
    mn = min(times)
    fps = 1000 / avg
    ok = 'YES' if avg < 16.6 else ('CLOSE' if avg < 25 else 'NO')
    print(f"{name:<12} {avg:>8.1f} {mn:>8.1f} {fps:>8.1f}  {ok}")
