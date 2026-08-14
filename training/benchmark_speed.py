"""
Benchmark SPAN-Lite inference speed at various resolutions.
Tests both PyTorch and ONNX Runtime to estimate browser performance.

Usage:
    python benchmark_speed.py --checkpoint best_phase2.pth
"""
import argparse
import time
import torch
from model_span import SPANLite

RESOLUTIONS = [
    ('360p', 360, 640),
    ('480p', 480, 854),
    ('720p', 720, 1280),
    ('1080p', 1080, 1920),
]


def benchmark_pytorch(model, device, height, width, warmup=10, runs=50):
    dummy = torch.randn(1, 3, height, width, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return times


def benchmark_onnx(onnx_path, height, width, warmup=5, runs=30):
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed, skipping ONNX benchmark")
        return None

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    available = ort.get_available_providers()
    providers = [p for p in providers if p in available]

    sess = ort.InferenceSession(onnx_path, providers=providers)
    used_ep = sess.get_providers()[0]

    import numpy as np
    dummy = np.random.randn(1, 3, height, width).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {'input': dummy})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {'input': dummy})
        times.append((time.perf_counter() - t0) * 1000)

    return times, used_ep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='D:/webvsr/checkpoints/best_phase2.pth')
    parser.add_argument('--onnx', default=None, help='ONNX model path (optional)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    model = SPANLite(num_in_ch=3, num_out_ch=3, feature_channels=32, upscale=2)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state)
    model.to(device).eval()

    print(f"\n{'Resolution':<12} {'Avg ms':>8} {'Min ms':>8} {'Max ms':>8} {'FPS':>8}  {'Can 60fps?'}")
    print('-' * 68)

    for name, h, w in RESOLUTIONS:
        times = benchmark_pytorch(model, device, h, w)
        avg = sum(times) / len(times)
        mn = min(times)
        mx = max(times)
        fps = 1000 / avg
        ok = 'YES' if avg < 16.6 else ('CLOSE' if avg < 25 else 'NO')
        print(f"{name:<12} {avg:>8.1f} {mn:>8.1f} {mx:>8.1f} {fps:>8.1f}  {ok}")

    if args.onnx:
        print(f"\n--- ONNX Runtime ---")
        for name, h, w in RESOLUTIONS:
            result = benchmark_onnx(args.onnx, h, w)
            if result:
                times, ep = result
                avg = sum(times) / len(times)
                mn = min(times)
                fps = 1000 / avg
                print(f"{name:<12} {avg:>8.1f} {mn:>8.1f} {fps:>8.1f} FPS  [{ep}]")


if __name__ == '__main__':
    main()
