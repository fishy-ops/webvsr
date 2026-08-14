"""
Post-training automation script.
Run after training completes to:
1. Evaluate on DIV2K validation
2. Export ONNX model
3. Export WebGPU weights
4. Generate comparison report
5. Print summary
"""
import subprocess
import sys
import os

PYTHON = os.path.join('D:', os.sep, 'webvsr', '.venv', 'Scripts', 'python.exe')
CWD = r'D:\webvsr'
BEST_CKPT = r'D:\webvsr\checkpoints\best_phase2.pth'


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=CWD, capture_output=False)
    if result.returncode != 0:
        print(f"  WARNING: {desc} failed with code {result.returncode}")
    return result.returncode


def main():
    if not os.path.exists(BEST_CKPT):
        # Fall back to phase 1
        alt = r'D:\webvsr\checkpoints\best_phase1.pth'
        if os.path.exists(alt):
            global BEST_CKPT
            BEST_CKPT = alt
            print(f"Using phase 1 checkpoint: {alt}")
        else:
            print("No checkpoint found!")
            return

    # 1. Evaluate
    run([PYTHON, 'evaluate.py', '--checkpoint', BEST_CKPT, '--dataset', 'all'],
        'Evaluate on DIV2K validation')

    # 2. Export ONNX
    run([PYTHON, 'export_onnx.py', BEST_CKPT],
        'Export ONNX model')

    # 3. Export WebGPU weights
    run([PYTHON, 'export_webgpu_weights.py', '--checkpoint', BEST_CKPT],
        'Export WebGPU binary weights')

    # 4. Generate comparison page
    run([PYTHON, 'generate_comparison_page.py', '--checkpoint', BEST_CKPT, '--count', '8'],
        'Generate visual comparison report')

    # 5. Summary
    print(f"\n{'='*60}")
    print(f"  POST-TRAINING COMPLETE")
    print(f"{'='*60}")

    onnx_dir = r'D:\webvsr\onnx'
    if os.path.isdir(onnx_dir):
        for f in os.listdir(onnx_dir):
            size = os.path.getsize(os.path.join(onnx_dir, f)) / 1024
            print(f"  ONNX: {f} ({size:.0f} KB)")

    weights_path = r'D:\webvsr\extension\models\span_lite_2x.bin'
    if os.path.exists(weights_path):
        size = os.path.getsize(weights_path) / 1024
        print(f"  WebGPU weights: {size:.0f} KB")

    report = r'D:\webvsr\comparison_report.html'
    if os.path.exists(report):
        size = os.path.getsize(report) / 1024 / 1024
        print(f"  Comparison report: {size:.1f} MB")

    eval_results = r'D:\webvsr\eval_results.json'
    if os.path.exists(eval_results):
        import json
        with open(eval_results) as f:
            results = json.load(f)
        for r in results:
            print(f"  {r['dataset']}: PSNR={r['avg_psnr']:.2f} dB, SSIM={r['avg_ssim']:.4f}")


if __name__ == '__main__':
    main()
