"""Export trained SPAN-Lite to ONNX with FP16 quantization for browser deployment."""

import argparse
import torch
import torch.nn as nn
from pathlib import Path

from model_span import SPANLite, count_params


def fuse_reparam(model):
    """Switch all Conv3XC to inference mode (fused single 3x3 conv)."""
    model.eval()
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, '_update_params') and hasattr(module, 'eval_conv'):
                module._update_params()
    return model


def count_inference_params(model):
    """Count params that would be in the ONNX model (inference mode)."""
    model = fuse_reparam(model)
    total = 0
    for name, p in model.named_parameters():
        if 'eval_conv' in name or 'conv_cat' in name or 'upsampler' in name:
            total += p.numel()
    for name, buf in model.named_buffers():
        if 'eval_conv' in name:
            total += buf.numel()
    return total


def export(checkpoint_path, output_dir="D:\\webvsr\\onnx", scale=2,
           input_h=360, input_w=640):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SPANLite(upscale=scale, feature_channels=32)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model = fuse_reparam(model)

    print(f"Training params:  {count_params(model):,}")

    dummy = torch.randn(1, 3, input_h, input_w)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")

    # FP32 export
    fp32_path = output_dir / "span_lite_2x_fp32.onnx"
    torch.onnx.export(
        model, dummy, str(fp32_path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
    )
    fp32_size = fp32_path.stat().st_size / 1024
    print(f"FP32 ONNX: {fp32_path} ({fp32_size:.0f} KB)")

    # FP16 quantization
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        fp16_path = output_dir / "span_lite_2x_fp16.onnx"
        quantize_dynamic(
            str(fp32_path), str(fp16_path),
            weight_type=QuantType.QUInt8,
        )
        fp16_size = fp16_path.stat().st_size / 1024
        print(f"Quantized ONNX: {fp16_path} ({fp16_size:.0f} KB)")
    except ImportError:
        print("onnxruntime not installed, skipping quantization")
        print("  pip install onnxruntime")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .pth checkpoint")
    parser.add_argument("--output-dir", default=r"D:\webvsr\onnx")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    export(args.checkpoint, args.output_dir, input_h=args.height, input_w=args.width)
