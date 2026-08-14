"""Export RIFE-Lite frame interpolation model to ONNX."""
import argparse
import torch
from pathlib import Path
from model_rife_lite import RIFELite


def export(checkpoint_path, output_dir="D:\\webvsr\\onnx", input_h=360, input_w=640):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = RIFELite()
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)

    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"RIFE-Lite params: {params:,}")

    img0 = torch.randn(1, 3, input_h, input_w)
    img1 = torch.randn(1, 3, input_h, input_w)

    with torch.no_grad():
        out = model(img0, img1, t=0.5)
    print(f"Input: {img0.shape} x2, Output: {out.shape}")

    class RIFEWrapper(torch.nn.Module):
        def __init__(self, rife):
            super().__init__()
            self.rife = rife

        def forward(self, img0, img1):
            return self.rife(img0, img1, t=0.5)

    wrapper = RIFEWrapper(model)
    onnx_path = output_dir / "rife_lite_fp32.onnx"

    torch.onnx.export(
        wrapper, (img0, img1), str(onnx_path),
        opset_version=17,
        input_names=["frame0", "frame1"],
        output_names=["interpolated"],
        dynamic_axes={
            "frame0": {0: "batch", 2: "height", 3: "width"},
            "frame1": {0: "batch", 2: "height", 3: "width"},
            "interpolated": {0: "batch", 2: "height", 3: "width"},
        },
    )
    size_kb = onnx_path.stat().st_size / 1024
    print(f"ONNX saved: {onnx_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=r"D:\webvsr\onnx")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    export(args.checkpoint, args.output_dir, args.height, args.width)
