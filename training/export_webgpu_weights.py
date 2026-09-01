"""
Export reparameterized SPAN-Lite weights as a flat binary for WebGPU shaders.

The WebGPU shader loads weights as a single Float32Array. This script:
1. Loads the training checkpoint
2. Reparameterizes Conv3XC → single 3x3 conv
3. Dumps all weights/biases in network order as a .bin file

Weight order (CHW layout):
  conv_first.weight [32, 3, 3, 3]
  conv_first.bias [32]
  block_1.c1.weight, bias
  block_1.c2.weight, bias
  block_1.c3.weight, bias
  ... (blocks 2-4)
  conv_cat.weight [32, 128, 1, 1]
  conv_cat.bias [32]
  conv_last.weight [48, 32, 3, 3]  (48 = 3 * 2^2 for PixelShuffle)
  conv_last.bias [48]
"""
import argparse
import struct
import torch
import numpy as np
from model_span import SPANLite


def export_weights(checkpoint_path, output_path, channels=32, scale=2):
    model = SPANLite(num_in_ch=3, num_out_ch=3, feature_channels=channels, upscale=scale)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state)
    model.eval()

    # Reparameterize
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, '_update_params') and hasattr(module, 'eval_conv'):
                module._update_params()

    # Collect weights in order
    weight_list = []
    param_names = []

    def add(name, tensor):
        weight_list.append(tensor.detach().cpu().numpy().astype(np.float32))
        param_names.append((name, tensor.shape))

    # conv_first
    add('conv_first.weight', model.conv_first.eval_conv.weight)
    add('conv_first.bias', model.conv_first.eval_conv.bias)

    # 4 SPAB blocks
    for i, block in enumerate([model.block_1, model.block_2, model.block_3, model.block_4]):
        for j, conv in enumerate([block.c1, block.c2, block.c3]):
            add(f'block_{i+1}.c{j+1}.weight', conv.eval_conv.weight)
            add(f'block_{i+1}.c{j+1}.bias', conv.eval_conv.bias)

    # conv_cat (1x1)
    add('conv_cat.weight', model.conv_cat.weight)
    add('conv_cat.bias', model.conv_cat.bias)

    # conv_last (upsampler input)
    add('conv_last.weight', model.conv_last.eval_conv.weight)
    add('conv_last.bias', model.conv_last.eval_conv.bias)

    # upsampler conv
    add('upsampler.weight', model.upsampler[0].weight)
    add('upsampler.bias', model.upsampler[0].bias)

    # Flatten and save
    all_weights = np.concatenate([w.flatten() for w in weight_list])
    total_params = all_weights.shape[0]

    with open(output_path, 'wb') as f:
        f.write(all_weights.tobytes())

    # Sibling manifest read by the WebGPU engine to set channel count.
    import json as _json
    manifest_path = output_path.rsplit('.', 1)[0] + '.json'
    with open(manifest_path, 'w') as f:
        _json.dump({"channels": channels, "scale": scale, "blocks": 4}, f)
    print(f"Wrote manifest {manifest_path}: channels={channels}")

    print(f"Exported {total_params:,} parameters ({total_params * 4 / 1024:.0f} KB)")
    print(f"\nWeight layout:")
    offset = 0
    for name, shape in param_names:
        count = 1
        for s in shape:
            count *= s
        print(f"  [{offset:>7d}] {name}: {list(shape)} ({count:,} params)")
        offset += count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best_phase2.pth')
    parser.add_argument('--output', default='extension/models/span_lite_2x.bin')
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    export_weights(args.checkpoint, args.output, args.channels, args.scale)
