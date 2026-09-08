"""SRVGGNetCompact — the architecture behind realesr-animevideov3.

Included to answer a direct question: what does a pretrained, GAN-trained model
built for anime video actually look like on our benchmark, and could we run it?
It is a plain stack of conv+PReLU with a single pixel-shuffle at the end, which
is why it is worth measuring -- unlike a transformer, this shape maps cleanly to
the WGSL compute pipeline we already have. The only question is cost.

Weights are BSD-3 (Real-ESRGAN). Architecture reimplemented here from the
published description rather than vendored.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        body += [nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1)]
        self.body = nn.Sequential(*body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = self.upsampler(self.body(x))
        # the reference adds a nearest-neighbour skip of the input
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def load_realesr_anime(path, device="cuda"):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob.get("params", blob.get("params_ema", blob))
    m = SRVGGNetCompact(num_feat=64, num_conv=16, upscale=4)
    m.load_state_dict(sd)
    return m.eval().to(device)


if __name__ == "__main__":
    import sys, time
    m = load_realesr_anime(sys.argv[1])
    n = sum(p.numel() for p in m.parameters())
    print(f"  realesr-animevideov3: {n:,} params  (SPAN-Lite c16 = 33,388)")
    print(f"  ratio: {n/33388:.0f}x larger")
    # what it costs at the resolution the extension actually runs
    x = torch.rand(1, 3, 540, 960, device="cuda")
    with torch.no_grad():
        for _ in range(3):
            m(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            m(x)
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 5
    print(f"  540p->4x on a 2070S: {dt*1000:.0f} ms/frame")
    print(f"  frame budget at 30fps = 33 ms, at 60fps = 17 ms")
