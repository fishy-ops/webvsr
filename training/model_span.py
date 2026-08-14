import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Conv3XC(nn.Module):
    """Reparameterizable conv: decomposed 1x1->3x3->1x1 + skip during training,
    fused single 3x3 at inference."""

    def __init__(self, c_in, c_out, gain=2, s=1, bias=True):
        super().__init__()
        self.stride = s

        self.sk = nn.Conv2d(c_in, c_out, 1, stride=s, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_in * gain, 1, bias=bias),
            nn.Conv2d(c_in * gain, c_out * gain, 3, stride=s, padding=0, bias=bias),
            nn.Conv2d(c_out * gain, c_out, 1, bias=bias),
        )
        self.eval_conv = nn.Conv2d(c_in, c_out, 3, padding=1, stride=s, bias=bias)
        self.eval_conv.weight.requires_grad = False
        self.eval_conv.bias.requires_grad = False
        self._update_params()

    def _update_params(self):
        w1 = self.conv[0].weight.data.clone()
        b1 = self.conv[0].bias.data.clone()
        w2 = self.conv[1].weight.data.clone()
        b2 = self.conv[1].bias.data.clone()
        w3 = self.conv[2].weight.data.clone()
        b3 = self.conv[2].bias.data.clone()

        w = F.conv2d(
            w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2
        ).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        weight = F.conv2d(
            w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0
        ).flip(2, 3).permute(1, 0, 2, 3)
        bias = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = F.pad(self.sk.weight.data.clone(), [1, 1, 1, 1])
        sk_b = self.sk.bias.data.clone()

        self.eval_conv.weight.data = weight + sk_w
        self.eval_conv.bias.data = bias + sk_b

    def forward(self, x):
        if self.training:
            x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
            return self.conv(x_pad) + self.sk(x)
        else:
            self._update_params()
            return self.eval_conv(x)


class SPAB(nn.Module):
    """Swift Parameter-free Attention Block.
    Attention = sigmoid(features) - 0.5, element-wise multiply."""

    def __init__(self, channels, bias=True):
        super().__init__()
        self.c1 = Conv3XC(channels, channels, gain=2, bias=bias)
        self.c2 = Conv3XC(channels, channels, gain=2, bias=bias)
        self.c3 = Conv3XC(channels, channels, gain=2, bias=bias)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        out1 = self.act(self.c1(x))
        out2 = self.act(self.c2(out1))
        out3 = self.c3(out2)
        att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * att
        return out, out1, att


class SPANLite(nn.Module):
    """SPAN-Lite: reduced SPAN for real-time browser inference.
    4 SPAB blocks, 32 channels (vs original 6 blocks, 48 channels)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, feature_channels=32,
                 upscale=2, img_range=1.0):
        super().__init__()
        self.img_range = img_range
        self.upscale = upscale
        self.mean = torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1)

        self.conv_first = Conv3XC(num_in_ch, feature_channels, gain=2)

        self.block_1 = SPAB(feature_channels)
        self.block_2 = SPAB(feature_channels)
        self.block_3 = SPAB(feature_channels)
        self.block_4 = SPAB(feature_channels)

        self.conv_cat = nn.Conv2d(feature_channels * 4, feature_channels, 1, bias=True)
        self.conv_last = Conv3XC(feature_channels, feature_channels, gain=2)

        self.upsampler = nn.Sequential(
            nn.Conv2d(feature_channels, num_out_ch * (upscale ** 2), 3, padding=1),
            nn.PixelShuffle(upscale),
        )

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        f0 = self.conv_first(x)
        b1, b1_mid, _ = self.block_1(f0)
        b2, _, _ = self.block_2(b1)
        b3, b3_mid, _ = self.block_3(b2)
        b4, _, _ = self.block_4(b3)

        b4 = self.conv_last(b4)
        out = self.conv_cat(torch.cat([f0, b1_mid, b3_mid, b4], dim=1))
        out = self.upsampler(out)
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SPANLite(upscale=2, feature_channels=32)
    print(f"SPAN-Lite (2x, 32ch, 4 blocks)")
    print(f"  Trainable params: {count_params(model):,}")

    x = torch.randn(1, 3, 360, 640)
    with torch.no_grad():
        model.eval()
        y = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Scale:  {y.shape[-1] // x.shape[-1]}x")
