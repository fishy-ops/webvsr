import torch
import torch.nn as nn
import torch.nn.functional as F
from model_span import Conv3XC, SPAB


class SPANLiteUnshuffle(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, feature_channels=32, upscale=2, img_range=1.0):
        super().__init__()
        self.img_range = img_range

        mean = torch.zeros(num_in_ch)
        rgb_mean = torch.tensor([0.4488, 0.4371, 0.4040])
        n = min(3, num_in_ch)
        mean[:n] = rgb_mean[:n]
        self.register_buffer('mean', mean.view(1, num_in_ch, 1, 1))

        self.unshuffle = nn.PixelUnshuffle(2)
        self.conv_first = Conv3XC(num_in_ch * 4, feature_channels)

        self.spab1 = SPAB(feature_channels)
        self.spab2 = SPAB(feature_channels)
        self.spab3 = SPAB(feature_channels)
        self.spab4 = SPAB(feature_channels)

        self.conv_cat = nn.Conv2d(feature_channels * 4, feature_channels, kernel_size=1)
        self.conv_last = Conv3XC(feature_channels, feature_channels)

        self.upsampler = nn.Conv2d(
            feature_channels,
            num_out_ch * (2 * upscale) ** 2,
            kernel_size=3,
            padding=1,
        )
        self.pixel_shuffle = nn.PixelShuffle(2 * upscale)

    def forward(self, x):
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        x = (x - mean) * self.img_range

        x = self.unshuffle(x)
        f0 = self.conv_first(x)

        b1, b1_mid, _ = self.spab1(f0)
        b2, b2_mid, _ = self.spab2(b1)
        b3, b3_mid, _ = self.spab3(b2)
        b4, b4_mid, _ = self.spab4(b3)

        x = torch.cat([f0, b1_mid, b3_mid, b4], dim=1)
        x = self.conv_cat(x)
        x = self.conv_last(x)

        x = self.upsampler(x)
        x = self.pixel_shuffle(x)

        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = SPANLiteUnshuffle(feature_channels=32, upscale=2)
    print(count_params(model))

    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 3, 512, 512)
