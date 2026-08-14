"""
RIFE-Lite: Lightweight frame interpolation for WebVSR.

Based on RIFE (Real-Time Intermediate Flow Estimation) but reduced for browser
deployment. Estimates bidirectional optical flow between two frames and warps
them to produce an intermediate frame.

Architecture: IFNet with reduced channels for <500K params total.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def warp(img, flow):
    B, _, H, W = img.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=img.dtype),
        torch.arange(W, device=img.device, dtype=img.dtype),
        indexing='ij',
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    vgrid_x = 2.0 * (grid_x + flow[:, 0]) / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * (grid_y + flow[:, 1]) / max(H - 1, 1) - 1.0
    vgrid = torch.stack([vgrid_x, vgrid_y], dim=-1)
    return F.grid_sample(img, vgrid, mode='bilinear', padding_mode='border', align_corners=True)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, 1)
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        return self.act(self.conv(x))


class IFBlock(nn.Module):
    """Single scale of the IFNet. Predicts flow + mask at one resolution."""
    def __init__(self, in_ch, mid_ch=32):
        super().__init__()
        self.encode = nn.Sequential(
            ConvBlock(in_ch, mid_ch, stride=2),
            ConvBlock(mid_ch, mid_ch),
            ConvBlock(mid_ch, mid_ch),
        )
        # 4 flow channels (2 for each direction) + 1 mask
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(mid_ch, mid_ch, 4, 2, 1),
            nn.PReLU(mid_ch),
            nn.Conv2d(mid_ch, 5, 3, 1, 1),
        )

    def forward(self, x):
        feat = self.encode(x)
        out = self.decode(feat)
        return out


class RIFELite(nn.Module):
    """
    3-scale IFNet for frame interpolation.

    Input: (img0, img1) each [B, 3, H, W], timestep t in [0, 1]
    Output: interpolated frame [B, 3, H, W]

    Total params: ~180K
    """
    def __init__(self):
        super().__init__()
        # 3 pyramid levels with increasing channels
        # Input at each level: img0 + img1 + flow_prev + mask_prev
        self.block0 = IFBlock(6, mid_ch=32)         # full res: 6 (img0+img1)
        self.block1 = IFBlock(6 + 5, mid_ch=32)     # + prev flow+mask
        self.block2 = IFBlock(6 + 5, mid_ch=32)     # + prev flow+mask

    def forward(self, img0, img1, t=0.5):
        # Scale 0: 1/4 resolution
        h, w = img0.shape[2], img0.shape[3]
        s0 = F.interpolate(img0, scale_factor=0.25, mode='bilinear', align_corners=False)
        s1 = F.interpolate(img1, scale_factor=0.25, mode='bilinear', align_corners=False)

        inp = torch.cat([s0, s1], dim=1)
        flow_mask = self.block0(inp)
        flow = flow_mask[:, :4] * 2.0
        mask = torch.sigmoid(flow_mask[:, 4:5])

        # Scale 1: 1/2 resolution
        s0 = F.interpolate(img0, scale_factor=0.5, mode='bilinear', align_corners=False)
        s1 = F.interpolate(img1, scale_factor=0.5, mode='bilinear', align_corners=False)
        flow_up = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=False) * 2.0
        mask_up = F.interpolate(mask, scale_factor=2, mode='bilinear', align_corners=False)

        warped0 = warp(s0, flow_up[:, :2] * t)
        warped1 = warp(s1, flow_up[:, 2:4] * (1 - t))
        inp = torch.cat([warped0, warped1, flow_up, mask_up], dim=1)
        delta = self.block1(inp)
        flow = flow_up + delta[:, :4]
        mask = torch.sigmoid(delta[:, 4:5] + mask_up)

        # Scale 2: full resolution
        flow_up = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False) * 2.0
        mask_up = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)

        warped0 = warp(img0, flow_up[:, :2] * t)
        warped1 = warp(img1, flow_up[:, 2:4] * (1 - t))
        inp = torch.cat([warped0, warped1, flow_up, mask_up], dim=1)
        delta = self.block2(inp)
        flow = flow_up + delta[:, :4]
        mask = torch.sigmoid(delta[:, 4:5] + mask_up)

        # Final blending
        warped0 = warp(img0, flow[:, :2] * t)
        warped1 = warp(img1, flow[:, 2:4] * (1 - t))
        output = mask * warped0 + (1 - mask) * warped1

        return output.clamp(0, 1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    model = RIFELite()
    print(f"RIFE-Lite params: {count_params(model):,}")

    x0 = torch.randn(1, 3, 360, 640)
    x1 = torch.randn(1, 3, 360, 640)
    with torch.no_grad():
        out = model(x0, x1, t=0.5)
    print(f"Input: {x0.shape}, Output: {out.shape}")
