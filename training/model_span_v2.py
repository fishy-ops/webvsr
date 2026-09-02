"""SPANV2-style attention: a learned channel-mixing gate instead of a free one.

From the NTIRE 2026 Efficient SR challenge report (arXiv:2604.03198), whose
runtime track was won by XiaomiMM's SPANV2 -- the same SPAN family this project
uses. Its SPABV2 block replaces SPAN's parameter-free attention with a learned
1x1 projection to a full CxC channel-mixing map, giving "content-adaptive
suppression and cross-channel gating" for C^2 extra parameters per block.

Why it might matter here. SPAN computes att = sigmoid(out3) - 0.5, so channel i
is gated only by what channel i itself saw; a gate can never be informed by
another channel. On a 16-channel trunk that is a real constraint, because the
channels are few enough that each carries a broad mixture. The 1x1 projection
lets the gate read all channels, at 256 parameters per block -- 1,024 across
four blocks, about 3% on top of 33,388.

The attention projection is initialised to the identity, so at step zero this
model is *exactly* SPAN, numerically. That matters for two reasons: a SPAN
checkpoint warm-starts it without any loss jump, and any measured difference is
attributable to the added freedom rather than to a different starting point.
"""

import torch
import torch.nn as nn

from model_span import Conv3XC, SPANLite


class SPABV2(nn.Module):
    """SPAB with a learned cross-channel attention projection."""

    def __init__(self, channels, bias=True):
        super().__init__()
        self.c1 = Conv3XC(channels, channels, gain=2, bias=bias)
        self.c2 = Conv3XC(channels, channels, gain=2, bias=bias)
        self.c3 = Conv3XC(channels, channels, gain=2, bias=bias)
        self.act = nn.SiLU(inplace=True)
        # No bias: sigmoid(x) - 0.5 is centred on zero, and a bias here would
        # shift the whole gate off centre at init, which is the one thing the
        # identity initialisation is meant to avoid.
        self.att_proj = nn.Conv2d(channels, channels, 1, bias=False)
        with torch.no_grad():
            self.att_proj.weight.copy_(
                torch.eye(channels).view(channels, channels, 1, 1))

    def forward(self, x):
        out1 = self.act(self.c1(x))
        out2 = self.act(self.c2(out1))
        out3 = self.c3(out2)
        att = torch.sigmoid(self.att_proj(out3)) - 0.5
        out = (out3 + x) * att
        return out, out1, att


class SPANLiteV2(SPANLite):
    """SPANLite with SPABV2 blocks. Everything else is unchanged."""

    def __init__(self, num_in_ch=3, num_out_ch=3, feature_channels=32,
                 upscale=2, img_range=1.0):
        super().__init__(num_in_ch=num_in_ch, num_out_ch=num_out_ch,
                         feature_channels=feature_channels,
                         upscale=upscale, img_range=img_range)
        self.block_1 = SPABV2(feature_channels)
        self.block_2 = SPABV2(feature_channels)
        self.block_3 = SPABV2(feature_channels)
        self.block_4 = SPABV2(feature_channels)


if __name__ == "__main__":
    torch.manual_seed(0)
    C = 16
    a = SPANLite(upscale=2, feature_channels=C).eval()
    b = SPANLiteV2(upscale=2, feature_channels=C).eval()

    # Copy the shared weights across so the only difference is att_proj.
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=False)
    print(f"loaded SPAN weights into V2: {len(missing)} missing "
          f"(expect 4 att_proj), {len(unexpected)} unexpected")

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        ya, yb = a(x), b(x)
    d = (ya - yb).abs().max().item()
    print(f"max |SPAN - SPANV2| at identity init: {d:.3e}  (must be ~0)")

    pa = sum(p.numel() for p in a.parameters() if p.requires_grad)
    pb = sum(p.numel() for p in b.parameters() if p.requires_grad)
    print(f"trainable params: {pa:,} -> {pb:,}  (+{pb - pa:,}, "
          f"{100 * (pb - pa) / pa:.1f}%)")
    assert d < 1e-5, "identity init broken: V2 must start numerically equal to SPAN"
    print("OK")
