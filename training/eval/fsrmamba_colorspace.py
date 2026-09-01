"""Colour transforms, matched bit-for-bit to FSR 3.1.4.

Reference: fsr-upstream/sdk/include/FidelityFX/gpu/fsr3upscaler/
           ffx_fsr3upscaler_common.h:201-222

FSR does its rectification (the anti-ghosting colour clamp) in YCoCg rather than
RGB, because separating luma from chroma lets it use an anisotropic clipping box
-- see the `FfxFloat32x3(1.7f, 1.0f, 1.0f)` scale in accumulate.h:57, which
deliberately allows more luma variation than chroma variation.

Note this is the *unnormalised* YCoCg used by FSR (Co/Cg are not scaled to a
[-0.5, 0.5] range), so round-tripping is exact but the channel magnitudes are
not what you would get from a textbook definition.
"""

from __future__ import annotations

import torch

__all__ = ["rgb_to_ycocg", "ycocg_to_rgb"]


def rgb_to_ycocg(rgb: torch.Tensor) -> torch.Tensor:
    """RGB -> YCoCg. Operates on the last dimension, which must be size 3."""
    r, g, b = rgb.unbind(-1)
    return torch.stack(
        (
            0.25 * r + 0.5 * g + 0.25 * b,
            0.5 * r - 0.5 * b,
            -0.25 * r + 0.5 * g - 0.25 * b,
        ),
        dim=-1,
    )


def ycocg_to_rgb(ycocg: torch.Tensor) -> torch.Tensor:
    """YCoCg -> RGB. Exact inverse of :func:`rgb_to_ycocg`."""
    y, co, cg = ycocg.unbind(-1)
    return torch.stack((y + co - cg, y + cg, y - co - cg), dim=-1)
