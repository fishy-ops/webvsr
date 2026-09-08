"""The look grade, mirrored from the WGSL in extension/webgpu-sr.js.

Kept byte-for-byte equivalent in operation order (exposure -> contrast ->
saturation -> temperature -> clamp) so marketing imagery shows what the shipped
shader actually produces. If the table here and the LOOKS table in the engine
ever disagree, the images become a lie, so both carry the same numbers.
"""
import numpy as np

LOOKS = {
    "natural":   dict(exposure=0.00, contrast=0.00, saturation=0.00, temp=0.00),
    "bright":    dict(exposure=0.07, contrast=0.10, saturation=0.12, temp=0.02),
    "vivid":     dict(exposure=0.02, contrast=0.18, saturation=0.30, temp=0.00),
    "cinematic": dict(exposure=-0.02, contrast=0.24, saturation=-0.05, temp=-0.05),
}
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _smoothstep01(x):
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def grade(img, look="natural"):
    """img: float array (H,W,3) in [0,1]. Returns the graded image."""
    g = LOOKS.get(look, LOOKS["natural"])
    c = img.astype(np.float32) * (1.0 + g["exposure"])
    c = c * (1.0 - g["contrast"]) + _smoothstep01(c) * g["contrast"]
    luma = (c * LUMA).sum(axis=-1, keepdims=True)
    c = luma + (c - luma) * (1.0 + g["saturation"])
    c = c + np.array([g["temp"], 0.0, -g["temp"]], dtype=np.float32)
    return np.clip(c, 0.0, 1.0)
