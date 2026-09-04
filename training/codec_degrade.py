"""
Codec-aware degradation — the domain fix.

dataset.py degrades with JPEG. The extension runs on H.264/H.265 web video,
whose artifacts are different in kind: JPEG quantises an 8x8 DCT per block in
isolation, while a video codec adds deblocking-filter smear, coarser chroma
handling, and — the part JPEG cannot imitate — the residual structure left by
inter-frame prediction.

APISR's observation (arXiv:2403.01598) is that you do not need video to get
those artifacts: running a *single frame* through a video codec at a punishing
QP reproduces most of them, because the codec still applies its transform,
quantisation and in-loop deblocking to an I-frame. That is what this module
does, in-process via PyAV, so it is cheap enough to run on-the-fly during
training instead of precomputing a fixed degraded copy.

Public entry point is `codec_compress(tensor, ...)`, shaped to drop into
dataset.py's degradation chain in place of (or after) `jpeg_compress_tensor`.
"""

import contextlib
import io
import os
import random

import numpy as np
import torch

import av
import av.logging

# Encoders write banners and per-frame stats to stderr; at one encode per
# training sample that is thousands of lines a second.
av.logging.set_level(av.logging.PANIC)


@contextlib.contextmanager
def _quiet_fd2():
    """Silence writes to fd 2 for the duration of the block.

    x265 and SVT-AV1 both write banners straight to the file descriptor rather
    than through libav's logging, so av.logging cannot reach them. At one encode
    per training sample that is thousands of lines a second. Redirecting the
    descriptor is the only thing that catches it; the cost is two dup2 calls.
    """
    try:
        saved = os.dup(2)
    except OSError:
        yield          # no fd 2 to redirect (rare, but do not take training down)
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)

# (codec name, pixel format). This has to match what the extension actually
# meets, which is a browser playing web video -- and that is no longer only
# H.264. YouTube serves VP9 to most desktop browsers and AV1 to a growing share,
# and their artifacts differ from x264's: VP9 and AV1 use larger transforms and
# stronger in-loop filtering, so they smear where x264 blocks. Training only on
# x264 artifacts is the same class of domain gap this project has been bitten by
# twice already (bicubic-vs-codec validation, render-vs-camera benchmarking).
#
# Weighted by a mix of web prevalence and encode cost, measured per 256x256
# frame through PyAV: x264 8.0ms, VP9 16.0ms, SVT-AV1 34.7ms. The expensive
# ones are garnish for distribution width, not parity.
CODECS = [
    ("libx264", "yuv420p"),
    ("libvpx-vp9", "yuv420p"),
    ("libsvtav1", "yuv420p"),
    ("libx265", "yuv420p"),
    ("mpeg4", "yuv420p"),
]
CODEC_WEIGHTS = [0.50, 0.22, 0.08, 0.05, 0.15]

# Per-codec quality range. Scales differ between encoders, so each carries its
# own: x264/x265 take CRF 0-51, VP9 and AV1 take CRF 0-63, mpeg4 takes a qscale
# where higher is worse. The VP9/AV1 ranges are set higher because the same
# numeric CRF is markedly less destructive on those encoders.
QUALITY = {
    "libx264": (23, 42),
    "libvpx-vp9": (35, 58),
    "libsvtav1": (40, 62),
    "libx265": (26, 45),
    "mpeg4": (4, 18),
}


def _to_uint8_hwc(t):
    """(C,H,W) float 0..1 tensor -> (H,W,3) uint8 array."""
    a = (t.detach().cpu().clamp(0, 1) * 255.0).round().to(torch.uint8)
    return a.permute(1, 2, 0).numpy()


def _to_tensor_chw(a, device, dtype):
    """(H,W,3) uint8 array -> (C,H,W) float 0..1 tensor."""
    t = torch.from_numpy(np.ascontiguousarray(a)).to(device)
    return t.permute(2, 0, 1).to(dtype) / 255.0


def codec_compress(img, codec=None, quality=None):
    """Round-trip a CHW float tensor through a real video encoder.

    img: (3,H,W) float tensor in [0,1]. Returns the same shape/dtype/device.

    Encodes a single frame as an intra picture and decodes it straight back.
    Dimensions are padded up to even numbers because yuv420p needs them, then
    cropped back, so odd-sized training crops do not crash.
    """
    if codec is None:
        codec, pix_fmt = random.choices(CODECS, weights=CODEC_WEIGHTS, k=1)[0]
    else:
        pix_fmt = dict(CODECS).get(codec, "yuv420p")
    lo, hi = QUALITY[codec]
    q = random.randint(lo, hi) if quality is None else quality

    device, dtype = img.device, img.dtype
    arr = _to_uint8_hwc(img)
    h, w, _ = arr.shape

    ph, pw = h + (h & 1), w + (w & 1)
    if (ph, pw) != (h, w):
        arr = np.pad(arr, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")

    opts = {}
    if codec == "libvpx-vp9":
        # b=0 is what puts libvpx in constant-quality mode; without it CRF is
        # only an upper bound and the encode is bitrate-targeted instead.
        opts.update({"crf": str(q), "b": "0",
                     "deadline": "realtime", "cpu-used": "8"})
    elif codec == "libsvtav1":
        opts.update({"crf": str(q), "preset": "12"})   # 12 = fastest preset
    elif codec in ("libx264", "libx265"):
        opts["crf"] = str(q)
        # Kill lookahead/threading so a one-frame encode is deterministic and
        # does not sit waiting for frames that will never arrive.
        opts["preset"] = "veryfast"
        if codec == "libx264":
            opts["tune"] = "fastdecode"
        else:
            # x265 writes its banner straight to stderr, bypassing libav's
            # logging entirely, so av.logging cannot mute it -- only this can.
            opts["x265-params"] = "log-level=none"
    buf = io.BytesIO()
    try:
      with _quiet_fd2():
          container = av.open(buf, mode="w", format="mp4")
          stream = container.add_stream(codec, rate=25)
          stream.width, stream.height = pw, ph
          stream.pix_fmt = pix_fmt
          if opts:
              stream.options = opts
          if codec == "mpeg4":
              # mpeg4 ignores CRF; drive it with the qscale bitstream knob.
              stream.codec_context.qmin = q
              stream.codec_context.qmax = q

          frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
          for packet in stream.encode(frame):
              container.mux(packet)
          for packet in stream.encode():          # flush
              container.mux(packet)
          container.close()

          buf.seek(0)
          container = av.open(buf, mode="r")
          out = None
          for f in container.decode(video=0):
              out = f.to_ndarray(format="rgb24")
              break
          container.close()
    except Exception:
        # A codec missing from this ffmpeg build should degrade the sample,
        # not kill the training run.
        return img

    if out is None:
        return img
    out = out[:h, :w]
    return _to_tensor_chw(out, device, dtype)


def second_order_codec(img, scale=2, jpeg_fn=None):
    """Degradation chain with a codec stage where the JPEG stage used to be.

    Mirrors the shape of dataset.degrade_second_order but ends in a real
    encoder. `jpeg_fn` lets the caller keep one JPEG pass in the mix, since
    some web video really is a JPEG-ish still upscaled into a video container.
    """
    import torch.nn.functional as F

    x = img
    _, h, w = x.shape
    th, tw = h // scale, w // scale

    # Only the *post*-resize encode is applied by default: it is the pass that
    # leaves the blocking the network actually sees, and one encode per sample
    # keeps the data loader ahead of the GPU.
    if random.random() < 0.25:
        x = codec_compress(x, codec="libx264")

    mode = random.choice(["bilinear", "bicubic", "area"])
    kw = {} if mode == "area" else {"align_corners": False}
    x = F.interpolate(x.unsqueeze(0), size=(th, tw), mode=mode, **kw).squeeze(0).clamp(0, 1)

    # Second pass at the *target* resolution: this is the one that leaves the
    # blocking the network actually sees, because it happens after the resize.
    if random.random() < 0.8:
        x = codec_compress(x)
    elif jpeg_fn is not None:
        x = jpeg_fn(x, (50, 95))

    return x.clamp(0, 1)
