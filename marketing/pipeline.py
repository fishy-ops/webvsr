"""Turn a source clip into the exact before/after a user sees, on this Mac.

The degradation here is deliberately the same one the measurements in
RESEARCH.md came from: downscale to half height with bicubic, then a real x264
encode at CRF 28, then decode. That order matters -- encoding *after* the
downscale is what puts codec artifacts at the resolution the network actually
sees. Anything simpler (a plain resize, a JPEG round trip) produces a prettier
"before" than real web video and would make the comparison a lie.

  HR      the original 1080p frame                      "the truth"
  LR      540p, x264 CRF 28                             what the browser receives
  before  LR upscaled 2x with bicubic                   what the browser shows today
  after   LR upscaled 2x by the shipped network         what the extension shows
"""
import io
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model_span import SPANLite

HERE = Path(__file__).parent
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_shipped(scale=2, channels=16):
    """The checkpoint PROVENANCE.json ties to the shipped .bin, so what these
    images show is what an installed user actually gets -- not a better
    research checkpoint."""
    m = SPANLite(num_in_ch=3, num_out_ch=3, feature_channels=channels, upscale=scale)
    blob = torch.load(HERE / "shipped_2x_c16.pth", map_location="cpu", weights_only=False)
    state = blob.get("model", blob.get("model_state_dict", blob))
    m.load_state_dict(state)
    return m.eval().to(DEV)


def read_frame(path, index):
    """Decode frame `index` as a float tensor (1,3,H,W) in [0,1]."""
    with av.open(str(path)) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i == index:
                arr = frame.to_ndarray(format="rgb24")
                break
        else:
            raise IndexError(f"{path} has fewer than {index + 1} frames")
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


def codec_downscale(hr, scale=2, crf=28):
    """Half the resolution, then a genuine x264 encode at `crf`.

    Encodes a single frame as an intra picture. Even at one frame the codec
    still applies its transform, quantisation and in-loop deblocking, which is
    where the blocking and ringing in real web video comes from.
    """
    h, w = hr.shape[-2:]
    lh, lw = (h // scale) & ~1, (w // scale) & ~1
    small = F.interpolate(hr, size=(lh, lw), mode="bicubic",
                          align_corners=False).clamp(0, 1)
    arr = (small[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as out:
        st = out.add_stream("libx264", rate=25)
        st.width, st.height, st.pix_fmt = lw, lh, "yuv420p"
        st.options = {"crf": str(crf), "preset": "veryfast"}
        out.mux(st.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")))
        out.mux(st.encode())
    buf.seek(0)
    with av.open(buf) as c:
        dec = next(c.decode(video=0)).to_ndarray(format="rgb24")
    t = torch.from_numpy(dec).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


def bicubic_up(lr, scale=2):
    return F.interpolate(lr, scale_factor=scale, mode="bicubic",
                         align_corners=False).clamp(0, 1)


@torch.no_grad()
def model_up(model, lr):
    return model(lr.to(DEV)).clamp(0, 1).cpu()


def to_pil(t):
    return Image.fromarray(
        (t[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def make_set(clip, frame_index, crf=28, scale=2, sharpen=3.5, look="vivid"):
    """Everything needed for one comparison, aligned pixel-for-pixel.

    The "after" side runs the FULL shipped chain -- network, then the
    contrast-adaptive sharpen, then the look grade -- because that is what a
    user with those settings actually sees. Earlier versions of this function
    stopped after the network and therefore under-sold the extension.
    """
    from looks import grade
    import numpy as _np
    hr = read_frame(clip, frame_index)
    h, w = hr.shape[-2:]
    hr = hr[:, :, : (h // (2 * scale)) * (2 * scale), : (w // (2 * scale)) * (2 * scale)]
    lr = codec_downscale(hr, scale, crf)
    model = load_shipped(scale)
    before = bicubic_up(lr, scale)
    after = model_up(model, lr)
    if sharpen:
        after = rcas_sharpen(after, sharpen)
    if look and look != "natural":
        a = grade(after[0].permute(1, 2, 0).numpy(), look)
        after = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
    # The encode can land a pixel off on odd sizes; crop all to the common size
    # rather than resampling, which would blur one side of the comparison.
    hh = min(hr.shape[-2], before.shape[-2], after.shape[-2])
    ww = min(hr.shape[-1], before.shape[-1], after.shape[-1])
    cut = lambda t: t[:, :, :hh, :ww]
    return {"hr": cut(hr), "before": cut(before), "after": cut(after), "lr": lr}


def rcas_sharpen(img, strength):
    """The extension's contrast-adaptive sharpen, mirrored from SHADER_SHARPEN.

    This was missing from every comparison image built before now: the Python
    pipeline ran the network only, while the extension applies this pass on top.
    So the "after" side was showing LESS than a real user gets. The clamp to the
    local min/max is what keeps a high strength from ringing -- it is why 3.5 is
    usable here at all, where a plain unsharp mask would halo badly.

    img: (1,3,H,W) in [0,1].
    """
    import torch.nn.functional as _F
    p = _F.pad(img, (1, 1, 1, 1), mode="replicate")
    c = img
    l = p[:, :, 1:-1, :-2]
    r = p[:, :, 1:-1, 2:]
    t = p[:, :, :-2, 1:-1]
    b = p[:, :, 2:, 1:-1]
    sharp = c + strength * (4.0 * c - l - r - t - b)
    mn = torch.minimum(c, torch.minimum(torch.minimum(l, r), torch.minimum(t, b)))
    mx = torch.maximum(c, torch.maximum(torch.maximum(l, r), torch.maximum(t, b)))
    return torch.clamp(torch.max(torch.min(sharp, mx), mn), 0.0, 1.0)
