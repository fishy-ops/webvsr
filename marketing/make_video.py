"""The demo video: four scenes, each revealed by one left-to-right wipe.

Stills prove the difference exists; only video proves it survives motion, which
is what people actually doubt about frame-by-frame enhancement. Every frame is
processed independently by the real shipped network -- no temporal smoothing, no
frame hand-picked for flattery.

Four different clips rather than one long one, because the Xiph test sequences
are only ~2 seconds each and because breadth is the more persuasive claim: the
same network, unchanged, on architecture, crowds, machinery and foliage.

Structure per scene: hold on "without", one wipe, hold on "with". A sweeping
divider was tried first and rejected -- it draws the eye to the divider instead
of to the picture, and at two seconds a scene there is no time for both.
"""
import sys
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from brand3 import INK, GROUND, GREEN, DULL, font, track, track_width
from pipeline import load_shipped, codec_downscale, bicubic_up, model_up

# (clip, crop-x, crop-y) -- each window is 100% pixels, chosen to hold detail
SCENES = [
    ("src/old_town_cross_1080p50.mp4", 520, 320),
    ("src/park_joy_1080p50.mp4",       440, 300),
    ("src/tractor_1080p25.mp4",        560, 300),
    ("src/station2_1080p25.mp4",       420, 300),
]
CW, CH = 1280, 720
FPS = 25
HOLD_A, WIPE, HOLD_B = 10, 18, 11      # frames; 39 total ~1.6s a scene
OUT = Path("assets/crisp_demo.mp4")


def ease(t):
    return t * t * (3 - 2 * t)          # smoothstep, no snap at either end


def label(d, x, y, text, bg, fg, size=16):
    f = font(size, 700)
    tw = track_width(d, text, f, 1.3)
    d.rounded_rectangle([x, y, x + tw + 28, y + 34], 17, fill=bg)
    track(d, (x + 14, y + 8), text, f, fg, 1.3)


def to_img(t, x0, y0):
    a = (t[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(a).crop((x0, y0, x0 + CW, y0 + CH))


def main():
    model = load_shipped()
    out = av.open(str(OUT), mode="w")
    st = out.add_stream("libx264", rate=FPS)
    st.width, st.height, st.pix_fmt = CW, CH, "yuv420p"
    st.options = {"crf": "18", "preset": "slow"}
    total = 0

    for path, x0, y0 in SCENES:
        pairs = []
        with av.open(path) as c:
            for i, frame in enumerate(c.decode(video=0)):
                if i >= HOLD_A + WIPE + HOLD_B:
                    break
                arr = frame.to_ndarray(format="rgb24")
                hr = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                H, W = hr.shape[-2:]
                hr = hr[:, :, : (H // 4) * 4, : (W // 4) * 4]
                lr = codec_downscale(hr, 2, 28)
                pairs.append((to_img(bicubic_up(lr, 2), x0, y0),
                              to_img(model_up(model, lr), x0, y0)))
        if not pairs:
            print(f"  !! no frames from {path}"); continue

        n = len(pairs)
        for i, (bi, ai) in enumerate(pairs):
            if i < HOLD_A:
                pos = 0.0
            elif i < HOLD_A + WIPE:
                pos = ease((i - HOLD_A + 1) / WIPE)
            else:
                pos = 1.0
            px = int(pos * CW)
            comp = bi.copy()
            if px > 0:
                comp.paste(ai.crop((0, 0, px, CH)), (0, 0))
            d = ImageDraw.Draw(comp, "RGBA")
            if 0 < px < CW:
                # flat signal green: one accent, used once per frame. A gradient
                # here would undo the whole reason the system is flat.
                d.rectangle([px - 2, 0, px + 1, CH], fill=GREEN)
            # the label always describes the side it sits on
            if px < CW - 190:
                label(d, CW - 176, 24, "WITHOUT", (255, 255, 255, 235), INK)
            if px > 190:
                label(d, 24, 24, "WITH CRISP", (255, 255, 255, 235), INK)
            for pkt in st.encode(av.VideoFrame.from_ndarray(np.asarray(comp), format="rgb24")):
                out.mux(pkt)
            total += 1
        print(f"  {Path(path).stem}: {n} frames")

    for pkt in st.encode():
        out.mux(pkt)
    out.close()
    print(f"wrote {OUT}  {OUT.stat().st_size/1024/1024:.1f} MB  "
          f"{total} frames = {total/FPS:.1f}s @ {FPS}fps")


if __name__ == "__main__":
    main()
