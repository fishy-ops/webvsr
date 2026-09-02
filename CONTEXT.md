# WebVSR — working context

Written 2026-08-31. Supersedes `HANDOFF.md`, which is from the Windows era and
has stale paths and at least one wrong number (it says the shipped model is
128K parameters; it is 33,388).

This file exists so a model or a new session can do useful work here without
being re-briefed. Read it top to bottom before touching anything.

---

## 1. What the project is

A Chrome MV3 extension, published as **"Video Upscaler & Enhancer" v1.0.1**,
that runs real-time video super-resolution in the page on the user's own GPU
via hand-written WGSL compute shaders. No server, no ONNX Runtime, no WASM.

Repo: `github.com/fishy-ops/webvsr`, working copy `aurora:~/dev/webvsr`, at
commit `1a127b8`, which matches `origin/main` and the shipped build (v1.0.2).

The value proposition is narrow and deliberate: it removes **compression
artifacts** from low-resolution web video. On already-sharp video it is meant
to stay out of the way. It is not, and must not become, a detail-invention
tool — see §7.

---

## 2. Hardware and environment

| | |
|---|---|
| Training / eval box | `aurora` — Ubuntu 26.04, RTX 2070 SUPER 8 GB, 8 CPU cores |
| Python | `~/dev/webvsr/.venv` (3.12 via `uv`; system python is 3.14, which torch has no wheels for) |
| torch | 2.13.0+cu130, CUDA working |
| Bulk storage | ZFS pool `tank`, ~7 TB free, WebVSR data under `/tank/webvsr` |
| ffmpeg | 8.0.1, with libx264/libx265/mpeg4 |
| Mac | M4 Pro, 48 GB — used only for Core ML / Neural Engine probes |

Run **all** compute on aurora. The Mac is a laptop and is not the training box.

Reach aurora as `ssh aurora`. Two Windows partitions may be mounted read-only
at `/mnt/windows-data` (C:) and `/mnt/windows-d` (D:); the D: one holds the
older `FSR-Mamba` research project, which is a useful source of prior results.

---

## 3. Repo map

```
extension/            the shipped Chrome extension
  manifest.json       MV3
  webgpu-sr.js        THE ENGINE. class WebGPUSR: WGSL builders, weight
                      loading, pipelines, render(). This file is the project's
                      main asset -- see §6.
  content.js          video detection, overlay UI, frame-time governor,
                      auto-engage, passthrough
  background.js       settings hub (chrome.storage) + broadcast
  popup.html/js       settings UI
  models/*.bin+.json  weights + manifest {channels, scale, blocks}
training/
  model_span.py       SPAN-Lite: Conv3XC (reparameterised), 4 SPAB blocks,
                      PixelShuffle
  train_span.py       trainer
  dataset.py          degradation pipeline
  codec_degrade.py    NEW -- real-codec degradation, see §8
  losses.py           Charbonnier + VGG perceptual + FFT
  export_webgpu_weights.py   checkpoint -> .bin + .json the engine loads
  eval/
    stratified_eval.py       THE EVAL HARNESS -- see §5
    fsrmamba_metrics.py      metrics lifted from the FSR-Mamba project
dev/                  browser benchmark harnesses (perf.html, bench.html)
```

Data and artefacts live on `/tank/webvsr`, never in the repo:
`clips/` eval video, `train_hr/` + `val_hr/` (9500/500 symlink split of
`data/vimeo_frames`), `ckpt_c16_codec/` current run, `eval_crf*.json`,
`evidence/` saved comparison images.

---

## 4. The problem being solved

**Symptom, reported by the user:** output looks good on text and simple
content, then "completely falls apart" the moment a scene gets busy. Faces
could be better.

**Measured, on the eval harness (this is the finding that matters):**

```
CRF 28, four clips, PSNR by content complexity (dB)
model            flat        edge     texture         all
bicubic         40.05       36.33       23.66       27.81
c16             40.08       36.74       24.61       28.77
```

The advantage over bicubic *shrinks as compression gets worse*:

| CRF | c16 texture PSNR advantage | c16 DISTS advantage |
|-----|---------------------------|---------------------|
| 20  | +1.16 dB                  | +0.0105             |
| 28  | +0.95 dB                  | +0.0069             |
| 36  | +0.59 dB                  | +0.0017             |

That is backwards from the product claim. The extension says it "shines where
plain math upscaling can't: removing compression artifacts", but it helps
*least* exactly where compression is worst.

**Diagnosis:** domain mismatch. The model trains on still images degraded with
JPEG and validates on DIV2K. It is deployed on codec-compressed video. Three
independent sources say fixing the training target beats any architecture
change:

- FSR-Mamba (the user's own prior project): fixing the supervision target "was
  worth more than kernel prediction, Swin, FiLM, multi-path data, capacity,
  distillation and adversarial training **combined**".
- arXiv:2602.11339 — retraining **SPAN**, this exact architecture, on real
  compressed pairs instead of DIV2K bicubic: **32.645 -> 33.511 dB**.
- Real-CUGAN is byte-identical to waifu2x-CUNet, retrained on domain data,
  and visibly better.

That is what the current experiment tests. See §8.

---

## 5. The eval harness — how quality is judged

`training/eval/stratified_eval.py`. Run it on aurora:

```
cd ~/dev/webvsr/training/eval
../../.venv/bin/python stratified_eval.py \
  --clips /tank/webvsr/clips \
  --model c16=../../checkpoints_c16/best_phase1.pth:16:2 \
  --scale 2 --height 1024 --crf 28 --frames 32 \
  --json /tank/webvsr/eval_out.json
```

It differs from the old `evaluate.py` in two ways that matter:

1. **Ground truth is real video.** An HR clip is downscaled and re-encoded with
   x264 to make the LR input, so the degradation has genuine inter-frame codec
   artifacts, not per-image JPEG.
2. **Pixels are bucketed by local gradient energy** into flat / edge / texture
   using global percentile thresholds, and every metric is reported per bucket.
   A whole-image mean cannot see the failure this project has.

Metrics, and how to read them:

| Metric | Meaning |
|---|---|
| PSNR per bucket | fidelity. **texture** is the column that matters |
| sharpness ratio | mean SR gradient / mean GT gradient. 1.00 matches GT, <1 over-smoothed, >1 over-sharpened. Only counts pixels where GT gradient >= 0.01, because a near-flat bucket otherwise divides by ~0 and reports ratios in the thousands |
| DISTS | perceptual distance, lower better. **This is the selection metric** |
| tLP | LPIPS between consecutive output frames minus the same for GT. Flicker. <= 0 is ideal; positive means the model flickers more than the source |

**Never select a checkpoint on PSNR.** PSNR is minimised by the conditional
mean, which is the blurry answer. The trainer now selects on DISTS and falls
back to PSNR only if DISTS will not import. Fixing this one line was worth
more than quadrupling the model; every run before it had been shipping its
blurriest good checkpoint.

**Validation must use the same degradation as training.** It did not until
2026-09-01, and the cost was concrete: a 200-epoch run improved bicubic-val
DISTS 0.0656 -> 0.0587 while its DISTS on codec-degraded video got *worse*,
0.0988 -> 0.1023. `ValidationDataset` takes `degrade_fn` and seeds it per
index so image i draws the same codec and quality every epoch; without that
seeding, "best checkpoint" drifts toward whichever epoch drew easy encoders.

Always keep the visual evidence. Comparison crops go to `/tank/webvsr/evidence`
and are never deleted.

---

## 6. The engine, and why not to rewrite it

`extension/webgpu-sr.js` runs the whole network as compute passes in a single
command encoder: `importExternalTexture` (zero copy from the video) -> pre ->
conv_first -> 4x SPAB -> conv_last -> conv_cat -> upsampler -> PixelShuffle ->
optional Catmull-Rom finish -> optional RCAS sharpen -> canvas.

The conv kernel is the crown jewel. Each thread computes a **2x2 output block
x 8 output channels = 32 statically-named register accumulators**, loading a
4x4 input patch once per input channel. Naive was ~460 ms at 720p; this is
21.8 ms. Two hard-won rules live in it:

- Never dynamically index a private array in WGSL — it spills to local memory
  and costs 5x. Use statically-named accumulators.
- Scalar f16 is *not* faster than f32 on Turing. `USE_F16` is implemented and
  deliberately off. Packed `vec2<f16>` is the untried real 2x.

Any proposal that discards this kernel — a transformer, Mamba, a full U-Net
rewrite — is throwing away the project's main asset and restarting the
optimisation from zero. The bar for that is very high and has not been met.

---

## 7. Settled decisions — do not re-propose these

| Decision | Evidence |
|---|---|
| **No GAN / adversarial training** | A prior ESRGAN fabricated fence lines. The perception-distortion bound makes it explicit: moving far along the curve *means* synthesising unverified detail |
| **No contextual loss / CoBi** | Matches feature *distributions* not locations, so it relocates and substitutes texture — the exact fabrication mode being avoided |
| **No per-pixel mixture-of-experts** | Failed across ~27 FSR-Mamba versions: without load balancing it starved 2 of 4 experts, with it the router went uniform |
| **No depthwise-separable convs** | Representation-limited at 32-48 channels; NTIRE runtime tracks favour dense 3x3. Wins on params, loses on latency |
| **No NR metrics as a training objective** | Fine-tuning against MANIQA/HyperIQA/DBCNN/NIMA produced adversarial colour-dot and green-edge artifacts. CLIP-IQA read-only is fine |
| **No NPU / WebNN path yet** | Measured on an M4 Pro via Core ML: GPU 10.56 ms, Neural Engine 12.83 ms, ALL 21.93 ms. The NPU is *slower* for this network, because a 16-channel 3x3 conv is a tiny memory-bound GEMM. WebNN is also origin-trial-only, production ~2027 |
| **No frame interpolation yet** | Costs about what SR costs on an already-saturated GPU, needs a ~20 GB Vimeo re-download, and requires running the canvas a frame behind an audio clock the extension does not own |
| **Sharpening stays at 1.4** | User's explicit decision. It is doing visible work. The harness reports a sharpness ratio if it is ever revisited |

Anime support, when it happens, is a **separately trained model behind a manual
toggle**, not a learned router. The engine already hot-swaps models by
manifest. `Anime4K-WebGPU` (npm `anime4k-webgpu`) is a working reference.

---

## 8. Current experiment — codec-realistic retraining

`training/codec_degrade.py` round-trips frames through real H.264/H.265/MPEG-4
encoders in-process with PyAV. This is APISR's observation: you do not need
video to get codec artifacts, because a single frame pushed through a video
encoder still gets the transform, quantisation and in-loop deblocking. Median
~19.5 ms per sample, cheap enough to run on-the-fly in the data loader.

Two things to know if you touch it:
- x265 writes its banner straight to stderr and bypasses libav's logger, so
  `av.logging.set_level` cannot mute it. Only `x265-params=log-level=none` can.
- x265 is ~4x the encode cost of x264, so `CODEC_WEIGHTS` keeps it a garnish.

Wired in via `SRDataset(degrade_fn=...)` and `train_span.py --codec-degrade`.
The trainer also gained `--init-from` (weights-only warm start; `--resume`
restores the epoch counter and would exit immediately) and CLI overrides for
the dead Windows data paths.

**Concluded 2026-09-01. It shipped.** The winning recipe, ~41 s/epoch:

```
.venv/bin/python -u training/train_span.py --channels 16 --scale 2 \
  --ckpt-dir /tank/webvsr/ckpt_c16_sharp2 \
  --train-dirs /tank/webvsr/train_hr --val-dir /tank/webvsr/val_hr \
  --codec-degrade --w-fft 0.5 --init-from checkpoints_c16/best_phase1.pth \
  --total-epochs 120 --phase1-epochs 120 --num-workers 8
```

Two changes carried it, and neither was architectural: DISTS selection
instead of PSNR, and `w_fft` 0.01 -> 0.5, which ships at a weight low enough
to be effectively off. Against the model it replaced:

| CRF | texture PSNR | DISTS |
|-----|--------------|-------|
| 20  | 25.84 -> 26.37 | 0.0986 -> 0.0938 |
| 28  | 24.61 -> 25.05 | 0.1248 -> 0.1206 |
| 36  | 22.39 -> 22.69 | 0.1650 -> 0.1625 |

**The test it did not pass is the slope.** The advantage over bicubic still
shrinks as compression worsens -- texture +1.69 / +1.40 / +0.89 dB at CRF
20 / 28 / 36 -- which is backwards from the product claim. Six runs across
capacity, degradation, loss weighting, selection and trunk layout have all
failed to bend it. Treat it as structural until something demonstrates
otherwise, and do not spend another capacity run on it.

An intermediate candidate (`ckpt_c16_sharp`, "sharp16") beat the shipped
model on texture while *regressing* flat and edge below it. Whole-image
means hid that; the per-bucket table caught it. Always read all three
buckets before shipping a checkpoint.

---

## 9. Parameter counts, since three circulate

| Count | What it is |
|---|---|
| 33,388 | **deployed** — after Conv3XC reparameterisation. `span_lite_2x_c16.bin` is 133,552 bytes / 4 |
| 142,852 | trainable during training (Conv3XC branches, `requires_grad=True`) |
| 173,460 | all tensors including the frozen `eval_conv` copies |
| 128,716 | the **32-channel** model, `span_lite_2x.bin`. This is where the wrong "128K" for c16 came from |

---

## 10. Known outstanding issues

- Adding a PSNR floor and a tLP ceiling to DISTS selection is still open.
  Selection is on DISTS alone today.
- Training converges far sooner than the schedule assumes. Under corrected
  validation, DISTS moved 0.2487 -> 0.2252 in five epochs and then sat in a
  +/-0.003 band for the remaining 105. Long runs are buying noise; shorten
  them before spending GPU time on anything else.
- Eval clips are 3 rendered UE5 scenes + 1 real video. The rendered ones
  outvote the real one, and the model behaves differently on them. **More real
  video clips are the single biggest improvement available to the harness.**
- `~6 px edge darkening at 4x` — zero-padding conv artifact, unfixed.
- The extension's UI has never been validated in a real Chrome session; aurora
  has no browser and no display.

The 4x model (`span_lite_4x_c16.bin`) is still from the old JPEG-degraded
recipe; only the 2x default has been retrained. `span_lite_2x_c16p2.bin` is
in the package but referenced by no code.

---

## 11. Conventions

- **Commits carry no co-author or generated-by trailers on this project**, from
  any tool. Every commit is authored by `fishy-ops` and nothing else. This is a
  standing instruction.
- Generated code is verified by **running** it — typecheck, import, execute —
  never by reading it back. If it fails, then open the file.
- Do not delete benchmark images or eval JSON. They are the evidence.
- Prefer editing on aurora over copying files back and forth.
