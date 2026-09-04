# WebVSR — research findings

Written 2026-09-02, from a grounded literature sweep (arXiv retrieval, not model
recall) plus direct reading of the NTIRE 2026 challenge report. Every claim here
names its source. Read `CONTEXT.md` first; this file is the *why* behind several
of the decisions recorded there.

---

## 1. The slope problem is answered, and it is not a bug

**The finding.** This project's advantage over bicubic shrinks as compression
worsens — texture PSNR +1.69 / +1.40 / +0.89 dB at CRF 20 / 28 / 36. Six
training runs varying capacity, degradation, loss weighting, checkpoint
selection and trunk layout failed to bend it.

**Why.** A benchmark of SR models on compressed video (arXiv:2305.04844) finds
that "many SR models are unable to deal with compression artifacts", and that
generative models such as Real-ESRGAN do *better* at low bitrates specifically
because they "generate plausible details rather than restoring original
content".

That is the whole mechanism. At CRF 36 the high-frequency information has been
quantised away; it is not in the input. A model that only restores has nothing
left to restore, so its advantage over a smooth baseline must fall toward zero.
The only thing that still helps at that point is **synthesising** detail that
was never there.

**So the slope is the perception–distortion bound, showing up as a product
constraint.** `CONTEXT.md` §7 rules out GANs precisely because a prior ESRGAN
fabricated fence lines. The slope is the price of that decision, and it is the
correct decision. Treat the slope as closed:

> Do not spend further capacity, loss-weighting or architecture runs on the
> slope. It is not reachable without fabricating detail, which is forbidden.

**Two things that follow.**

1. **The product claim is backwards and should be corrected.** The extension
   says it "shines where plain math upscaling can't: removing compression
   artifacts". Measurement says it helps *least* where compression is worst.
   Fixing the wording is free and honest.
2. **One non-fabricating lever remains: degradation-aware conditioning.**
   Telling the model how compressed its input is, rather than making it infer
   this blind, recurs in the literature — hierarchical encoding across
   quantisation parameters (arXiv:2506.14381) and soft defect masks guiding
   temporal fusion (arXiv:2607.21219). It cannot recover destroyed
   information, but it should stop one model having to compromise across the
   whole CRF range at once. Untested here.

---

## 2. Efficiency: the bottleneck is memory bandwidth, confirmed externally

The NTIRE 2026 Efficient SR challenge (arXiv:2604.03198) runtime track was won
by **XiaomiMM's SPANV2** — the same SPAN family this project uses. Their stated
conclusion:

> the performance bottleneck lies not in FLOPs but in **memory bandwidth**.

This independently confirms `CONTEXT.md` §6's own diagnosis that a 16-channel
3×3 conv is a tiny memory-bound GEMM. It also says where to spend effort.

**Their three changes, and what each means here:**

| SPANV2 change | Relevance to this project |
|---|---|
| `span_attn_op` fuses the 1×1 attention conv, an add and a multiply into **one CUDA kernel, eliminating 3× redundant DRAM round-trips** | The direct analogue is **pass fusion in WGSL**. The engine runs pre → conv_first → 4× SPAB → conv_last → conv_cat → upsampler as separate passes, and every pass is a DRAM round-trip. This is the highest-value untried efficiency work, and it is now evidence-backed rather than speculative |
| Parameter-free attention replaced by a **learned 1×1 projection** to a full C×C channel-mixing map, +C² per block | Implemented here as `--arch spanv2`. On C=16 that is 1,024 parameters, ~3% of the deployed 33,388 |
| **Nearest-neighbour-initialised** parallel branch on the upsampler, letting the deep branch learn only residual detail | Untried. Note it uses a depthwise branch; `CONTEXT.md` §7 rules out depthwise *in the trunk* on representation grounds, which is a different argument from an upsampling branch |

**Recurring patterns across the winning entries:** pruning plus distillation to
recover the loss; multi-branch reparameterisation collapsed to a single 3×3 at
inference (this project already does this via Conv3XC).

**Settled negatively:** state space models (Mamba) were tried by two teams and
showed "relatively high runtimes compared to top-ranked entries". Given a prior
FSR-Mamba project exists, this is worth knowing before revisiting: **Mamba is
not the efficiency answer for this size of network.**

---

## 2a. How bandwidth-bound this engine actually is

Counted from the pass graph in `webgpu-sr.js` (`this.passes`): 17 conv-like
passes each reading and writing a full C-channel feature buffer, plus 4
attention passes reading two and writing one. At C=16 and f32 a single feature
buffer is `16 * px * 4` bytes, and nothing close to it fits in L2 (59 MB at
720p against 4 MB of L2 on a 2070 SUPER), so this traffic really does reach
DRAM.

| Neural input | One buffer | Intermediate traffic / frame | At 60 fps |
|---|---|---|---|
| 720p | 59 MB | 2,713 MB | **163 GB/s** |
| 1080p | 133 MB | 6,105 MB | **366 GB/s** |

Against 448 GB/s on the RTX 2070 SUPER and 273 GB/s unified on an M4 Pro.

**This changes the f16 decision.** `USE_F16 = false` records that scalar f16 was
no faster than f32 -- but that was measured at 720p, which is 36% of the 2070
SUPER's peak bandwidth and therefore not bandwidth-bound; halving the bytes of
a workload with bandwidth to spare buys nothing.

`NEURAL_CAP` is now 1080. At 1080p the same engine sits at **82% of peak on the
same GPU**, and *above* the M4 Pro's total bandwidth. f16 halves every one of
those bytes.

> Testable prediction: f16 should now help at 1080p on the very GPU where it did
> not help at 720p, because raising the cap moved the workload across the
> roofline. `dev/gpu_probe.html` measures both resolutions and both precisions.
> If this is wrong, the roofline reasoning is wrong and should be recorded as
> such.

**Fusing the attention passes** -- SPANV2's exact optimisation, and here the
`attn` pass is already the add-and-multiply -- removes each block's write of
`out3` and the attention's read back of it: 2 buffers per block across 4
blocks, **17% of all intermediate traffic**, independent of precision. It
composes with f16 rather than competing.

Caveats: these are analytic figures, not measurements. They ignore weight
traffic (133 KB, cached) and assume no cache reuse between passes, which is
close to true at these buffer sizes but not exactly true. Treat them as the
shape of the problem and as a prediction to falsify, not as a benchmark.

---

## 2b. WebGPU-specific evidence for fusion, and a dispatch-count cost

A study of WebGPU inference overhead (arXiv:2604.02344) measures what the
browser layer itself costs, which no SR paper covers:

- Per-dispatch API overhead: **24-36 us on Vulkan, 32-71 us on Metal.**
- **Kernel fusion improved throughput by 53%** on Vulkan in their WebGPU
  context. CUDA-style fusion gave no benefit there, so the win is specific to
  reducing WebGPU dispatches, not to fusion in the abstract.
- At batch size 1, per-operation overhead dominates *regardless of kernel
  quality*. This engine is effectively batch size 1.
- A reference WebGPU implementation reached 11-12% of CUDA performance. Useful
  for calibrating expectations about any browser-side number.

This engine issues **22 dispatches per frame** (pre, 20 in `this.passes`,
shuffle). On Metal that is 0.7-1.6 ms of pure API overhead against a 21.8 ms
720p budget -- not dominant, but not nothing, and it falls as passes are fused.

**Three independent lines now point at fusion**: the traffic count in section
2a (17% of intermediate bytes), SPANV2's fused CUDA kernel winning the NTIRE
runtime track, and a measured 53% from fusion in WebGPU specifically. Fusing
the four attention passes into the convolutions that feed them addresses all
three at once.

One methodological warning worth carrying: they find naive single-operation
benchmarks **overestimate dispatch cost by about 20x** versus sequential
dispatch. `dev/gpu_probe.html` times whole `render()` calls, which is the
sequential case, so its numbers should not inherit that error.

---

## 3. If an "advanced mode" is ever built

For users with compute to spare, the evidence points at transformers, not at a
bigger convnet, and specifically at ones designed for *compressed* input:

- **FTVSR** (arXiv:2212.14046) — self-attention over a combined space-time-
  **frequency** domain, explicitly to separate real texture from compression
  artifacts. The most on-domain result found.
- **Swin2SR** (arXiv:2209.11345) — compressed-image SR, top-5 in its challenge,
  explicitly targets compression artifacts.
- **VSR-HE** (arXiv:2506.14381) — hierarchical encoding transformer for H.265
  artifacts across quantisation parameters; 180p→720p, 270p→1080p.

For cheap temporal information, **SWRN** (arXiv:2208.11608) runs three
neighbouring frames plus a hidden state at real-time rates on mobile. Note the
warning from arXiv:2206.07687: in recurrent VSR, pruning error is *amplified as
the hidden state propagates*, so a pruned recurrent model needs temporal
finetuning.

**Caveat on all of the above:** none of these report browser or WebGPU numbers,
and none were measured against this engine. They are candidates, not plans, and
`CONTEXT.md` §6's bar for discarding the existing kernel still applies.

---

## 3a. The one idea that addresses efficiency and the slope together

**Adaptive depth driven by estimated compression level.** Run fewer SPAB blocks
on lightly-compressed video and more on heavily-compressed video.

**Supporting evidence.** A blind compressed-video enhancement method
(arXiv:2511.16137) learns a degradation representation, because the
quantisation parameter is usually unavailable, then uses "a sequential
inference strategy that adaptively adjusts the number of artifact reduction
stages according to the estimated compression level". It raises PSNR gain from
0.31 to 0.65 dB at QP 22 over blind methods **and cuts average inference time
by 50%**. Same mechanism, both problems.

It also fits this codebase: `content.js` already runs a governor that adapts
internal *resolution* to the frame-time budget. Adapting *depth* is the same
control one level down.

**The caveat that decides whether it works.** The early-exit literature is
sobering: exiting at the first *correct* exit is worth ~10% accuracy under
corruption, but realistic confidence-based strategies deliver only **~1%**,
because confidence is badly calibrated under distribution shift
(arXiv:2212.01562). Exit mechanisms studied are confidence thresholds,
conformal risk control, and learned gates (arXiv:2602.03043, arXiv:2506.21103).

**Why this project is not in that trap.** Those results are about routing on
the *model's own confidence*, which is exactly the unreliable signal. Here the
routing signal is **exogenous and directly measurable**: how compressed the
input is. The browser knows the video's resolution, and decoded-byte counters
give a bitrate estimate over time. A cheap blockiness measure on the decoded
frame is another option that never asks the network what it thinks.

So the honest position: the *mechanism* has strong support and the *routing
signal* is better-posed here than in the papers reporting 1%. That is a reason
to try it, not a reason to assume it works.

**What it needs.** A model trained to be truncatable, since blocks cannot
simply be dropped from the current one -- `conv_cat` consumes features from
blocks 1, 3 and 4 by construction, so skipping a block leaves it reading
tensors that were never produced. Either train early-exit heads jointly, or
train with stochastic depth. Note the literature does not settle whether a
truncatable model loses quality at full depth: the slimmable-network paper
retrieved reports its widest configuration but no single-width baseline
(arXiv:2605.22677), so that control has to be run here.

---

## 4. Training schedule: current runs are far too long

Two independent runs under corrected codec-domain validation:

| Run | Converged by | What the rest bought |
|---|---|---|
| 2× (`ckpt_c16_sharp2`) | ~epoch 15 | DISTS sat in a ±0.003 band for 105 more epochs |
| 4× (`ckpt_c16x4_sharp`) | ~epoch 35 | DISTS got **worse** — 0.2774 at ep35, 0.2813 by ep80 |

The 4× case is the stronger evidence: the extra epochs actively degraded the
selection metric. **Use 40–60 epochs, not 120**, and spend the saved GPU time on
more experiments rather than longer ones.

---

## 5a. SPANV2 attention: tested here, does not transfer

Trained 60 epochs, warm-started from the shipped 2× model with the attention
projection identity-initialised, so it began numerically equal to it.

| CRF 20 (all clips) | texture | DISTS | flat | edge |
|---|---|---|---|---|
| sharp2 | **26.37** | 0.0938 | **42.31** | **38.91** |
| spanv2 | 26.34 | **0.0934** | 41.92 | 38.38 |

At CRF 36 it also loses on DISTS (0.1627 vs 0.1625). Validation agreed: DISTS
moved 0.2206 to 0.2204 across the whole run, which is noise. Not shipped.

The identity init is what makes this conclusive. The model started at exactly
the shipped quality, so 60 epochs of strictly greater freedom yielding nothing
is a statement about the change itself.

**This sharpens rather than weakens section 2.** SPANV2 won the *runtime*
track, and the report's own insight is memory bandwidth, not FLOPs -- the win
came from kernel fusion. The learned attention rode alongside it. The quality
half is now tested and empty here; **pass fusion is the part that has not been
tried, and it remains the lever.**

---

## 6. The 2× recipe does not transfer to 4× as-is

The 4× retrain (`ckpt_c16x4_sharp`) used the identical recipe that won at 2×,
and **lost to the shipped 4× model on every metric at every CRF**:

| CRF | texture PSNR | DISTS | tLP |
|-----|--------------|-------|-----|
| 20 | 20.29 → 20.01 | 0.1922 → 0.1926 | −0.0075 → −0.0030 |
| 28 | 19.76 → 19.54 | 0.2109 → 0.2122 | −0.0090 → −0.0055 |
| 36 | 18.53 → 18.48 | 0.2419 → 0.2430 | −0.0142 → −0.0122 |

It is blurrier (texture sharpness ratio 0.576 vs 0.606 at CRF 20) and flickers
more. It was not shipped.

**The tell is that validation and the harness disagreed again.** Validation
DISTS improved 0.2904 → 0.2774 while harness DISTS worsened — the same shape as
the bug fixed in §5 of `CONTEXT.md`, so a second domain gap is still open at 4×.

**Most likely cause: LR crop size.** `crop_size` is in HR pixels, so the LR crop
the model sees is `crop_size / scale`:

| Run | crop_size | LR crop trained on | LR the harness feeds | Outcome |
|---|---|---|---|---|
| 2× | 256 | **128** | 256 | won |
| 4× | 256 | **64** | 256 | lost |

At 4× the model trained on a quarter of the context it was judged on, while the
2× run trained on half. `--crop-size` now exists for exactly this; the retry
uses 512 (LR 128) to match the run that worked.

**General lesson: when changing scale, hold the LR crop constant, not the HR
crop.** Everything the network sees is on the LR side.

---

## 6a. The crop-size retry is invalid: the dataset cannot supply a 512px crop

The retry ran to completion (`ckpt_c16x4_crop512`, 60 epochs, phase-1 best
26.43 dB) and again lost to the shipped model:

| CRF | texture PSNR | DISTS | tLP | texture sharpness |
|-----|--------------|-------|-----|-------------------|
| 20 | 20.283 -> 20.100 | 0.1922 -> 0.1915 | -0.00749 -> -0.00740 | 0.6064 -> 0.5829 |
| 28 | 19.758 -> 19.599 | 0.2109 -> 0.2113 | -0.00902 -> -0.00917 | 0.5852 -> 0.5522 |
| 36 | 18.529 -> 18.508 | 0.2419 -> 0.2426 | -0.01421 -> -0.01526 | 0.5200 -> 0.4915 |

**This does not falsify the hypothesis in §6 — the experiment never tested it.**

Every training frame is Vimeo-90K at **448x256** (10,000 frames, `data/vimeo_frames`).
`random_crop` (`training/dataset.py:19`) upscales any frame smaller than the
requested crop rather than refusing it:

```python
w, h = img.size                              # 448 x 256
if w < crop_size or h < crop_size:           # 448 < 512 -> True
    img = img.resize((max(w, crop_size), max(h, crop_size)), Image.BICUBIC)
```

So `--crop-size 512` bicubically stretched every frame to 512x512 — **1.14x on x,
2.00x on y** — and used that as the HR *target*. The run trained a
super-resolution model to reproduce bicubic upsampling, on geometrically
distorted frames.

The measured artifact is precisely what that bug predicts: crop512 is blurrier
than shipped at every CRF, because bicubic blur is what its targets contained.
Its DISTS/tLP wins at CRF 20 are 0.0007 and 0.0001 — noise.

**The real constraint is the dataset ceiling: max usable crop = min(448, 256) = 256 HR px.**

| scale | HR crop needed for LR 128 | available from Vimeo-90K |
|---|---|---|
| 2x | 256 | yes — this is why the 2x run worked |
| 4x | 512 | **no** |

The shipped 2x recipe sits exactly on that ceiling by luck, not design. The 4x
LR-crop hypothesis remains **untested and still plausible**; testing it needs
source frames of at least 512x512.

**Consequences:**
1. `random_crop` must raise on an oversized crop instead of silently
   fabricating HR detail. A trainer that invents its own targets from bicubic
   can only teach blur, and it does so without a single warning in the log.
2. **Training data must be re-sourced at >=512px before any further 4x recipe
   work.** Vimeo-90K cannot express the experiment, and it also caps the busy-scene
   work in §3a: 448x256 frames carry little of the dense high-frequency texture
   that the slope problem is about.

---

## 7. The busy-scene failure is real, and it is over-sharpening — not blur

> **RETRACTED in part — see §9.** The busyness trend below was measured on a
> 4-clip set in which busyness was perfectly confounded with clip identity, a
> limitation flagged at the time. On a 15-clip set spanning busyness 0.004-0.684
> the trend disappears (`corr(busyness, gain) = -0.297`, bins non-monotonic) and
> the two worst clips are among the *least* busy. The over-sharpening
> observation survives; the busyness explanation does not.

`training/eval/busy_eval.py` makes the *frame* the unit instead of the pixel.
`stratified_eval` pools every frame's texture pixels into one number, which
cannot distinguish a calm frame with a small detailed region from a frame that
is detailed everywhere — exactly the distinction the complaint is about. Each
frame is scored by **busyness** (fraction of pixels above the global texture
threshold) and frames are binned by it.

Shipped 2x c16, 160 frames, 4 clips, CRF 28:

| bin | busyness | n | bicubic tex PSNR | shipped | gain | sharp bicubic | sharp model |
|-----|----------|---|------------------|---------|------|---------------|-------------|
| 0 | 0.097-0.121 | 32 | 21.185 | 22.604 | **+1.420** | 0.8191 | 0.8609 |
| 1 | 0.121-0.268 | 32 | 21.937 | 23.351 | **+1.414** | 0.8294 | 0.8667 |
| 2 | 0.268-0.293 | 32 | 23.259 | 24.655 | **+1.396** | 0.8230 | 0.8611 |
| 3 | 0.293-0.677 | 32 | 24.531 | 25.296 | +0.765 | 0.8101 | 0.8857 |
| 4 | 0.677-0.695 | 32 | 23.988 | 23.413 | **-0.575** | 0.8007 | 0.9498 |

**The complaint reproduces.** The gain is flat at ~+1.4 dB up to a third of the
frame being textured, halves by bin 3, and goes **negative** in bin 4 — on that
content the model is measurably worse than doing nothing.

**The mechanism is the opposite of what "falls apart" usually means.** In bin 4
the model's sharpness ratio is **0.9498 against bicubic's 0.8007** — it emits
*more* gradient energy than the baseline while losing 0.58 dB. It is not mushing
detail; it is manufacturing detail in the wrong places. That is why it looks bad
enough to switch off: wrong high-frequency structure is far more objectionable
than softness, and PSNR understates how bad it looks.

Note this is the same failure the project already rejected GANs for
(`CONTEXT.md` 7, fabricated fence lines) — appearing in a non-GAN model, driven
by content density rather than by the loss.

### What this does not yet establish

**Busyness is confounded with clip identity, and the confound is total.**

| clip | n | busyness range | mean gain | model sharpness |
|---|---|---|---|---|
| bistro_30s | 40 | 0.097-0.134 | +1.413 | 0.8621 |
| chess_30s | 40 | 0.287-0.305 | +1.304 | 0.8510 |
| locomotive_30s | 40 | 0.228-0.276 | +1.444 | 0.8694 |
| vsr_test_video | 40 | 0.675-0.695 | **-0.625** | **0.9568** |

Each clip spans only 0.018-0.048 of busyness, so bin 4 *is* `vsr_test_video` and
nothing else. Within-clip correlations between busyness and gain are weak and
inconsistent in sign (+0.17, -0.06, +0.53, +0.38) over ranges too narrow to
mean anything.

So the honest claim is: **one clip out of four is catastrophically bad, it is
also by far the busiest, and its failure mode is excess fabricated
high-frequency detail.** Whether busyness *causes* this, or merely correlates
with whatever that clip contains, needs clips that span 0.1-0.7 busyness
*within* the same evaluation. The current 4-clip set cannot separate them, and
no conclusion should be built on the busyness axis until it does.

---

## 8. Training data is the binding constraint, and the fix is already on disk

Three separate problems above reduce to one cause:

- 6a: the 4x LR-crop experiment cannot run — it needs 512px HR crops, and
  Vimeo-90K is 448x256.
- 7: the busy-scene axis cannot be tested — it needs clips spanning a wide
  busyness range.
- Training happens on 448x256 frames while evaluation runs at 1080p, so the
  model is judged far outside the resolution it was fitted on.

**`/tank/webvsr/datasets/DIV2K_train_HR` already holds 800 images (3.4 GB) with
a median short side of 1356 px and a minimum of 1140 — every one of them
supports a 512 crop, and most support 1024.** DIV2K_valid_HR adds 100 more.
They were downloaded and then never wired into training: `CONFIG["train_dirs"]`
points only at the Vimeo symlinks.

That makes the 4x retry runnable today with no download at all. The caveat is
that DIV2K is stills, so it carries no temporal signal and none of the codec
artifacts the codec-domain degradation was built for — it should be *mixed*
with the video frames, not swapped for them.

Downloading real high-resolution video is still worth doing, but for section 7
rather than section 6a: what is missing there is busyness *coverage* across many
clips, which stills cannot provide and which 4K adds nothing to on its own — a
1080p clip already exceeds the 512 crop requirement several times over.

---

## 9. The +1.4 dB advantage is a property of three benchmark clips

Section 7's confound was resolved by adding 11 Xiph/derf clips chosen to span
scene busyness, giving 15 clips over busyness 0.004-0.684. Re-measuring the
shipped 2x model at CRF 28, 480 frames, `--height 1024` (no HR enlargement):

| clip set | n | mean gain vs bicubic | sd |
|---|---|---|---|
| the 3 original render clips | 96 | **+1.383 dB** | 0.085 |
| `vsr_test_video` | 32 | -0.416 dB | — |
| **the 11 newly added clips** | 352 | **-0.077 dB** | 0.345 |
| all 15 | 480 | +0.192 dB | 0.673 |

**On the new clips the model is worse than bicubic on 162 of 352 frames.** Its
measured advantage is not a general property of the model; it is a property of
`bistro_30s`, `chess_30s` and `locomotive_30s`, on which it is remarkably
consistent (sd 0.085) and remarkably good.

### What separates those three clips

Four candidate explanations were tested and three were rejected outright:

| candidate | test | verdict |
|---|---|---|
| busyness | 15 clips over 0.004-0.684 | **rejected**, corr -0.297, non-monotonic |
| HR reference enlarged by the harness | re-ran at `--height 1024` | **rejected**, the three still win by ~+1.4 |
| HR already codec-compressed | 8x8 block-energy ratio | **rejected**, those three are the *least* block-aligned |
| synthetic / denoised source | noise sigma in flat regions | **not rejected**, but incomplete |

The three winners measure **exactly 0.0000** noise in flat regions, as does
`dinner_1080p30` — which loses 0.475 dB. So zero noise is necessary-looking but
plainly not sufficient, and the honest position is that *the cause is not yet
identified*. What is established is the split itself, and that it is large,
consistent, and aligned with clip provenance rather than with any scene property
measured so far.

### Why this matters more than the busy-scene question

Every model comparison in this file — the slope in §1, the SPANV2 result in §5a,
both 4x retrains in §6 and §6a — was scored on `/tank/webvsr/clips`: those three
clips plus `vsr_test_video`. Their verdicts are internally consistent, because
all models were ranked on the same set. But the *magnitudes* are not
transferable, and any claim of the form "the extension gains ~1.4 dB" describes
three clips rather than the deployment domain.

The user-reported failure — content bad enough to switch the extension off — now
has a plausible reading that costs nothing to accept: on ordinary camera footage
the model was never delivering the benchmark's gain in the first place.

**Actions:**
1. Re-run the model comparisons that matter against `/tank/webvsr/clips_busy`
   (15 clips) before any further architecture work; the current selection signal
   is measured on unrepresentative content.
2. Do not quote +1.4 dB. On the broader set the shipped 2x model is
   -0.077 dB against bicubic on texture PSNR.
3. Identify what those three clips have that the others do not. Until that is
   known, neither set can be assumed to be the representative one.

---

## 10. Fusion and f16, measured — §2a and §2b are no longer analytic

Sections 2a and 2b were explicitly "the shape of the problem and a prediction to
falsify". Both predictions have now been run on hardware, in a browser, on an
**Apple M4 Pro (metal-3, `shader-f16` available)**, 1080p and 720p, 20 timed
iterations each, timed to `queue.onSubmittedWorkDone()` rather than to submit.

### Attention fusion: confirmed, and free

`c3` and the attention that consumes it are now one pass (`convattn`). Dispatches
per frame drop **22 → 18**.

| | 720p | 1080p |
|---|---|---|
| unfused f32 | 33.2 ms | 73.2 ms |
| fused f32 | 30.9 ms | 66.1 ms |
| speedup | **1.074x** | **1.107x** |

**Output is bit-identical: `max_abs_diff = 0` at both resolutions.** Same
arithmetic, same order, one less DRAM round-trip.

**On Turing the same change is roughly neutral.** Measured through Chrome 152 +
Xvfb + Vulkan on the RTX 2070 SUPER (the correctness check passes there: 22
passes unfused, 18 fused, both rendering):

| RTX 2070 SUPER | 720p | 1080p |
|---|---|---|
| unfused | 27.4 ms | 50.8 ms |
| fused | 24.7 ms | 52.0 ms |
| speedup | 1.109x | 0.977x |

Do not read the 1080p number as a regression. Two separate unfused runs measured
50.8 ms and 53.4 ms at 1080p — **~5% run-to-run spread, larger than the 2.4%
"slowdown"**. The honest statement is that fusion is a clear win on Apple
silicon, and within noise of neutral on Turing.

That split is consistent with §2b: per-dispatch overhead is 32-71us on Metal
against 24-36us on Vulkan, so removing four dispatches is worth roughly twice as
much on Apple. The traffic saving should help both, but it is evidently not
large enough on Turing to clear the noise floor at 1080p. Fusion stays enabled:
it is bit-identical, clearly positive on one GPU family, and not measurably
negative on the other. The gain is larger at 1080p,
which is the direction §2a predicts: the more bandwidth-bound the workload, the
more removing traffic is worth. §2a's estimate was 17% of *intermediate* traffic;
10.7% end-to-end is consistent, since pre, shuffle, upsampler, finish and
sharpen are untouched.

Fusing c3 changes which scratch buffer each block lands in, so the pass list is
re-planned rather than patched — naively fusing block 2 would have it read and
write `sB` in a single pass. `convAttn()` throws if an output aliases an input.

### f16: a large win here, but not for the predicted reason

| | 720p | 1080p |
|---|---|---|
| fused f32 | 29.8 ms | 66.2 ms |
| fused f16 | 21.4 ms | 48.0 ms |
| speedup | **1.393x** | **1.379x** |

Quality cost at 1080p on a synthetic frame: **`max_abs_diff` 1/255, mean
0.00315**.

That frame was flat colour and 9px squares, which is not evidence about video.
`dev/f16_quality.html` repeats the question properly: 24 frames from 6 clips
spanning busyness 0.005-0.68, each put through a real x264 encode at CRF 28 by
the same `make_pair` the evaluation harness uses, scored as **PSNR against the
ground truth** rather than against f32. Frames are cropped to the highest-
gradient 512px window, where quantisation error has the most to act on.

| clip | PSNR f32 | PSNR f16 | delta | max pixel diff |
|---|---|---|---|---|
| dinner_1080p30 | 39.5179 | 39.5235 | +0.0056 | 1 |
| bistro_30s | 29.2242 | 29.2236 | -0.0007 | 1 |
| blue_sky_1080p25 | 24.5215 | 24.5207 | -0.0008 | 1 |
| life_1080p30 | 25.2431 | 25.2432 | +0.0001 | 2 |
| park_joy_1080p50 | 20.3777 | 20.3779 | +0.0002 | 1 |
| crowd_run_1080p50 | 20.4312 | 20.4312 | 0.0000 | 1 |

**Mean delta +0.0007 dB; the worst clip loses 0.0008 dB; worst pixel 2/255.**
Both signs appear, which is what noise looks like rather than degradation.
f16 is quality-neutral on real video.

**The specific §2a prediction is not supported.** It said f16 should help at
1080p *and not at 720p*, because raising `NEURAL_CAP` moved the workload across
the roofline. f16 in fact helps slightly **more** at 720p (1.393 vs 1.379), i.e.
uniformly. On Apple silicon f16 also doubles ALU throughput and halves register
and cache pressure, so it pays whether or not DRAM bandwidth is saturated. The
roofline argument explains the *fusion* result well and the *f16* result poorly.

This does not overturn `USE_F16 = false`: that was measured on Turing, a
different GPU, and is not retested here. What it establishes is that the default
is wrong **for Apple silicon**, where f16 is the single largest efficiency win
available — 1.38x for a quality change of 0.0007 dB.

**Shipped enabled wherever `shader-f16` is reported.** The narrower option was
to gate on `adapter.info.vendor`, since Apple is the only family measured. That
was rejected on the asymmetry of the risk:

- **Quality does not depend on the precision.** That is measured, on real video,
  against ground truth — +0.0007 dB. It is not a per-GPU question.
- **Speed does vary**, 1.38x on Apple against no gain on Turing. But "no gain"
  leaves a GPU exactly where it already was; it is not a regression.
- **Memory always halves**, every feature buffer, which is pure benefit on the
  memory-tight GPUs most likely to be running this.
- A device without the feature takes the existing f32 path. Verified by
  intercepting `requestAdapter` to hide `shader-f16`: the engine initialises at
  4 bytes/scalar and renders a full frame correctly.

Turing remains unmeasured under the new default — the box that has one runs
headless with no WebGPU-capable browser. `F16_VENDORS` can be set back to a
vendor-substring list if a GPU is ever found where f16 is actively slower.

Stacked, on the M4 Pro at 1080p: **73.2 ms unfused f32 -> 48.0 ms fused f16, a
1.525x speedup** for zero perceptible quality change.

### The tooling was broken, and any earlier number from it is void

`dev/gpu_probe.html` could not have produced a valid measurement. Three
independent faults, all now fixed:

1. **It never called `configure()`**, so `render()` returned at its `!this.ctx`
   guard. It was timing a function that did nothing.
2. **It passed an `ImageBitmap` to `importExternalTexture`**, which accepts only
   a `VideoFrame` or `HTMLVideoElement`.
3. **It reloaded the engine with a `<script>` tag.** The engine declares
   top-level consts, so the second load threw `Identifier 'MEAN' has already
   been declared`, the reload silently did nothing, and the comparison measured
   the *first* build twice. Observed directly: the first fusion run reported 22
   passes for both builds, a 1.0 speedup and a 0-pixel difference — a perfect
   null result produced by comparing a build with itself.

Fault 3 is the dangerous one, because it fails as a *plausible* answer rather
than an error. The probe now loads the engine by evaluating its source in a
fresh function scope, and `window.fusion()` refuses to report at all when both
builds return the same pass count.

---

## 11. Re-scored on 15 clips: the perceptual win is real, the PSNR win is not

> **The model called "shipped" below is not the one the extension ships — see
> §14.** Every figure in this section describes
> `checkpoints_c16/best_phase2.pth`. The deployed binary was exported from a
> different checkpoint. The section's *reasoning* stands and its comparisons
> between candidates are still valid, but its magnitudes belong to a model users
> do not run.

Section 9's action item, executed. Shipped 2x against bicubic on the 15-clip set
at `--height 1024`, 32 frames per clip, full metric suite.

| CRF | texture PSNR | overall PSNR | DISTS | tLP | texture sharpness |
|-----|--------------|--------------|-------|-----|-------------------|
| 20 | +0.155 dB | +0.135 dB | **+8.9%** | +0.00235 | +0.0526 |
| 28 | +0.197 dB | +0.190 dB | **+6.1%** | +0.00299 | +0.0591 |
| 36 | +0.092 dB | +0.117 dB | **+3.7%** | +0.00354 | +0.0432 |

The model wins on every metric at every CRF. But the PSNR margin is a fifth of
what the old 4-clip set reported (+0.95 dB at CRF 28), while DISTS actually
holds up slightly *better* (+6.1% against +5.2%). Splitting by clip provenance
at CRF 28 shows why:

| group | n | DISTS | texture PSNR | tLP | DISTS wins |
|---|---|---|---|---|---|
| 3 render clips | 3 | **+16.2%** | **+1.323 dB** | +0.0030 | 3/3 |
| 12 real-camera clips | 12 | **+3.7%** | **-0.087 dB** | +0.0030 | 10/12 |

**Three separate conclusions, and they differ:**

1. **The PSNR advantage is entirely the render clips.** On real camera footage
   it is -0.087 dB — zero. Any claim of the form "+1.4 dB" describes synthetic
   content only. Section 9's finding stands.
2. **The perceptual advantage survives but shrinks 4x**, +16.2% to +3.7% DISTS,
   winning 10 of 12 real clips. Small, consistent, real. Two clips lose:
   `controlled_burn` (-8.4%) and `vsr_test_video` (-4.1%).
3. **The flicker advantage is completely unaffected by the split: +0.0030 on
   both groups.** It is the only metric that does not care whether the content
   is synthetic, which makes temporal stability the most transferable thing
   this model does — and it is not what the project has been selecting on.

**This resolves the question section 9 left open.** The model earns its GPU time,
but on perceptual and temporal grounds worth roughly 4% DISTS, not on the
distortion headline. That matches the original product goal — visibly better
rather than higher PSNR — so the honest framing is a modest perceptual and
temporal improvement, and PSNR should stop being quoted as the benefit.

---

## 12. Adaptive depth: a usable 41%-cheaper exit, and what it costs

Two multi-exit runs. The first co-adapted the shared trunk (8 frozen epochs, then
joint); the second froze the trunk for all 60 epochs and trained only the early
heads, on codec-domain degradation rather than the legacy JPEG chain.

| | take 1 d4 | take 2 d4 | take 1 d2 | take 2 d2 |
|---|---|---|---|---|
| DISTS @ CRF 28 | 0.1674 | **0.1629** | 0.1668 | **0.1661** |

**Freezing the trunk is the whole result.** In take 2 `d4` is *identical* to the
shipped model on every metric at every CRF — not close, identical — because the
deep head and trunk never moved. Take 1's co-adaptation cost the deep exit 2.8%
DISTS and bought the shallow exit essentially nothing. The early exit should be
built as an addition, never as a joint retrain.

Take 2, 15 clips, `--height 1024`:

| CRF | model | DISTS | vs bicubic | texture PSNR | tLP | sharpness |
|-----|-------|-------|------------|--------------|-----|-----------|
| 20 | bicubic | 0.1309 | — | 26.932 | -0.01421 | 0.7640 |
| 20 | shipped / d4 | 0.1192 | +8.9% | 27.087 | -0.01186 | 0.8166 |
| 20 | **d2** | 0.1234 | **+5.7%** | 26.691 | -0.01579 | 0.7197 |
| 28 | bicubic | 0.1735 | — | 24.985 | -0.01282 | 0.6783 |
| 28 | shipped / d4 | 0.1629 | +6.1% | 25.181 | -0.00982 | 0.7374 |
| 28 | **d2** | 0.1661 | **+4.3%** | 24.884 | -0.01439 | 0.6435 |

**The trade is slightly better than proportional**: the shallow exit keeps ~64%
of the perceptual gain for 59% of the compute.

**But it flickers more than bicubic** (-0.01439 against -0.01282 at CRF 28) and is
softer than bicubic too (0.6435 against 0.6783). Given §11 found flicker to be the
only advantage that transfers across content types, a cheap mode that *worsens*
it is a real trade. Depth 2 is a thermal/battery fallback, not a default.

### Two process findings from the same run

**Phase 1 improved PSNR and degraded perception, monotonically.** Over nine
validations the shallow exit went 28.87 -> 29.57 dB while its DISTS went
0.2382 -> 0.2436. Phase 2's perceptual loss reversed it, giving up 0.15 dB to
recover 0.007 DISTS. Selecting that run on PSNR would have picked its worst
perceptual checkpoint.

**Selecting on the worst exit goes degenerate when an exit is frozen.** `d4`'s
score never changes, so it is permanently the worst once `d2` passes it, and
selection stops tracking the head that is actually training — it was driven by
numerical jitter in the frozen exit (0.2388-0.2391 between identical
evaluations). It cost 0.0006 DISTS here, which is luck rather than design.
Selection should consider only the trainable exits.

---

## 13. Why the three render clips behave differently: bicubic rings on them

Section 9 left this open after rejecting busyness, harness enlargement and prior
codec compression. The answer was already in the eval output, in a column nobody
had read across clips: **what bicubic itself does on each one.**

Sharpness ratio against ground truth in the edge bucket, where 1.00 means the
output carries the same gradient energy as the reference:

| clip | bicubic | shipped | |
|---|---|---|---|
| locomotive_30s | **2.442** | 1.514 | render |
| bistro_30s | **2.313** | 1.614 | render |
| chess_30s | **1.873** | 1.382 | render |
| dinner_1080p30 | 1.020 | 1.056 | |
| vsr_test_video | 1.007 | 0.963 | |
| life_1080p30 | 0.833 | 0.694 | |
| ducks_take_off | 0.780 | 0.784 | |
| in_to_tree_1080p50 | 0.290 | 0.269 | |

**On the three render clips bicubic emits 1.9-2.4x the gradient energy of the
reference.** That is not softness, it is *ringing* — overshoot at edges. Every
real-camera clip sits at or below 1.0, where bicubic is merely soft, which is
what bicubic normally does.

The renders are anti-aliased synthetic images: their edges are smooth,
band-limited ramps. Downscale and compress one, upsample it bicubically, and the
result overshoots those ramps badly. The model does not overshoot, so it recovers
2.44 -> 1.51 and books a large win.

**So the model's headline advantage on these clips is mostly ringing
suppression, not super-resolution.** It is winning against a baseline that is
failing unusually badly, rather than restoring unusual amounts of detail.

That also explains where the gain sits. Averaged over the two groups:

| group | flat | edge | texture | bicubic edge PSNR |
|---|---|---|---|---|
| 3 renders | — | +0.21 | **+1.32** | 39.76 |
| 12 real | -0.02 | +0.00 | **-0.09** | 30.14 |

The renders' texture is synthetic: regular, band-limited, and therefore
predictable by a small convolutional model. Natural texture — foliage, crowds,
water, grain — is stochastic, destroyed by quantisation, and unrecoverable. This
is exactly the argument of §1, with content type as the axis instead of
compression level.

**Consequences.**

1. `dinner_1080p30` is no longer anomalous. It has zero sensor noise like the
   renders, but it is camera footage, so bicubic does not ring on it (1.020) and
   there is nothing for the model to fix.
2. **A benchmark should not be built from rendered content**, because it measures
   ringing suppression on a baseline that misbehaves there. The 15-clip set stays
   the reference set; the three renders are worth keeping only as a labelled
   subset.
3. If ringing suppression is a genuine strength, it is worth knowing *where else*
   bicubic rings — animation, screen content, game capture and UI are all
   band-limited and anti-aliased in the same way. That is a plausible content
   niche where this model is strong for a reason that now has a mechanism, and
   it is untested.

---

## 14. The wrong checkpoint is deployed, and it is the weaker one

Verifying the multi-exit export byte-for-byte turned up something else: the
`.bin` files are tracked in git, the `.pth` checkpoints are not, and nothing
recorded which produced which. They had drifted.

`extension/models/span_lite_2x_c16.bin` was exported from
`ckpt_c16_sharp2/best_phase1.pth`. Every 2x evaluation across two sessions —
§9, §11, §12, §13 — scored `checkpoints_c16/best_phase2.pth` instead. The 4x
binary does trace to `checkpoints_c16x4/best_phase1.pth`, which the 4x
evaluations already used, so only 2x was affected.

Both models against bicubic, 15 clips, CRF 28:

| | DISTS | texture PSNR | overall PSNR | tLP | sharpness |
|---|---|---|---|---|---|
| bicubic | 0.1735 | 24.985 | 28.169 | -0.01282 | 0.6784 |
| **deployed** (`sharp2`) | 0.1717 | **25.461** | **28.603** | -0.01060 | 0.6866 |
| the one evaluated (`best_phase2`) | **0.1629** | 25.182 | 28.359 | **-0.00982** | **0.7374** |

**The deployed model beats bicubic by 1.0% DISTS, not the 6.1% reported.** Split
by clip origin, the gap is worse than that average suggests:

| clip group | deployed | evaluated |
|---|---|---|
| 3 render clips | +10.1% DISTS, 3/3 | +16.2% DISTS, 3/3 |
| **12 real-camera clips** | **-1.2% DISTS, 7/12** | **+3.7% DISTS, 10/12** |
| all 15 | +1.1%, 10/15 | +6.2%, 13/15 |

**On real camera footage the deployed model is perceptually worse than bicubic**,
by 1.2%, winning 7 of 12 clips — a coin flip. Its whole measured advantage is
the ringing suppression on synthetic content that §13 explains.

**The checkpoint that was merely being evaluated is the better one on real
content**, on both metrics that transfer: DISTS +3.7% against -1.2%, and flicker
+0.0030 against +0.0018. It loses only on PSNR (+0.20 dB against +0.48 dB) —
the axis §11 established does not transfer.

That inverts how the two look under PSNR selection, which is the most likely way
the wrong one came to be shipped: `sharp2` is the better model by distortion and
the worse one by perception, and v1.0.2 was chosen before the 15-clip set
existed, on the 4-clip benchmark the renders dominate.

**Actions.**

1. `extension/models/PROVENANCE.json` now records checkpoint, channels, scale
   and binary sha256 for each shipped model; `training/eval/verify_shipped.py`
   re-exports and byte-compares, so the mapping is proved rather than asserted.
   Run it before reporting any "shipped" number — `--spec` prints the eval model
   spec so a chain script cannot point somewhere else by hand.
2. **Consider shipping `checkpoints_c16/best_phase2.pth` instead**, pending
   confirmation at CRF 20 and 36. On the evidence here it is better on the two
   metrics that generalise and worse only on the one that does not.
3. Nothing in §§9-13 needs re-deriving. Those sections compare candidates against
   each other on one consistent reference, and that comparison is unaffected.
   What changes is any sentence of the form "the shipped model scores X".

---

## 15. Prior art on GitHub, which the arXiv-only search missed

> **The content gate proposed below does not work as specified — see §21.** The
> ringing signal separates renders from camera footage but cannot identify which
> *camera* clips the model loses on, which is the distinction that would matter.

§5 notes that arXiv was the wrong corpus for the engineering questions. The
correction was never made: the research stayed on arXiv, and the working code
that talented people publish went unread. Three things found in one pass.

### Anime4K designed around the split we measured in §13

[bloc97/Anime4K](https://github.com/bloc97/Anime4K) (MIT) is a real-time line
reconstruction upscaler, ~3ms on a Vega 64, temporally coherent by construction
because it is a deterministic local operator with no state. Its README states it
is optimised for native 1080p anime and explicitly **not** for content with
"film grain, older MPEG compression artifacts".

That is §13's finding from the other direction. We measured that this model wins
on anti-aliased, band-limited content where bicubic rings (+10-16% DISTS) and is
worth roughly nothing on natural camera footage (+1%, and -1.2% for the deployed
model). Anime4K's authors reached the same content boundary and responded by
**specialising**: a line detector gates the enhancement so gradient maximisation
is applied near lines rather than indiscriminately, and one iteration of targeted
FXAA on those lines suppresses the ringing it would otherwise introduce.

**This suggests the strategic option this project has not considered: stop trying
to be a general upscaler.** The measurements say we are a strong line/edge
restorer and a break-even general one. A content gate — run the network where it
helps, fall back to bicubic where it does not — would raise average delivered
quality *and* cut GPU time, rather than trading one for the other.

And §13 already supplies the detector. The mechanism there was that **bicubic
rings on this content**, 1.9-2.4x the ground truth's gradient energy at edges
against <=1.0 on camera footage. Ringing is measurable from the upscaled frame
alone, with no reference: compare gradient energy at edges against the
neighbourhood. High ringing means band-limited content, which means the model
earns its time. That is a cheap per-scene classifier built out of a number we
already compute.

Related and directly useful to the engine:
[SegaraRai/anime4k-wgpu](https://github.com/SegaraRai/anime4k-wgpu) is a
WGSL/wgpu port — the same shader language this engine is written in.

### Temporal consistency: my twin loss was the version without motion

The VSR work that targets flicker ([StableVSR](https://github.com/claudiom4sir/stablevsr),
[MGLD-VSR](https://github.com/IanYeung/MGLD-VSR)) uses a **flow-guided temporal
loss**: warp the previous frame by estimated motion, then penalise disagreement
with the current output. The methods themselves are diffusion-based and far too
heavy for a 33k-parameter browser model, but the loss structure transfers.

§12's twin-consistency experiment was that idea with the motion removed — two
degradations of one static crop. Penalising disagreement everywhere has a trivial
minimum: make the network less responsive to its input. It found it, converging
onto bicubic's sharpness (0.6745 vs 0.6784) and bicubic's tLP (-0.01279 vs
-0.01282), and gave up the model's entire advantage.

**The fix is the same principle as Anime4K's line gate: apply the penalty where
it means something.** Either use genuine consecutive frames with motion
compensation, or restrict the twin penalty to regions where the two degradations
actually differ, instead of over the whole frame.

---

## 16. The network is not just an anti-ringing clamp — and the codec mix was wrong

### Ringing suppression does not explain the win after all

§13 found the model's large advantage on render clips is that bicubic *rings*
there. The obvious follow-up: an anti-ringing clamp costs almost nothing — an
upscaled pixel cannot legitimately fall outside the min-max of the source
neighbourhood it came from, so clamp it there. If that captured most of the
gain, much of what a 33k-parameter network buys would be available free.

It does not. CRF 28, DISTS against bicubic:

| variant | render clips | camera clips | edge sharpness (render) |
|---|---|---|---|
| bicubic | — | — | 2.1520 |
| bicubic + clamp | **-0.7%** | **-0.3%** | 1.9617 |
| deployed | +9.6% | +0.9% | 1.4012 |
| candidate | **+15.9%** | **+5.1%** | 1.4238 |

The clamp works as a clamp — edge sharpness falls 2.15 -> 1.96, so overshoot is
genuinely being suppressed — and perceptual quality gets slightly *worse* on both
groups. Clamping removes the overshoot without putting anything in its place;
the network removes it and reconstructs plausible edge structure underneath.

**So the network is doing real restoration, not cheap artifact suppression, and
the neural route is justified on this content.** The content-gate idea from §15
survives, but the gate should select between *network and bicubic*, not between
*network and a clamp*.

This is also the third independent measurement showing the candidate checkpoint
beats the deployed one — here by 15.9% against 9.6% on renders and 5.1% against
0.9% on camera footage (§14, and CRF 20/28 in chain10).

### The degradation chain modelled the wrong codecs

`codec_degrade.py` sampled libx264 (80%), mpeg4 (15%), libx265 (5%), with a
comment stating "H.264 and H.265 are what web video actually is". That has not
been true for the deployment target for years: **YouTube serves VP9 to most
desktop browsers and AV1 to a growing share**, and their artifacts differ in kind
— larger transforms and stronger in-loop filtering, so they smear where x264
blocks. A browser extension trained only on x264 artifacts is the same class of
domain gap this project has already been bitten by twice.

Both encoders were available and both work through PyAV, measured per 256x256
frame: x264 8.0ms, VP9 16.0ms, SVT-AV1 34.7ms (libaom is not exposed by PyAV).
The chain now samples x264 0.50 / VP9 0.22 / mpeg4 0.15 / AV1 0.08 / x265 0.05,
costing 74.8ms per sample against roughly 55ms before.

SVT-AV1 writes its banner straight to the file descriptor, as x265 does, so
`av.logging` cannot reach it. The encode is now wrapped in a file-descriptor
redirect, which handles both.

---

## 17. Swapped the shipped 2x model

§14 found the extension shipped `ckpt_c16_sharp2/best_phase1.pth` while every
evaluation scored `checkpoints_c16/best_phase2.pth`. Measured across all three
CRFs on the 15-clip set, split by clip origin, DISTS against bicubic:

| CRF | deployed (camera) | candidate (camera) | deployed (render) | candidate (render) |
|---|---|---|---|---|
| 20 | **-1.6%**, 8/12 | **+5.7%**, 10/12 | +16.1% | +21.3% |
| 28 | **-1.2%**, 7/12 | **+3.7%**, 10/12 | +10.1% | +16.2% |
| 36 | **-0.9%**, **4/12** | **+2.5%**, 10/12 | +3.7% | +9.9% |

**The deployed model was perceptually worse than bicubic on real camera footage
at every compression level**, and at CRF 36 it beat bicubic on only 4 of 12
clips. The candidate wins 10 of 12 at every CRF and is better on both clip
groups, on DISTS and on |tLP|. It gives up 0.3-0.5 dB of texture PSNR, on the
axis §11 showed does not transfer.

`extension/models/span_lite_2x_c16.bin` is now exported from
`checkpoints_c16/best_phase2.pth`, `PROVENANCE.json` records the swap, and
`verify_shipped.py` confirms the binary traces to it. The previous weights are
kept at `checkpoints_c16/deployed_2x_c16.pth`, so reverting is one re-export.

Note what this was *not*: no retraining, no architecture change. The improvement
was a checkpoint that already existed in the repo and had been measured against
for two sessions without anyone noticing it was not the one shipping.

---

## 18. Folding conv_last into conv_cat: exact, and slower

`conv_cat` is a 1x1 over the concat of four buffers and `conv_last` is the 3x3
that produces the fourth, so the pair composes exactly:

    W3 @ (L * b4 + bl) = (W3 @ L) * b4 + W3 @ bl

giving one 3x3 kernel `K[o,i] = sum_m W3[o,m] L[m,i]` and a bias shift. Verified
in PyTorch at `max|ref - fused| = 6e-7`, and composable at *load* time from the
existing tensors, so no re-export and every shipped `.bin` keeps working.

It removes a whole pass — 18 dispatches to 17, one full C-channel write and the
read that followed it. Measured on an M4 Pro at 720p, output `max_abs_diff` 1/255:

| build | frame time |
|---|---|
| unfolded | **21.2 ms** |
| folded, one thread per output channel | 30.2 ms |
| folded, 8 output channels per thread | 23.1 ms |

**Both folded variants are slower.** The first was 42% slower because every
thread re-read the same 3x3 neighbourhood; amortising across 8 accumulators
recovered most of that and still lost by 9%.

The reason is worth recording, because it bounds a whole family of ideas.
`buildConv` is tuned hard: it computes a **2x2 pixel block per thread** and loads
a **4x4 input patch once per input channel**, reusing it across all 9 taps and
all 4 pixels. A fused kernel that does not replicate that blocking pays far more
in redundant loads than one removed buffer round-trip saves.

**The general lesson: pass-count is not the cost model here.** §10's attention
fusion won because it removed a pass *without* giving up any blocking — the
attention was elementwise and folded into an already-optimal conv's write. This
fold instead replaced an optimal 3x3 with a worse one. Future fusion candidates
should be judged on whether they preserve the inner-loop reuse, not on how many
dispatches they delete.

Left in behind `FOLD_CONV_LAST`, defaulting off.

---

## 19. Training on the codecs browsers decode: better perceptually, worse temporally

> **The flicker regression here is largely an artifact — see §26.** The tLP
> figures in this section use a summary convention that inflates gains on
> clips where the model crosses zero. Corrected, the codec model gives up
> about 5% of the flicker advantage, not three quarters.

§16 changed the degradation chain from x264/x265/mpeg4 to a mix including VP9 and
AV1, on the grounds that a browser extension meets what YouTube serves. Retrained
from the newly-shipped checkpoint, 40 epochs, everything else held constant.

On the **12 real-camera clips**, DISTS against bicubic:

| CRF | shipped | codec-retrained | shipped tLP | retrained tLP |
|-----|---------|-----------------|-------------|---------------|
| 20 | +5.7%, 10/12 | **+7.9%, 11/12** | +0.0022 | +0.0012 |
| 28 | +3.7%, 10/12 | **+5.1%, 11/12** | +0.0030 | +0.0008 |
| 36 | +2.5%, 10/12 | **+3.7%, 11/12** | +0.0036 | +0.0009 |

**It wins on DISTS at every CRF and takes an extra clip at every CRF.** The
degradation domain was genuinely wrong, and fixing it was worth 1.2-2.2
percentage points on the metric this project selects on.

**But it gives up most of the flicker advantage.** At CRF 36 the shipped model's
tLP improvement over bicubic is +0.0036 and the retrained model's is +0.0009 —
roughly three quarters of it gone. It is also less sharp (0.7775 against 0.8166
at CRF 20).

Those three facts are consistent with one description: **the retrained model is
smoother**, spatially and temporally, and DISTS rewards that on camera footage
while |tLP| and the sharpness ratio penalise it. VP9 and AV1 smear where x264
blocks, so a model fitted to them learns a gentler correction.

**Not swapped.** The DISTS gain is real and consistent, but §11 established that
flicker is the one advantage that transfers across content types, and this trades
away three quarters of it. Trading the metric that generalises for the metric
that is easier to move is the mistake this project already made once, in the
opposite direction, when it shipped on PSNR.

The right resolution is not to pick one — it is to get both, which is what the
masked twin-consistency run (§12, fixed) is for: it targets flicker directly and
composes with this change. If it holds the DISTS gain while recovering tLP, the
swap becomes obvious. If it does not, the choice needs to be made deliberately
rather than by whichever run finished last.

---

## 20. The engine matches PyTorch, so the measurements describe what ships

Every quality number in this file is measured in PyTorch. Users get the WGSL
engine. Nothing had ever checked that the two agree, and a mismatch in the mean
subtraction, the `img_range` scale, the pixel-shuffle indexing or the final
colour conversion would have been invisible to every evaluation here and visible
in every frame a user sees.

One real codec-degraded 256x256 LR crop, the shipped 2x weights, PyTorch output
against the browser's, **with f16 active in the engine**:

| | |
|---|---|
| max abs diff | **1** / 255 |
| mean abs diff | 0.0243 |
| pixels off by more than 8 | **0** |

That is float rounding and nothing else. The evaluation apparatus predicts
delivered quality, and the f16 decision holds on real content end to end rather
than only against an f32 engine.

`dev/parity_check.html` is the check. Worth re-running after any change to the
preprocessing pass, the shuffle, or the weight export -- those are the three
places where a silent divergence could open up, and none of them are covered by
the shader-level comparisons in §10 or §18, which only ever compare the engine
against itself.

---

## 21. The ringing gate does not work, and what the remaining loss actually is

§15 proposed gating the network on content: run it where bicubic rings, fall back
where it does not, using §13's mechanism as a no-reference detector. Tested
against the per-clip results already on disk, no GPU required.

Camera clips only, CRF 28, bicubic's edge sharpness ratio against measured DISTS
gain:

| model | corr(edge sharpness, gain) | separable by a threshold |
|---|---|---|
| shipped | **+0.075** | no |
| codec-retrained | **+0.259** | no |

**No usable signal.** `dinner_1080p30` has the highest bicubic edge sharpness of
any camera clip (1.021) and is the largest winner (+15.3%); `controlled_burn`
loses at 0.765, sitting in the middle of the winners' range (0.290-1.021). The
ringing measure separates renders from camera footage, which §13 already
established — and the renders were never the problem.

**The gate idea is not dead, but the signal is wrong.** After the codec retrain
only **one** camera clip still loses: `controlled_burn_1080p`, at -4.1%. It is
fire and smoke — pure stochastic texture, temporally chaotic, with no stable
structure to restore. That is §1's argument again: information that was
quantised away cannot be recovered, and smoke is nearly all such information.

So a gate would need to detect **stochasticity**, not ringing — something like
the fraction of high-frequency energy that is temporally incoherent between
frames, rather than a spatial overshoot measure. Untested, and worth far less
than it was before: the model now wins 11 of 12 camera clips, so a perfect gate
would recover about 4% DISTS on one twelfth of content.

**Recorded mainly so the gate is not built on the wrong signal.** The mechanism
in §13 is real and explains the render clips; it simply does not generalise into
a runtime router.

---

## 22. EMA and DISTS-as-loss added nothing — and validation stopped predicting

Two techniques taken from the NTIRE 2026 efficient-SR recipes, both used by the
winner on this same SPAN family: EMA of weights at decay 0.999, and DISTS added
as a training term (checkpoints were already being *selected* on DISTS while
nothing in the loss pointed at it). Trained on the corrected codec mix, 40
epochs, identical to §19's run in every other respect.

Real-camera clips, DISTS against bicubic:

| CRF | codec fix alone | + EMA + DISTS |
|-----|-----------------|---------------|
| 20 | +7.9%, 11/12 | +7.9%, 11/12 |
| 28 | +5.1%, 11/12 | +5.2%, 10/12 |
| 36 | +3.7%, 11/12 | +3.7%, 10/12 |

**Indistinguishable.** The entire gain was already delivered by the degradation
fix. Two plausible reasons: EMA earns its keep over long from-scratch schedules
and these runs fine-tune a converged checkpoint for 40 epochs, and DISTS overlaps
heavily with the VGG perceptual term already in phase 2 — a second perceptual
loss pointed at nearly the same thing.

### The part that matters more: validation stopped tracking the benchmark

**Validation DISTS said this run was 4.8% better** — 0.2011 against 0.2113 — and
the 15-clip video benchmark says it is identical. The validation set is Vimeo
still frames put through the codec chain; the benchmark is real clips through a
real encode, scored per clip and split by origin.

That gap is not academic. **Checkpoint selection runs on validation DISTS.**
Every "best" checkpoint this project has ever saved was chosen by a signal that,
on this evidence, can move 4.8% while the thing being optimised does not move at
all. §14 already showed the wrong checkpoint can ship; this shows the selection
metric itself can be pointing somewhere the benchmark does not follow.

Worth noting the failure is one-directional here — validation was *optimistic*,
not pessimistic — so nothing already shipped is called into question. But the
next time validation reports a gain, it should not be believed until the clip
benchmark agrees.

**What follows:** run the clip benchmark, not validation, when a decision
depends on the answer. `rank_models.py` exists for exactly this, and it is
cheap enough — one pass over all candidates — that there is no reason to trust
the proxy.

---

## 23. A video validation set, so selection can see what the benchmark sees

§22 found the still-frame validation set reporting a 4.8% DISTS gain on a run the
clip benchmark scored as identical — while **checkpoint selection runs on that
signal**. §11 found flicker is the one advantage that transfers, and a still-frame
set cannot measure it at all, so no run has ever been able to select on it.

`training/video_val.py` replaces it with **consecutive frame pairs from five
held-out clips** — `pedestrian_area`, `red_kayak`, `riverbed`, `rush_hour` and
`sintel_trailer`, four camera plus one animation to keep §13's band-limited class
represented. None appear in `clips_busy`: selecting on the benchmark would make
every number in this file self-confirming. 30 pairs, 5.3 MB, preloaded.

### The 9 dB PSNR spread was a defect in the set, now fixed

The first version reported PSNR of 33.83 / 42.57 / 45.21 for three checkpoints
the benchmark separates by ~0.3 dB. The per-clip breakdown found it immediately:

| clip | deployed | shipped | webcodec |
|---|---|---|---|
| pedestrian_area | 27.59 | 27.55 | 27.61 |
| red_kayak | 24.86 | 24.82 | 24.84 |
| riverbed | 25.50 | 25.41 | 25.47 |
| rush_hour | 28.21 | 27.87 | 28.14 |
| **sintel_trailer** | **62.99** | **107.22** | **120.00** |

The four camera clips agree within 0.35 dB. `sintel_trailer`'s fades are
near-constant frames — MSE around 1e-12, so PSNR runs to 120 dB — and one clip
was moving the five-clip mean by nine decibels while carrying no information
about any model. Pairs whose ground truth has variance below 1e-4 are now
dropped (6 of 30), and aggregation is by median rather than mean.

| model | PSNR | DISTS | \|tLP\| |
|---|---|---|---|
| deployed (pre-§17) | 26.51 | 0.2040 | 0.00765 |
| shipped | 26.43 | **0.1916** | 0.00797 |
| codec-retrained | 26.51 | 0.1919 | **0.00594** |

`|tLP|` rather than `tLP`: §10 established 0 is the target and "lower is better"
is optimised by a constant grey frame.

### It is NOT yet fit to select on

PSNR is now sane and DISTS still ranks the pre-§17 model last, which is the
ordering that matters most. **But flicker disagrees with the benchmark.** Here
the codec-retrained model has the best |tLP| (0.00594 against 0.00797); §19
measured it as clearly *worse* than shipped on the 15-clip set.

Four held-out camera clips against twelve benchmark clips is a plausible reason —
small samples, different content, median against per-clip mean — and the
benchmark should be weighted higher on count alone. But the disagreement is on
**the exact metric this set was built to measure**, which is disqualifying for
now. Ordering three known checkpoints correctly was necessary, not sufficient,
and on flicker it does not even do that.

**Not wired into `train_span.py`.** Two reasons: swapping the selection signal
mid-queue would make the running comparisons incommensurable, and it has not
earned the job. The test it has to pass is whether the checkpoint it picks beats
the checkpoint the old signal picks — measured on the benchmark, not argued from
principle.

---

## 24. Flicker does not rank models consistently across content

> **Partly an artifact — see §26.** The benchmark side of this comparison
> used the raw tLP difference while the held-out side used |tLP|. Corrected,
> the benchmark says the two models are near-tied rather than clearly
> opposite, so the disagreement is much smaller than recorded.

§23 disqualified the video validation set because it ranked the codec-retrained
model best on |tLP| where the benchmark ranked it worst, and blamed the likely
cause: four held-out clips against twelve. That was testable. The held-out set
was widened to **eleven clips, ten of them camera** — adding video-conference
content, which is a large share of real web video and was absent from both sets —
giving 60 usable pairs.

**DISTS agreement became strong:**

| model | DISTS (held-out) | benchmark ordering |
|---|---|---|
| deployed (pre-§17) | 0.2130 | worst |
| shipped | 0.2022 | better |
| codec-retrained | **0.2008** | best |
| EMA+DISTS | 0.2011 | tied with codec-retrained |

That reproduces every ordering the 15-clip benchmark reports, including §22's
null result — EMA+DISTS at 0.2011 against the codec run's 0.2008 is the same
"indistinguishable" the benchmark found, arrived at independently.

**The flicker disagreement did not go away:**

| model | benchmark \|tLP\| | held-out \|tLP\| |
|---|---|---|
| shipped | 0.00982 (better) | 0.00913 |
| codec-retrained | 0.01202 (worse) | **0.00822** (better) |

Opposite orderings on two sets of real camera clips, both put through real
encodes. With 10 clips against 12 this is no longer a small-sample story.

**The honest reading is that tLP ranks models differently depending on content.**
The benchmark clips are largely high-motion nature — `park_joy`, `crowd_run`,
`ducks_take_off`; the held-out set is more mixed, including near-static
conference video. A model's temporal behaviour evidently depends on how much
motion it is given, and the two sets disagree about which model handles that
better.

**This qualifies §11.** That section found tLP improving by an identical +0.0030
on both the render and camera groups *within one clip set*, and concluded flicker
was the advantage that transfers. It transfers across content *type* within that
set; it does not transfer across clip sets. That is a weaker claim than the one
recorded, and any plan resting on "select on flicker" — including the one
proposed at the end of §11 — needs it.

**Consequences.** The held-out set is now usable for **DISTS** selection, where it
agrees with the benchmark on four checkpoints including a null. It is not usable
for **flicker** selection, and neither is the benchmark, until it is understood
why they disagree. The cheapest next probe is whether tLP tracks scene motion —
if it does, the metric needs normalising by motion before it can rank anything.

---

## 25. Twin consistency retired, and the shape of what worked tonight

The masked version of §12's consistency loss — penalty restricted to the half of
pixels where the two degradations already agree, so the model cannot buy
agreement by becoming unresponsive — trained cleanly and did not collapse onto
bicubic the way the unmasked one did. It is still worse than doing nothing:

| CRF | shipped | codec fix alone | + masked twin |
|-----|---------|-----------------|---------------|
| 20 | +5.7%, tLP +0.0022 | **+7.9%, +0.0012** | +7.0%, +0.0008 |
| 28 | +3.7%, +0.0030 | **+5.1%, +0.0008** | +4.7%, +0.0003 |

Worse on DISTS *and* worse on flicker, which is the axis it exists to improve.
The masking removed the trivial minimum without making the penalty useful — the
term sat flat at 0.0096 for the whole run, never driven down, so it acted as a
constant drag rather than a gradient toward anything.

**Twin consistency is retired.** Two attempts, two regressions.

### The pattern across the night

| change | kind | result |
|---|---|---|
| ship the checkpoint that was already being evaluated (§17) | correction | **+4.9pp** |
| model the codecs browsers actually decode (§19) | correction | **+1.4pp** |
| refuse to enlarge the HR reference (§10) | correction | fixed a measurement |
| tie binaries to checkpoints (§14) | correction | caught the wrong model shipping |
| unmasked twin consistency (§12) | addition | regression |
| conv_last folded into conv_cat (§18) | addition | 9% slower |
| EMA + DISTS-as-loss (§22) | addition | null |
| masked twin consistency (§25) | addition | regression |

**Four corrections, four gains. Four additions, zero gains.** Every technique
added because it works elsewhere — two of them straight from the NTIRE 2026
winner's recipe on this same architecture family — produced nothing or worse.
Every gain came from finding something that was already wrong.

That is not an argument that techniques never work. It is an argument about where
to look first in *this* codebase at *this* stage: the measurement apparatus and
the deployment path had accumulated four separate defects, and each was worth
more than any recipe change tried against them. The corrections were also far
cheaper — three of the four cost no GPU at all.

The remaining known-wrong thing is §24: two clip sets disagree about which model
flickers less, and neither is authoritative. On the night's evidence that is
where the next gain is, not in a fifth technique.

---

## 26. A sign error in how tLP was summarised, and what it changed

`split_by_origin.py` reported `mean(model_tlp - bicubic_tlp)`. §10 established
that **0 is the target** for tLP: positive is added flicker, negative is temporal
over-smoothing, and "lower is better" is optimised by a constant grey frame. The
raw difference therefore credits a model *unboundedly* for flickering less than
the truth — exactly the failure §10 already identified, reintroduced one level up
in the aggregation.

The clip where it bites:

| | bicubic | model | raw difference | |tLP| deviation |
|---|---|---|---|---|
| `vsr_test_video` | -0.00710 | +0.00440 | **+0.01150** | **-0.00270** |

Bicubic over-smooths; the model adds flicker. Raw scores that as a large gain
because the number went up. By deviation it went 0.0071 to 0.0044 — a real
improvement, roughly a quarter the size.

First suspected as measurement noise. It is not: re-evaluating the same
checkpoints on the same clips reproduces every per-clip tLP **exactly**, standard
deviation 0.00000. The degradation is seeded and the metric is deterministic. The
discrepancy was entirely in the summary.

**What it changes.**

**§19's magnitude was wrong, but its direction was not — and this correction was
itself over-stated on one CRF.** Under the corrected convention:

| CRF | shipped | codec-retrained | gap |
|---|---|---|---|
| 20 | -0.00156 | -0.00148 | **5%** |
| 28 | -0.00178 | -0.00032 | **82%** |

At CRF 20 the flicker cost is negligible and §19's "three quarters" was indeed an
artifact. At CRF 28 — closer to what web video actually uses — shipped reduces
flicker deviation **5.5x more**, and the regression is real. Generalising the
CRF 20 figure to "about 5%" was a second error on top of the first; both CRFs had
to be checked before the claim was worth making.

**§24 shrinks.** Its "opposite orderings" compared the benchmark computed with the
raw convention against the held-out set computed with |tLP|. Corrected, the
benchmark has the two models near-tied (0.00008 apart) where the held-out set
prefers the codec model — a mild disagreement, not a contradiction. §24's broader
claim that flicker is content-dependent is weakened accordingly, and §11's
"advantage that transfers" needs less qualification than §24 imposed.

**§22 is unaffected in its conclusion** — EMA+DISTS was null on DISTS, which
carries the finding — though its tLP column shares the error.

`rank_models.py` used |tLP| from the start, so the ranking in §27 is computed
correctly. `split_by_origin.py` is now fixed.

**The general lesson repeats §18's.** A metric's direction has to be enforced at
every level it is aggregated, not just where it is defined. §10 got the
per-frame definition right and the error reappeared in the mean over clips.

---

## 27. Final ranking, and a decision that is a real trade

`rank_models.py` over the 12 real-camera clips, criteria fixed in §14/§11 before
any of these numbers existed: camera DISTS first, then |tLP| deviation, then
clips won.

**CRF 20**

| model | cam DISTS | wins | \|tLP\| dev | render DISTS |
|---|---|---|---|---|
| **webcodec** | **+7.9%** | 11/12 | -0.00148 | +19.8% |
| ema_dists | +7.9% | 11/12 | -0.00138 | +19.4% |
| masked_twin | +7.0% | 11/12 | -0.00121 | +18.1% |
| shipped | +5.7% | 10/12 | **-0.00156** | +21.3% |

**CRF 28**

| model | cam DISTS | wins | \|tLP\| dev | render DISTS |
|---|---|---|---|---|
| ema_dists | **+5.2%** | 10/12 | -0.00023 | +13.1% |
| **webcodec** | +5.1% | **11/12** | -0.00032 | +13.3% |
| masked_twin | +4.7% | 10/12 | -0.00013 | +12.0% |
| shipped | +3.7% | 10/12 | **-0.00178** | +16.2% |

**Among the new models, `webcodec` wins.** It ties `ema_dists` on DISTS at CRF 20,
trails by 0.1pp at CRF 28, and beats it on both |tLP| and clips won — consistent
with §22, where EMA and DISTS-as-loss added nothing over the codec fix alone.

**Against the shipped model it is a genuine trade, not a clear win:**

- **Perceptual: clearly better.** +7.9% against +5.7% at CRF 20, +5.1% against
  +3.7% at CRF 28, and 11 of 12 clips against 10.
- **Temporal: clearly worse at CRF 28.** 5.5x less reduction in flicker deviation,
  at the compression level most web video uses.
- **Render content: worse.** +13.3% against +16.2%.

The pre-registered rule ranks DISTS first and therefore selects `webcodec`. But
that ordering was justified by §11's claim that flicker is the advantage which
transfers — a claim §24 weakened and §26 partly restored, and which is now known
to hold at some CRFs and not others.

**Not swapped.** §17's swap was made unilaterally because the evidence was
one-sided: the deployed model was *worse than bicubic* on real footage and the
alternative won on everything. This is not that. Trading a measurable temporal
regression for a measurable perceptual gain is a product judgement about which
artifact a viewer minds more, and no metric here answers it. The comparison is
recorded; the choice belongs to whoever owns the product.

---

## 5. How this research was produced, and what to trust

Retrieved from the arXiv API and answered strictly from retrieved abstracts, via
`llm-search` in the `claude-local-llm` plugin. Two limits worth knowing:

- **Abstracts only.** Where a fact lives in a paper's body — such as which
  architecture won a challenge track — abstract-level retrieval cannot find it.
  The SPANV2 details above came from reading the challenge report directly, and
  the local agent independently confirmed no standalone SPANV2 paper exists.
- **arXiv is the wrong corpus for GPU kernel engineering.** Queries about
  operator fusion and memory traffic returned multimodal NAS and radar
  classification papers. That knowledge lives in vendor documentation and
  challenge reports, not in the SR literature.
