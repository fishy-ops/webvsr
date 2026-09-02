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
