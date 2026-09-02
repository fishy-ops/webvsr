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
