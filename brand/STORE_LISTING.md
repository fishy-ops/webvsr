# Crisp — store listing copy

Everything here is claimable from measurements in `RESEARCH.md`. Two things the
old listing said are deliberately gone, both flagged in the notes at the bottom.

---

## Extension name  (Chrome limit: 75 characters)

```
Crisp — AI Video Quality Enhancer
```
32 characters. Brand word first so it can be recognised and recommended by name,
then the exact phrase people type. "Upscaler" is deliberately absent from the
title: it is the technical word, and the mainstream tools people already use
(Topaz, Canva, ElevenLabs) all say "enhance video quality" in their consumer
naming even when the engine underneath is an upscaler.

**Short name** (toolbar, 12 char limit): `Crisp`

---

## Short description  (Chrome limit: 132 characters)

```
Sharpens blurry, low-quality video as it plays, on any site. Free, runs on your own computer, nothing uploaded.
```
110 characters. Leads with the problem word ("blurry"), covers the two things
people check before installing (price, privacy).

---

## Detailed description

**Blurry video, fixed as you watch.**

A lot of web video is sent at a low resolution and squeezed hard to save
bandwidth. Stretched onto a big screen it goes soft, blocky and smeared. Crisp
rebuilds a sharper picture from it, live, while the video plays — on any site,
in any HTML5 player.

It is not a contrast slider or a sharpening filter. Crisp runs a real neural
network on your graphics card, through WebGPU, and reconstructs detail that a
plain stretch just enlarges.

**What you get**

- **Sharper video everywhere.** Works on any site with a normal HTML5 video
  player. Turn it on with the on-video button or Alt+S.
- **It stays out of the way.** Crisp watches its own frame budget. If your
  machine cannot keep up, it steps aside and passes the original through
  untouched — it will never make playback stutter to look better.
- **It only runs when it helps.** If a video is already about as sharp as your
  screen, Crisp stays off instead of burning your battery for nothing.
- **Adjustable.** Quality level, target resolution, sharpening amount, and a
  per-site off switch.

**Private by construction, not by promise**

Everything happens on your own computer. Your video is never uploaded, because
there is nothing to upload it to — Crisp has no server. No account, no sign-up,
no tracking, no data collected. It works offline.

**Free.** No trial, no subscription, no watermark, no paid tier.

**For the curious**

The model is a 33,388-parameter SPAN-Lite network, hand-written as WebGPU
compute shaders. A 1080p frame takes about 48 ms on an Apple M4 Pro; a 2019
desktop GPU does 720p→1440p at roughly 46 frames per second. It is trained on
video put through real H.264, H.265, VP9 and AV1 encoders, so it learns to undo
the artifacts real web video actually has rather than a synthetic imitation of
them.

Quality is measured against plain bicubic upscaling on 19 public-domain test
clips, not on hand-picked examples.

**Honest about the limits**

- You need a reasonably recent Chrome and a GPU with WebGPU. Without it, Crisp
  simply stays off.
- The gain is largest on low-resolution video that is not also crushed by
  compression. On very heavily compressed video there is less real detail left
  to recover, and the improvement is smaller.
- It does not invent detail that was never there. It reconstructs; it does not
  hallucinate a face into a blur.
- DRM-protected video (Netflix, Disney+ and similar) cannot be processed by any
  extension, including this one.

---

## Notes on what changed and why

**Removed: "shines exactly where plain math upscaling can't — removing
compression artifacts."** Our own measurement says the opposite. The advantage
over bicubic is +9.3% at CRF 20 and falls to +5.9% at CRF 28 — the gain *shrinks*
as compression gets worse, because heavy compression destroys the detail there
is to recover. The listing now says that plainly under "Honest about the limits".

**Removed: "up to 4K", "HD", and any resolution promise.** The shipped model is
2x. Claiming a resolution target invites a one-star review from anyone who
measures it.

**Kept and made specific:** the local/private claim, which is the strongest
honest differentiator against the cloud "AI video enhancer" services this will
sit next to in search results. "No server to upload to" is a structural claim,
not a policy promise, and is worth more than either.
