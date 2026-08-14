# WebVSR — Handoff (context transfer for a new chat)

> Self-contained so a new session can pick up immediately. Written 2026-08-10.
> Project root: **`D:\webvsr`**  ·  Public repo: **https://github.com/fishy-ops/webvsr**

---

## 0. TL;DR

**WebVSR** is a Chrome MV3 extension that does **real-time video super-resolution
in the browser on the user's own GPU** via **hand-written WebGPU (WGSL) compute
shaders** — no servers, no ONNX Runtime. It upscales low-res / compressed web
video live and passes already-sharp video through untouched.

**Status: COMPLETE and pushed** (7 commits, public). The engine + all models are
**verified on the real GPU** (RTX 2070 SUPER). The content-script UI/governor
logic is **syntax-checked only** — I cannot sideload the extension to click
through it, so it **needs a real Chrome reload to fully validate**.

---

## 1. What shipped

- **Real-time SR, 2× and 4×.** Default model is a **16-channel SPAN-Lite 2×**
  (128K params, ~32 dB). 720p→1440p in ~22 ms; 4× (16ch) native at ~170 fps
  (360p→1440p ~6 ms GPU). Target-scale slider 1.5×–4× auto-loads the right model.
- **DRS governor + passthrough** (`content.js`): runs the net at whatever internal
  resolution fits the video's frame budget (targets **wall-clock** time so it
  matches 60 fps sources); if it still can't keep up at min res, it hides SR and
  **passes the original through** (never worse than off).
- **RCAS-style adaptive sharpen** (`SHADER_SHARPEN` in `webgpu-sr.js`), clamped to
  local min/max so no halos. Strength setting (Off/Low/Med/High, default Med).
- **Auto-engage** (default on): skips SR (≈0 GPU, overlay shows "idle") when the
  source is already ≈ screen resolution — the fix for "100% GPU for no gain".
- **Full settings** (popup + on-video ⚙ flyout): GPU-load (light/balanced/max →
  governor budget), quality (fast/medium/quality → internal-res ceiling),
  target-scale (1.5–4×, >2× warning), sharpness, auto-pause, remember-state,
  show-stats, only-fullscreen, per-site disable. Alt+S toggles; ◐ hold-to-compare.
- **Engine is scale-aware + hot-swappable**: reads channels+scale from a sibling
  `.json` manifest; `switchModel(url)` disposes old weights and reloads.

## 2. The honest quality story (important — the user cares deeply about this)

- On **clean/sharp** video, this SR ≈ a good bicubic/Lanczos+sharpen. The user
  called this out and was right.
- On **realistically compressed low-res** video (what real web video is), it
  clearly **beats bicubic** by removing JPEG/codec blocking + noise —
  *reconstruction, not fabrication*. See `results/compare_fair.png`.
- **GAN is rejected.** The user trained an ESRGAN before; it fabricated fence
  lines / fake structure. GAN's adversarial objective inherently invents detail →
  not trustworthy. **Do not propose GAN.**
- **Perceptual-only (VGG, no GAN)** was tested (phase-2, `checkpoints_c16/best_phase2.pth`);
  it was only marginally sharper (+? subtle) and PSNR-optimized phase-1 is more
  temporally stable for video, so **phase-1 (PSNR) is the shipped model**.
- **More channels don't help.** 32ch ≈ 16ch at both 2× (32.30 vs 32.02 dB) and 4×
  (26.45 vs 26.16 dB) — visually indistinguishable. The 4× softness is
  architectural/task-inherent (tiny model reconstructing 4× from quarter-size
  compressed input), not a capacity problem.

## 3. Key files (`D:\webvsr`)

**Extension (`extension/`):**
- `manifest.json` — MV3, all_frames+match_about_blank, CSP allows wasm (legacy),
  content scripts = `webgpu-sr.js` + `content.js`, `models/*` web-accessible.
- `webgpu-sr.js` — **the engine.** `class WebGPUSR`: WGSL builders (buildPre/
  buildConv/buildAttn/buildCat/buildShuffle + SHADER_FINISH/SHADER_SHARPEN),
  `weightSpec(C,scale)`, `loadWeights`/`switchModel`, `configure`, `render`.
  Conv is 2×2 register-blocked (32 accumulators). GPU timestamps for governor.
- `content.js` — video detection, overlay UI, governor, auto-engage, model
  selection by target scale, settings plumbing.
- `background.js` — settings hub (chrome.storage) + broadcast. No inference.
- `popup.html` / `popup.js` — full settings page + "Run speed benchmark" link.
- `models/` — `span_lite_2x_c16.bin` (default 2×), `span_lite_2x_c16p2.bin`
  (perceptual option, not wired), `span_lite_4x_c16.bin` (4×), `span_lite_2x.bin`
  (original 32ch, used only by bench.html), each with a `.json` manifest.
- `bench.html`, `perf.html`, `test-live.html` — dev/test harnesses (not part of
  the shipped extension).

**Training / models (repo root):**
- `model_span.py` — SPAN-Lite (Conv3XC reparam, 4 SPAB blocks, PixelShuffle).
- `train_span.py` — trainer. Args: `--channels N --scale S --ckpt-dir DIR
  --total-epochs E --phase1-epochs E [--phase2]`.
- `dataset.py` — Real-ESRGAN-style second-order degradation (blur→resize→noise→JPEG).
- `losses.py` — Charbonnier + VGG perceptual (no GAN) + FFT.
- `export_webgpu_weights.py` — `--channels N --scale S --checkpoint X --output Y`;
  writes the `.bin` + sibling `.json` manifest the engine reads.
- `make_compare_*.py` — visual comparison montages → `results/`.
- `OPTIMIZATION_LOG.md` — full record of the real-time kernel work + scoreboard.

**NOT in git** (`.gitignore`): `data/`, `checkpoints*/` (`.pth`), `.venv/`,
`__pycache__/`, `*.log`, `comparisons*/`, `onnx/`.

## 4. How to build/verify

- **Test the engine** (no sideload needed): `cd extension && python -m http.server 8123`,
  then open `perf.html?v=X` and call `window.bench([360,720])` / custom JS.
  Numbers use GPU timestamps (median). Correctness = feed a gradient, check
  monotonic output + no NaN.
- **Train a model:** `python train_span.py --channels 16 --scale 4 --ckpt-dir
  D:/webvsr/checkpoints_c16x4 --total-epochs 150 --phase1-epochs 150`.
  Data present: DIV2K(800)+Flickr2K(2650) in `C:\Users\reach\OneDrive\Documents\
  mamba-sr\...` + `D:\webvsr\data\vimeo_frames`(10k).
- **Export:** `python export_webgpu_weights.py --channels 16 --scale 4
  --checkpoint <best_phase1.pth> --output extension/models/<name>.bin`.
- **Reparam note:** the model must be `.eval()` + `_update_params()` (Conv3XC → 3×3)
  before export; the export script does this.

## 5. Known issues / next steps

- **UI validation:** content.js (auto-engage, model-switch, governor, passthrough,
  all settings) is untested in a real extension context. First task: load unpacked
  in Chrome, play a low-res video, and fix anything in the `[WebVSR]` console.
- **4× edge-darkening (~6px):** zero-padding conv artifact, more visible at 4×.
  Not fixed (matching training). Fix idea: overscan-crop a few px in the finishing
  pass, or retrain with reflect padding.
- **Open offers to the user:** the 4× edge fix; a "Vivid" perceptual model as an
  opt-in; validate + tune governor budgets on real hardware.

## 6. Environment / gotchas

- **GPU:** RTX 2070 SUPER 8 GB, thermally marginal (throttles on sustained load).
- **Python:** repo `.venv` uses torch 2.7.1+cu118; `python` on PATH works for
  training/export here (CUDA available). (Historically a 3.11 was needed for the
  mamba-sr video project's shared_memory — not relevant to this repo.)
- **The in-app browser pane uses the real 2070S GPU** (confirmed NVIDIA Turing) —
  but it renders `D:\` files as static snapshots, so serve over `http://127.0.0.1:8123`
  (localhost is blocked for direct nav; use preview_start). It can't sideload the
  extension. Screenshots need the pane fronted; background tabs throttle timing.
- **WGSL lesson:** never dynamically index a private array (spills to local mem →
  5× slower). Use statically-named accumulators.

## 7. Git / attribution (IMPORTANT)

- Repo committed as **fishy-ops** with `fishy-ops@users.noreply.github.com`
  (local git config in the repo). **The user asked for NO Claude co-author** —
  do NOT add a `Co-Authored-By: Claude` trailer to commits on this project.
- Commit + push normally: `cd /d/webvsr && git add -A && git commit -m "..." &&
  git push origin main`.

## 8. Prior context

The older sibling project is `C:\Users\reach\OneDrive\Documents\mamba-sr` (image/
video SR in PyTorch: ESRGAN 22M/GAN, RCAN 5M, ResNetSR 1.5M, VSRNet 29.6M — all
**too heavy for real-time browser**; treat as learning assets). Its
`PROJECT_CONTEXT_HANDOFF.md` has the full history. WebVSR is the go-forward,
lightweight, WebGPU-native realization of that idea.
