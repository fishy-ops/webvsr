/**
 * Crisp Content Script
 *
 * Detects <video> elements, adds a floating SR button + settings flyout, and
 * runs the in-page WebGPU engine (webgpu-sr.js). Smoothness is the priority:
 * a frame-time governor keeps the neural net inside the video's frame budget so
 * playback never bogs down. Settings come from the background service worker.
 */

const DISPLAY_CAP = 2160;   // max finished (on-screen) height
const NEURAL_CAP = 1080;    // hard ceiling on neural input height (safety/VRAM).
// At 1080 the feature buffers are ~1.2 GB (9 x C x W*H x 4B); 1440 would be ~2.1 GB,
// too much to ask of an integrated GPU. Below this the governor decides -- and a
// source taller than the cap is downsampled before SR, which costs real detail,
// so the cap should stay at or above the resolution most sources actually are.
const MIN_NEURAL = 144;
const NEURAL_STEP = 16;
const START_NEURAL = 216;   // conservative start so first frames never stall

// GPU-load presets → fraction of the video's frame interval the net may use.
// Lower = smoother/lighter; 'max' removes the cap (native res, high GPU).
// ── Power ───────────────────────────────────────────────────────
// One control, three genuinely different behaviours.
//
// This replaces the old GPU-load + Quality pair. Those were built on a
// mechanism that shrank the neural input below the source resolution to buy
// frame time -- which cannot help, because reconstructing detail we just threw
// away is worse than not running. Once that was removed the two controls had
// nothing left to do: every option behaved identically on typical web video.
//
//   cap       ceiling on neural input height. Only bites on video taller than
//             it; the input is never below the source either way.
//   minRatio  share of the video's own frames SR must actually deliver, over a
//             measurement window, to stay on. Below it we step down a tier
//             (full model -> lite model -> original). 0 = never step down.
//   lite      start on the lite model instead of the full one.
//   headroom  also step down when the GPU is busy for more than this share of
//             each frame interval, even if every frame is still delivered: a
//             model that *just* fits keeps the GPU pinned at ~100%, with no
//             room left for the page, the compositor or anything else.
const POWER = {
  auto:    { cap: NEURAL_CAP, minRatio: 0.9,  lite: false, headroom: 0.8 },  // the default
  battery: { cap: 720,        minRatio: 0.97, lite: true,  headroom: 0.6 },  // fans and drain down
  max:     { cap: NEURAL_CAP, minRatio: 0,    lite: false, headroom: Infinity }, // quality over smoothness
};

// Keep-up measurement. The old governor compared a median per-frame wall time
// against the frame interval, which was wrong in both directions: the wall time
// included promise-resolution latency and first-frame shader compilation (false
// passthrough), and the loop ran on requestAnimationFrame, so a frame that took
// 1.02x the interval really delivered every other frame while still counting as
// "keeping up" (choppy SR left on). Leaving passthrough was decided by a lone
// probe frame on an idle GPU every 700ms, which is cheaper than sustained load,
// so it flapped on, stuttered, and flapped off.
//
// Now: count what the viewer sees -- SR frames delivered / frames the video
// presented (requestVideoFrameCallback's presentedFrames) -- over real playback
// windows, and leave passthrough only after a real sustained trial.
const WINDOW_MS = 2000;       // one measurement window of playback
const WARMUP_MS = 700;        // ignored after (re)configure: pipeline compile, first allocs
const FAIL_WINDOWS = 2;       // consecutive failing windows before giving up
const TRIAL_FIRST_MS = 4000;  // first retry after giving up; doubles per failed trial
const TRIAL_MAX_MS = 60000;
const GAP_MS = 400;           // a gap this long between frames = pause/seek, restart the window
const LITE_COST_RATIO = 2.8;  // full-model cost / lite-model cost, measured on an M4 Pro (14 ms / 5 ms at 720p)
const powerOf = () => POWER[settings.power] || POWER.auto;


let settings = {
  enabled: false, power: 'auto',
  targetScale: 2, autoPause: true, rememberState: true, showStats: true,
  onlyFullscreen: false, blockedSites: [], sharpness: 3.5, sharpnessCustom: false,
  autoEngage: true, showCompare: false, look: 'vivid',
};

// Models: a fast 2× and a native 4×. Target scale >2 uses the 4× model for real
// reconstruction; ≤2 uses the 2× model. (chrome.runtime is available here.)
const MODEL_2X = chrome.runtime.getURL('models/span_lite_2x_c16.bin');
const MODEL_4X = chrome.runtime.getURL('models/span_lite_4x_c16.bin');
// The lite tier: 8 channels instead of 16, ~3.3x cheaper per pixel (5.8ms vs
// 18.9ms at 720p on an M4 Pro). A GPU that can't hold the full model gets this
// before it gets the original video back. 2x only; 4x has no lite model.
const MODEL_2X_LITE = chrome.runtime.getURL('models/span_lite_2x_c8.bin');
let liteMissing = false;   // set if the lite weights fail to load; never retried
// Source heights at which the full model already failed on this page, so the
// next video at that size (playlist next, SPA navigation) starts on lite
// instead of stuttering for two windows first. Per page, never persisted: a
// different tab or a later session may have a quieter GPU.
const fullFailedAt = new Set();
const tierKey = (h) => h + '|' + (settings.power || 'auto') + '|' + (settings.targetScale || 2);
const liteAvailable = (s) => s <= 2 && !liteMissing;
const modelFor = (s, lite) => (s > 2 ? MODEL_4X : (lite && liteAvailable(s) ? MODEL_2X_LITE : MODEL_2X));

// ── Shared engine (one GPU device for the page) ───────────────────
let engine = null, enginePromise = null, engineError = null, activeOverlay = null;

async function getEngine() {
  if (engine) return engine;
  if (enginePromise) return enginePromise;
  enginePromise = (async () => {
    try {
      if (typeof WebGPUSR === 'undefined') throw new Error('webgpu-sr.js not loaded');
      if (!navigator.gpu) throw new Error('navigator.gpu missing (WebGPU disabled)');
      const e = new WebGPUSR();
      if (!await e.init()) throw new Error('no WebGPU adapter/device');
      // Start on the fast 2× model; switched to 4× on demand by target scale.
      await e.loadWeights(MODEL_2X);
      engine = e;
      console.log('[Crisp] Engine ready');
      return e;
    } catch (err) {
      engineError = err.message || String(err);
      console.error('[Crisp] Engine init failed:', err);
      enginePromise = null;
      return null;
    }
  })();
  return enginePromise;
}

class VideoOverlay {
  constructor(video) {
    this.video = video;
    this.active = false;
    this.processing = false;
    this.lastMs = 0;
    this.cfgInW = 0; this.cfgInH = 0; this.cfgDispW = 0; this.cfgDispH = 0;
    this._rvfcId = null;
    this._tick = null;
    this.comparing = false;
    this.cantKeepUp = false;   // SR can't match the source framerate → pass through
    this.resetGovernor();
    this._srcW = 0; this._srcH = 0;   // last seen intrinsic video resolution
    this._needsFirstFrame = true;     // hide SR output until a fresh frame is rendered
    this.buildUI();

    // Re-tune / reset when the media in this element changes without a page
    // reload: SPA navigation to another video, playlist "next", or an adaptive
    // quality switch. Keeps the same <video>, so we must react to its events.
    const onSrc = () => this.onSourceChange();
    this.video.addEventListener('loadstart', onSrc);
    this.video.addEventListener('emptied', onSrc);
    // Seeking while paused shows a new frame without playback, and
    // requestVideoFrameCallback is not guaranteed to fire for it -- measured
    // not firing in Chromium -- which left the previous SR frame covering the
    // new position. Enhance the seeked-to frame directly.
    this.video.addEventListener('seeked', () => {
      if (!this.active || !engine || !this.video.paused) return;
      if (this.gated() || (this.cantKeepUp && !this.trialing)) return;
      if (this.processing) this._pending = true;
      else this.processFrame();
    });

    // Remember-state: auto-enable where the user left it on.
    if (settings.rememberState && settings.enabled) {
      this.video.addEventListener('loadeddata', () => {
        if (!this.active) this.start();
      }, { once: true });
      if (this.video.readyState >= 2) this.start();
    }
  }

  // The <video> started loading different media. Drop stale SR output and reset
  // the governor/config so the next frames reconfigure for the new source.
  onSourceChange() {
    this._srcW = 0; this._srcH = 0;
    this.cfgInW = 0; this.cfgInH = 0; this.cfgDispW = 0; this.cfgDispH = 0;
    this.resetGovernor();
    this.processing = false;
    this._needsFirstFrame = true;
    this.outCanvas.style.display = 'none';   // reveal the original until SR is ready
  }

  // Forget everything measured about keeping up: new source, new resolution, or
  // a setting that changes the cost. A verdict about the old situation says
  // nothing about the new one.
  resetGovernor() {
    this.lite = !!((powerOf().lite || fullFailedAt.has(tierKey(this._srcH)))
                   && liteAvailable(settings.targetScale || 2));
    this._promoted = false;
    this.basic = false;        // no-network tier: bicubic + sharpen, for GPUs lite is too much for
    this.cantKeepUp = false;
    this.trialing = false;
    this._trialDelay = TRIAL_FIRST_MS;
    this._nextTrialAt = 0;
    this._failStreak = 0;
    this.lastRatio = null;
    this.srcFps = 0;
    this.restartWindow();
  }

  restartWindow(now = performance.now()) {
    this._winStart = now + WARMUP_MS;   // frames before this don't count
    this._winSrc = 0;                   // frames the video presented in the window
    this._winGpu = 0; this._winGpuN = 0; // GPU-timestamp ms of SR frames in the window
    this._winSr = 0;                    // SR frames finished in the window
    this._lastPresented = null;
    this._lastCb = now;
  }

  buildUI() {
    // Layer 1: output canvas, low z so site controls stay on top.
    this.canvasLayer = el('div', {
      position: 'absolute', inset: '0', pointerEvents: 'none', zIndex: '10', overflow: 'hidden',
    });
    this.outCanvas = el('canvas', {
      position: 'absolute', inset: '0', width: '100%', height: '100%',
      objectFit: 'contain', pointerEvents: 'none', display: 'none',
    });
    this.canvasLayer.appendChild(this.outCanvas);

    // Layer 2: UI, above site controls.
    this.uiLayer = el('div', {
      position: 'absolute', inset: '0', pointerEvents: 'none', zIndex: '2147483647',
    });

    // SR toggle button.
    this.btn = el('button', btnStyle());
    this.btn.innerHTML = '<span style="font-size:10px;font-weight:800;pointer-events:none">SR</span>';
    this.btn.title = 'Toggle Crisp (Alt+S)';
    this.btn.addEventListener('click', (e) => { e.stopPropagation(); e.preventDefault(); this.toggle(); });

    // Gear (settings) + compare buttons. Compare is opt-in (settings.showCompare);
    // spacing is set so the buttons don't crowd the SR toggle.
    this.gearBtn = el('button', miniBtnStyle('58px'));
    this.gearBtn.innerHTML = gearSvg();
    this.gearBtn.title = 'Crisp settings';
    this.gearBtn.addEventListener('click', (e) => { e.stopPropagation(); e.preventDefault(); this.toggleFlyout(); });

    this.cmpBtn = el('button', miniBtnStyle('98px'));
    this.cmpBtn.innerHTML = '<span style="font-size:13px;pointer-events:none">◐</span>';
    this.cmpBtn.title = 'Hold to compare with original';
    const cmpDown = (e) => { e.preventDefault(); this.setComparing(true); };
    const cmpUp = () => this.setComparing(false);
    this.cmpBtn.addEventListener('mousedown', cmpDown);
    this.cmpBtn.addEventListener('mouseup', cmpUp);
    this.cmpBtn.addEventListener('mouseleave', cmpUp);

    // Settings flyout panel.
    this.flyout = this.buildFlyout();

    // Stats HUD.
    this.statsEl = el('div', {
      position: 'absolute', top: '12px', left: '12px', padding: '7px 11px',
      background: 'rgba(255,255,255,0.6)', backdropFilter: 'blur(10px) saturate(135%)',
      border: '1px solid rgba(255,255,255,0.7)', borderRadius: '12px', color: '#242a33',
      fontFamily: '"SF Mono","Cascadia Code","Consolas",monospace', fontSize: '11px',
      lineHeight: '1.55', pointerEvents: 'none', display: 'none', whiteSpace: 'pre', opacity: '0',
      transition: 'opacity .25s ease',
      boxShadow: '0 6px 20px rgba(30,40,60,0.28), inset 0 1px 0 rgba(255,255,255,0.9)',
    });

    this.uiLayer.append(this.statsEl, this.flyout, this.cmpBtn, this.gearBtn, this.btn);
    this.gateControls();

    // Attach over the video.
    let target = this.video.parentElement;
    if (!target) return;
    for (let i = 0; i < 3 && target.parentElement; i++) {
      const r = target.getBoundingClientRect(), vr = this.video.getBoundingClientRect();
      if (r.width >= vr.width * 0.9 && r.height >= vr.height * 0.9) break;
      target = target.parentElement;
    }
    if (getComputedStyle(target).position === 'static') target.style.position = 'relative';
    target.append(this.canvasLayer, this.uiLayer);

    // Auto-hide: reveal the controls on pointer activity, then fade everything
    // out while the video plays undisturbed so nothing sits over the picture.
    target.addEventListener('mousemove', () => this.showChrome());
    target.addEventListener('mouseleave', () => {
      if (this.flyout.style.display === 'none' && !this.comparing) this.hideChrome();
    });
    this.showChrome();
  }

  buildFlyout() {
    const f = el('div', {
      position: 'absolute', top: '50%', left: '120px', transform: 'translateY(-50%)',
      background: 'rgba(255,255,255,0.6)', backdropFilter: 'blur(16px) saturate(135%)',
      border: '1px solid rgba(255,255,255,0.7)',
      borderRadius: '16px', padding: '14px', width: '196px', pointerEvents: 'auto',
      display: 'none', color: '#242a33',
      boxShadow: '0 16px 44px rgba(30,40,60,0.32), inset 0 1px 0 rgba(255,255,255,0.9)',
      fontFamily: '-apple-system,"Segoe UI",system-ui,sans-serif', fontSize: '12px',
    });
    f.addEventListener('click', (e) => e.stopPropagation());
    const label = (t) => el('div', { fontSize: '10px', textTransform: 'uppercase',
      letterSpacing: '0.09em', color: '#7a8494', margin: '12px 0 7px' }, t);

    f.appendChild(label('Power'));
    this.powerSeg = this.segment(['auto', 'battery', 'max'], ['Auto', 'Battery', 'Max'],
      () => settings.power || 'auto', (v) => setSetting({ power: v }));
    f.appendChild(this.powerSeg.root);

    f.appendChild(label('Look'));
    // A grade on top of the reconstruction, not part of it. 'Natural' is exactly
    // zero adjustment and stays the default, so the look never gets confused
    // with what the network actually recovers.
    this.lookSeg = this.segment(
      ['natural', 'bright', 'vivid', 'cinematic'],
      ['Natural', 'Bright', 'Vivid', 'Cinema'],
      () => settings.look || 'natural', (v) => setSetting({ look: v }));
    f.appendChild(this.lookSeg.root);

    const hint = el('div', { marginTop: '10px', fontSize: '10px', color: '#6b7280', lineHeight: '1.5' },
      'More options in the extension popup.');
    f.appendChild(hint);
    return f;
  }

  segment(values, labels, get, set) {
    const root = el('div', { position: 'relative', display: 'flex', gap: '4px', padding: '4px',
      background: 'rgba(120,132,155,0.18)', border: '1px solid rgba(255,255,255,0.5)',
      borderRadius: '11px', boxShadow: 'inset 0 1px 2px rgba(40,50,70,0.12)' });
    // Sliding white pill that animates to the selected option.
    const slider = el('div', {
      position: 'absolute', top: '4px', bottom: '4px', left: '0', width: '0',
      background: '#ffffff', borderRadius: '8px',
      boxShadow: '0 2px 8px rgba(45,55,80,0.2), inset 0 1px 0 rgba(255,255,255,1)',
      transition: 'left .28s cubic-bezier(.4,0,.2,1), width .28s cubic-bezier(.4,0,.2,1), opacity .2s ease',
      pointerEvents: 'none', zIndex: '0', opacity: '0',
    });
    root.appendChild(slider);
    const btns = values.map((v, i) => {
      const b = el('button', segBtnStyle(), labels[i]);
      b.style.position = 'relative'; b.style.zIndex = '1';
      b.addEventListener('click', (e) => { e.stopPropagation(); set(v); });
      root.appendChild(b);
      return { v, b };
    });
    const refresh = () => {
      let active = null;
      btns.forEach(({ v, b }) => {
        const on = get() === v;
        if (on) active = b;
        b.style.color = on ? '#242a33' : '#5c6472';
        b.style.fontWeight = on ? '700' : '500';
      });
      if (active) {
        slider.style.left = active.offsetLeft + 'px';
        slider.style.width = active.offsetWidth + 'px';
        slider.style.opacity = '1';
      } else { slider.style.opacity = '0'; }
    };
    refresh();
    return { root, refresh };
  }

  toggleFlyout() {
    if (this.flyout.style.display === 'none') this.openFlyout();
    else this.closeFlyout();
  }

  // Any press outside the flyout (and outside the gear, which toggles it
  // itself) closes it, as does Escape. Listened for in the capture phase on
  // pointerdown, so a site that stops click propagation -- most video players
  // do -- can't keep it open. composedPath() sees through shadow roots.
  openFlyout() {
    this.flyout.style.display = 'block';
    this.powerSeg.refresh(); this.lookSeg.refresh();
    this._onOutside = (e) => {
      const path = e.composedPath ? e.composedPath() : [];
      if (path.includes(this.flyout) || path.includes(this.gearBtn)) return;
      this.closeFlyout();
    };
    this._onEsc = (e) => { if (e.key === 'Escape') this.closeFlyout(); };
    document.addEventListener('pointerdown', this._onOutside, true);
    document.addEventListener('keydown', this._onEsc, true);
    // Leaving the page for another window/tab also counts as clicking away.
    window.addEventListener('blur', this._onOutside);
    this.showChrome();
  }

  closeFlyout() {
    this.flyout.style.display = 'none';
    if (this._onOutside) {
      document.removeEventListener('pointerdown', this._onOutside, true);
      document.removeEventListener('keydown', this._onEsc, true);
      window.removeEventListener('blur', this._onOutside);
      this._onOutside = this._onEsc = null;
    }
    this._armHide();
  }

  // Which controls may exist on screen at all (independent of the fade state).
  // Compare is opt-in and only while SR is running.
  gateControls() {
    this.gearBtn.style.display = 'flex';
    this.cmpBtn.style.display = (this.active && settings.showCompare) ? 'flex' : 'none';
  }

  _statsShouldShow() { return settings.showStats && (this.active || this.notWorthIt); }

  // Fade the controls + HUD in, and schedule them to fade back out.
  showChrome() {
    clearTimeout(this._hideTimer);
    this.btn.style.opacity = this.active ? '1' : '0.85';
    this.btn.style.pointerEvents = 'auto';
    this.gearBtn.style.opacity = '1'; this.gearBtn.style.pointerEvents = 'auto';
    if (this.active && settings.showCompare) {
      this.cmpBtn.style.opacity = '1'; this.cmpBtn.style.pointerEvents = 'auto';
    }
    if (this._statsShouldShow()) this.statsEl.style.opacity = '1';
    this._armHide();
  }

  _armHide() {
    clearTimeout(this._hideTimer);
    this._hideTimer = setTimeout(() => {
      // Stay put while paused, comparing, or the settings flyout is open.
      if (this.video.paused || this.comparing || this.flyout.style.display !== 'none') {
        this._armHide(); return;
      }
      this.hideChrome();
    }, 2600);
  }

  hideChrome() {
    clearTimeout(this._hideTimer);
    this.btn.style.opacity = '0'; this.btn.style.pointerEvents = 'none';
    this.gearBtn.style.opacity = '0'; this.gearBtn.style.pointerEvents = 'none';
    this.cmpBtn.style.opacity = '0'; this.cmpBtn.style.pointerEvents = 'none';
    this.statsEl.style.opacity = '0';
  }

  setComparing(on) {
    this.comparing = on;
    this.updateVisibility();   // reveal the original while held
  }

  async toggle() { if (this.active) this.stop(); else await this.start(); }

  async start() {
    this.active = true;
    this.styleBtn(true);
    this.gateControls();
    if (settings.showStats) { this.statsEl.style.display = 'block'; this.statsEl.textContent = 'Starting GPU…'; }
    this.showChrome();
    if (settings.rememberState) setSetting({ enabled: true });

    if (activeOverlay && activeOverlay !== this) activeOverlay.stop();
    activeOverlay = this;

    if (typeof this.video.requestVideoFrameCallback !== 'function') {
      this.fail('this browser has no requestVideoFrameCallback'); return;
    }
    const e = await Promise.race([getEngine(),
      new Promise((r) => setTimeout(() => r('__t__'), 10000))]);
    if (e === '__t__') { this.fail('GPU init timed out (10s)'); enginePromise = null; return; }
    if (!e) { this.fail(engineError || 'WebGPU init failed'); return; }
    if (!this.active) return;   // stopped while the engine was starting

    this.cfgInW = 0; this.cfgInH = 0; this.cfgDispW = 0; this.cfgDispH = 0;
    this._srcW = 0; this._srcH = 0; this._needsFirstFrame = true;
    this.resetGovernor();
    this.outCanvas.style.display = 'none';   // revealed after the first SR frame lands
    if (settings.showStats) this.statsEl.textContent = 'Processing…';
    this.schedule();
    // Visibility can change without a new frame (resize or fullscreen while
    // paused, the tab coming back), so re-check it on a slow timer too.
    clearInterval(this._tick);
    this._tick = setInterval(() => this.updateVisibility(), 500);
    // A frame is already on screen if the video is paused; enhance it now
    // rather than waiting for playback to present a new one.
    if (this.video.readyState >= 2 && !this.gated()) this.processFrame();
  }

  fail(msg) {
    this.statsEl.style.display = 'block';
    this.statsEl.style.opacity = '1';
    this.statsEl.textContent = 'Unavailable:\n' + msg;
    this.active = false; this.styleBtn(false); this.gateControls(); this.showChrome();
    this.unschedule();
  }

  stop() {
    this.active = false;
    this.styleBtn(false);
    this.gateControls();
    this.closeFlyout();
    this.unschedule();
    this.outCanvas.style.display = 'none';
    this.statsEl.style.display = 'none';
    this.showChrome();
    if (settings.rememberState) setSetting({ enabled: false });
    if (activeOverlay === this) activeOverlay = null;
  }

  styleBtn(on) {
    Object.assign(this.btn.style, on ? {
      background: 'linear-gradient(180deg, #3a4150, #2a2f3a)',
      color: '#fff', borderColor: 'rgba(255,255,255,0.35)',
      boxShadow: '0 6px 20px rgba(30,36,48,0.5), inset 0 1px 0 rgba(255,255,255,0.35)', opacity: '1',
    } : {
      background: 'rgba(255,255,255,0.55)', color: '#2b3242', borderColor: 'rgba(255,255,255,0.7)',
      boxShadow: '0 5px 16px rgba(30,40,60,0.28), inset 0 1px 0 rgba(255,255,255,0.9)', opacity: '0.85',
    });
  }

  // Frames are driven by the video itself: one callback per frame the browser
  // presents, instead of polling currentTime from requestAnimationFrame at the
  // display's refresh rate. Nothing runs while the video is paused.
  schedule() {
    if (this._rvfcId != null) return;
    this._rvfcId = this.video.requestVideoFrameCallback((now, meta) => this.onVideoFrame(now, meta));
  }

  unschedule() {
    if (this._rvfcId != null) {
      this.video.cancelVideoFrameCallback?.(this._rvfcId);
      this._rvfcId = null;
    }
    clearInterval(this._tick); this._tick = null;
  }

  gatedByFullscreen() {
    return settings.onlyFullscreen && !document.fullscreenElement;
  }

  // Reasons not to enhance at all right now (distinct from "can't keep up").
  gated() {
    return (settings.autoPause && document.hidden) || this.comparing
      || this.gatedByFullscreen() || this.notWorthIt;
  }

  // Device-pixel height of the picture itself inside the element: the element
  // box can be taller than the video when it is letterboxed.
  shownHeight() {
    const r = this.video.getBoundingClientRect();
    const vw = this.video.videoWidth, vh = this.video.videoHeight;
    if (!r.height || !vw || !vh) return 0;
    const cssH = Math.min(r.height, r.width * vh / vw);
    return Math.round(cssH * (window.devicePixelRatio || 1));
  }

  // Auto-engage: SR only helps when the source is clearly below the on-screen
  // size. If it's already ~screen resolution, skip it (no GPU spent for no gain).
  autoSkip() {
    if (!settings.autoEngage) return false;
    const vh = this.video.videoHeight;
    const dispH = this.shownHeight();
    if (!vh || dispH <= 0) return false;
    return vh >= dispH * 0.85;
  }

  // Show the SR canvas only when it holds a current, enhanced frame and we mean
  // to show it; otherwise the original video underneath is what's seen.
  updateVisibility() {
    if (!this.active) return;
    const wasSkipping = this.notWorthIt;
    this.notWorthIt = this.autoSkip();
    const hide = this.gated() || this.cantKeepUp || this.trialing || this._needsFirstFrame;
    const want = hide ? 'none' : '';
    if (this.outCanvas.style.display !== want) this.outCanvas.style.display = want;
    if (this.notWorthIt !== wasSkipping) this.restartWindow();
    if (settings.showStats) this.refreshStats();
  }

  onVideoFrame(now, meta) {
    this._rvfcId = null;
    if (!this.active) return;
    this.schedule();

    // presentedFrames counts every frame the compositor showed, including ones
    // whose callbacks were coalesced, so it is the true denominator.
    const presented = meta && meta.presentedFrames;
    if (now - this._lastCb > GAP_MS) this.restartWindow(now);   // resumed after a pause/seek
    this._lastCb = now;
    if (now >= this._winStart) {
      const known = presented != null && this._lastPresented != null;
      this._winSrc += known ? Math.max(1, presented - this._lastPresented) : 1;
    }
    this._lastPresented = presented ?? null;

    this.updateVisibility();
    if (this.gated()) { this.restartWindow(now); return; }

    // Given up: sit idle, and every so often run a real trial -- sustained
    // playback, canvas hidden -- rather than trusting a single probe frame.
    if (this.cantKeepUp && !this.trialing) {
      if (now < this._nextTrialAt) return;
      this.trialing = true;
      this.restartWindow(now);
    }

    // A frame that lands while the previous one is still being confirmed is not
    // dropped: it runs the moment that one completes. The confirmation
    // (onSubmittedWorkDone) resolves some milliseconds after the GPU actually
    // finishes, and skipping outright cost ~10% of frames at 720p30 on an M4
    // Pro with the GPU only 19ms busy per 33ms frame. When the GPU truly can't
    // keep up, this runs it back-to-back, and the window measures exactly that.
    if (this.processing) this._pending = true;
    else this.processFrame();
    this.maybeCloseWindow(now);
  }

  maybeCloseWindow(now) {
    const span = now - this._winStart;
    if (span < WINDOW_MS || this._winSrc < 8) return;
    const ratio = Math.min(1, this._winSr / this._winSrc);
    this.srcFps = this._winSrc * 1000 / span;
    this.lastRatio = ratio;
    const need = powerOf().minRatio;
    // Headroom is judged on GPU timestamps only: wall-clock time includes the
    // completion latency and would read as "busy" on an idle GPU.
    const gpuAvg = this._winGpuN ? this._winGpu / this._winGpuN : 0;
    this.gpuShare = gpuAvg && this.srcFps ? gpuAvg * this.srcFps / 1000 : 0;
    const tooBusy = this.gpuShare > powerOf().headroom;
    const ok = ratio >= need && !tooBusy;

    if (this.trialing) {
      this.trialing = false;
      if (ok) {
        this.cantKeepUp = false;
        this._trialDelay = TRIAL_FIRST_MS;
      } else {
        this._trialDelay = Math.min(this._trialDelay * 2, TRIAL_MAX_MS);
        this._nextTrialAt = now + this._trialDelay;
      }
    } else if (!this.cantKeepUp) {
      this._failStreak = ok ? 0 : this._failStreak + 1;
      if (this._failStreak >= FAIL_WINDOWS) {
        this._failStreak = 0;
        if (!this.lite && !this.basic && liteAvailable(settings.targetScale || 2)) {
          // Step down to the lite model rather than straight to the original.
          // The next frame swaps weights; the window restarts once it lands.
          this.lite = true;
          fullFailedAt.add(tierKey(this._srcH));
        } else if (!this.basic) {
          // Then to no network at all. Still sharper than the browser's own
          // scaling, and costs a few ms even on an integrated GPU.
          this.basic = true;
          this.restartWindow(now);
        } else {
          this.cantKeepUp = true;
          this._nextTrialAt = now + this._trialDelay;
        }
      } else if (ok && this.lite && !powerOf().lite && !this._promoted && this.lastGpuMs
                 && this.lastGpuMs * LITE_COST_RATIO < 0.7 * (1000 / this.srcFps)) {
        // Demoted by a hiccup (a heavy page, another tab) with plenty of room
        // now: try the full model once more. Once per reset, so a GPU that sits
        // on the boundary can't bounce between the two.
        this._promoted = true;
        this.lite = false;
      }
    }
    // Next window starts immediately; only a (re)configure needs a warm-up.
    this._winStart = now; this._winSrc = 0; this._winSr = 0;
    this._winGpu = 0; this._winGpuN = 0;
    this.updateVisibility();
  }

  processFrame() {
    const vw = this.video.videoWidth, vh = this.video.videoHeight;
    if (!vw || !vh) return;
    if (this.video.readyState < 2) return;   // no decoded frame yet (mid source-switch)

    // Intrinsic resolution changed (adaptive-streaming / quality switch, or a new
    // source in the same element): re-tune the governor and force a reconfigure so
    // we upscale from the video's *current* native size, not a stale one.
    if (vw !== this._srcW || vh !== this._srcH) {
      this._srcW = vw; this._srcH = vh;
      this.cfgInW = 0; this.cfgInH = 0;
      this.resetGovernor();
      this._needsFirstFrame = true;
    }

    // Load the model for the target scale (2x vs native 4x) and tier (full vs
    // lite). Swap is async; skip frames while it happens. A swap only restarts
    // the measurement window: resetting the governor here would put the tier
    // back to full and undo the step-down that asked for the swap.
    const want = modelFor(settings.targetScale || 2, this.lite);
    if (!this.basic && engine._modelUrl !== want && !this._switching) {
      this._switching = true;
      const done = () => { this.cfgInW = 0; this._switching = false; this.restartWindow(); };
      engine.switchModel(want)
        .then(done)
        .catch(() => {
          if (want === MODEL_2X_LITE) {
            // No usable lite weights. If this was a step-down, take the next
            // one (basic); if lite was just the Battery default, run full.
            liteMissing = true;
            if (this.lite && !powerOf().lite) this.basic = true;
            this.lite = false;
          }
          if (want !== MODEL_2X) {
            engine.switchModel(MODEL_2X).then(done).catch(() => { this._switching = false; });
          } else { this._switching = false; }
        });
    }
    if (this._switching) return;

    this.processing = true;
    const t0 = performance.now();
    try {
      // Never feed the network fewer pixels than the source has. Anything below
      // `srcH` is reconstructing detail we destroyed ourselves, so it is worse
      // than passthrough by construction; when SR cannot keep up at this height
      // the governor passes the original through instead.
      const srcH = Math.min(NEURAL_CAP, vh);
      const ceilH = Math.min(srcH, powerOf().cap);
      const nH = Math.max(MIN_NEURAL, ceilH) & ~1;
      const nW = Math.round(vw * nH / vh) & ~1;

      // Finish at the size the video is actually shown, never larger: pixels
      // past that are thrown away by the compositor after we paid for the
      // resample and the sharpen. A 1080p video in a 1440-device-pixel-tall
      // player was finishing at 2160 -- 2.25x the sharpen work for nothing.
      // Fullscreen or a resize changes the shown size, and the check below
      // reconfigures on the next frame.
      const scale = settings.targetScale || 2;
      const shownH = this.shownHeight();
      let dH = Math.min(Math.round(vh * scale), DISPLAY_CAP);
      if (shownH > 0) dH = Math.min(dH, Math.max(vh, shownH));
      dH &= ~1;
      const dW = Math.round(dH * vw / vh) & ~1;
      this._nW = nW; this._nH = nH; this._dW = dW; this._dH = dH;

      if (nW !== this.cfgInW || nH !== this.cfgInH || dW !== this.cfgDispW || dH !== this.cfgDispH
          || this.basic !== this.cfgBasic) {
        engine.configure(this.outCanvas, nW, nH, dW, dH, { neural: !this.basic });
        this.cfgInW = nW; this.cfgInH = nH; this.cfgDispW = dW; this.cfgDispH = dH;
        this.cfgBasic = this.basic;
        this.restartWindow();   // new buffers and a pipeline warm-up: don't score them
      }
      engine.sharpen = settings.sharpness || 0;
      if (engine.setLook) engine.setLook(settings.look || 'natural');
      if (this.basic) engine.renderBasic(this.video);
      else engine.render(this.video);
    } catch (err) {
      console.error('[Crisp] Frame error:', err);
      this.fail(err.message);
      this.processing = false;
      return;
    }

    engine.device.queue.onSubmittedWorkDone().then(() => {
      this.processing = false;
      if (!this.active) return;
      if (!engine.ready) { this.engineLost(); return; }
      if (engine.error) { const msg = engine.error; engine.error = null; this.gpuError(msg); return; }
      this.lastMs = performance.now() - t0;
      this.lastGpuMs = engine.gpuMs || this.lastMs;
      if (performance.now() >= this._winStart) {
        this._winSr++;
        if (engine.hasTS && engine.gpuMs > 0) { this._winGpu += engine.gpuMs; this._winGpuN++; }
      }
      this._needsFirstFrame = false;   // a valid frame is now on the canvas, safe to show
      this.updateVisibility();
      if (this._pending) {
        this._pending = false;
        if (!this.gated() && (!this.cantKeepUp || this.trialing)) this.processFrame();
      }
    }).catch(() => { this.processing = false; });
  }

  // The GPU rejected the frame. The likeliest cause is memory -- the network's
  // feature buffers are the only large allocation -- so fall back to the basic
  // tier, which has none. If basic fails too, stop and say so.
  gpuError(msg) {
    console.warn('[Crisp] frame rejected by the GPU:', msg);
    if (this.basic) { this.fail('GPU error: ' + msg); return; }
    this.basic = true;
    this.cfgBasic = null;       // force a reconfigure on the next frame
    this._needsFirstFrame = true;
    this.restartWindow();
    this.updateVisibility();
  }

  // The device is gone (driver reset, GPU switch, sleep). Everything built on
  // it is dead, so drop the shared engine; the next start makes a new one.
  engineLost() {
    engine = null; enginePromise = null;
    this.fail('the GPU was reset.\nClick SR to start again.');
  }

  refreshStats() {
    if (!this.active || !settings.showStats) return;
    const head = '<span style="color:#2b3242;font-weight:700">Crisp</span> ';
    if (this.notWorthIt) {
      this.statsEl.innerHTML = head + '<span style="color:#7a8494">idle</span>\n' +
        'already about as sharp as\nyour screen, nothing to fix here';
      return;
    }
    const fps = Math.round(this.srcFps);
    if (this.cantKeepUp) {
      const pct = this.lastRatio != null ? Math.round(this.lastRatio * 100) + '%' : '';
      const retry = Math.max(0, Math.ceil((this._nextTrialAt - performance.now()) / 1000));
      this.statsEl.innerHTML = head + '<span style="color:#c07d2a">passthrough</span>\n' +
        'your GPU kept up with ' + (pct || 'too few') + ' of frames,\nshowing the original instead\n' +
        (this.trialing ? 'testing again now…' : 'trying again in ' + retry + 's');
      return;
    }
    const nat = this.basic ? ' · basic, no AI'
      : (this._nH >= this.video.videoHeight ? ' native' : '') + (this.lite ? ' · lite' : '');
    const shp = (settings.sharpness || 0) > 0.01 ? ' · sharp' : '';
    // Delivered frames, not 1000/frame-time: that figure could read "45fps" on a
    // 30fps video, or look fine while half the frames were being dropped.
    const kept = this.lastRatio == null ? 'measuring…'
      : Math.round(this.lastRatio * fps) + '/' + fps + ' fps';
    this.statsEl.innerHTML = head + (settings.power || 'auto') +
      ' <span style="color:#6b7280">' + kept + shp + '</span>\n' +
      'SR ' + this._nW + '×' + this._nH + nat + ' → ' + this._dW + '×' + this._dH + '\n' +
      Math.round(this.lastGpuMs || this.lastMs) + 'ms gpu' +
      (fps ? ' · ' + Math.round(1000 / fps) + 'ms/frame' : '');
  }

  applySettings() {
    if (!settings.showStats) this.statsEl.style.display = 'none';
    else if (this.active) this.statsEl.style.display = 'block';
    this.gateControls();
    this.showChrome();
    if (this.flyout.style.display !== 'none') { this.powerSeg.refresh(); this.lookSeg.refresh(); }
    // Power and scale change what a frame costs, so an earlier "can't keep up"
    // (or "fine") verdict no longer applies. Measure again from scratch.
    const costKey = (settings.power || 'auto') + '|' + (settings.targetScale || 2);
    if (this._costKey && this._costKey !== costKey) this.resetGovernor();
    this._costKey = costKey;
    this.updateVisibility();
  }

  destroy() {
    this.stop();
    this.canvasLayer?.remove();
    this.uiLayer?.remove();
  }
}

// ── Style helpers ────────────────────────────────────────────────
function el(tag, style, text) {
  const e = document.createElement(tag);
  if (style) Object.assign(e.style, style);
  if (text != null) e.textContent = text;
  return e;
}
function btnStyle() {
  return {
    position: 'absolute', top: '50%', left: '12px', transform: 'translateY(-50%)',
    width: '38px', height: '38px', borderRadius: '50%', border: '1px solid rgba(255,255,255,0.7)',
    background: 'rgba(255,255,255,0.52)', backdropFilter: 'blur(9px) saturate(135%)', color: '#2b3242',
    fontFamily: '-apple-system,"Segoe UI",system-ui,sans-serif', fontSize: '10px', fontWeight: '800',
    letterSpacing: '0.06em', cursor: 'pointer', pointerEvents: 'auto', opacity: '0.85',
    boxShadow: '0 5px 16px rgba(30,40,60,0.28), inset 0 1px 0 rgba(255,255,255,0.9)',
    transition: 'opacity .15s, background .18s, color .15s, border-color .18s, box-shadow .2s',
    outline: 'none', padding: '0', display: 'flex', alignItems: 'center', justifyContent: 'center', lineHeight: '1',
  };
}
function miniBtnStyle(left) {
  return {
    position: 'absolute', top: '50%', left, transform: 'translateY(-50%)',
    width: '30px', height: '30px', borderRadius: '50%', border: '1px solid rgba(255,255,255,0.7)',
    background: 'rgba(255,255,255,0.52)', backdropFilter: 'blur(9px) saturate(135%)', color: '#2b3242',
    cursor: 'pointer', pointerEvents: 'none', opacity: '0',
    boxShadow: '0 4px 12px rgba(30,40,60,0.25), inset 0 1px 0 rgba(255,255,255,0.9)',
    transition: 'opacity .2s ease, background .18s, border-color .18s, box-shadow .2s',
    display: 'none', alignItems: 'center', justifyContent: 'center', padding: '0', outline: 'none',
  };
}
function segBtnStyle() {
  return {
    flex: '1', padding: '6px 0', background: 'transparent', border: '1px solid transparent',
    borderRadius: '8px', color: '#5c6472', fontSize: '11px', fontWeight: '500', cursor: 'pointer', outline: 'none',
    transition: 'background .18s, color .18s, box-shadow .2s',
    fontFamily: '-apple-system,"Segoe UI",system-ui,sans-serif',
  };
}
function gearSvg() {
  return '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
    'stroke-width="2" style="pointer-events:none"><circle cx="12" cy="12" r="3"/>' +
    '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>';
}

// ── Settings plumbing ────────────────────────────────────────────
function setSetting(patch) {
  Object.assign(settings, patch);
  chrome.runtime.sendMessage({ type: 'SET_SETTINGS', patch }, () => void chrome.runtime.lastError);
  onSettings();
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'SETTINGS_CHANGED' && msg.settings) {
    settings = msg.settings;
    onSettings();
  }
});

const siteBlocked = () => (settings.blockedSites || []).includes(location.hostname);

// Blocking a site from the popup takes effect on open tabs immediately, and
// unblocking brings the controls back without a reload.
function onSettings() {
  if (siteBlocked()) {
    overlays.forEach((o) => o.destroy());
    overlays.clear();
    return;
  }
  overlays.forEach((o) => o.applySettings?.());
  scan();
}

// ── Video discovery ──────────────────────────────────────────────
const overlays = new Map();

function hookVideo(video) {
  if (overlays.has(video) || siteBlocked()) return;
  if (!video.videoWidth || !video.videoHeight) {
    video.addEventListener('loadedmetadata', () => hookVideo(video), { once: true });
    return;
  }
  const rect = video.getBoundingClientRect();
  if (rect.width < 100 || rect.height < 60) return;
  overlays.set(video, new VideoOverlay(video));
}
// Drop overlays whose <video> left the document. Checking isConnected catches a
// video removed inside a container, which is how single-page sites remove them;
// matching only removed nodes that are themselves <video> missed those and
// leaked the overlay and its GPU canvas.
function prune() {
  overlays.forEach((o, v) => { if (!v.isConnected) { o.destroy(); overlays.delete(v); } });
}
function scan() {
  prune();
  if (siteBlocked() || !document.body) return;
  document.querySelectorAll('video').forEach(hookVideo);
}

const mo = new MutationObserver((muts) => {
  for (const m of muts) {
    for (const n of m.addedNodes) {
      if (n.nodeName === 'VIDEO') hookVideo(n);
      else if (n.querySelectorAll) n.querySelectorAll('video').forEach(hookVideo);
    }
    if (m.removedNodes.length) prune();
  }
});

// Keyboard shortcut: Alt+S toggles the largest visible video's SR.
document.addEventListener('keydown', (e) => {
  if (e.altKey && (e.key === 's' || e.key === 'S')) {
    let best = null, bestArea = 0;
    overlays.forEach((o) => {
      const r = o.video.getBoundingClientRect();
      const a = r.width * r.height;
      if (a > bestArea) { bestArea = a; best = o; }
    });
    if (best) { e.preventDefault(); best.toggle(); }
  }
});

function init() {
  if (siteBlocked()) console.log('[Crisp] disabled on', location.hostname);
  // Observe even when blocked, so unblocking from the popup needs no reload;
  // hookVideo refuses while the site is blocked.
  if (document.body) { mo.observe(document.body, { childList: true, subtree: true }); scan(); }
}

// Load settings, then start.
chrome.runtime.sendMessage({ type: 'GET_SETTINGS' }, (s) => {
  if (s && !chrome.runtime.lastError) settings = s;
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
});
setInterval(scan, 2000);

console.log('[Crisp] Content script loaded');
