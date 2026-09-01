/**
 * WebVSR Content Script
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
const PERF_BUDGET = { light: 0.55, balanced: 0.85, max: 100 };
// Model intensity → ceiling on internal resolution as a fraction of native.
const QUALITY_FRAC = { fast: 0.5, medium: 0.7, quality: 1.0 };

let settings = {
  enabled: false, perfMode: 'max', quality: 'quality',
  targetScale: 2, autoPause: true, rememberState: true, showStats: true,
  onlyFullscreen: false, blockedSites: [], sharpness: 1.4, sharpnessCustom: false,
  autoEngage: true, showCompare: false,
};

// Models: a fast 2× and a native 4×. Target scale >2 uses the 4× model for real
// reconstruction; ≤2 uses the 2× model. (chrome.runtime is available here.)
const MODEL_2X = chrome.runtime.getURL('models/span_lite_2x_c16.bin');
const MODEL_4X = chrome.runtime.getURL('models/span_lite_4x_c16.bin');
const modelForScale = (s) => (s > 2 ? MODEL_4X : MODEL_2X);

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
      console.log('[WebVSR] Engine ready');
      return e;
    } catch (err) {
      engineError = err.message || String(err);
      console.error('[WebVSR] Engine init failed:', err);
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
    this.animId = null;
    this.frameTimes = [];
    this.lastMs = 0;
    this.srH = START_NEURAL;
    this.budgetMs = 30;
    this.cfgInW = 0; this.cfgInH = 0; this.cfgDispW = 0; this.cfgDispH = 0;
    this.lastMediaTime = -1;
    this._rvfcOn = false;
    this.comparing = false;
    this.cantKeepUp = false;   // SR can't match the source framerate → pass through
    this._lastProbe = 0;
    this._srcW = 0; this._srcH = 0;   // last seen intrinsic video resolution
    this._needsFirstFrame = true;     // hide SR output until a fresh frame is rendered
    this.buildUI();

    // Re-tune / reset when the media in this element changes without a page
    // reload: SPA navigation to another video, playlist "next", or an adaptive
    // quality switch. Keeps the same <video>, so we must react to its events.
    const onSrc = () => this.onSourceChange();
    this.video.addEventListener('loadstart', onSrc);
    this.video.addEventListener('emptied', onSrc);

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
    this.srH = START_NEURAL;
    this.lastMediaTime = -1;
    this.frameTimes = [];
    this.cantKeepUp = false;
    this.processing = false;
    this._needsFirstFrame = true;
    this.outCanvas.style.display = 'none';   // reveal the original until SR is ready
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
    this.btn.title = 'Toggle WebVSR (Alt+S)';
    this.btn.addEventListener('click', (e) => { e.stopPropagation(); e.preventDefault(); this.toggle(); });

    // Gear (settings) + compare buttons. Compare is opt-in (settings.showCompare);
    // spacing is set so the buttons don't crowd the SR toggle.
    this.gearBtn = el('button', miniBtnStyle('58px'));
    this.gearBtn.innerHTML = gearSvg();
    this.gearBtn.title = 'WebVSR settings';
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

    f.appendChild(label('GPU load'));
    this.perfSeg = this.segment(['light', 'balanced', 'max'], ['Light', 'Balanced', 'Max'],
      () => settings.perfMode, (v) => setSetting({ perfMode: v }));
    f.appendChild(this.perfSeg.root);

    f.appendChild(label('Quality'));
    this.qualSeg = this.segment(['fast', 'medium', 'quality'], ['Fast', 'Medium', 'Quality'],
      () => settings.quality, (v) => setSetting({ quality: v }));
    f.appendChild(this.qualSeg.root);

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
    const show = this.flyout.style.display === 'none';
    this.flyout.style.display = show ? 'block' : 'none';
    if (show) { this.perfSeg.refresh(); this.qualSeg.refresh(); }
    this.showChrome();
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
    if (this.active) this.outCanvas.style.display = on ? 'none' : '';  // reveal original while held
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

    const e = await Promise.race([getEngine(),
      new Promise((r) => setTimeout(() => r('__t__'), 10000))]);
    if (e === '__t__') { this.fail('GPU init timed out (10s)'); enginePromise = null; return; }
    if (!e) { this.fail(engineError || 'WebGPU init failed'); return; }

    this.cfgInW = 0; this.cfgInH = 0; this.cfgDispW = 0; this.cfgDispH = 0;
    this.srH = START_NEURAL; this.frameTimes = [];
    this._srcW = 0; this._srcH = 0; this._needsFirstFrame = true;
    this.outCanvas.style.display = 'none';   // the loop reveals it after the first frame
    if (settings.showStats) this.statsEl.textContent = 'Processing…';
    this.measureFps();
    this.loop();
  }

  fail(msg) {
    this.statsEl.style.display = 'block';
    this.statsEl.style.opacity = '1';
    this.statsEl.textContent = 'Unavailable:\n' + msg;
    this.active = false; this.styleBtn(false); this.gateControls(); this.showChrome();
  }

  stop() {
    this.active = false;
    this.styleBtn(false);
    this.gateControls();
    this.flyout.style.display = 'none';
    if (this.animId) { cancelAnimationFrame(this.animId); this.animId = null; }
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

  measureFps() {
    if (this._rvfcOn || typeof this.video.requestVideoFrameCallback !== 'function') return;
    this._rvfcOn = true;
    let count = 0, t0 = 0;
    const cb = (now) => {
      if (!this.active) { this._rvfcOn = false; return; }
      if (!t0) t0 = now;
      count++;
      const el2 = now - t0;
      if (el2 >= 1000) {
        const fps = count * 1000 / el2;
        this._frameInterval = 1000 / Math.min(fps, 60);
        count = 0; t0 = now;
      }
      this.video.requestVideoFrameCallback(cb);
    };
    this.video.requestVideoFrameCallback(cb);
  }

  gatedByFullscreen() {
    return settings.onlyFullscreen && !document.fullscreenElement;
  }

  paused() {
    return this.video.paused || this.video.ended ||
      (settings.autoPause && document.hidden) || this.comparing || this.gatedByFullscreen();
  }

  // Auto-engage: SR only helps when the source is clearly below the on-screen
  // size. If it's already ~screen resolution, skip it (no GPU spent for no gain).
  autoSkip() {
    if (!settings.autoEngage) return false;
    const vh = this.video.videoHeight;
    if (!vh) return false;
    const rect = this.video.getBoundingClientRect();
    const dispH = (rect.height || 0) * (window.devicePixelRatio || 1);
    if (dispH <= 0) return false;
    return vh >= dispH * 0.85;
  }

  loop() {
    if (!this.active) return;
    this.notWorthIt = this.autoSkip();
    // Reveal the original when gated, comparing, can't-keep-up, not worth it, or
    // while we're still waiting on the first rendered frame after a (re)start.
    const hide = this.gatedByFullscreen() || this.comparing || this.cantKeepUp
      || this.notWorthIt || this._needsFirstFrame;
    if (hide && this.outCanvas.style.display !== 'none') this.outCanvas.style.display = 'none';
    else if (!hide && this.outCanvas.style.display === 'none') this.outCanvas.style.display = '';
    if (this.notWorthIt && settings.showStats) {
      this.statsEl.innerHTML =
        '<span style="color:#2b3242;font-weight:700">WebVSR</span> <span style="color:#7a8494">idle</span>\n' +
        'already about as sharp as\nyour screen, nothing to fix here';
    }

    if (!this.processing && !this.paused() && !this.notWorthIt
        && this.video.currentTime !== this.lastMediaTime) {
      if (this.cantKeepUp) {
        const now = performance.now();
        if (now - this._lastProbe > 700) { this._lastProbe = now; this.processFrame(); }
      } else {
        this.processFrame();
      }
    }
    this.animId = requestAnimationFrame(() => this.loop());
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
      this.srH = START_NEURAL;
      this.cfgInW = 0; this.cfgInH = 0;
      this.frameTimes = [];
      this.cantKeepUp = false;
      this._needsFirstFrame = true;
    }

    // Load the model that matches the target scale (2× vs native 4×). Swap is
    // async; skip frames while it happens, and fall back to 2× if 4× is missing.
    const want = modelForScale(settings.targetScale || 2);
    if (engine._modelUrl !== want && !this._switching) {
      this._switching = true;
      engine.switchModel(want)
        .then(() => { this.cfgInW = 0; this._switching = false; })
        .catch(() => {
          if (want !== MODEL_2X) {
            engine.switchModel(MODEL_2X).then(() => { this.cfgInW = 0; this._switching = false; })
              .catch(() => { this._switching = false; });
          } else { this._switching = false; }
        });
    }
    if (this._switching) return;

    this.processing = true;
    this.lastMediaTime = this.video.currentTime;
    const t0 = performance.now();
    try {
      const frameInt = this._frameInterval || 33;
      this.budgetMs = frameInt * (PERF_BUDGET[settings.perfMode] ?? 0.85);
      const ceilH = Math.min(NEURAL_CAP, vh, Math.round(vh * (QUALITY_FRAC[settings.quality] ?? 1)));

      let nH = Math.min(Math.round(this.srH / NEURAL_STEP) * NEURAL_STEP, ceilH);
      nH = Math.max(MIN_NEURAL, nH) & ~1;
      let nW = Math.round(vw * nH / vh) & ~1;

      const scale = settings.targetScale || 2;
      let dH = Math.min(Math.round(vh * scale), DISPLAY_CAP);
      let dW = Math.round(dH * vw / vh);
      this._nW = nW; this._nH = nH; this._dW = dW; this._dH = dH; this._ceilH = ceilH;

      if (nW !== this.cfgInW || nH !== this.cfgInH || dW !== this.cfgDispW || dH !== this.cfgDispH) {
        engine.configure(this.outCanvas, nW, nH, dW, dH);
        this.cfgInW = nW; this.cfgInH = nH; this.cfgDispW = dW; this.cfgDispH = dH;
      }
      engine.sharpen = settings.sharpness || 0;
      engine.render(this.video);
    } catch (err) {
      console.error('[WebVSR] Frame error:', err);
      this.fail(err.message);
      this.processing = false;
      return;
    }

    engine.device.queue.onSubmittedWorkDone().then(() => {
      // Govern on real wall-clock time (includes frame import + overhead), so the
      // governor targets true throughput and can actually match 60fps sources.
      const wallMs = performance.now() - t0;
      this.lastMs = wallMs;
      this.lastGpuMs = engine.gpuMs || wallMs;
      this._needsFirstFrame = false;   // a valid frame is now on the canvas, safe to show
      this.adjust(wallMs);
      if (settings.showStats) this.refreshStats();
      this.processing = false;
    }).catch(() => { this.processing = false; });
  }

  // Governor: keep the wall-clock frame time near the budget (a fraction of the
  // video's frame interval). cost ∝ pixels², so target srH·√(budget/measured);
  // damped, with a deadband. If pinned at min res and still can't match the
  // source framerate, mark cantKeepUp → the loop passes the original through
  // (never show choppy SR over smooth video).
  adjust(dt) {
    this.frameTimes.push(dt);
    if (this.frameTimes.length > 8) this.frameTimes.shift();
    const s = [...this.frameTimes].sort((a, b) => a - b);
    const med = s[s.length >> 1] || dt;
    const frameInt = this._frameInterval || 33;

    if (settings.perfMode === 'max') {           // no cap: climb to the ceiling
      this.srH = Math.min(this._ceilH, this.srH + (this._ceilH - this.srH) * 0.5);
      // 'max' lifts the *resolution* cap, not the promise that SR never makes
      // playback worse. At the ceiling there is no lower internal res left to
      // drop to, so passthrough is the only remedy there is -- keep it armed.
      if (med > frameInt * 1.25) this.cantKeepUp = true;
      else if (med < frameInt * 0.95) this.cantKeepUp = false;
      return;
    }
    const ratio = med / this.budgetMs;
    if (ratio > 1.05 || ratio < 0.9) {
      let target = this.srH * Math.sqrt(this.budgetMs / med);
      target = Math.max(MIN_NEURAL, Math.min(this._ceilH, target));
      this.srH += (target - this.srH) * 0.5;
    }
    // Passthrough safety net (hysteresis): can't hit source fps even at min res.
    const atMin = Math.round(this.srH / NEURAL_STEP) * NEURAL_STEP <= MIN_NEURAL;
    if (atMin && med > frameInt * 1.25) this.cantKeepUp = true;
    else if (med < frameInt * 0.95) this.cantKeepUp = false;
  }

  refreshStats() {
    if (!this.active) return;
    const gpu = Math.round(this.lastGpuMs || this.lastMs);
    if (this.cantKeepUp) {
      this.statsEl.innerHTML =
        '<span style="color:#2b3242;font-weight:700">WebVSR</span> ' +
        '<span style="color:#c07d2a">passthrough</span>\n' +
        'too much to keep up with here,\nshowing the original instead';
      return;
    }
    const avg = this.frameTimes.length
      ? this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length : 0;
    const fps = avg > 0 ? (1000 / avg).toFixed(0) : '…';
    const nat = this._nH >= this.video.videoHeight ? ' native' : '';
    const shp = (settings.sharpness || 0) > 0.01 ? ' · sharp' : '';
    this.statsEl.innerHTML =
      '<span style="color:#2b3242;font-weight:700">WebVSR</span> ' + settings.perfMode +
      ' <span style="color:#6b7280">' + fps + 'fps' + shp + '</span>\n' +
      'SR ' + this._nW + '×' + this._nH + nat + ' → ' + this._dW + '×' + this._dH + '\n' +
      gpu + 'ms gpu · ' + Math.round(this.budgetMs) + 'ms budget';
  }

  applySettings() {
    if (!settings.showStats) this.statsEl.style.display = 'none';
    else if (this.active) this.statsEl.style.display = 'block';
    this.gateControls();
    this.showChrome();
    if (this.flyout.style.display !== 'none') { this.perfSeg.refresh(); this.qualSeg.refresh(); }
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
  overlays.forEach((o) => o.applySettings?.());
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'SETTINGS_CHANGED' && msg.settings) {
    settings = msg.settings;
    overlays.forEach((o) => o.applySettings?.());
  }
});

// ── Video discovery ──────────────────────────────────────────────
const overlays = new Map();

function hookVideo(video) {
  if (overlays.has(video)) return;
  if (!video.videoWidth || !video.videoHeight) {
    video.addEventListener('loadedmetadata', () => hookVideo(video), { once: true });
    return;
  }
  const rect = video.getBoundingClientRect();
  if (rect.width < 100 || rect.height < 60) return;
  overlays.set(video, new VideoOverlay(video));
}
function scan() { document.querySelectorAll('video').forEach(hookVideo); }

const mo = new MutationObserver((muts) => {
  for (const m of muts) {
    for (const n of m.addedNodes) {
      if (n.nodeName === 'VIDEO') hookVideo(n);
      else if (n.querySelectorAll) n.querySelectorAll('video').forEach(hookVideo);
    }
    for (const n of m.removedNodes) {
      if (n.nodeName === 'VIDEO' && overlays.has(n)) {
        overlays.get(n).destroy(); overlays.delete(n);
      }
    }
  }
});

// Pause processing immediately when the tab is hidden (saves GPU).
document.addEventListener('visibilitychange', () => {});

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
  if ((settings.blockedSites || []).includes(location.hostname)) {
    console.log('[WebVSR] disabled on', location.hostname);
    return;
  }
  if (document.body) { mo.observe(document.body, { childList: true, subtree: true }); scan(); }
}

// Load settings, then start.
chrome.runtime.sendMessage({ type: 'GET_SETTINGS' }, (s) => {
  if (s && !chrome.runtime.lastError) settings = s;
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
});
setInterval(scan, 2000);

console.log('[WebVSR] Content script loaded');
