/**
 * WebVSR Content Script
 *
 * Detects <video> elements, adds a floating SR button + settings flyout, and
 * runs the in-page WebGPU engine (webgpu-sr.js). Smoothness is the priority:
 * a frame-time governor keeps the neural net inside the video's frame budget so
 * playback never bogs down. Settings come from the background service worker.
 */

const DISPLAY_CAP = 2160;   // max finished (on-screen) height
const NEURAL_CAP = 720;     // hard ceiling on neural input height (safety/VRAM)
const MIN_NEURAL = 144;
const NEURAL_STEP = 16;
const START_NEURAL = 216;   // conservative start so first frames never stall

// GPU-load presets → fraction of the video's frame interval the net may use.
// Lower = smoother/lighter; 'max' removes the cap (native res, high GPU).
const PERF_BUDGET = { light: 0.55, balanced: 0.85, max: 100 };
// Model intensity → ceiling on internal resolution as a fraction of native.
const QUALITY_FRAC = { fast: 0.5, medium: 0.7, quality: 1.0 };

let settings = {
  enabled: false, perfMode: 'balanced', quality: 'quality',
  targetScale: 2, autoPause: true, rememberState: true, showStats: true,
  onlyFullscreen: false, blockedSites: [], sharpness: 0.35, autoEngage: true,
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
    this.buildUI();

    // Remember-state: auto-enable where the user left it on.
    if (settings.rememberState && settings.enabled) {
      this.video.addEventListener('loadeddata', () => {
        if (!this.active) this.start();
      }, { once: true });
      if (this.video.readyState >= 2) this.start();
    }
  }

  buildUI() {
    // Layer 1: output canvas — low z so site controls stay on top.
    this.canvasLayer = el('div', {
      position: 'absolute', inset: '0', pointerEvents: 'none', zIndex: '10', overflow: 'hidden',
    });
    this.outCanvas = el('canvas', {
      position: 'absolute', inset: '0', width: '100%', height: '100%',
      objectFit: 'contain', pointerEvents: 'none', display: 'none',
    });
    this.canvasLayer.appendChild(this.outCanvas);

    // Layer 2: UI — above site controls.
    this.uiLayer = el('div', {
      position: 'absolute', inset: '0', pointerEvents: 'none', zIndex: '2147483647',
    });

    // SR toggle button.
    this.btn = el('button', btnStyle());
    this.btn.innerHTML = '<span style="font-size:10px;font-weight:800;pointer-events:none">SR</span>';
    this.btn.title = 'Toggle WebVSR (Alt+S)';
    this.btn.addEventListener('mouseenter', () => { this.btn.style.opacity = '1'; this.showControls(true); });
    this.btn.addEventListener('mouseleave', () => { this.btn.style.opacity = this.active ? '1' : '0.65'; });
    this.btn.addEventListener('click', (e) => { e.stopPropagation(); e.preventDefault(); this.toggle(); });

    // Gear (settings flyout) + compare buttons — appear on hover, left of SR.
    this.gearBtn = el('button', miniBtnStyle('84px'));
    this.gearBtn.innerHTML = gearSvg();
    this.gearBtn.title = 'WebVSR settings';
    this.gearBtn.addEventListener('click', (e) => { e.stopPropagation(); e.preventDefault(); this.toggleFlyout(); });

    this.cmpBtn = el('button', miniBtnStyle('48px'));
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
      position: 'absolute', top: '12px', left: '12px', padding: '5px 10px',
      background: 'rgba(0,0,0,0.75)', borderRadius: '5px', color: '#ccc',
      fontFamily: '"SF Mono","Cascadia Code","Consolas",monospace', fontSize: '11px',
      lineHeight: '1.6', pointerEvents: 'none', display: 'none', whiteSpace: 'pre',
    });

    this.uiLayer.append(this.statsEl, this.flyout, this.cmpBtn, this.gearBtn, this.btn);
    this.setControlsVisible(false);

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
  }

  buildFlyout() {
    const f = el('div', {
      position: 'absolute', top: '50%', left: '120px', transform: 'translateY(-50%)',
      background: 'rgba(18,20,26,0.96)', border: '1px solid rgba(255,255,255,0.12)',
      borderRadius: '10px', padding: '12px 14px', width: '190px', pointerEvents: 'auto',
      display: 'none', color: '#e0e0e0', boxShadow: '0 8px 30px rgba(0,0,0,0.5)',
      fontFamily: '-apple-system,"Segoe UI",system-ui,sans-serif', fontSize: '12px',
    });
    f.addEventListener('click', (e) => e.stopPropagation());
    const label = (t) => el('div', { fontSize: '10px', textTransform: 'uppercase',
      letterSpacing: '0.08em', color: '#7c8390', margin: '10px 0 6px' }, t);

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
    const root = el('div', { display: 'flex', gap: '4px' });
    const btns = values.map((v, i) => {
      const b = el('button', segBtnStyle(), labels[i]);
      b.addEventListener('click', (e) => { e.stopPropagation(); set(v); });
      root.appendChild(b);
      return { v, b };
    });
    const refresh = () => btns.forEach(({ v, b }) => {
      const on = get() === v;
      b.style.background = on ? 'rgba(0,165,200,0.22)' : '#23262e';
      b.style.color = on ? '#00c4e8' : '#aaa';
      b.style.borderColor = on ? 'rgba(0,196,232,0.4)' : 'rgba(255,255,255,0.06)';
    });
    refresh();
    return { root, refresh };
  }

  toggleFlyout() {
    const show = this.flyout.style.display === 'none';
    this.flyout.style.display = show ? 'block' : 'none';
    if (show) { this.perfSeg.refresh(); this.qualSeg.refresh(); }
  }

  setControlsVisible(v) {
    const d = v ? 'flex' : 'none';
    this.gearBtn.style.display = d;
    this.cmpBtn.style.display = this.active ? d : 'none';
  }
  showControls(v) { this.setControlsVisible(v); }

  setComparing(on) {
    this.comparing = on;
    if (this.active) this.outCanvas.style.display = on ? 'none' : '';  // reveal original while held
  }

  async toggle() { if (this.active) this.stop(); else await this.start(); }

  async start() {
    this.active = true;
    this.styleBtn(true);
    this.setControlsVisible(true);
    if (settings.showStats) { this.statsEl.style.display = 'block'; this.statsEl.textContent = 'Starting GPU…'; }
    if (settings.rememberState) setSetting({ enabled: true });

    if (activeOverlay && activeOverlay !== this) activeOverlay.stop();
    activeOverlay = this;

    const e = await Promise.race([getEngine(),
      new Promise((r) => setTimeout(() => r('__t__'), 10000))]);
    if (e === '__t__') { this.fail('GPU init timed out (10s)'); enginePromise = null; return; }
    if (!e) { this.fail(engineError || 'WebGPU init failed'); return; }

    this.cfgInW = 0; this.cfgInH = 0; this.cfgDispW = 0; this.cfgDispH = 0;
    this.srH = START_NEURAL; this.frameTimes = [];
    this.outCanvas.style.display = '';
    if (settings.showStats) this.statsEl.textContent = 'Processing…';
    this.measureFps();
    this.loop();
  }

  fail(msg) {
    this.statsEl.style.display = 'block';
    this.statsEl.textContent = 'Unavailable:\n' + msg;
    this.active = false; this.styleBtn(false); this.setControlsVisible(false);
  }

  stop() {
    this.active = false;
    this.styleBtn(false);
    this.setControlsVisible(false);
    this.flyout.style.display = 'none';
    if (this.animId) { cancelAnimationFrame(this.animId); this.animId = null; }
    this.outCanvas.style.display = 'none';
    this.statsEl.style.display = 'none';
    if (settings.rememberState) setSetting({ enabled: false });
    if (activeOverlay === this) activeOverlay = null;
  }

  styleBtn(on) {
    Object.assign(this.btn.style, on ? {
      background: 'rgba(0,165,200,0.9)', color: '#fff', borderColor: 'rgba(0,200,240,0.6)',
      boxShadow: '0 0 12px rgba(0,196,232,0.4)', opacity: '1',
    } : {
      background: 'rgba(20,22,28,0.8)', color: '#bbb', borderColor: 'rgba(255,255,255,0.3)',
      boxShadow: 'none', opacity: '0.65',
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
    // Reveal the original when gated, comparing, can't-keep-up, or not worth it.
    const hide = this.gatedByFullscreen() || this.comparing || this.cantKeepUp || this.notWorthIt;
    if (hide && this.outCanvas.style.display !== 'none') this.outCanvas.style.display = 'none';
    else if (!hide && this.outCanvas.style.display === 'none') this.outCanvas.style.display = '';
    if (this.notWorthIt && settings.showStats) {
      this.statsEl.innerHTML =
        '<span style="color:#00c4e8;font-weight:700">WebVSR</span> <span style="color:#7c8f7c">idle</span>\n' +
        'source already ≈ screen res —\nSR would not help (no GPU used)';
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
      this.cantKeepUp = false;
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
        '<span style="color:#00c4e8;font-weight:700">WebVSR</span> ' +
        '<span style="color:#e0a24a">passthrough</span>\n' +
        'source too fast for SR here —\nshowing original (no slowdown)';
      return;
    }
    const avg = this.frameTimes.length
      ? this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length : 0;
    const fps = avg > 0 ? (1000 / avg).toFixed(0) : '…';
    const nat = this._nH >= this.video.videoHeight ? ' native' : '';
    const shp = (settings.sharpness || 0) > 0.01 ? ' · sharp' : '';
    this.statsEl.innerHTML =
      '<span style="color:#00c4e8;font-weight:700">WebVSR</span> ' + settings.perfMode +
      ' <span style="color:#888">' + fps + 'fps' + shp + '</span>\n' +
      'SR ' + this._nW + '×' + this._nH + nat + ' → ' + this._dW + '×' + this._dH + '\n' +
      gpu + 'ms gpu · ' + Math.round(this.budgetMs) + 'ms budget';
  }

  applySettings() {
    if (!settings.showStats) this.statsEl.style.display = 'none';
    else if (this.active) this.statsEl.style.display = 'block';
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
    width: '38px', height: '38px', borderRadius: '50%', border: '2px solid rgba(255,255,255,0.3)',
    background: 'rgba(20,22,28,0.8)', color: '#bbb',
    fontFamily: '-apple-system,"Segoe UI",system-ui,sans-serif', fontSize: '10px', fontWeight: '800',
    letterSpacing: '0.05em', cursor: 'pointer', pointerEvents: 'auto', opacity: '0.65',
    transition: 'opacity .15s, background .15s, color .15s, border-color .15s, box-shadow .15s',
    outline: 'none', padding: '0', display: 'flex', alignItems: 'center', justifyContent: 'center', lineHeight: '1',
  };
}
function miniBtnStyle(left) {
  return {
    position: 'absolute', top: '50%', left, transform: 'translateY(-50%)',
    width: '30px', height: '30px', borderRadius: '50%', border: '1px solid rgba(255,255,255,0.25)',
    background: 'rgba(20,22,28,0.8)', color: '#ddd', cursor: 'pointer', pointerEvents: 'auto',
    display: 'none', alignItems: 'center', justifyContent: 'center', padding: '0', outline: 'none',
  };
}
function segBtnStyle() {
  return {
    flex: '1', padding: '6px 0', background: '#23262e', border: '1px solid rgba(255,255,255,0.06)',
    borderRadius: '6px', color: '#aaa', fontSize: '11px', cursor: 'pointer', outline: 'none',
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
