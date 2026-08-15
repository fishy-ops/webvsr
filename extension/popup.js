let settings = {
  perfMode: 'balanced', quality: 'quality', targetScale: 2,
  autoPause: true, rememberState: true, showStats: true,
};

const PERF_HINT = {
  light: 'Lowest GPU use, smoothest. Runs the model at a lower internal resolution.',
  balanced: 'Recommended. Keeps the video smooth while adding detail.',
  max: 'Native-resolution model every frame. Highest quality — may reduce FPS / use full GPU.',
};
const QUAL_HINT = {
  fast: 'Cheapest — processes a downscaled frame. Best for weak GPUs.',
  medium: 'A balance of detail and speed.',
  quality: 'Full detail (native internal resolution). Pair with Balanced/Light to stay smooth.',
};

function renderSeg(id, value) {
  document.querySelectorAll('#' + id + ' button').forEach((b) => {
    b.classList.toggle('active', b.dataset.v === String(value));
  });
}

// Sliding pill for each segmented control — animates to the active option.
document.querySelectorAll('.seg').forEach((seg) => {
  const s = document.createElement('span');
  s.className = 'seg-slider';
  seg.insertBefore(s, seg.firstChild);
});
let slidersPrimed = false;
function moveSliders() {
  const first = !slidersPrimed;
  document.querySelectorAll('.seg').forEach((seg) => {
    const s = seg.querySelector('.seg-slider');
    const a = seg.querySelector('button.active');
    if (!s) return;
    if (first) s.style.transition = 'none';   // place instantly on first paint
    if (!a) { s.style.opacity = '0'; return; }
    s.style.opacity = '1';
    s.style.left = a.offsetLeft + 'px';
    s.style.width = a.offsetWidth + 'px';
  });
  if (first) {
    void document.body.offsetWidth;            // flush layout, then re-enable animation
    requestAnimationFrame(() => {
      document.querySelectorAll('.seg-slider').forEach((s) => { s.style.transition = ''; });
    });
    slidersPrimed = true;
  }
}
function scaleHint(s) {
  s = parseFloat(s);
  if (s > 2) return '<span class="warn">⚠ Above 2× the model adds no new detail (it reconstructs up to 2×) — it just interpolates, and costs more GPU.</span>';
  if (s === 2) return 'Recommended: matches the model\'s true 2× reconstruction.';
  return 'Below 2×: lighter, slightly softer than the model\'s full output.';
}

const SHARP_PRESETS = [[0, 'Off'], [0.55, 'Low'], [0.9, 'Med'], [1.4, 'High']];
function updateSharpVal(v) {
  document.getElementById('sharpVal').textContent = Number(v).toFixed(2) + '×';
  let near = '';
  for (const [pv, name] of SHARP_PRESETS) { if (Math.abs(v - pv) < 0.06) { near = name; break; } }
  const el = document.getElementById('sharpPreset');
  el.textContent = near ? '≈ ' + near : '';
  el.style.opacity = near ? '1' : '0';   // preset label only shows near a preset
}
function renderSharp() {
  const custom = !!settings.sharpnessCustom;
  document.querySelectorAll('#sharp button').forEach((b) => {
    const on = custom ? (b.dataset.v === 'custom') : (b.dataset.v === String(settings.sharpness));
    b.classList.toggle('active', on);
  });
  document.getElementById('customWrap').classList.toggle('open', custom);
  if (custom) {
    const v = settings.sharpness || 0;
    const slider = document.getElementById('sharpSlider');
    if (document.activeElement !== slider) slider.value = v;   // don't fight an active drag
    updateSharpVal(v);
  }
}

function render() {
  renderSeg('perf', settings.perfMode);
  renderSeg('quality', settings.quality);
  renderSeg('scale', settings.targetScale);
  renderSharp();
  document.getElementById('perfHint').textContent = PERF_HINT[settings.perfMode] || '';
  document.getElementById('qualHint').textContent = QUAL_HINT[settings.quality] || '';
  document.getElementById('scaleHint').innerHTML = scaleHint(settings.targetScale);
  document.getElementById('autoEngage').checked = !!settings.autoEngage;
  document.getElementById('autoPause').checked = !!settings.autoPause;
  document.getElementById('rememberState').checked = !!settings.rememberState;
  document.getElementById('showStats').checked = !!settings.showStats;
  document.getElementById('showCompare').checked = !!settings.showCompare;
  document.getElementById('onlyFullscreen').checked = !!settings.onlyFullscreen;
  document.getElementById('blockSite').checked =
    !!(activeHost && (settings.blockedSites || []).includes(activeHost));
  moveSliders();
}

let activeHost = '';

function save(patch) {
  Object.assign(settings, patch);
  chrome.runtime.sendMessage({ type: 'SET_SETTINGS', patch }, (s) => { if (s) settings = s; render(); });
  render();
}

document.querySelectorAll('#perf button').forEach((b) =>
  b.addEventListener('click', () => save({ perfMode: b.dataset.v })));
document.querySelectorAll('#quality button').forEach((b) =>
  b.addEventListener('click', () => save({ quality: b.dataset.v })));
document.querySelectorAll('#scale button').forEach((b) =>
  b.addEventListener('click', () => save({ targetScale: parseFloat(b.dataset.v) })));
document.querySelectorAll('#sharp button').forEach((b) =>
  b.addEventListener('click', () => {
    if (b.dataset.v === 'custom') save({ sharpnessCustom: true });   // keep value, reveal slider
    else save({ sharpness: parseFloat(b.dataset.v), sharpnessCustom: false });
  }));
document.getElementById('sharpSlider').addEventListener('input', (e) => {
  const v = parseFloat(e.target.value);
  updateSharpVal(v);
  save({ sharpness: v, sharpnessCustom: true });
});
document.getElementById('autoEngage').addEventListener('change', (e) => save({ autoEngage: e.target.checked }));
document.getElementById('autoPause').addEventListener('change', (e) => save({ autoPause: e.target.checked }));
document.getElementById('rememberState').addEventListener('change', (e) => save({ rememberState: e.target.checked }));
document.getElementById('showStats').addEventListener('change', (e) => save({ showStats: e.target.checked }));
document.getElementById('showCompare').addEventListener('change', (e) => save({ showCompare: e.target.checked }));
document.getElementById('onlyFullscreen').addEventListener('change', (e) => save({ onlyFullscreen: e.target.checked }));
document.getElementById('blockSite').addEventListener('change', (e) => {
  let list = (settings.blockedSites || []).slice();
  if (e.target.checked) { if (activeHost && !list.includes(activeHost)) list.push(activeHost); }
  else { list = list.filter((h) => h !== activeHost); }
  save({ blockedSites: list });
});

chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  try { activeHost = tabs[0] && tabs[0].url ? new URL(tabs[0].url).hostname : ''; }
  catch (_) { activeHost = ''; }
  document.getElementById('siteHost').textContent = activeHost || 'this site';
  render();
});

chrome.runtime.sendMessage({ type: 'GET_SETTINGS' }, (s) => {
  if (s && !chrome.runtime.lastError) settings = s;
  render();
});
