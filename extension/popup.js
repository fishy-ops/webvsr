let settings = {
  perfMode: 'max', quality: 'quality', targetScale: 4,
  autoPause: true, rememberState: true, showStats: true, sharpness: 1.4,
};

const PERF_HINT = {
  light: 'Easiest on your graphics card and the smoothest. Does a little less work each frame.',
  balanced: 'The one we recommend. Keeps video smooth while still cleaning it up.',
  max: 'Goes all out on every frame for the best quality. Can lower the frame rate on slower machines.',
};
const QUAL_HINT = {
  fast: 'The lightest setting. Good if your computer is on the slower side.',
  medium: 'A nice middle ground between detail and speed.',
  quality: 'Full detail. Pair it with Balanced or Light if you want to keep things smooth.',
};

function renderSeg(id, value) {
  document.querySelectorAll('#' + id + ' button').forEach((b) => {
    b.classList.toggle('active', b.dataset.v === String(value));
  });
}

// Sliding pill for each segmented control; animates to the active option.
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
  if (s >= 4) return 'Uses the dedicated 4× model. The sharpest option, and the hardest on your graphics card.';
  if (s > 2) return '<span class="warn">⚠ 3× runs the 4× model and scales it back down. 4× looks sharper for the same effort.</span>';
  if (s === 2) return 'The sweet spot. This is what the main model is actually built for.';
  return 'A bit lighter than 2×, and a touch softer.';
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
