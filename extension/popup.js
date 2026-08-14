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
function scaleHint(s) {
  s = parseFloat(s);
  if (s > 2) return '<span class="warn">⚠ Above 2× the model adds no new detail (it reconstructs up to 2×) — it just interpolates, and costs more GPU.</span>';
  if (s === 2) return 'Recommended: matches the model\'s true 2× reconstruction.';
  return 'Below 2×: lighter, slightly softer than the model\'s full output.';
}

function render() {
  renderSeg('perf', settings.perfMode);
  renderSeg('quality', settings.quality);
  renderSeg('scale', settings.targetScale);
  renderSeg('sharp', settings.sharpness);
  document.getElementById('perfHint').textContent = PERF_HINT[settings.perfMode] || '';
  document.getElementById('qualHint').textContent = QUAL_HINT[settings.quality] || '';
  document.getElementById('scaleHint').innerHTML = scaleHint(settings.targetScale);
  document.getElementById('autoEngage').checked = !!settings.autoEngage;
  document.getElementById('autoPause').checked = !!settings.autoPause;
  document.getElementById('rememberState').checked = !!settings.rememberState;
  document.getElementById('showStats').checked = !!settings.showStats;
  document.getElementById('onlyFullscreen').checked = !!settings.onlyFullscreen;
  document.getElementById('blockSite').checked =
    !!(activeHost && (settings.blockedSites || []).includes(activeHost));
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
  b.addEventListener('click', () => save({ sharpness: parseFloat(b.dataset.v) })));
document.getElementById('autoEngage').addEventListener('change', (e) => save({ autoEngage: e.target.checked }));
document.getElementById('autoPause').addEventListener('change', (e) => save({ autoPause: e.target.checked }));
document.getElementById('rememberState').addEventListener('change', (e) => save({ rememberState: e.target.checked }));
document.getElementById('showStats').addEventListener('change', (e) => save({ showStats: e.target.checked }));
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
