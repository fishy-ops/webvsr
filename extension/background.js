/**
 * Crisp Background (service worker)
 *
 * Owns persistent settings and broadcasts changes to all tabs. No inference
 * happens here anymore - the engine runs entirely in the content script.
 */

const DEFAULT_SETTINGS = {
  enabled: false,        // last SR on/off state (for "remember state")
  // GPU load / smoothness - maps to the governor's frame-time budget.
  // Model intensity - caps the neural net's internal resolution (until real
  // lighter models are trained, this is a resolution ceiling on one model).
  power: 'auto',         // 'auto' | 'battery' | 'max' — see POWER in content.js
  targetScale: 2,        // output = base × this (1.5, 2, 3, 4) - >2 warns
  autoPause: true,       // pause SR when tab hidden or video paused
  rememberState: true,   // re-enable SR automatically where it was on
  showStats: true,       // show the on-video stats HUD
  onlyFullscreen: false, // only run SR while the video is fullscreen
  blockedSites: [],      // hostnames where Crisp stays off entirely
  sharpness: 3.5,        // contrast-adaptive sharpen strength (0=off … 3.5=high, or custom)
  sharpnessCustom: false,// true = user is on the custom slider (sharpness can exceed High)
  autoEngage: true,      // only run SR when the source is clearly lower-res than the display
  showCompare: false,    // show the on-video hold-to-compare button (opt-in)
  look: 'vivid',       // colour/tone grade on top of the upscale.
                         // 'natural' is exactly zero adjustment, so the
                         // default output is only what the model recovered.
};

let SETTINGS = { ...DEFAULT_SETTINGS };

// Carry existing installs across the GPU-load/Quality -> Power change. Anyone
// who had deliberately turned the load down wanted less heat, so they land on
// 'battery'; everyone else gets 'auto'. Without this they would silently keep
// two dead keys and lose the intent they had expressed.
function migrate(st) {
  // GPU load + Quality -> Power. Anyone who had deliberately turned the load
  // down wanted less heat, so they land on 'battery'; everyone else on 'auto'.
  if (!st.power) {
    const wasLight = st.perfMode === 'light' || st.quality === 'fast';
    st.power = wasLight ? 'battery' : 'auto';
    delete st.perfMode;
    delete st.quality;
  }
  // Punchier defaults (sharpness 3.5, Vivid). Only move installs that are still
  // sitting on the OLD defaults exactly -- those values were never chosen, they
  // were just what shipped. Anyone who picked their own sharpness or look keeps
  // it, because overriding a deliberate choice on update is not ours to make.
  if (st.sharpness === 1.4 && !st.sharpnessCustom) st.sharpness = 3.5;
  if (!st.look || st.look === 'natural') st.look = 'vivid';
  return st;
}

chrome.storage.local.get('webvsrSettings', (r) => {
  if (r.webvsrSettings) {
    SETTINGS = { ...DEFAULT_SETTINGS, ...migrate(r.webvsrSettings) };
    chrome.storage.local.set({ webvsrSettings: SETTINGS });
  }
});

function broadcast() {
  chrome.tabs.query({}, (tabs) => {
    for (const t of tabs) {
      if (t.id != null) {
        chrome.tabs.sendMessage(t.id, { type: 'SETTINGS_CHANGED', settings: SETTINGS },
          () => void chrome.runtime.lastError);  // ignore tabs without the content script
      }
    }
  });
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === 'GET_SETTINGS') {
    sendResponse(SETTINGS);
    return true;
  }
  if (msg.type === 'SET_SETTINGS') {
    SETTINGS = { ...SETTINGS, ...msg.patch };
    chrome.storage.local.set({ webvsrSettings: SETTINGS });
    sendResponse(SETTINGS);
    broadcast();
    return true;
  }
});
