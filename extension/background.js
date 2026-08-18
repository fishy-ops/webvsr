/**
 * WebVSR Background (service worker)
 *
 * Owns persistent settings and broadcasts changes to all tabs. No inference
 * happens here anymore - the engine runs entirely in the content script.
 */

const DEFAULT_SETTINGS = {
  enabled: false,        // last SR on/off state (for "remember state")
  // GPU load / smoothness - maps to the governor's frame-time budget.
  perfMode: 'balanced',  // 'light' | 'balanced' | 'max'
  // Model intensity - caps the neural net's internal resolution (until real
  // lighter models are trained, this is a resolution ceiling on one model).
  quality: 'quality',    // 'fast' | 'medium' | 'quality'
  targetScale: 2,        // output = base × this (1.5, 2, 3) - >2 warns
  autoPause: true,       // pause SR when tab hidden or video paused
  rememberState: true,   // re-enable SR automatically where it was on
  showStats: true,       // show the on-video stats HUD
  onlyFullscreen: false, // only run SR while the video is fullscreen
  blockedSites: [],      // hostnames where WebVSR stays off entirely
  sharpness: 0.9,        // contrast-adaptive sharpen strength (0=off … 1.4=high, or custom)
  sharpnessCustom: false,// true = user is on the custom slider (sharpness can exceed High)
  autoEngage: true,      // only run SR when the source is clearly lower-res than the display
  showCompare: false,    // show the on-video hold-to-compare button (opt-in)
};

let SETTINGS = { ...DEFAULT_SETTINGS };

chrome.storage.local.get('webvsrSettings', (r) => {
  if (r.webvsrSettings) SETTINGS = { ...DEFAULT_SETTINGS, ...r.webvsrSettings };
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
