# WebVSR — Privacy Policy

**Effective date:** 15 August 2026

WebVSR is a Chrome extension that upscales and sharpens video in real time, on
your own device, using your GPU (via WebGPU). This policy explains what the
extension does and does not do with your data. In short: **WebVSR collects
nothing, sends nothing, and has no servers.**

## Data we collect

**None.** WebVSR does not collect, store, transmit, sell, or share any personal
data, browsing data, or usage data. There is no account, no login, no analytics,
no advertising, and no third-party services of any kind.

## How your video is processed

All video processing happens **locally, in your browser, on your GPU**. When you
turn WebVSR on for a video, the extension reads the video frames already playing
on the page and runs its super-resolution and sharpening entirely on-device using
WebGPU compute shaders. **Video frames never leave your computer** — nothing is
uploaded, streamed, or sent to any server (there are no WebVSR servers to send it
to). WebVSR does not read, record, or transmit the contents of the pages you
visit.

## Settings

Your preferences (such as GPU-load level, quality, target scale, sharpness, and
any per-site enable/disable choices) are saved locally on your device using the
browser's `chrome.storage.local` API. These settings stay on your machine, are
used only to remember your choices, and are not transmitted anywhere.

## Permissions and why they are needed

- **Host access to all sites (`<all_urls>`):** Video can appear on any website, so
  the extension needs to be able to detect and enhance `<video>` elements
  wherever you watch. This access is used **only** to find videos and draw the
  upscaled result over them locally. WebVSR does not read page content or send any
  site data anywhere.
- **`storage`:** To save your settings locally on your device (see above).
- **`activeTab`:** To identify the current site so the "disable on this site"
  toggle can apply to the page you are viewing.

## Children's privacy

WebVSR does not collect any data from anyone, including children.

## Changes to this policy

If this policy changes, the updated version will be posted at this location with a
new effective date.

## Contact

Questions about this policy can be raised via the project's issue tracker on
GitHub: <https://github.com/fishy-ops/webvsr/issues>.
