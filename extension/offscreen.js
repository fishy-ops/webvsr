/**
 * WebVSR Offscreen Document
 *
 * Runs ONNX Runtime Web inference on the WebGPU execution provider.
 * Frames are transferred as WebP data URLs (compact) rather than JSON
 * pixel arrays. Listener is registered immediately; ORT is lazy-loaded.
 */

let ort = null;
let session = null;
let usingWebGPU = false;

// Reused scratch canvases to avoid per-frame allocation.
const inCanvas = new OffscreenCanvas(2, 2);
const inCtx = inCanvas.getContext('2d', { willReadFrequently: true });
const outCanvas = new OffscreenCanvas(2, 2);
const outCtx = outCanvas.getContext('2d');

async function loadOrt() {
  if (ort) return;
  // Full build includes the JSEP (WebGPU) backend.
  ort = await import('./lib/ort.min.mjs');
  ort.env.wasm.wasmPaths = chrome.runtime.getURL('lib/');
  ort.env.wasm.numThreads = 1;
  console.log('[WebVSR] ORT loaded');
}

async function handleLoadModel(modelPath) {
  try {
    await loadOrt();
    try {
      session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['webgpu'],
      });
      usingWebGPU = true;
      console.log('[WebVSR] Model loaded on WebGPU');
    } catch (gpuErr) {
      console.warn('[WebVSR] WebGPU EP failed, falling back to WASM:', gpuErr.message);
      session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['wasm'],
      });
      usingWebGPU = false;
      console.log('[WebVSR] Model loaded on WASM');
    }
    return { success: true, backend: usingWebGPU ? 'webgpu' : 'wasm' };
  } catch (e) {
    console.error('[WebVSR] Model load failed:', e);
    return { success: false, error: e.message };
  }
}

async function decodeToImageData(dataUrl, width, height) {
  const blob = await (await fetch(dataUrl)).blob();
  const bmp = await createImageBitmap(blob);
  inCanvas.width = width;
  inCanvas.height = height;
  inCtx.drawImage(bmp, 0, 0, width, height);
  bmp.close();
  return inCtx.getImageData(0, 0, width, height).data;
}

async function handleInfer(dataUrl, width, height, quality) {
  if (!session || !ort) return null;

  const rgba = await decodeToImageData(dataUrl, width, height);

  const t0 = performance.now();
  const pixels = width * height;
  const float32 = new Float32Array(3 * pixels);
  for (let i = 0; i < pixels; i++) {
    float32[i] = rgba[i * 4] / 255.0;
    float32[pixels + i] = rgba[i * 4 + 1] / 255.0;
    float32[2 * pixels + i] = rgba[i * 4 + 2] / 255.0;
  }

  const tensor = new ort.Tensor('float32', float32, [1, 3, height, width]);
  const results = await session.run({ input: tensor });
  const outData = results.output.data;
  const inferMs = performance.now() - t0;

  const outW = width * 2;
  const outH = height * 2;
  const outPixels = outW * outH;

  const rgbaOut = new Uint8ClampedArray(outPixels * 4);
  for (let i = 0; i < outPixels; i++) {
    rgbaOut[i * 4] = clamp255(outData[i]);
    rgbaOut[i * 4 + 1] = clamp255(outData[outPixels + i]);
    rgbaOut[i * 4 + 2] = clamp255(outData[2 * outPixels + i]);
    rgbaOut[i * 4 + 3] = 255;
  }

  outCanvas.width = outW;
  outCanvas.height = outH;
  outCtx.putImageData(new ImageData(rgbaOut, outW, outH), 0, 0);
  const outBlob = await outCanvas.convertToBlob({
    type: 'image/webp',
    quality: quality || 0.92,
  });
  const outUrl = await blobToDataUrl(outBlob);

  return { dataUrl: outUrl, width: outW, height: outH, inferMs, backend: usingWebGPU ? 'webgpu' : 'wasm' };
}

function clamp255(v) {
  v = v * 255;
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = reject;
    fr.readAsDataURL(blob);
  });
}

// Register listener IMMEDIATELY (before any async work)
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === 'OFFSCREEN_LOAD_MODEL') {
    handleLoadModel(msg.modelPath).then(sendResponse);
    return true;
  }
  if (msg.type === 'OFFSCREEN_INFER') {
    handleInfer(msg.dataUrl, msg.width, msg.height, msg.quality)
      .then((result) => sendResponse(result ? { result, inferMs: result.inferMs } : null))
      .catch((e) => {
        console.error('[WebVSR] Infer error:', e);
        sendResponse({ error: e.message });
      });
    return true;
  }
});

console.log('[WebVSR] Offscreen document ready');
