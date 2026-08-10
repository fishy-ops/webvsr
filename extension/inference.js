/**
 * WebVSR Inference Manager
 *
 * Handles tiled inference for video frames. Instead of processing the entire
 * frame at once (which may exceed the 16.6ms budget), we split the frame into
 * tiles and process them over multiple animation frames.
 *
 * Strategy:
 * - "quality" mode: full-frame SR, may drop to lower fps
 * - "balanced" mode: tile-based SR, processes N tiles per frame
 * - "fast" mode: alternating frames SR (SR every other frame, hold previous)
 * - "anime" mode: same as quality but with anime-optimized model
 */

class InferenceManager {
  constructor() {
    this.session = null;
    this.modelLoaded = false;
    this.tileSize = 128;
    this.tileOverlap = 8;
    this.mode = 'quality';
    this.stats = { fps: 0, inferMs: 0, resolution: '' };
    this.frameCount = 0;
  }

  async loadModel() {
    try {
      const response = await chrome.runtime.sendMessage({
        type: 'LOAD_MODEL',
        modelPath: chrome.runtime.getURL('models/span_lite_2x.onnx'),
      });
      this.modelLoaded = response === true;
      return this.modelLoaded;
    } catch (e) {
      console.error('[WebVSR] Failed to load model:', e);
      return false;
    }
  }

  async inferFullFrame(imageData, width, height) {
    const t0 = performance.now();
    const response = await chrome.runtime.sendMessage({
      type: 'INFER',
      imageData: Array.from(imageData),
      width,
      height,
    });
    const ms = performance.now() - t0;
    this.stats.inferMs = ms;
    return response?.result || null;
  }

  splitToTiles(width, height) {
    const tiles = [];
    const step = this.tileSize - this.tileOverlap;

    for (let y = 0; y < height; y += step) {
      for (let x = 0; x < width; x += step) {
        const tw = Math.min(this.tileSize, width - x);
        const th = Math.min(this.tileSize, height - y);
        tiles.push({ x, y, w: tw, h: th });
      }
    }
    return tiles;
  }

  async inferTiled(captureCanvas, outputCtx, tilesPerFrame = 4) {
    const width = captureCanvas.width;
    const height = captureCanvas.height;
    const captureCtx = captureCanvas.getContext('2d');
    const tiles = this.splitToTiles(width, height);

    let tileIdx = 0;
    const processBatch = async () => {
      const batch = tiles.slice(tileIdx, tileIdx + tilesPerFrame);
      for (const tile of batch) {
        const tileData = captureCtx.getImageData(tile.x, tile.y, tile.w, tile.h);
        const result = await this.inferFullFrame(tileData.data, tile.w, tile.h);
        if (result) {
          const imgData = new ImageData(
            new Uint8ClampedArray(result.data),
            result.width,
            result.height
          );
          outputCtx.putImageData(imgData, tile.x * 2, tile.y * 2);
        }
      }
      tileIdx += tilesPerFrame;
      return tileIdx >= tiles.length;
    };

    return processBatch;
  }

  shouldProcess() {
    if (this.mode === 'fast') {
      this.frameCount++;
      return this.frameCount % 2 === 0;
    }
    return true;
  }

  getStats() {
    return this.stats;
  }
}

if (typeof globalThis !== 'undefined') {
  globalThis.WebVSRInference = InferenceManager;
}
