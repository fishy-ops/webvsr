/**
 * WebVSR — in-page WebGPU super-resolution engine.
 *
 * Runs SPAN-Lite (fused, 2x) entirely as WGSL compute shaders. No ONNX
 * Runtime, no WASM, no message passing: the video frame is pulled straight
 * into the GPU (importExternalTexture, zero-copy) and the result is written
 * to an overlay canvas. Runs inside the content script, so strict page CSPs
 * (YouTube) don't apply — WebGPU shader creation is not eval.
 *
 * Inference dataflow (all at input resolution WxH until PixelShuffle):
 *   pre:        video -> x (3ch, mean-subtracted)
 *   conv_first: x -> f0                                    (3x3, 3->32)
 *   4x SPAB:    c1(+SiLU), c2(+SiLU), c3, att=(sigmoid(c3)-0.5)*(c3+in)
 *   conv_last:  b4pre -> b4                                (3x3, 32->32)
 *   conv_cat:   concat[f0,b1_mid,b3_mid,b4] -> cat         (1x1, 128->32)
 *   upsampler:  cat -> up (3x3, 32->12) then PixelShuffle(2) -> RGB 2Wx2H
 */

const MEAN = [0.4488, 0.4371, 0.4040];

// Half precision: implemented with f32 fallback, but off by default — see init().
const USE_F16 = false;

// Weight tensors in export order (matches export_webgpu_weights.py), for a
// SPAN-Lite with C feature channels. [name, outC, inC, k]
function weightSpec(C) {
  return [
    ['conv_first', C, 3, 3],
    ['b1c1', C, C, 3], ['b1c2', C, C, 3], ['b1c3', C, C, 3],
    ['b2c1', C, C, 3], ['b2c2', C, C, 3], ['b2c3', C, C, 3],
    ['b3c1', C, C, 3], ['b3c2', C, C, 3], ['b3c3', C, C, 3],
    ['b4c1', C, C, 3], ['b4c2', C, C, 3], ['b4c3', C, C, 3],
    ['conv_cat', C, 4 * C, 1],
    ['conv_last', C, C, 3],
    ['upsampler', 12, C, 3],
  ];
}

class WebGPUSR {
  constructor() {
    this.device = null;
    this.w = {};        // weight/bias GPU buffers by name
    this.buf = {};      // feature GPU buffers
    this.pipe = {};     // compute pipelines
    this.bg = {};       // cached bind groups
    this.params = {};   // uniform buffers
    this.inW = 0;
    this.inH = 0;
    this.ready = false;
    this.sampler = null;
    this.outTex = null;
    this.C = 32;        // feature channels (from the model manifest; 32 by default)
    this.sharpen = 0;   // 0..1 contrast-adaptive sharpen strength (0 = off)
  }

  async init() {
    if (!navigator.gpu) { console.warn('[WebVSR] WebGPU unavailable'); return false; }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) { console.warn('[WebVSR] No WebGPU adapter'); return false; }
    this.hasTS = adapter.features.has('timestamp-query');
    // f16 is fully implemented with an automatic f32 fallback, but defaults OFF:
    // on this GPU class scalar f16 is NOT faster than f32 (measured — the 2× only
    // comes from packed vec2<f16>, a bigger rewrite with quality risk). Set
    // USE_F16 = true to trade a little accuracy for ~half the VRAM on tight GPUs.
    this.f16 = USE_F16 && adapter.features.has('shader-f16');
    this.FS = this.f16 ? 2 : 4;                        // bytes per scalar
    const feats = [];
    if (this.hasTS) feats.push('timestamp-query');
    if (this.f16) feats.push('shader-f16');
    this.device = await adapter.requestDevice({ requiredFeatures: feats });
    console.log('[WebVSR] precision:', this.f16 ? 'f16' : 'f32');
    this.device.lost.then((info) => {
      console.error('[WebVSR] GPU device lost:', info.message);
      this.ready = false;
    });
    this.sampler = this.device.createSampler({
      magFilter: 'linear', minFilter: 'linear',
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
    });
    // Accurate GPU timing via timestamp queries (immune to CPU↔GPU sync latency).
    this.gpuMs = 0;
    if (this.hasTS) {
      this.querySet = this.device.createQuerySet({ type: 'timestamp', count: 2 });
      this.tsResolve = this.device.createBuffer({
        size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.tsRead = this.device.createBuffer({
        size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this.tsBusy = false;
    }
    // Pipelines are built in loadWeights, once the channel count is known.
    return true;
  }

  async loadWeights(url) {
    // Optional sibling manifest (e.g. span_lite_2x.json) sets the channel count.
    try {
      const mUrl = url.replace(/\.bin(\?.*)?$/, '.json$1');
      const mResp = await fetch(mUrl);
      if (mResp.ok) { const m = await mResp.json(); if (m.channels) this.C = m.channels | 0; }
    } catch (_) { /* no manifest → keep default 32 */ }
    this._buildPipelines();

    const buffer = await (await fetch(url)).arrayBuffer();
    const all = new Float32Array(buffer);
    const FS = this.FS;
    // In f16 mode, convert each tensor to half-float bytes before upload.
    const toBytes = (sub) => this.f16 ? f32ArrayToF16(sub) : sub;
    let off = 0;
    for (const [name, outC, inC, k] of weightSpec(this.C)) {
      const wLen = outC * inC * k * k;
      const bLen = outC;
      const wBuf = this.device.createBuffer({
        size: wLen * FS, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(wBuf, 0, toBytes(all.subarray(off, off + wLen)));
      off += wLen;
      const bBuf = this.device.createBuffer({
        size: Math.max(bLen, 4) * FS, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(bBuf, 0, toBytes(all.subarray(off, off + bLen)));
      off += bLen;
      this.w[name] = { weight: wBuf, bias: bBuf };
    }
    if (off !== all.length) {
      console.warn(`[WebVSR] weight size mismatch: read ${off} of ${all.length}`);
    }
    this.ready = true;
    return true;
  }

  // ── Pipelines (resolution-independent) ──────────────────────────
  _buildPipelines() {
    const d = this.device;
    const mk = (code, entry = 'main') =>
      d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code }), entryPoint: entry } });

    const T = this.f16 ? 'f16' : 'f32';
    const EN = this.f16 ? 'enable f16;\n' : '';
    this.pipe.pre = mk(EN + buildPre(T));
    this.pipe.conv = mk(EN + buildConv(T));
    this.pipe.attn = mk(EN + buildAttn(T));
    this.pipe.cat = mk(EN + buildCat(T, this.C));
    this.pipe.shuffle = mk(EN + buildShuffle(T));
    this.pipe.finish = mk(SHADER_FINISH);     // reads a texture, always f32
    this.pipe.sharpen = mk(SHADER_SHARPEN);   // contrast-adaptive sharpen
  }

  // ── Allocate buffers + bind groups ──
  // inW/inH: neural input resolution (governed). dispW/dispH: final display
  // size (defaults to 2× input). A Catmull-Rom finishing pass resamples the
  // neural 2× output to the display size when they differ.
  configure(canvas, inW, inH, dispW, dispH) {
    const d = this.device;
    inW &= ~1; inH &= ~1;
    this.inW = inW; this.inH = inH;
    const px = inW * inH;
    const outW = inW * 2, outH = inH * 2;
    dispW = Math.max(2, Math.round(dispW || outW));
    dispH = Math.max(2, Math.round(dispH || outH));
    this.dispW = dispW; this.dispH = dispH;
    this.needFinish = (dispW !== outW || dispH !== outH);

    // Free previous resources if resizing.
    Object.values(this.buf).forEach((b) => b.destroy?.());
    this.buf = {};
    this.outTex?.destroy?.();
    this.dispTex?.destroy?.();

    const feat = (ch) => d.createBuffer({ size: ch * px * this.FS, usage: GPUBufferUsage.STORAGE });
    for (const name of ['x3', 'f0', 'mid1', 'mid3', 'bb4', 'sA', 'sB', 'sC', 'catout']) {
      this.buf[name] = feat(name === 'x3' ? 3 : this.C);
    }
    this.buf.up = feat(12);

    const TEX = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING;
    this.outTex = d.createTexture({ size: [outW, outH], format: 'rgba8unorm', usage: TEX });
    if (this.needFinish) {
      this.dispTex = d.createTexture({ size: [dispW, dispH], format: 'rgba8unorm', usage: TEX });
    }
    // Sharpen output (display size) + its strength uniform (updated per frame).
    this.sharpTex?.destroy?.();
    this.sharpTex = d.createTexture({ size: [dispW, dispH], format: 'rgba8unorm', usage: TEX });
    if (!this.sharpParams) {
      this.sharpParams = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }

    this.ctx = canvas.getContext('webgpu');
    canvas.width = dispW; canvas.height = dispH;
    this.ctx.configure({
      device: d, format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
      alphaMode: 'opaque',
    });

    this._buildBindGroups();
  }

  _u(arr) {
    const b = this.device.createBuffer({
      size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const data = new Uint32Array(8);
    arr.forEach((v, i) => { data[i] = v >>> 0; });
    this.device.queue.writeBuffer(b, 0, data);
    return b;
  }

  _buildBindGroups() {
    const d = this.device, B = this.buf, W = this.w;
    const P = this.pipe;
    const { inW, inH } = this;
    const C = this.C;

    // conv3x3 / conv1x1 bind group: [in, weight, bias, out, params]
    const conv = (inBuf, name, outBuf, inC, outC, k, act) => {
      const params = this._u([inW, inH, inC, outC, act, k]);
      return d.createBindGroup({
        layout: P.conv.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inBuf } },
          { binding: 1, resource: { buffer: W[name].weight } },
          { binding: 2, resource: { buffer: W[name].bias } },
          { binding: 3, resource: { buffer: outBuf } },
          { binding: 4, resource: { buffer: params } },
        ],
      });
    };
    const attn = (t3, xb, out) => {
      const params = this._u([inW, inH, C, 0]);
      return d.createBindGroup({
        layout: P.attn.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: t3 } },
          { binding: 1, resource: { buffer: xb } },
          { binding: 2, resource: { buffer: out } },
          { binding: 3, resource: { buffer: params } },
        ],
      });
    };

    // Ordered pass list: [pipelineKey, bindGroup, dispatchZ (= output channels)]
    this.passes = [
      ['conv', conv(B.x3, 'conv_first', B.f0, 3, C, 3, 0), C],
      // block 1 (in f0)
      ['conv', conv(B.f0, 'b1c1', B.mid1, C, C, 3, 1), C],
      ['conv', conv(B.mid1, 'b1c2', B.sA, C, C, 3, 1), C],
      ['conv', conv(B.sA, 'b1c3', B.sB, C, C, 3, 0), C],
      ['attn', attn(B.sB, B.f0, B.sC), C],
      // block 2 (in sC)
      ['conv', conv(B.sC, 'b2c1', B.sA, C, C, 3, 1), C],
      ['conv', conv(B.sA, 'b2c2', B.sB, C, C, 3, 1), C],
      ['conv', conv(B.sB, 'b2c3', B.sA, C, C, 3, 0), C],
      ['attn', attn(B.sA, B.sC, B.sB), C],
      // block 3 (in sB)
      ['conv', conv(B.sB, 'b3c1', B.mid3, C, C, 3, 1), C],
      ['conv', conv(B.mid3, 'b3c2', B.sA, C, C, 3, 1), C],
      ['conv', conv(B.sA, 'b3c3', B.sC, C, C, 3, 0), C],
      ['attn', attn(B.sC, B.sB, B.sA), C],
      // block 4 (in sA)
      ['conv', conv(B.sA, 'b4c1', B.sB, C, C, 3, 1), C],
      ['conv', conv(B.sB, 'b4c2', B.sC, C, C, 3, 1), C],
      ['conv', conv(B.sC, 'b4c3', B.sB, C, C, 3, 0), C],
      ['attn', attn(B.sB, B.sA, B.sC), C],
      // tail
      ['conv', conv(B.sC, 'conv_last', B.bb4, C, C, 3, 0), C],
      ['cat', this._catBG(), C],
      ['conv', conv(B.catout, 'upsampler', B.up, C, 12, 3, 0), 12],
    ];

    // PixelShuffle -> neural output texture
    this.shuffleBG = d.createBindGroup({
      layout: P.shuffle.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: B.up } },
        { binding: 1, resource: this.outTex.createView() },
        { binding: 2, resource: { buffer: this._u([inW, inH, 2, 3]) } },
      ],
    });

    // Finishing: Catmull-Rom resample neural output (2in) -> display size.
    if (this.needFinish) {
      this.finishBG = d.createBindGroup({
        layout: P.finish.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.outTex.createView() },
          { binding: 1, resource: this.dispTex.createView() },
          { binding: 2, resource: { buffer: this._u([inW * 2, inH * 2, this.dispW, this.dispH]) } },
        ],
      });
    }

    // Sharpen: reads the final image (dispTex when finishing, else outTex) -> sharpTex.
    this.sharpenBG = d.createBindGroup({
      layout: P.sharpen.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: (this.needFinish ? this.dispTex : this.outTex).createView() },
        { binding: 1, resource: this.sharpTex.createView() },
        { binding: 2, resource: { buffer: this.sharpParams } },
      ],
    });
  }

  _catBG() {
    const d = this.device, B = this.buf, W = this.w;
    return d.createBindGroup({
      layout: this.pipe.cat.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: B.f0 } },
        { binding: 1, resource: { buffer: B.mid1 } },
        { binding: 2, resource: { buffer: B.mid3 } },
        { binding: 3, resource: { buffer: B.bb4 } },
        { binding: 4, resource: { buffer: W.conv_cat.weight } },
        { binding: 5, resource: { buffer: W.conv_cat.bias } },
        { binding: 6, resource: { buffer: B.catout } },
        { binding: 7, resource: { buffer: this._u([this.inW, this.inH, 4 * this.C, this.C]) } },
      ],
    });
  }

  // ── Run one frame: video -> canvas ──────────────────────────────
  render(video) {
    if (!this.ready || !this.ctx) return;
    const d = this.device;
    const { inW, inH } = this;
    const gx16 = Math.ceil(inW / 16), gy16 = Math.ceil(inH / 16);
    const gx8 = Math.ceil(inW / 8), gy8 = Math.ceil(inH / 8);

    // Preprocess bind group must be rebuilt each frame (external texture).
    const extTex = d.importExternalTexture({ source: video });
    const preBG = d.createBindGroup({
      layout: this.pipe.pre.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: extTex },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.buf.x3 } },
        { binding: 3, resource: { buffer: this._preParams } },
      ],
    });

    const enc = d.createCommandEncoder();
    // Timestamp: mark start on pass 1; end goes on pass 1 (no finish) or pass 2.
    const pass1ts = this.hasTS
      ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 0,
          ...(this.needFinish ? {} : { endOfPassWriteIndex: 1 }) } }
      : undefined;
    const pass = enc.beginComputePass(pass1ts);

    pass.setPipeline(this.pipe.pre);
    pass.setBindGroup(0, preBG);
    pass.dispatchWorkgroups(gx16, gy16, 1);

    for (const [key, bg, oc] of this.passes) {
      pass.setPipeline(this.pipe[key]);
      pass.setBindGroup(0, bg);
      if (key === 'attn') pass.dispatchWorkgroups(gx16, gy16, oc);        // per-channel (oc = C)
      else if (key === 'cat') pass.dispatchWorkgroups(gx8, gy8, Math.ceil(oc / 8)); // 8 ch/thread
      else pass.dispatchWorkgroups(gx16, gy16, Math.ceil(oc / 8));        // conv: 2×2px×8ch/thread
    }

    // PixelShuffle at neural output resolution.
    pass.setPipeline(this.pipe.shuffle);
    pass.setBindGroup(0, this.shuffleBG);
    pass.dispatchWorkgroups(Math.ceil(inW * 2 / 16), Math.ceil(inH * 2 / 16), 1);
    pass.end();

    if (this.needFinish) {
      // Separate pass: read the freshly-written outTex as a sampled texture.
      const finTs = this.hasTS
        ? { timestampWrites: { querySet: this.querySet, endOfPassWriteIndex: 1 } }
        : undefined;
      const fin = enc.beginComputePass(finTs);
      fin.setPipeline(this.pipe.finish);
      fin.setBindGroup(0, this.finishBG);
      fin.dispatchWorkgroups(Math.ceil(this.dispW / 16), Math.ceil(this.dispH / 16), 1);
      fin.end();
    }

    // Optional contrast-adaptive sharpen, then present to the canvas.
    const canvasTex = this.ctx.getCurrentTexture();
    if (this.sharpen > 0.001) {
      d.queue.writeBuffer(this.sharpParams, 0,
        new Uint32Array([this.dispW, this.dispH, Math.round(this.sharpen * 4096), 0]));
      const sp = enc.beginComputePass();
      sp.setPipeline(this.pipe.sharpen);
      sp.setBindGroup(0, this.sharpenBG);
      sp.dispatchWorkgroups(Math.ceil(this.dispW / 16), Math.ceil(this.dispH / 16), 1);
      sp.end();
      enc.copyTextureToTexture({ texture: this.sharpTex }, { texture: canvasTex }, [this.dispW, this.dispH, 1]);
    } else {
      const srcTex = this.needFinish ? this.dispTex : this.outTex;
      enc.copyTextureToTexture({ texture: srcTex }, { texture: canvasTex }, [this.dispW, this.dispH, 1]);
    }

    // Resolve GPU timestamps and read back asynchronously (never blocks render).
    const sampleTs = this.hasTS && !this.tsBusy;
    if (sampleTs) {
      enc.resolveQuerySet(this.querySet, 0, 2, this.tsResolve, 0);
      enc.copyBufferToBuffer(this.tsResolve, 0, this.tsRead, 0, 16);
    }
    d.queue.submit([enc.finish()]);

    if (sampleTs) {
      this.tsBusy = true;
      this.tsRead.mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigInt64Array(this.tsRead.getMappedRange());
        const ns = Number(t[1] - t[0]);
        if (ns > 0) this.gpuMs = ns / 1e6;
        this.tsRead.unmap();
        this.tsBusy = false;
      }).catch(() => { this.tsBusy = false; });
    }
  }

  // Cache preprocess params buffer (created lazily in configure via getter).
  get _preParams() {
    if (!this.__pre || this.__preW !== this.inW || this.__preH !== this.inH) {
      this.__pre = this._u([this.inW, this.inH, 0, 0]);
      this.__preW = this.inW; this.__preH = this.inH;
    }
    return this.__pre;
  }

  async waitIdle() { await this.device.queue.onSubmittedWorkDone(); }

  dispose() {
    Object.values(this.buf).forEach((b) => b.destroy?.());
    this.outTex?.destroy?.();
    this.ready = false;
  }
}

// ── WGSL shaders ──────────────────────────────────────────────────
// Storage buffers use the scalar type T (f16 when available, else f32); all
// arithmetic is done in f32 (values cast on load/store) so half precision never
// costs accuracy — the win is halved weight/feature memory traffic, which is the
// measured bottleneck. The f32 path makes every T(...) cast a no-op.

// f32 -> IEEE-754 half-float bits, for uploading weights in f16 mode.
const _f16buf = new ArrayBuffer(4);
const _f16f32 = new Float32Array(_f16buf);
const _f16u32 = new Uint32Array(_f16buf);
function f32ToHalf(val) {
  _f16f32[0] = val;
  const x = _f16u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);   // inf/nan
  const e = exp - 112;                                            // 127 - 15
  if (e >= 0x1f) return sign | 0x7c00;                           // overflow -> inf
  if (e <= 0) {
    if (e < -10) return sign;                                     // underflow -> 0
    mant |= 0x800000;
    const shift = 14 - e;
    let half = mant >> shift;
    if ((mant >> (shift - 1)) & 1) half += 1;                     // round
    return sign | half;
  }
  let half = (e << 10) | (mant >> 13);
  if (mant & 0x1000) half += 1;                                   // round
  return sign | half;
}
function f32ArrayToF16(arr) {
  const out = new Uint16Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = f32ToHalf(arr[i]);
  return out;
}

const buildPre = (T) => /* wgsl */`
@group(0) @binding(0) var inTex: texture_external;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read_write> outp: array<${T}>;
struct P { W: u32, H: u32, a: u32, b: u32 };
@group(0) @binding(3) var<uniform> p: P;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.W || gid.y >= p.H) { return; }
  let uv = (vec2f(f32(gid.x), f32(gid.y)) + 0.5) / vec2f(f32(p.W), f32(p.H));
  let c = textureSampleBaseClampToEdge(inTex, samp, uv);
  let idx = gid.y * p.W + gid.x;
  let px = p.W * p.H;
  outp[idx]           = ${T}(c.r - ${MEAN[0]});
  outp[px + idx]      = ${T}(c.g - ${MEAN[1]});
  outp[2u * px + idx] = ${T}(c.b - ${MEAN[2]});
}`;

// Each thread computes a 2x2 output block for a group of 8 output channels =
// 32 register accumulators (statically named → no spill). A 4x4 input patch is
// loaded once per input channel and reused across all 9 taps × 4 pixels, and
// every weight fetch is reused across all 4 pixels — maximizing arithmetic
// intensity (the measured bottleneck). Grid: ceil(W/2) × ceil(H/2) threads.
const buildConv = (T) => {
  const L = [];
  L.push(`
@group(0) @binding(0) var<storage, read> inp: array<${T}>;
@group(0) @binding(1) var<storage, read> wgt: array<${T}>;
@group(0) @binding(2) var<storage, read> bia: array<${T}>;
@group(0) @binding(3) var<storage, read_write> outp: array<${T}>;
struct P { W:u32, H:u32, inC:u32, outC:u32, act:u32, k:u32 };
@group(0) @binding(4) var<uniform> pp: P;
fn silu(v: f32) -> f32 { return v / (1.0 + exp(-v)); }
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let bx = gid.x * 2u; let by = gid.y * 2u;
  if (bx >= pp.W || by >= pp.H) { return; }
  let W = pp.W; let H = pp.H; let IC = pp.inC; let OC = pp.outC; let px = W * H;
  let o = gid.z * 8u; let stride = IC * 9u;
  let vx1 = (bx + 1u) < W; let vy1 = (by + 1u) < H;`);
  const acc = [];
  for (let pI = 0; pI < 4; pI++) for (let k = 0; k < 8; k++) acc.push(`a${pI}_${k}`);
  L.push('  ' + acc.map((a) => 'var ' + a + ' = 0.0;').join(' '));
  for (let k = 0; k < 8; k++)
    L.push(`  if (o+${k}u<OC) { let bv=f32(bia[o+${k}u]); a0_${k}=bv; a1_${k}=bv; a2_${k}=bv; a3_${k}=bv; }`);
  L.push('  for (var ic = 0u; ic < IC; ic++) {');
  L.push('    let cb = ic * px;');
  for (let c = 0; c < 4; c++)
    L.push(`    let cx${c}=i32(bx)+${c - 1}; let cok${c}=cx${c}>=0 && cx${c}<i32(W); let cc${c}=u32(clamp(cx${c},0,i32(W)-1));`);
  for (let r = 0; r < 4; r++)
    L.push(`    let ry${r}=i32(by)+${r - 1}; let rok${r}=ry${r}>=0 && ry${r}<i32(H); let rr${r}=cb+u32(clamp(ry${r},0,i32(H)-1))*W;`);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++)
    L.push(`    let p${r}_${c}=select(0.0, f32(inp[rr${r}+cc${c}]), rok${r} && cok${c});`);
  for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
    L.push(`    { let wc = ic*9u + ${ky * 3 + kx}u;`);
    for (let k = 0; k < 8; k++) L.push(`      let w${k}=f32(wgt[(o+${k}u)*stride+wc]);`);
    const s = [`p${ky}_${kx}`, `p${ky}_${kx + 1}`, `p${ky + 1}_${kx}`, `p${ky + 1}_${kx + 1}`];
    for (let k = 0; k < 8; k++)
      L.push(`      a0_${k}+=${s[0]}*w${k}; a1_${k}+=${s[1]}*w${k}; a2_${k}+=${s[2]}*w${k}; a3_${k}+=${s[3]}*w${k};`);
    L.push('    }');
  }
  L.push('  }');
  L.push('  let doAct = pp.act == 1u;');
  for (let k = 0; k < 8; k++) {
    L.push(`  if (o+${k}u<OC) {`);
    L.push(`    var v0=a0_${k}; var v1=a1_${k}; var v2=a2_${k}; var v3=a3_${k};`);
    L.push('    if (doAct) { v0=silu(v0); v1=silu(v1); v2=silu(v2); v3=silu(v3); }');
    L.push(`    let oc=(o+${k}u)*px;`);
    L.push(`    outp[oc+by*W+bx]=${T}(v0);`);
    L.push(`    if (vx1) { outp[oc+by*W+bx+1u]=${T}(v1); }`);
    L.push(`    if (vy1) { outp[oc+(by+1u)*W+bx]=${T}(v2); }`);
    L.push(`    if (vx1 && vy1) { outp[oc+(by+1u)*W+bx+1u]=${T}(v3); }`);
    L.push('  }');
  }
  L.push('}');
  return L.join('\n');
};

const buildAttn = (T) => /* wgsl */`
@group(0) @binding(0) var<storage, read> t3: array<${T}>;
@group(0) @binding(1) var<storage, read> xb: array<${T}>;
@group(0) @binding(2) var<storage, read_write> outp: array<${T}>;
struct P { W: u32, H: u32, C: u32, a: u32 };
@group(0) @binding(3) var<uniform> p: P;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y; let c = gid.z;
  if (x >= p.W || y >= p.H || c >= p.C) { return; }
  let idx = c * p.W * p.H + y * p.W + x;
  let v = f32(t3[idx]);
  let att = 1.0 / (1.0 + exp(-v)) - 0.5;
  outp[idx] = ${T}((v + f32(xb[idx])) * att);
}`;

// conv_cat: 1×1 over the concat of 4 C-channel buffers (4C inputs → C outputs).
// Channel boundaries are baked from C. 8 output channels per thread.
const buildCat = (T, C) => {
  const IN = 4 * C;
  const pick = `if (ic < ${C}u) { pix = f32(in0[ic * px + idx]); }
    else if (ic < ${2 * C}u) { pix = f32(in1[(ic - ${C}u) * px + idx]); }
    else if (ic < ${3 * C}u) { pix = f32(in2[(ic - ${2 * C}u) * px + idx]); }
    else { pix = f32(in3[(ic - ${3 * C}u) * px + idx]); }`;
  let macs = '';
  for (let k = 0; k < 8; k++) macs += `    a${k} += pix * f32(wgt[(o + ${k}u) * ${IN}u + ic]);\n`;
  return `
@group(0) @binding(0) var<storage, read> in0: array<${T}>;
@group(0) @binding(1) var<storage, read> in1: array<${T}>;
@group(0) @binding(2) var<storage, read> in2: array<${T}>;
@group(0) @binding(3) var<storage, read> in3: array<${T}>;
@group(0) @binding(4) var<storage, read> wgt: array<${T}>;
@group(0) @binding(5) var<storage, read> bia: array<${T}>;
@group(0) @binding(6) var<storage, read_write> outp: array<${T}>;
struct P { W: u32, H: u32, inC: u32, outC: u32 };
@group(0) @binding(7) var<uniform> p: P;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= p.W || y >= p.H) { return; }
  let px = p.W * p.H; let idx = y * p.W + x; let OC = p.outC;
  let o = gid.z * 8u;
  var a0=0.0; var a1=0.0; var a2=0.0; var a3=0.0; var a4=0.0; var a5=0.0; var a6=0.0; var a7=0.0;
  if (o+0u<OC){a0=f32(bia[o+0u]);} if (o+1u<OC){a1=f32(bia[o+1u]);}
  if (o+2u<OC){a2=f32(bia[o+2u]);} if (o+3u<OC){a3=f32(bia[o+3u]);}
  if (o+4u<OC){a4=f32(bia[o+4u]);} if (o+5u<OC){a5=f32(bia[o+5u]);}
  if (o+6u<OC){a6=f32(bia[o+6u]);} if (o+7u<OC){a7=f32(bia[o+7u]);}
  for (var ic = 0u; ic < ${IN}u; ic++) {
    var pix: f32;
    ${pick}
${macs}  }
  if (o+0u<OC){ outp[(o+0u)*px+idx]=${T}(a0); } if (o+1u<OC){ outp[(o+1u)*px+idx]=${T}(a1); }
  if (o+2u<OC){ outp[(o+2u)*px+idx]=${T}(a2); } if (o+3u<OC){ outp[(o+3u)*px+idx]=${T}(a3); }
  if (o+4u<OC){ outp[(o+4u)*px+idx]=${T}(a4); } if (o+5u<OC){ outp[(o+5u)*px+idx]=${T}(a5); }
  if (o+6u<OC){ outp[(o+6u)*px+idx]=${T}(a6); } if (o+7u<OC){ outp[(o+7u)*px+idx]=${T}(a7); }
}`;
};

const buildShuffle = (T) => /* wgsl */`
@group(0) @binding(0) var<storage, read> inp: array<${T}>;
@group(0) @binding(1) var outTex: texture_storage_2d<rgba8unorm, write>;
struct P { W: u32, H: u32, scale: u32, ch: u32 };
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let X = gid.x; let Y = gid.y;
  let outW = p.W * p.scale; let outH = p.H * p.scale;
  if (X >= outW || Y >= outH) { return; }
  let ix = X / p.scale; let iy = Y / p.scale;
  let sx = X % p.scale; let sy = Y % p.scale;
  let s2 = p.scale * p.scale;
  let base = sy * p.scale + sx;
  let px = p.W * p.H;
  let idx = iy * p.W + ix;
  let r = clamp(f32(inp[(0u * s2 + base) * px + idx]), 0.0, 1.0);
  let g = clamp(f32(inp[(1u * s2 + base) * px + idx]), 0.0, 1.0);
  let b = clamp(f32(inp[(2u * s2 + base) * px + idx]), 0.0, 1.0);
  textureStore(outTex, vec2u(X, Y), vec4f(r, g, b, 1.0));
}`;

// Catmull-Rom bicubic resample from the neural output to the display size —
// a cheap, sharp spatial finish (in the spirit of FSR's EASU) so a
// governed-down internal resolution still fills the screen crisply.
const SHADER_FINISH = /* wgsl */`
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
struct P { srcW: u32, srcH: u32, dstW: u32, dstH: u32 };
@group(0) @binding(2) var<uniform> p: P;

fn crw(t: f32) -> vec4<f32> {
  let t2 = t * t; let t3 = t2 * t;
  return vec4<f32>(
    -0.5 * t3 + t2 - 0.5 * t,
     1.5 * t3 - 2.5 * t2 + 1.0,
    -1.5 * t3 + 2.0 * t2 + 0.5 * t,
     0.5 * t3 - 0.5 * t2,
  );
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.dstW || gid.y >= p.dstH) { return; }
  let W = i32(p.srcW); let H = i32(p.srcH);
  let sx = (f32(gid.x) + 0.5) * f32(p.srcW) / f32(p.dstW) - 0.5;
  let sy = (f32(gid.y) + 0.5) * f32(p.srcH) / f32(p.dstH) - 0.5;
  let ix = i32(floor(sx)); let iy = i32(floor(sy));
  let wx = crw(sx - f32(ix)); let wy = crw(sy - f32(iy));

  var col = vec3<f32>(0.0);
  for (var m = 0; m < 4; m++) {
    let yy = clamp(iy - 1 + m, 0, H - 1);
    var row = vec3<f32>(0.0);
    for (var n = 0; n < 4; n++) {
      let xx = clamp(ix - 1 + n, 0, W - 1);
      row += wx[n] * textureLoad(src, vec2i(xx, yy), 0).rgb;
    }
    col += wy[m] * row;
  }
  textureStore(dst, vec2u(gid.x, gid.y), vec4f(clamp(col, vec3(0.0), vec3(1.0)), 1.0));
}`;

// Contrast-adaptive sharpen (FSR RCAS spirit): unsharp with a 5-tap cross,
// clamped to the local min/max so it boosts edge contrast without ringing/halos.
const SHADER_SHARPEN = /* wgsl */`
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
struct P { W: u32, H: u32, strq: u32, pad: u32 };
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.W || gid.y >= p.H) { return; }
  let x = i32(gid.x); let y = i32(gid.y);
  let W = i32(p.W); let H = i32(p.H);
  let c = textureLoad(src, vec2i(x, y), 0).rgb;
  let l = textureLoad(src, vec2i(max(x - 1, 0), y), 0).rgb;
  let r = textureLoad(src, vec2i(min(x + 1, W - 1), y), 0).rgb;
  let t = textureLoad(src, vec2i(x, max(y - 1, 0)), 0).rgb;
  let b = textureLoad(src, vec2i(x, min(y + 1, H - 1)), 0).rgb;
  let strength = f32(p.strq) / 4096.0;
  let sharp = c + strength * (4.0 * c - l - r - t - b);
  let mn = min(c, min(min(l, r), min(t, b)));
  let mx = max(c, max(max(l, r), max(t, b)));
  let outc = clamp(sharp, mn, mx);   // no overshoot beyond local neighborhood
  textureStore(dst, vec2u(gid.x, gid.y), vec4f(outc, 1.0));
}`;

if (typeof globalThis !== 'undefined') {
  globalThis.WebGPUSR = WebGPUSR;
}
