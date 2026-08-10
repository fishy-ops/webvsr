/**
 * FXAA — Fast Approximate Anti-Aliasing (WebGPU)
 * Based on FXAA 3.11 by Timothy Lottes (NVIDIA).
 *
 * Detects edges via luminance contrast and blends along them
 * to smooth jagged staircase patterns from pixel-shuffle upsampling.
 *
 * Pipeline: SR → FXAA (smooth edges) → RCAS (sharpen textures)
 */

class FXAAPass {
  constructor() {
    this.device = null;
    this.pipeline = null;
    this.bindGroupLayout = null;
    this.edgeThreshold = 0.125;
    this.edgeThresholdMin = 0.0312;
    this.subpixQuality = 0.75;
  }

  async init(device) {
    this.device = device;

    const shaderModule = device.createShaderModule({
      code: FXAA_SHADER,
    });

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    this.pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });
  }

  async apply(inputBuffer, outputBuffer, width, height) {
    const uniformData = new ArrayBuffer(32);
    const u32 = new Uint32Array(uniformData, 0, 2);
    const f32 = new Float32Array(uniformData, 8, 3);
    u32[0] = width;
    u32[1] = height;
    f32[0] = this.edgeThreshold;
    f32[1] = this.edgeThresholdMin;
    f32[2] = this.subpixQuality;

    const uniformBuf = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(uniformBuf, 0, uniformData);

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: uniformBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(width / 16),
      Math.ceil(height / 16),
      1
    );
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    uniformBuf.destroy();
  }
}

const FXAA_SHADER = `
struct Params {
  width: u32,
  height: u32,
  edge_threshold: f32,
  edge_threshold_min: f32,
  subpix_quality: f32,
  _pad1: f32,
  _pad2: f32,
  _pad3: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn load(ch: u32, x: i32, y: i32) -> f32 {
  let cx = clamp(x, 0, i32(params.width) - 1);
  let cy = clamp(y, 0, i32(params.height) - 1);
  return input[ch * params.width * params.height + u32(cy) * params.width + u32(cx)];
}

fn luma(x: i32, y: i32) -> f32 {
  return load(0u, x, y) * 0.299 + load(1u, x, y) * 0.587 + load(2u, x, y) * 0.114;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);

  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  let W = params.width;
  let H = params.height;

  // Sample 3x3 luma neighborhood
  let luma_c  = luma(x, y);
  let luma_n  = luma(x, y - 1);
  let luma_s  = luma(x, y + 1);
  let luma_w  = luma(x - 1, y);
  let luma_e  = luma(x + 1, y);
  let luma_nw = luma(x - 1, y - 1);
  let luma_ne = luma(x + 1, y - 1);
  let luma_sw = luma(x - 1, y + 1);
  let luma_se = luma(x + 1, y + 1);

  // Local contrast
  let range_max = max(max(max(luma_n, luma_s), max(luma_w, luma_e)), luma_c);
  let range_min = min(min(min(luma_n, luma_s), min(luma_w, luma_e)), luma_c);
  let local_range = range_max - range_min;

  // Skip if below edge threshold
  let threshold = max(params.edge_threshold_min, range_max * params.edge_threshold);
  if (local_range < threshold) {
    for (var ch = 0u; ch < 3u; ch++) {
      output[ch * W * H + gid.y * W + gid.x] = load(ch, x, y);
    }
    return;
  }

  // Edge direction: horizontal vs vertical
  let edge_h = abs(luma_n + luma_s - 2.0 * luma_c) * 2.0
             + abs(luma_ne + luma_se - 2.0 * luma_e)
             + abs(luma_nw + luma_sw - 2.0 * luma_w);
  let edge_v = abs(luma_w + luma_e - 2.0 * luma_c) * 2.0
             + abs(luma_nw + luma_ne - 2.0 * luma_n)
             + abs(luma_sw + luma_se - 2.0 * luma_s);
  let is_horizontal = edge_h >= edge_v;

  // Sub-pixel blend factor
  let luma_avg = (luma_n + luma_s + luma_w + luma_e) * 0.25;
  var sub_pix = abs(luma_avg - luma_c) / (local_range + 1e-8);
  sub_pix = clamp(sub_pix, 0.0, 1.0);
  sub_pix = (-2.0 * sub_pix + 3.0) * sub_pix * sub_pix;
  sub_pix = sub_pix * sub_pix * params.subpix_quality;

  // Determine blend neighbor (strongest gradient side)
  var blend_x = x;
  var blend_y = y;
  if (is_horizontal) {
    let grad_n = abs(luma_n - luma_c);
    let grad_s = abs(luma_s - luma_c);
    if (grad_n > grad_s) {
      blend_y = y - 1;
    } else {
      blend_y = y + 1;
    }
  } else {
    let grad_w = abs(luma_w - luma_c);
    let grad_e = abs(luma_e - luma_c);
    if (grad_w > grad_e) {
      blend_x = x - 1;
    } else {
      blend_x = x + 1;
    }
  }

  // Blend center with neighbor
  for (var ch = 0u; ch < 3u; ch++) {
    let center = load(ch, x, y);
    let neighbor = load(ch, blend_x, blend_y);
    output[ch * W * H + gid.y * W + gid.x] = center * (1.0 - sub_pix) + neighbor * sub_pix;
  }
}
`;

if (typeof globalThis !== 'undefined') {
  globalThis.FXAAPass = FXAAPass;
}
