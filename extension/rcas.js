/**
 * RCAS — Robust Contrast-Adaptive Sharpening
 * Based on AMD FidelityFX CAS/RCAS (MIT licensed).
 *
 * Single-pass edge-aware sharpening that enhances texture detail
 * without creating halos around edges. Runs as a post-process
 * after SR upscaling.
 *
 * The algorithm:
 * 1. Sample center pixel + 4 cardinal neighbors
 * 2. Compute local contrast (max - min of the 5 samples)
 * 3. Derive a sharpening weight that's inversely proportional to contrast
 *    (sharpen flat textures more, edges less)
 * 4. Apply a negative-lobe filter clamped to prevent ringing
 */

class RCASPass {
  constructor() {
    this.device = null;
    this.pipeline = null;
    this.bindGroupLayout = null;
    this.sharpness = 0.25;
  }

  async init(device) {
    this.device = device;

    const shaderModule = device.createShaderModule({
      code: RCAS_SHADER,
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
    const uniformData = new Float32Array([this.sharpness]);
    const sizeData = new Uint32Array([width, height]);
    const uniformBuf = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const combined = new ArrayBuffer(16);
    new Uint32Array(combined, 0, 2).set(sizeData);
    new Float32Array(combined, 8, 1).set(uniformData);
    this.device.queue.writeBuffer(uniformBuf, 0, combined);

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

const RCAS_SHADER = `
struct Params {
  width: u32,
  height: u32,
  sharpness: f32,
  _pad: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn load(ch: u32, x: i32, y: i32) -> f32 {
  let cx = clamp(x, 0, i32(params.width) - 1);
  let cy = clamp(y, 0, i32(params.height) - 1);
  return input[ch * params.width * params.height + u32(cy) * params.width + u32(cx)];
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

  for (var ch = 0u; ch < 3u; ch++) {
    // 5-tap pattern: center + N/S/E/W
    let c = load(ch, x, y);
    let n = load(ch, x, y - 1);
    let s = load(ch, x, y + 1);
    let w = load(ch, x - 1, y);
    let e = load(ch, x + 1, y);

    // Local min/max for contrast detection
    let mn = min(min(n, s), min(w, e));
    let mx = max(max(n, s), max(w, e));
    let mn2 = min(mn, c);
    let mx2 = max(mx, c);

    // Peak = distance from center to nearest extreme, normalized by range
    let range = mx2 - mn2;
    let peak = select(0.0, (1.0 - min(mn2, 1.0 - mx2) / range), range > 0.001);

    // Sharpening weight: less sharpening near edges (high contrast)
    // More sharpening in flat areas (low contrast)
    let wt = sqrt(min(peak, 1.0)) * (-1.0 / (4.0 * params.sharpness + 4.0));

    // Apply negative-lobe sharpening filter
    let sum = n * wt + s * wt + w * wt + e * wt;
    let result = (c + sum) / (1.0 + 4.0 * wt);

    // Clamp to prevent ringing beyond local min/max
    let final_val = clamp(result, mn2, mx2);

    output[ch * W * H + gid.y * W + gid.x] = final_val;
  }
}
`;

if (typeof globalThis !== 'undefined') {
  globalThis.RCASPass = RCASPass;
}
