import computeWgsl from "../../shaders/particlesCompute.wgsl?raw";
import renderWgsl from "../../shaders/particlesRenderInstanced.wgsl?raw";

const STRIDE = 4; 
const BYTES = STRIDE * 4;
const SPEED = 2;

type FpsCb = (fps: number | null) => void;
type ErrCb = (msg: string) => void;

interface GpuState {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;

  computePipeline: GPUComputePipeline;
  renderPipeline: GPURenderPipeline;

  particleBuf: GPUBuffer;
  paramsBuf: GPUBuffer;

  computeBG: GPUBindGroup;
  renderBG: GPUBindGroup;

  count: number;
  workgroups: number;
}

export class GpuParticles {
  private readonly canvas: HTMLCanvasElement;
  private readonly onFps: FpsCb;
  private readonly onError: ErrCb;

  private gpu: GpuState | null = null;
  private rafId: number | null = null;

  private fpsTimestamps: number[] = [];
  private lastFpsUpdate = 0;

  constructor(canvas: HTMLCanvasElement, onFps: FpsCb, onError: ErrCb) {
    this.canvas = canvas;
    this.onFps = onFps;
    this.onError = onError;
  }

  async start(count: number): Promise<void> {
    this.stop();

    if (!navigator.gpu) {
      this.onError("WebGPU is not supported in this browser.");
      return;
    }

    try {
      await this.init(count);
      this.fpsTimestamps = [];
      this.lastFpsUpdate = 0;
      this.onError(""); 
      this.rafId = requestAnimationFrame(this.frame);
    } catch (e) {
      this.onError(e instanceof Error ? e.message : "WebGPU init failed.");
    }
  }

  stop(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    if (this.gpu) {
      this.gpu.particleBuf.destroy();
      this.gpu.paramsBuf.destroy();
      this.gpu.device.destroy();
      this.gpu = null;
    }
    this.onFps(null);
  }

  destroy(): void {
    this.stop();
  }

  private async init(count: number): Promise<void> {
    const canvas = this.canvas;

    const W = canvas.width;
    const H = canvas.height;

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter found.");

    const device = await adapter.requestDevice();

    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("Failed to get WebGPU context.");

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: "opaque" });

    // Particles init (CPU -> GPU)
    const data = new Float32Array(count * STRIDE);
    for (let i = 0; i < count; i++) {
      const o = i * STRIDE;
      data[o + 0] = Math.random() * W;
      data[o + 1] = Math.random() * H;
      data[o + 2] = (Math.random() * 2 - 1) * SPEED;
      data[o + 3] = (Math.random() * 2 - 1) * SPEED;
    }

    const particleBuf = device.createBuffer({
      size: count * BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(particleBuf, 0, data);

    const paramsBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const paramsRaw = new ArrayBuffer(16);
    new Float32Array(paramsRaw, 0, 2).set([W, H]);
    new Uint32Array(paramsRaw, 8, 1).set([count]);
    device.queue.writeBuffer(paramsBuf, 0, paramsRaw);

    const computePipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: computeWgsl }),
        entryPoint: "cs_main",
      },
    });

    const renderModule = device.createShaderModule({ code: renderWgsl });
    const renderPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: renderModule, entryPoint: "vs_main" },
      fragment: {
        module: renderModule,
        entryPoint: "fs_main",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
    });

    const computeBG = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: particleBuf } },
        { binding: 1, resource: { buffer: paramsBuf } },
      ],
    });

    const renderBG = device.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: particleBuf } },
        { binding: 1, resource: { buffer: paramsBuf } },
      ],
    });

    this.gpu = {
      device,
      context,
      format,
      computePipeline,
      renderPipeline,
      particleBuf,
      paramsBuf,
      computeBG,
      renderBG,
      count,
      workgroups: Math.ceil(count / 64),
    };
  }

  private frame = (now: number): void => {
    const g = this.gpu;
    if (!g) return;

    const enc = g.device.createCommandEncoder();

    const cp = enc.beginComputePass();
    cp.setPipeline(g.computePipeline);
    cp.setBindGroup(0, g.computeBG);
    cp.dispatchWorkgroups(g.workgroups);
    cp.end();

    const rp = enc.beginRenderPass({
      colorAttachments: [
        {
          view: g.context.getCurrentTexture().createView(),
          clearValue: { r: 0.012, g: 0.012, b: 0.016, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    rp.setPipeline(g.renderPipeline);
    rp.setBindGroup(0, g.renderBG);

    // INSTANCED: 6 vertices per quad, instanceCount = particle count
    rp.draw(6, g.count);

    rp.end();

    g.device.queue.submit([enc.finish()]);

    // FPS (throttled ~4x/s)
    this.fpsTimestamps.push(now);
    if (this.fpsTimestamps.length > 60) this.fpsTimestamps.shift();
    if (now - this.lastFpsUpdate >= 250 && this.fpsTimestamps.length >= 2) {
      const elapsed = this.fpsTimestamps.at(-1)! - this.fpsTimestamps[0]!;
      this.onFps(
        Math.round(((this.fpsTimestamps.length - 1) / elapsed) * 1000),
      );
      this.lastFpsUpdate = now;
    }

    this.rafId = requestAnimationFrame(this.frame);
  };
}
