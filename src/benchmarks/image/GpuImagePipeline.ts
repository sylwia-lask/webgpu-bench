import passGray from "../../shaders/imagePassGrayscale.wgsl?raw";
import passBlurX from "../../shaders/imagePassBlurX.wgsl?raw";
import passBlurY from "../../shaders/imagePassBlurY.wgsl?raw";
import passSobel from "../../shaders/imagePassSobel.wgsl?raw";
import passThr from "../../shaders/imagePassThreshold.wgsl?raw";
import presentWgsl from "../../shaders/imagePresent.wgsl?raw";

export type ImagePipelineFlags = {
  grayscale: boolean;
  blur: boolean;
  sobel: boolean;
  threshold: boolean;
};

type ErrCb = (msg: string) => void;

export class GpuImagePipeline {
  private canvas: HTMLCanvasElement;
  private onError: ErrCb;

  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private format: GPUTextureFormat | null = null;

  private paramsBuf: GPUBuffer | null = null;

  private srcTex: GPUTexture | null = null;
  private ping: GPUTexture | null = null;
  private pong: GPUTexture | null = null;

  private pGray: GPUComputePipeline | null = null;
  private pBlurX: GPUComputePipeline | null = null;
  private pBlurY: GPUComputePipeline | null = null;
  private pSobel: GPUComputePipeline | null = null;
  private pThr: GPUComputePipeline | null = null;

  private pPresent: GPURenderPipeline | null = null;
  private presentSampler: GPUSampler | null = null;

  // OFFSCREEN render target we control
  private presentTex: GPUTexture | null = null;

  // Intermediate copy destination we control
  private blitTex: GPUTexture | null = null;

  private width = 0;
  private height = 0;

  constructor(canvas: HTMLCanvasElement, onError: ErrCb) {
    this.canvas = canvas;
    this.onError = onError;
  }

  async init(width: number, height: number): Promise<void> {
    this.destroy();

    this.width = width;
    this.height = height;

    if (!navigator.gpu) throw new Error("WebGPU is not supported in this browser.");

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter found.");

    const device = await adapter.requestDevice();
    this.device = device;

    const ctx = this.canvas.getContext("webgpu");
    if (!ctx) throw new Error("Failed to get WebGPU canvas context.");
    this.context = ctx;

    const format = navigator.gpu.getPreferredCanvasFormat();
    this.format = format;

    // Surface: use as render target only (most reliable in Chrome/Dawn)
    ctx.configure({
      device,
      format,
      alphaMode: "opaque",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.presentSampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });

    this.paramsBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Source texture (upload image)
    // IMPORTANT (Chrome/Dawn): copyExternalImageToTexture() requires COPY_DST + RENDER_ATTACHMENT.
    this.srcTex = device.createTexture({
      size: { width, height },
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Ping/pong (compute)
    const ppUsage =
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.COPY_DST;

    this.ping = device.createTexture({
      size: { width, height },
      format: "rgba8unorm",
      usage: ppUsage,
    });

    this.pong = device.createTexture({
      size: { width, height },
      format: "rgba8unorm",
      usage: ppUsage,
    });

    // OFFSCREEN present texture (render target + copy source)
    this.presentTex = device.createTexture({
      size: { width, height },
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });

    // Intermediate copy destination + sampled in the final blit render pass
    this.blitTex = device.createTexture({
      size: { width, height },
      format,
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    });

    // Compute pipelines
    this.pGray = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: passGray }),
        entryPoint: "cs_main",
      },
    });

    this.pBlurX = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: passBlurX }),
        entryPoint: "cs_main",
      },
    });

    this.pBlurY = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: passBlurY }),
        entryPoint: "cs_main",
      },
    });

    this.pSobel = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: passSobel }),
        entryPoint: "cs_main",
      },
    });

    this.pThr = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: passThr }),
        entryPoint: "cs_main",
      },
    });

    // Present pipeline (fullscreen triangle)
    const presentModule = device.createShaderModule({ code: presentWgsl });
    this.pPresent = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: presentModule, entryPoint: "vs_main" },
      fragment: {
        module: presentModule,
        entryPoint: "fs_main",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  async setSourceFromImageBitmap(bitmap: ImageBitmap): Promise<void> {
    const device = this.device;
    const src = this.srcTex;
    if (!device || !src) return;

    // Chrome/Dawn validates dst usage for CopyExternalImageToTexture very strictly.
    device.queue.copyExternalImageToTexture(
      { source: bitmap },
      { texture: src },
      { width: this.width, height: this.height },
    );

    await device.queue.onSubmittedWorkDone();
  }

  async run(flags: ImagePipelineFlags): Promise<number> {
    const device = this.device;
    const ctx = this.context;
    const paramsBuf = this.paramsBuf;
    const presentTex = this.presentTex;
    const blitTex = this.blitTex;

    if (!device || !ctx || !paramsBuf || !presentTex || !blitTex) {
      throw new Error("GPU pipeline not initialized.");
    }

    const srcTex = this.srcTex!;
    const ping = this.ping!;
    const pong = this.pong!;

    const f =
      (flags.grayscale ? 1 : 0) |
      (flags.blur ? 2 : 0) |
      (flags.sobel ? 4 : 0) |
      (flags.threshold ? 8 : 0);

    const raw = new ArrayBuffer(16);
    new Uint32Array(raw, 0, 4).set([this.width, this.height, f, 0]);
    device.queue.writeBuffer(paramsBuf, 0, raw);

    const workX = Math.ceil(this.width / 16);
    const workY = Math.ceil(this.height / 16);

    const t0 = performance.now();
    const enc = device.createCommandEncoder();

    // acquire surface ONCE
    const canvasTex = ctx.getCurrentTexture();

    const doPass = (
      pipe: GPUComputePipeline,
      input: GPUTexture,
      output: GPUTexture,
    ) => {
      const bg = device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 1, resource: input.createView() },
          { binding: 2, resource: output.createView() },
          { binding: 3, resource: { buffer: paramsBuf } },
        ],
      });

      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(workX, workY);
      pass.end();
    };

    let curIn: GPUTexture = srcTex;
    let curOut: GPUTexture = ping;

    if (flags.grayscale) {
      doPass(this.pGray!, curIn, curOut);
      curIn = ping;
      curOut = pong;
    }

    if (flags.blur) {
      doPass(this.pBlurX!, curIn, curOut);
      [curIn, curOut] = [curOut, curOut === ping ? pong : ping];

      doPass(this.pBlurY!, curIn, curOut);
      [curIn, curOut] = [curOut, curOut === ping ? pong : ping];
    }

    if (flags.sobel) {
      doPass(this.pSobel!, curIn, curOut);
      [curIn, curOut] = [curOut, curOut === ping ? pong : ping];
    }

    if (flags.threshold) {
      doPass(this.pThr!, curIn, curOut);
      [curIn, curOut] = [curOut, curOut === ping ? pong : ping];
    }

    const anyEnabled =
      flags.grayscale || flags.blur || flags.sobel || flags.threshold;

    const finalTex = anyEnabled ? curIn : srcTex;

    // PASS 1: render finalTex -> presentTex
    const presentBG = device.createBindGroup({
      layout: this.pPresent!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.presentSampler! },
        { binding: 1, resource: finalTex.createView() },
      ],
    });

    const pass1 = enc.beginRenderPass({
      colorAttachments: [
        {
          view: presentTex.createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0.07, g: 0.07, b: 0.09, a: 1.0 },
        },
      ],
    });

    pass1.setPipeline(this.pPresent!);
    pass1.setBindGroup(0, presentBG);
    pass1.draw(3);
    pass1.end();

    // COPY: presentTex -> blitTex
    enc.copyTextureToTexture(
      { texture: presentTex },
      { texture: blitTex },
      { width: this.width, height: this.height },
    );

    // PASS 2: render blitTex -> canvas
    const blitBG = device.createBindGroup({
      layout: this.pPresent!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.presentSampler! },
        { binding: 1, resource: blitTex.createView() },
      ],
    });

    const pass2 = enc.beginRenderPass({
      colorAttachments: [
        {
          view: canvasTex.createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 1.0 },
        },
      ],
    });

    pass2.setPipeline(this.pPresent!);
    pass2.setBindGroup(0, blitBG);
    pass2.draw(3);
    pass2.end();

    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();

    return performance.now() - t0;
  }

  destroy(): void {
    this.srcTex?.destroy();
    this.ping?.destroy();
    this.pong?.destroy();
    this.presentTex?.destroy();
    this.blitTex?.destroy();
    this.paramsBuf?.destroy();
    this.device?.destroy();

    this.srcTex = null;
    this.ping = null;
    this.pong = null;
    this.presentTex = null;
    this.blitTex = null;
    this.paramsBuf = null;

    this.pGray = null;
    this.pBlurX = null;
    this.pBlurY = null;
    this.pSobel = null;
    this.pThr = null;

    this.pPresent = null;
    this.presentSampler = null;

    this.context = null;
    this.format = null;
    this.device = null;
  }
}