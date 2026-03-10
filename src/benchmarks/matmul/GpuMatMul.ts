import matmulWgsl from "../../shaders/matmulCompute.wgsl?raw";

export interface GpuMatMulRunResult {
  ms: number;
}

export class GpuMatMul {
  private device: GPUDevice | null = null;
  private pipeline: GPUComputePipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;

  private aBuf: GPUBuffer | null = null;
  private bBuf: GPUBuffer | null = null;
  private cBuf: GPUBuffer | null = null;
  private paramsBuf: GPUBuffer | null = null;

  private bindGroup: GPUBindGroup | null = null;
  private workgroupsX = 0;
  private workgroupsY = 0;

  async init(n: number): Promise<void> {
    this.destroy();

    if (!navigator.gpu)
      throw new Error("WebGPU is not supported in this browser.");

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter found.");

    this.device = await adapter.requestDevice();

    const device = this.device;

    this.pipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: matmulWgsl }),
        entryPoint: "cs_main",
      },
    });
    this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);

    const count = n * n;
    const bytes = count * 4;

    // Use deterministic-ish data (not all zeros) to avoid extreme compiler/driver weirdness.
    const A = new Float32Array(count);
    const B = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      A[i] = (i % 97) * 0.01;
      B[i] = (i % 89) * 0.02;
    }

    this.aBuf = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.bBuf = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.cBuf = device.createBuffer({
      size: bytes,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(this.aBuf, 0, A);
    device.queue.writeBuffer(this.bBuf, 0, B);

    this.paramsBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const paramsRaw = new ArrayBuffer(16);
    new Uint32Array(paramsRaw, 0, 1).set([n]);
    device.queue.writeBuffer(this.paramsBuf, 0, paramsRaw);

    this.bindGroup = device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.aBuf } },
        { binding: 1, resource: { buffer: this.bBuf } },
        { binding: 2, resource: { buffer: this.cBuf } },
        { binding: 3, resource: { buffer: this.paramsBuf } },
      ],
    });

    this.workgroupsX = Math.ceil(n / 16);
    this.workgroupsY = Math.ceil(n / 16);
  }

  /**
   * Run one dispatch (C = A*B) and wait until GPU work completes.
   * Returns elapsed time measured on CPU around submission+completion.
   */
  async runOnce(): Promise<GpuMatMulRunResult> {
    const device = this.device;
    const pipeline = this.pipeline;
    const bindGroup = this.bindGroup;
    if (!device || !pipeline || !bindGroup)
      throw new Error("GpuMatMul not initialized.");

    const t0 = performance.now();

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this.workgroupsX, this.workgroupsY, 1);
    pass.end();

    device.queue.submit([enc.finish()]);

    await device.queue.onSubmittedWorkDone();

    const t1 = performance.now();
    return { ms: t1 - t0 };
  }

  /**
   * Optional: read back a few values to ensure we don't accidentally optimize away the work.
   * (Not needed for correctness, but good sanity.)
   */
  async readBackSample(sampleCount = 8): Promise<number[]> {
    const device = this.device;
    const cBuf = this.cBuf;
    if (!device || !cBuf) throw new Error("GpuMatMul not initialized.");

    const bytes = sampleCount * 4;
    const readBuf = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(cBuf, 0, readBuf, 0, bytes);
    device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();
    readBuf.destroy();

    return Array.from(copy);
  }

  destroy(): void {
    if (this.aBuf) this.aBuf.destroy();
    if (this.bBuf) this.bBuf.destroy();
    if (this.cBuf) this.cBuf.destroy();
    if (this.paramsBuf) this.paramsBuf.destroy();

    this.aBuf = null;
    this.bBuf = null;
    this.cBuf = null;
    this.paramsBuf = null;
    this.bindGroup = null;
    this.bindGroupLayout = null;
    this.pipeline = null;

    if (this.device) this.device.destroy();
    this.device = null;
    this.workgroupsX = 0;
    this.workgroupsY = 0;
  }
}
