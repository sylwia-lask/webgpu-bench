import triangleWgsl from "../shaders/triangle.wgsl?raw";

export async function initWebGPU(
  canvas: HTMLCanvasElement,
): Promise<() => void> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not supported in this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No WebGPU adapter found.");
  }

  const device = await adapter.requestDevice();

  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new Error("Failed to get WebGPU context from canvas.");
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format });

  const shaderModule = device.createShaderModule({ code: triangleWgsl });

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
  });

  function frame() {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context!.getCurrentTexture().createView(),
          clearValue: { r: 0.07, g: 0.07, b: 0.09, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(pipeline);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  frame();

  // Return cleanup to destroy the device on unmount
  return () => device.destroy();
}
