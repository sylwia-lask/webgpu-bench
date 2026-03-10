import type { ImagePipelineFlags } from "./GpuImagePipeline";

export class JsImagePipeline {
  private ctx: CanvasRenderingContext2D;

  private w = 0;
  private h = 0;

  private src: Uint8ClampedArray | null = null;
  private a: Float32Array | null = null;
  private b: Float32Array | null = null;

  constructor(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Failed to get 2D context.");
    this.ctx = ctx;
  }

  init(w: number, h: number): void {
    this.w = w;
    this.h = h;
    this.src = null;
    this.a = new Float32Array(w * h * 3);
    this.b = new Float32Array(w * h * 3);
  }

  setSourceFromImageData(img: ImageData): void {
    this.src = img.data;
  }

  run(flags: ImagePipelineFlags): number {
    if (!this.src || !this.a || !this.b)
      throw new Error("JS pipeline not initialized.");

    const t0 = performance.now();

    const w = this.w;
    const h = this.h;
    const src = this.src;
    let cur = this.a;
    let tmp = this.b;

    // Load source as RGB (3 floats per pixel)
    for (let i = 0; i < w * h; i++) {
      cur[i * 3 + 0] = src[i * 4 + 0] / 255;
      cur[i * 3 + 1] = src[i * 4 + 1] / 255;
      cur[i * 3 + 2] = src[i * 4 + 2] / 255;
    }

    if (flags.grayscale) {
      for (let i = 0; i < w * h; i++) {
        const luma =
          0.2126 * cur[i * 3 + 0] +
          0.7152 * cur[i * 3 + 1] +
          0.0722 * cur[i * 3 + 2];
        cur[i * 3 + 0] = luma;
        cur[i * 3 + 1] = luma;
        cur[i * 3 + 2] = luma;
      }
    }

    if (flags.blur) {
      const w0 = 0.227027;
      const w1 = 0.1945946;
      const w2 = 0.1216216;
      const w3 = 0.054054;
      const w4 = 0.016216;

      // Horizontal pass: cur -> tmp
      for (let y = 0; y < h; y++) {
        const row = y * w;
        for (let x = 0; x < w; x++) {
          const xm1 = Math.max(0, x - 1);
          const xp1 = Math.min(w - 1, x + 1);
          const xm2 = Math.max(0, x - 2);
          const xp2 = Math.min(w - 1, x + 2);
          const xm3 = Math.max(0, x - 3);
          const xp3 = Math.min(w - 1, x + 3);
          const xm4 = Math.max(0, x - 4);
          const xp4 = Math.min(w - 1, x + 4);
          const dst = (row + x) * 3;
          for (let c = 0; c < 3; c++) {
            tmp[dst + c] =
              cur[(row + x) * 3 + c] * w0 +
              cur[(row + xm1) * 3 + c] * w1 +
              cur[(row + xp1) * 3 + c] * w1 +
              cur[(row + xm2) * 3 + c] * w2 +
              cur[(row + xp2) * 3 + c] * w2 +
              cur[(row + xm3) * 3 + c] * w3 +
              cur[(row + xp3) * 3 + c] * w3 +
              cur[(row + xm4) * 3 + c] * w4 +
              cur[(row + xp4) * 3 + c] * w4;
          }
        }
      }

      // Vertical pass: tmp -> cur
      for (let y = 0; y < h; y++) {
        const ym1 = Math.max(0, y - 1);
        const yp1 = Math.min(h - 1, y + 1);
        const ym2 = Math.max(0, y - 2);
        const yp2 = Math.min(h - 1, y + 2);
        const ym3 = Math.max(0, y - 3);
        const yp3 = Math.min(h - 1, y + 3);
        const ym4 = Math.max(0, y - 4);
        const yp4 = Math.min(h - 1, y + 4);
        for (let x = 0; x < w; x++) {
          const dst = (y * w + x) * 3;
          for (let c = 0; c < 3; c++) {
            cur[dst + c] =
              tmp[(y * w + x) * 3 + c] * w0 +
              tmp[(ym1 * w + x) * 3 + c] * w1 +
              tmp[(yp1 * w + x) * 3 + c] * w1 +
              tmp[(ym2 * w + x) * 3 + c] * w2 +
              tmp[(yp2 * w + x) * 3 + c] * w2 +
              tmp[(ym3 * w + x) * 3 + c] * w3 +
              tmp[(yp3 * w + x) * 3 + c] * w3 +
              tmp[(ym4 * w + x) * 3 + c] * w4 +
              tmp[(yp4 * w + x) * 3 + c] * w4;
          }
        }
      }
    }

    if (flags.sobel) {
      const luma = (x: number, y: number) => {
        x = Math.max(0, Math.min(w - 1, x));
        y = Math.max(0, Math.min(h - 1, y));
        const i = (y * w + x) * 3;
        return 0.2126 * cur[i] + 0.7152 * cur[i + 1] + 0.0722 * cur[i + 2];
      };

      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const tl = luma(x - 1, y - 1);
          const t = luma(x + 0, y - 1);
          const tr = luma(x + 1, y - 1);
          const l = luma(x - 1, y + 0);
          const r = luma(x + 1, y + 0);
          const bl = luma(x - 1, y + 1);
          const b0 = luma(x + 0, y + 1);
          const br = luma(x + 1, y + 1);

          const gx = -tl + tr - 2 * l + 2 * r - bl + br;
          const gy = -tl - 2 * t - tr + bl + 2 * b0 + br;

          const e = Math.max(0, Math.min(1, Math.sqrt(gx * gx + gy * gy) * 1.2));
          const dst = (y * w + x) * 3;
          tmp[dst + 0] = e;
          tmp[dst + 1] = e;
          tmp[dst + 2] = e;
        }
      }

      [cur, tmp] = [tmp, cur];
    }

    if (flags.threshold) {
      const thr = 0.35;
      for (let i = 0; i < w * h; i++) {
        const luma =
          0.2126 * cur[i * 3 + 0] +
          0.7152 * cur[i * 3 + 1] +
          0.0722 * cur[i * 3 + 2];
        const v = luma > thr ? 1 : 0;
        cur[i * 3 + 0] = v;
        cur[i * 3 + 1] = v;
        cur[i * 3 + 2] = v;
      }
    }

    // Present to canvas (convert RGB float array to RGBA)
    const out = new ImageData(w, h);
    const d = out.data;
    for (let i = 0; i < w * h; i++) {
      d[i * 4 + 0] = Math.max(0, Math.min(255, Math.floor(cur[i * 3 + 0] * 255)));
      d[i * 4 + 1] = Math.max(0, Math.min(255, Math.floor(cur[i * 3 + 1] * 255)));
      d[i * 4 + 2] = Math.max(0, Math.min(255, Math.floor(cur[i * 3 + 2] * 255)));
      d[i * 4 + 3] = 255;
    }
    this.ctx.putImageData(out, 0, 0);

    const t1 = performance.now();
    return t1 - t0;
  }
}
