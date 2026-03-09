import type { ImagePipelineFlags } from './GpuImagePipeline'

export class JsImagePipeline {
  private ctx: CanvasRenderingContext2D

  private w = 0
  private h = 0

  private src: Uint8ClampedArray | null = null
  private a: Float32Array | null = null
  private b: Float32Array | null = null

  constructor(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Failed to get 2D context.')
    this.ctx = ctx
  }

  init(w: number, h: number): void {
    this.w = w
    this.h = h
    this.src = null
    this.a = new Float32Array(w * h) // grayscale space
    this.b = new Float32Array(w * h)
  }

  setSourceFromImageData(img: ImageData): void {
    this.src = img.data
  }

  run(flags: ImagePipelineFlags): number {
    if (!this.src || !this.a || !this.b) throw new Error('JS pipeline not initialized.')

    const t0 = performance.now()

    const w = this.w
    const h = this.h
    const src = this.src
    const a = this.a
    const b = this.b

    // 1) Grayscale (or just luma baseline)
    if (flags.grayscale || flags.blur || flags.sobel || flags.threshold) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const i = (y * w + x)
          const o = i * 4
          const r = src[o + 0] / 255
          const g = src[o + 1] / 255
          const bb = src[o + 2] / 255
          a[i] = 0.2126 * r + 0.7152 * g + 0.0722 * bb
        }
      }
    }

    // 2) Blur (separable 9-tap)
    if (flags.blur) {
      // blur X: a -> b
      const w0 = 0.227027
      const w1 = 0.1945946
      const w2 = 0.1216216
      const w3 = 0.054054
      const w4 = 0.016216

      for (let y = 0; y < h; y++) {
        const row = y * w
        for (let x = 0; x < w; x++) {
          const xm1 = Math.max(0, x - 1)
          const xp1 = Math.min(w - 1, x + 1)
          const xm2 = Math.max(0, x - 2)
          const xp2 = Math.min(w - 1, x + 2)
          const xm3 = Math.max(0, x - 3)
          const xp3 = Math.min(w - 1, x + 3)
          const xm4 = Math.max(0, x - 4)
          const xp4 = Math.min(w - 1, x + 4)

          const i = row + x
          b[i] =
            a[row + x]   * w0 +
            a[row + xm1] * w1 + a[row + xp1] * w1 +
            a[row + xm2] * w2 + a[row + xp2] * w2 +
            a[row + xm3] * w3 + a[row + xp3] * w3 +
            a[row + xm4] * w4 + a[row + xp4] * w4
        }
      }

      // blur Y: b -> a
      for (let y = 0; y < h; y++) {
        const ym1 = Math.max(0, y - 1)
        const yp1 = Math.min(h - 1, y + 1)
        const ym2 = Math.max(0, y - 2)
        const yp2 = Math.min(h - 1, y + 2)
        const ym3 = Math.max(0, y - 3)
        const yp3 = Math.min(h - 1, y + 3)
        const ym4 = Math.max(0, y - 4)
        const yp4 = Math.min(h - 1, y + 4)

        for (let x = 0; x < w; x++) {
          const i = y * w + x
          a[i] =
            b[y * w + x]     * w0 +
            b[ym1 * w + x]   * w1 + b[yp1 * w + x] * w1 +
            b[ym2 * w + x]   * w2 + b[yp2 * w + x] * w2 +
            b[ym3 * w + x]   * w3 + b[yp3 * w + x] * w3 +
            b[ym4 * w + x]   * w4 + b[yp4 * w + x] * w4
        }
      }
    }

    // 3) Sobel (a -> b)
    if (flags.sobel) {
      const get = (x: number, y: number) => {
        x = Math.max(0, Math.min(w - 1, x))
        y = Math.max(0, Math.min(h - 1, y))
        return a[y * w + x]
      }

      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const tl = get(x - 1, y - 1)
          const t  = get(x + 0, y - 1)
          const tr = get(x + 1, y - 1)
          const l  = get(x - 1, y + 0)
          const r  = get(x + 1, y + 0)
          const bl = get(x - 1, y + 1)
          const b0 = get(x + 0, y + 1)
          const br = get(x + 1, y + 1)

          const gx = (-1 * tl) + (1 * tr) + (-2 * l) + (2 * r) + (-1 * bl) + (1 * br)
          const gy = (-1 * tl) + (-2 * t ) + (-1 * tr) + ( 1 * bl) + (2 * b0) + (1 * br)

          const mag = Math.sqrt(gx * gx + gy * gy)
          b[y * w + x] = Math.max(0, Math.min(1, mag * 1.2))
        }
      }

      // swap b -> a (so next pass reads from a)
      a.set(b)
    }

    // 4) Threshold (a -> a)
    if (flags.threshold) {
      const thr = 0.35
      for (let i = 0; i < a.length; i++) {
        a[i] = a[i] > thr ? 1 : 0
      }
    }

    // Present to canvas (convert grayscale a[] to RGBA)
    const out = new ImageData(w, h)
    const d = out.data
    for (let i = 0; i < a.length; i++) {
      const v = Math.max(0, Math.min(255, Math.floor(a[i] * 255)))
      const o = i * 4
      d[o + 0] = v
      d[o + 1] = v
      d[o + 2] = v
      d[o + 3] = 255 
    }
    this.ctx.putImageData(out, 0, 0)

    const t1 = performance.now()
    return t1 - t0
  }
}