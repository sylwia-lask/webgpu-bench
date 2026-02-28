const STRIDE = 4 // x,y,vx,vy
const SPEED = 2

type FpsCb = (fps: number | null) => void

export class JsParticles {
  private readonly canvas: HTMLCanvasElement
  private readonly ctx: CanvasRenderingContext2D
  private readonly onFps: FpsCb

  private buf: Float32Array | null = null
  private count = 0
  private rafId: number | null = null

  private fpsTimestamps: number[] = []
  private lastFpsUpdate = 0

  constructor(canvas: HTMLCanvasElement, onFps: FpsCb) {
    this.canvas = canvas
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Failed to get 2D context.')
    this.ctx = ctx
    this.onFps = onFps
  }

  start(count: number): void {
    this.stop()

    const W = this.canvas.width
    const H = this.canvas.height

    this.count = count
    this.buf = new Float32Array(count * STRIDE)
    for (let i = 0; i < count; i++) {
      const o = i * STRIDE
      this.buf[o + 0] = Math.random() * W
      this.buf[o + 1] = Math.random() * H
      this.buf[o + 2] = (Math.random() * 2 - 1) * SPEED
      this.buf[o + 3] = (Math.random() * 2 - 1) * SPEED
    }

    this.fpsTimestamps = []
    this.lastFpsUpdate = 0
    this.rafId = requestAnimationFrame(this.frame)
  }

  stop(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId)
      this.rafId = null
    }
    this.onFps(null)
  }

  destroy(): void {
    this.stop()
  }

  private frame = (now: number): void => {
    const buf = this.buf
    if (!buf) return

    const W = this.canvas.width
    const H = this.canvas.height

    // update
    for (let i = 0; i < this.count; i++) {
      const o = i * STRIDE
      buf[o + 0] += buf[o + 2]
      buf[o + 1] += buf[o + 3]

      if (buf[o + 0] < 0) { buf[o + 0] = 0; buf[o + 2] *= -1 }
      if (buf[o + 0] > W) { buf[o + 0] = W; buf[o + 2] *= -1 }
      if (buf[o + 1] < 0) { buf[o + 1] = 0; buf[o + 3] *= -1 }
      if (buf[o + 1] > H) { buf[o + 1] = H; buf[o + 3] *= -1 }
    }

    // render
    const ctx = this.ctx
    ctx.fillStyle = '#030712'
    ctx.fillRect(0, 0, W, H)

    ctx.fillStyle = '#f59e0b'
    for (let i = 0; i < this.count; i++) {
      const o = i * STRIDE
      ctx.fillRect(buf[o], buf[o + 1], 2, 2)
    }

    // fps (throttled)
    this.fpsTimestamps.push(now)
    if (this.fpsTimestamps.length > 60) this.fpsTimestamps.shift()
    if (now - this.lastFpsUpdate >= 250 && this.fpsTimestamps.length >= 2) {
      const elapsed = this.fpsTimestamps.at(-1)! - this.fpsTimestamps[0]!
      this.onFps(Math.round(((this.fpsTimestamps.length - 1) / elapsed) * 1000))
      this.lastFpsUpdate = now
    }

    this.rafId = requestAnimationFrame(this.frame)
  }
}