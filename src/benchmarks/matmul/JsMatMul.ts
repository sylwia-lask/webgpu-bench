export interface JsMatMulRunResult {
  ms: number
}

export class JsMatMul {
  private n = 0
  private A: Float32Array | null = null
  private B: Float32Array | null = null
  private C: Float32Array | null = null

  init(n: number): void {
    this.n = n
    const count = n * n

    const A = new Float32Array(count)
    const B = new Float32Array(count)
    const C = new Float32Array(count)

    for (let i = 0; i < count; i++) {
      A[i] = (i % 97) * 0.01
      B[i] = (i % 89) * 0.02
      C[i] = 0
    }

    this.A = A
    this.B = B
    this.C = C
  }

  runOnce(): JsMatMulRunResult {
    const A = this.A
    const B = this.B
    const C = this.C
    const n = this.n
    if (!A || !B || !C || !n) throw new Error('JsMatMul not initialized.')

    // Reset output
    C.fill(0)

    const t0 = performance.now()

    // Tiled matrix multiply (block size 32 tends to be ok in JS)
    const BS = 32
    for (let ii = 0; ii < n; ii += BS) {
      for (let kk = 0; kk < n; kk += BS) {
        for (let jj = 0; jj < n; jj += BS) {
          const iMax = Math.min(ii + BS, n)
          const kMax = Math.min(kk + BS, n)
          const jMax = Math.min(jj + BS, n)

          for (let i = ii; i < iMax; i++) {
            const iRow = i * n
            for (let k = kk; k < kMax; k++) {
              const a = A[iRow + k]
              const kRow = k * n
              for (let j = jj; j < jMax; j++) {
                C[iRow + j] += a * B[kRow + j]
              }
            }
          }
        }
      }
    }

    const t1 = performance.now()
    return { ms: t1 - t0 }
  }

  sample(sampleCount = 8): number[] {
    const C = this.C
    if (!C) return []
    return Array.from(C.slice(0, sampleCount))
  }

  destroy(): void {
    this.A = null
    this.B = null
    this.C = null
    this.n = 0
  }
}