import { useEffect, useMemo, useRef, useState } from 'react'
import { GpuMatMul } from './matmul/GpuMatMul'
import { JsMatMul } from './matmul/JsMatMul'

function msToPretty(ms: number | null) {
  if (ms === null) return '—'
  if (ms < 1) return `${(ms * 1000).toFixed(0)} µs`
  if (ms < 1000) return `${ms.toFixed(2)} ms`
  return `${(ms / 1000).toFixed(2)} s`
}

type Result = {
  lastMs: number | null
  avgMs: number | null
  sample: number[]
}

const emptyResult: Result = { lastMs: null, avgMs: null, sample: [] }

export default function MatMulBenchmark() {
  const [n, setN] = useState(1024)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [cpuRes, setCpuRes] = useState<Result>(emptyResult)
  const [gpuRes, setGpuRes] = useState<Result>(emptyResult)

  const gpu = useRef<GpuMatMul | null>(null)
  const cpu = useRef<JsMatMul | null>(null)

  const canUseWebGPU = useMemo(() => !!navigator.gpu, [])

  useEffect(() => {
    return () => {
      gpu.current?.destroy()
      cpu.current?.destroy()
    }
  }, [])

  async function prepareBoth() {
    setError(null)
    setCpuRes(emptyResult)
    setGpuRes(emptyResult)

    // CPU
    if (!cpu.current) cpu.current = new JsMatMul()
    cpu.current.init(n)

    // GPU
    if (!canUseWebGPU) throw new Error('WebGPU not available in this browser.')
    if (!gpu.current) gpu.current = new GpuMatMul()
    await gpu.current.init(n)
  }

  async function runBoth(iterations: number) {
    if (running) return
    setRunning(true)

    try {
      await prepareBoth()

      // Warmup both (important!)
      cpu.current!.runOnce()
      await gpu.current!.runOnce()

      const cpuTimes: number[] = []
      const gpuTimes: number[] = []

      for (let i = 0; i < iterations; i++) {
        // CPU
        const cr = cpu.current!.runOnce()
        cpuTimes.push(cr.ms)
        setCpuRes((prev) => ({ ...prev, lastMs: cr.ms }))

        // GPU
        const gr = await gpu.current!.runOnce()
        gpuTimes.push(gr.ms)
        setGpuRes((prev) => ({ ...prev, lastMs: gr.ms }))
      }

      const cpuAvg = cpuTimes.reduce((a, b) => a + b, 0) / cpuTimes.length
      const gpuAvg = gpuTimes.reduce((a, b) => a + b, 0) / gpuTimes.length

      setCpuRes({ lastMs: cpuTimes.at(-1) ?? null, avgMs: cpuAvg, sample: cpu.current!.sample(8) })
      setGpuRes({ lastMs: gpuTimes.at(-1) ?? null, avgMs: gpuAvg, sample: await gpu.current!.readBackSample(8) })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Benchmark failed.')
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="px-6 py-5 border-b border-gray-800">
        <h2 className="text-lg font-semibold">Matrix Multiply</h2>
        <p className="text-xs text-gray-500 mt-1">
          Runs CPU (JS) and GPU (WebGPU) back-to-back and shows results side-by-side.
        </p>
      </div>

      <div className="flex-1 overflow-auto px-6 py-6">
        <div className="flex flex-wrap gap-3 items-end">
          <div className="flex flex-col">
            <label className="text-xs text-gray-500 mb-1">Matrix size (N×N)</label>
            <select
              value={n}
              onChange={(e) => setN(parseInt(e.target.value, 10))}
              className="bg-gray-900 border border-gray-800 rounded px-3 py-2 text-sm text-gray-200"
            >
              {[256, 512, 768, 1024, 1536, 2048].map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </div>

          <div className="flex gap-2">
            <button
              disabled={running}
              onClick={() => runBoth(3)}
              className="px-4 py-2 rounded bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-sm font-medium"
            >
              Run both ×3
            </button>
            <button
              disabled={running}
              onClick={() => runBoth(10)}
              className="px-4 py-2 rounded border border-gray-800 hover:border-gray-700 disabled:opacity-50 text-sm font-medium"
            >
              Run both ×10
            </button>
          </div>

          {!canUseWebGPU && (
            <div className="text-xs text-yellow-400 border border-yellow-900/40 bg-yellow-950/30 rounded px-3 py-2">
              WebGPU not available
            </div>
          )}
        </div>

        {error && (
          <div className="mt-5 text-sm text-red-400 border border-red-900/40 bg-red-950/30 rounded p-3">
            {error}
          </div>
        )}

        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-3">
          {/* CPU */}
          <div className="rounded border border-gray-800 bg-gray-900/30 p-4">
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold">JS (CPU)</div>
              <div className="text-xs text-gray-500">typed arrays + tiled</div>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <div className="rounded border border-gray-800 bg-gray-950/30 p-3">
                <div className="text-xs text-gray-500">Last</div>
                <div className="text-xl font-semibold mt-1">{msToPretty(cpuRes.lastMs)}</div>
              </div>
              <div className="rounded border border-gray-800 bg-gray-950/30 p-3">
                <div className="text-xs text-gray-500">Avg</div>
                <div className="text-xl font-semibold mt-1">{msToPretty(cpuRes.avgMs)}</div>
              </div>
            </div>

            <div className="mt-4">
              <div className="text-xs text-gray-500">Result sample</div>
              <div className="text-xs text-gray-400 mt-2 font-mono break-all">
                {cpuRes.sample.length ? cpuRes.sample.map((x) => x.toFixed(3)).join(', ') : '—'}
              </div>
            </div>
          </div>

          {/* GPU */}
          <div className="rounded border border-gray-800 bg-gray-900/30 p-4">
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold">WebGPU (GPU)</div>
              <div className="text-xs text-gray-500">compute shader + tiled</div>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <div className="rounded border border-gray-800 bg-gray-950/30 p-3">
                <div className="text-xs text-gray-500">Last</div>
                <div className="text-xl font-semibold mt-1">{msToPretty(gpuRes.lastMs)}</div>
              </div>
              <div className="rounded border border-gray-800 bg-gray-950/30 p-3">
                <div className="text-xs text-gray-500">Avg</div>
                <div className="text-xl font-semibold mt-1">{msToPretty(gpuRes.avgMs)}</div>
              </div>
            </div>

            <div className="mt-4">
              <div className="text-xs text-gray-500">Result sample</div>
              <div className="text-xs text-gray-400 mt-2 font-mono break-all">
                {gpuRes.sample.length ? gpuRes.sample.map((x) => x.toFixed(3)).join(', ') : '—'}
              </div>
            </div>
          </div>
        </div>

        <div className="mt-6 text-xs text-gray-500 leading-relaxed">
          Tip: jeśli wyniki nadal podobne, zwiększ N (np. 1536–2048). Przy małym N narzut i synchronizacja mogą dominować.
        </div>
      </div>
    </div>
  )
}