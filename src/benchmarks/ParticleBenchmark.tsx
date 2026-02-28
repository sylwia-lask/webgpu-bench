import { useEffect, useMemo, useRef, useState } from 'react'
import { GpuParticles } from './particles/GpuParticles'
import { JsParticles } from './particles/JsParticles'

function setupCanvasDpr(canvas: HTMLCanvasElement, cssW: number, cssH: number) {
  const dpr = window.devicePixelRatio || 1
  canvas.style.width = `${cssW}px`
  canvas.style.height = `${cssH}px`
  canvas.width = Math.floor(cssW * dpr)
  canvas.height = Math.floor(cssH * dpr)
  return dpr
}

export default function ParticleBenchmark() {
  const cpuCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const gpuCanvasRef = useRef<HTMLCanvasElement | null>(null)

  const cpuEngineRef = useRef<JsParticles | null>(null)
  const gpuEngineRef = useRef<GpuParticles | null>(null)

  const [count, setCount] = useState(50_000)
  const [running, setRunning] = useState(false)

  const [cpuFps, setCpuFps] = useState<number | null>(null)
  const [gpuFps, setGpuFps] = useState<number | null>(null)
  const [gpuErr, setGpuErr] = useState<string>('')

  const canUseWebGPU = useMemo(() => !!navigator.gpu, [])

  useEffect(() => {
    // init canvases sizes once
    const cpuCanvas = cpuCanvasRef.current
    const gpuCanvas = gpuCanvasRef.current
    if (cpuCanvas) setupCanvasDpr(cpuCanvas, 520, 320)
    if (gpuCanvas) setupCanvasDpr(gpuCanvas, 520, 320)

    return () => {
      cpuEngineRef.current?.destroy()
      gpuEngineRef.current?.destroy()
      cpuEngineRef.current = null
      gpuEngineRef.current = null
    }
  }, [])

  async function startBoth() {
    if (running) return
    setRunning(true)
    setGpuErr('')

    const cpuCanvas = cpuCanvasRef.current
    const gpuCanvas = gpuCanvasRef.current
    if (!cpuCanvas || !gpuCanvas) return

    // (re)create engines fresh
    cpuEngineRef.current?.destroy()
    gpuEngineRef.current?.destroy()

    cpuEngineRef.current = new JsParticles(cpuCanvas, setCpuFps)

    if (!canUseWebGPU) {
      setGpuErr('WebGPU not available in this browser.')
      cpuEngineRef.current.start(count)
      return
    }

    gpuEngineRef.current = new GpuParticles(gpuCanvas, setGpuFps, (msg) => setGpuErr(msg || ''))

    // Start CPU immediately, GPU async init and start
    cpuEngineRef.current.start(count)
    await gpuEngineRef.current.start(count)
  }

  function stopBoth() {
    cpuEngineRef.current?.stop()
    gpuEngineRef.current?.stop()
    setRunning(false)
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="px-6 py-5 border-b border-gray-800">
        <h2 className="text-lg font-semibold">Particle Simulation</h2>
        <p className="text-xs text-gray-500 mt-1">
          CPU (Canvas2D) vs WebGPU (compute + instanced render). Same canvas size (DPR-aware).
        </p>
      </div>

      <div className="flex-1 overflow-auto px-6 py-6">
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex flex-col">
            <label className="text-xs text-gray-500 mb-1">Particles</label>
            <select
              value={count}
              onChange={(e) => setCount(parseInt(e.target.value, 10))}
              className="bg-gray-900 border border-gray-800 rounded px-3 py-2 text-sm text-gray-200"
              disabled={running}
            >
              {[1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 500_000].map((v) => (
                <option key={v} value={v}>{v.toLocaleString()}</option>
              ))}
            </select>
          </div>

          <div className="flex gap-2">
            <button
              onClick={startBoth}
              disabled={running}
              className="px-4 py-2 rounded bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-sm font-medium"
            >
              Run both
            </button>
            <button
              onClick={stopBoth}
              disabled={!running}
              className="px-4 py-2 rounded border border-gray-800 hover:border-gray-700 disabled:opacity-50 text-sm font-medium"
            >
              Stop
            </button>
          </div>

          {!canUseWebGPU && (
            <div className="text-xs text-yellow-400 border border-yellow-900/40 bg-yellow-950/30 rounded px-3 py-2">
              WebGPU not available
            </div>
          )}
        </div>

        {gpuErr && (
          <div className="mt-4 text-sm text-red-400 border border-red-900/40 bg-red-950/30 rounded p-3">
            {gpuErr}
          </div>
        )}

        <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="rounded border border-gray-800 bg-gray-900/30 p-4">
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold">JS (CPU) — Canvas2D</div>
              <div className="text-xs text-gray-500">FPS: {cpuFps ?? '—'}</div>
            </div>
            <div className="mt-3 rounded border border-gray-800 overflow-hidden">
              <canvas ref={cpuCanvasRef} />
            </div>
          </div>

          <div className="rounded border border-gray-800 bg-gray-900/30 p-4">
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold">WebGPU (GPU) — compute + instancing</div>
              <div className="text-xs text-gray-500">FPS: {gpuFps ?? '—'}</div>
            </div>
            <div className="mt-3 rounded border border-gray-800 overflow-hidden">
              <canvas ref={gpuCanvasRef} />
            </div>
          </div>
        </div>

        <div className="mt-6 text-xs text-gray-500 leading-relaxed">
          Tip: jeśli wyniki są “dziwnie podobne”, sprawdź czy oba canvasy mają identyczny <code>width/height</code> (po DPR),
          i zwiększ particles aż CPU zacznie spadać.
        </div>
      </div>
    </div>
  )
}