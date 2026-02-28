import { forwardRef, useEffect, useRef } from 'react'
import { initWebGPU } from '../gpu/renderer'

const BenchCanvas = forwardRef<HTMLCanvasElement>((_, ref) => {
  const internalRef = useRef<HTMLCanvasElement>(null)
  const canvasRef = (ref as React.RefObject<HTMLCanvasElement>) ?? internalRef

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    let cleanup: (() => void) | undefined

    initWebGPU(canvas)
      .then((destroy) => { cleanup = destroy })
      .catch((err: unknown) => console.error('WebGPU init failed:', err))

    return () => cleanup?.()
  }, [canvasRef])

  return (
    <div className="relative w-full max-w-2xl aspect-video bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        width={1280}
        height={720}
      />
    </div>
  )
})

BenchCanvas.displayName = 'BenchCanvas'

export default BenchCanvas
