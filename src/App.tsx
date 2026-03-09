import { useState } from 'react'
import ParticleBenchmark from './benchmarks/ParticleBenchmark'
import MatMulBenchmark from './benchmarks/MatMulBenchmark'
import ImageBenchmark from './benchmarks/ImageBenchmark'

const TABS = [
  { id: 'particles',  label: 'Particle Simulation' },
  { id: 'matrix',     label: 'Matrix Multiply' },
  { id: 'image',      label: 'Image Processing' },
] as const

type TabId = typeof TABS[number]['id']

export default function App() {
  const [active, setActive] = useState<TabId>('particles')

  return (
    <div className="h-screen bg-gray-950 text-gray-100 flex flex-col overflow-hidden">
      <header className="border-b border-gray-800 px-6 py-4 flex items-center justify-between shrink-0">
        <div>
          <h1 className="text-xl font-bold tracking-tight">WebGPU Bench</h1>
          <p className="text-xs text-gray-500 mt-0.5">GPU vs CPU benchmarks in the browser</p>
        </div>
        <span className="text-xs bg-gray-800 text-gray-400 px-2 py-1 rounded">v0.1.0</span>
      </header>

      {/* Tabs */}
      <nav className="border-b border-gray-800 px-6 shrink-0">
        <div className="flex gap-1">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActive(tab.id)}
              className={[
                'px-4 py-3 text-sm font-medium border-b-2 transition-colors',
                active === tab.id
                  ? 'border-indigo-500 text-indigo-400'
                  : 'border-transparent text-gray-500 hover:text-gray-300',
              ].join(' ')}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </nav>

      {/* Tab content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {active === 'particles'  && <ParticleBenchmark />}
        {active === 'matrix' && <MatMulBenchmark />}
        {active === 'image' && <ImageBenchmark />}
      </main>
    </div>
  )
}
