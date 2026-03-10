# WebGPU Bench

A browser-based benchmark suite comparing GPU (WebGPU) vs CPU (JavaScript) performance across three workloads.

**Live demo:** https://sylwia-lask.github.io/webgpu-bench/

> Requires a browser with WebGPU support — Chrome 113+, Edge 113+, or Chrome Canary. Firefox and Safari have limited support.

---

## Benchmarks

### Particle Simulation

Simulates N-body particles bouncing off canvas boundaries and renders them in real time.

| | CPU | GPU |
|---|---|---|
| **Update** | JavaScript (typed arrays) | WGSL compute shader |
| **Render** | Canvas 2D API | Instanced rendering (2×2 px quads) |
| **Metric** | FPS | FPS |

The GPU pipeline runs physics in a compute pass and renders with a vertex/fragment shader pair. Particle count is configurable from 1 000 to 500 000.

---

### Matrix Multiplication

Multiplies two square N×N matrices (C = A × B) and compares execution time.

| | CPU | GPU |
|---|---|---|
| **Algorithm** | Tiled loop, typed arrays | Tiled 16×16 WGSL compute shader with shared memory |
| **Metric** | ms (last + avg over N runs) | ms (last + avg over N runs) |

The GPU shader uses workgroup shared memory to reduce global memory bandwidth. Matrix size is configurable from 256 to 2048. Both implementations use the same tiling strategy for a fair comparison.

---

### Image Processing

Runs a multi-pass image processing pipeline on a user-supplied or default image.

Passes (each individually toggleable):

1. **Grayscale** — luminance-weighted RGB → gray (BT.709 coefficients)
2. **Blur** — separable 5-tap Gaussian (horizontal + vertical pass)
3. **Edge Detect** — Sobel operator (X + Y gradient magnitude)
4. **Threshold** — binary threshold at 0.35 luminance

| | CPU | GPU |
|---|---|---|
| **Implementation** | JavaScript on `ImageData` | WGSL compute shaders with ping-pong textures |
| **Metric** | ms (last + avg) | ms (last + avg) |

Processing resolution is configurable from 512×512 to 2048×2048.

---

## Tech Stack

- **React 19** + **TypeScript**
- **Vite 6** for bundling
- **Tailwind CSS v4**
- **WebGPU API** (`@webgpu/types`)

---

## Running locally

```bash
npm install
npm run dev
```

Then open http://localhost:5173.

## Building

```bash
npm run build
```

Output goes to `dist/`.

---

## Deployment

Deployed automatically to GitHub Pages on every push to `main` via GitHub Actions (`.github/workflows/deploy.yml`).

To enable in your own fork: go to **Settings → Pages → Source** and set it to **GitHub Actions**.
