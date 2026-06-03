import { useEffect, useRef } from 'react'
import styles from './ParticleWave.module.css'

// Height-based color: deep purple (valley) → magenta → pink → cyan (peak)
const HEIGHT_STOPS = [
  [20,  0,   60 ],  // deep purple — valley
  [140, 0,   160],  // violet
  [220, 40,  160],  // magenta-pink
  [0,   200, 220],  // teal-cyan — peak
]

function lerpColor(a, b, t) {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ]
}

function heightColor(t) {
  // t: 0 = low, 1 = high
  const s = Math.max(0, Math.min(1, t)) * (HEIGHT_STOPS.length - 1)
  const i = Math.min(Math.floor(s), HEIGHT_STOPS.length - 2)
  return lerpColor(HEIGHT_STOPS[i], HEIGHT_STOPS[i + 1], s - i)
}

const COLS  = 70   // horizontal grid points
const ROWS  = 32   // depth grid points
const FOV   = 320  // perspective field of view

function project(gx, gy, gz, cx, cy) {
  const scale = FOV / (FOV + gz)
  return { sx: cx + gx * scale, sy: cy + gy * scale, scale }
}

function waveHeight(col, row, W, time) {
  const xf = col / (COLS - 1)
  const zf = row / (ROWS - 1)
  return (
    Math.sin(xf * 3.5 * Math.PI + time * 0.5) * W * 0.055 +
    Math.sin(xf * 2.0 * Math.PI - zf * 2.5 + time * 0.35) * W * 0.035 +
    Math.sin(zf * 3.0 * Math.PI + time * 0.4) * W * 0.025
  )
}

export default function ParticleWave() {
  const canvasRef = useRef(null)
  const rafRef    = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx    = canvas.getContext('2d')
    let   time   = 0

    function resize() {
      const dpr     = window.devicePixelRatio || 1
      canvas.width  = canvas.offsetWidth  * dpr
      canvas.height = canvas.offsetHeight * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }

    function draw() {
      const W  = canvas.offsetWidth
      const H  = canvas.offsetHeight
      ctx.clearRect(0, 0, W, H)

      // Camera anchor — push the grid toward the bottom of the canvas
      const cx = W * 0.5
      const cy = H * 0.82

      // 3D grid extents
      const gridW  = W * 1.3
      const gridD  = W * 0.75   // z depth
      const gridH  = H * 0.65   // max height above baseline

      // Pre-compute all projected points
      const pts = []
      let minH = Infinity, maxH = -Infinity

      for (let row = 0; row < ROWS; row++) {
        pts[row] = []
        for (let col = 0; col < COLS; col++) {
          const gx = (col / (COLS - 1) - 0.5) * gridW
          const gz = (1 - row / (ROWS - 1)) * gridD   // back rows have high z
          const h  = waveHeight(col, row, W, time)
          const gy = -h                                 // up = negative y in screen space
          if (h < minH) minH = h
          if (h > maxH) maxH = h
          pts[row][col] = { ...project(gx, gy, gz, cx, cy), h }
        }
      }

      const hRange = maxH - minH || 1

      // Draw back-to-front (painter's algorithm)
      for (let row = 0; row < ROWS; row++) {
        // Draw horizontal lines connecting dots along this row
        for (let col = 0; col < COLS - 1; col++) {
          const a = pts[row][col]
          const b = pts[row][col + 1]
          const t = (a.h - minH) / hRange

          const [r, g, bC] = heightColor(t)
          const lineAlpha = 0.15 + 0.2 * t
          ctx.beginPath()
          ctx.moveTo(a.sx, a.sy)
          ctx.lineTo(b.sx, b.sy)
          ctx.strokeStyle = `rgba(${r},${g},${bC},${lineAlpha.toFixed(2)})`
          ctx.lineWidth   = a.scale * 0.8
          ctx.stroke()
        }

        // Draw dots on top of lines
        for (let col = 0; col < COLS; col++) {
          const p = pts[row][col]
          const t = (p.h - minH) / hRange
          const [r, g, bC] = heightColor(t)

          const dotR  = p.scale * (1.4 + t * 1.8)   // peaks have bigger dots
          const alpha = 0.5 + 0.5 * t

          // Soft glow for high points only
          if (t > 0.6) {
            ctx.beginPath()
            ctx.arc(p.sx, p.sy, dotR * 2.8, 0, Math.PI * 2)
            ctx.fillStyle = `rgba(${r},${g},${bC},0.07)`
            ctx.fill()
          }

          // Core dot
          ctx.beginPath()
          ctx.arc(p.sx, p.sy, dotR, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(${r},${g},${bC},${alpha.toFixed(2)})`
          ctx.fill()
        }
      }

      // Floating sparkles above the wave
      for (let s = 0; s < 18; s++) {
        const sf   = s / 18
        const sx   = W * (0.1 + sf * 0.8)
        const sy   = H * (0.05 + Math.sin(sf * 7 + time * 0.3) * 0.12)
        const sr   = 0.8 + Math.sin(sf * 13 + time) * 0.5
        const sa   = 0.3 + 0.3 * Math.sin(sf * 9 + time * 1.5)
        const [r, g, bC] = heightColor(0.7 + sf * 0.3)
        ctx.beginPath()
        ctx.arc(sx, sy, Math.max(0.3, sr), 0, Math.PI * 2)
        ctx.fillStyle = `rgba(${r},${g},${bC},${sa.toFixed(2)})`
        ctx.fill()
      }

      time += 0.01
      rafRef.current = requestAnimationFrame(draw)
    }

    resize()
    window.addEventListener('resize', resize)
    draw()

    return () => {
      cancelAnimationFrame(rafRef.current)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <div className={styles.wrapper}>
      <canvas ref={canvasRef} className={styles.canvas} />
    </div>
  )
}
