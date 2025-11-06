import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react'

export interface BacktestResult {
  fastPeriod: number
  slowPeriod: number
  totalReturn: number
  trades: number
  maxDrawdown: number
  meanRet?: number
  stdRet?: number
  // Optional display axes values when using generic axes
  displayX?: number
  displayY?: number
}

interface HeatmapChartProps {
  results: BacktestResult[]
  metric?: keyof BacktestResult
  onCellClick?: (fastPeriod: number, slowPeriod: number) => void
  selectedParams?: { fast: number; slow: number } | null
  xLabel?: string
  yLabel?: string
}

export function HeatmapChart({ results, metric = 'totalReturn', onCellClick, selectedParams, xLabel = 'Fast MA Period', yLabel = 'Slow MA Period' }: HeatmapChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hoveredCell, setHoveredCell] = useState<{ x: number; y: number; value: number } | null>(null)

  const fastPeriods = useMemo(
    () => Array.from(new Set(results.map(r => r.displayX ?? r.fastPeriod))).sort((a, b) => a - b),
    [results]
  )
  const slowPeriods = useMemo(
    () => Array.from(new Set(results.map(r => r.displayY ?? r.slowPeriod))).sort((a, b) => a - b),
    [results]
  )

  const createMatrix = useCallback((): number[][] => {
    const rows = slowPeriods.length
    const cols = fastPeriods.length
    const matrix: number[][] = Array.from({ length: rows }, () => Array(cols).fill(NaN))
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const r = results.find(x => (x.displayX ?? x.fastPeriod) === fastPeriods[j] && (x.displayY ?? x.slowPeriod) === slowPeriods[i])
        matrix[i][j] = r ? (r[metric] as number) : NaN
      }
    }
    return matrix
  }, [results, metric, fastPeriods, slowPeriods])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const margin = { top: 40, right: 100, bottom: 60, left: 60 }
    const chartWidth = canvas.width - margin.left - margin.right
    const chartHeight = canvas.height - margin.top - margin.bottom
    const cols = fastPeriods.length
    const rows = slowPeriods.length
    const cellW = chartWidth / cols
    const cellH = chartHeight / rows

    const matrix = createMatrix()
    let minV = Infinity, maxV = -Infinity
    for (const row of matrix) for (const v of row) if (isFinite(v)) { if (v < minV) minV = v; if (v > maxV) maxV = v }

    const color = (v: number) => {
      if (!isFinite(v)) return '#2a2a2a'
      if (metric === 'maxDrawdown') {
        const n = (v - minV) / (maxV - minV + 1e-9)
        const r = Math.floor(255 * n), g = Math.floor(255 * (1 - n))
        return `rgb(${r},${g},0)`
      } else {
        if (minV < 0 && maxV > 0) {
          if (v < 0) { const t = v / (minV || -1); const r = Math.floor(100 + 155 * t); return `rgb(${r},0,0)` }
          const t = v / (maxV || 1); const g = Math.floor(100 + 155 * t); return `rgb(0,${g},0)`
        } else {
          const n = (v - minV) / (maxV - minV + 1e-9)
          const r = Math.floor(255 * (1 - n)), g = Math.floor(255 * n)
          return `rgb(${r},${g},0)`
        }
      }
    }

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const x = margin.left + j * cellW
        const y = margin.top + i * cellH
        const v = matrix[i][j]
        ctx.fillStyle = color(v)
        ctx.fillRect(x, y, cellW - 1, cellH - 1)
        if (selectedParams && fastPeriods[j] === selectedParams.fast && slowPeriods[i] === selectedParams.slow) {
          ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; ctx.strokeRect(x, y, cellW - 1, cellH - 1)
        }
      }
    }

    // Axes
    ctx.fillStyle = '#888'; ctx.font = '12px sans-serif'
    ctx.textAlign = 'center'
    for (let j = 0; j < cols; j++) {
      if (j % Math.ceil(cols / 10) === 0) ctx.fillText(String(fastPeriods[j]), margin.left + j * cellW + cellW / 2, canvas.height - margin.bottom + 20)
    }
    ctx.textAlign = 'right'
    for (let i = 0; i < rows; i++) {
      if (i % Math.ceil(rows / 10) === 0) ctx.fillText(String(slowPeriods[i]), margin.left - 10, margin.top + i * cellH + cellH / 2)
    }
    ctx.fillStyle = '#aaa'; ctx.font = '14px sans-serif'; ctx.textAlign = 'center'
    ctx.fillText(xLabel, canvas.width / 2, canvas.height - 10)
    ctx.save(); ctx.translate(15, canvas.height / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(yLabel, 0, 0); ctx.restore()
  }, [createMatrix, fastPeriods, slowPeriods, metric, selectedParams])

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return
    const parent = canvas.parentElement
    if (parent) { canvas.width = parent.clientWidth; canvas.height = Math.min(parent.clientWidth * 0.75, 600) }
    draw()
  }, [draw])

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current; if (!canvas) return
    const rect = canvas.getBoundingClientRect(); const x = e.clientX - rect.left; const y = e.clientY - rect.top
    const margin = { top: 40, right: 100, bottom: 60, left: 60 }
    const chartW = canvas.width - margin.left - margin.right
    const chartH = canvas.height - margin.top - margin.bottom
    const cols = fastPeriods.length, rows = slowPeriods.length
    const cellW = chartW / cols, cellH = chartH / rows
    if (x >= margin.left && x <= canvas.width - margin.right && y >= margin.top && y <= canvas.height - margin.bottom) {
      const col = Math.floor((x - margin.left) / cellW); const row = Math.floor((y - margin.top) / cellH)
      const r = results.find(v => (v.displayX ?? v.fastPeriod) === fastPeriods[col] && (v.displayY ?? v.slowPeriod) === slowPeriods[row])
      if (r) setHoveredCell({ x: fastPeriods[col], y: slowPeriods[row], value: (r[metric] as number) })
      else setHoveredCell(null)
    } else { setHoveredCell(null) }
  }

  const onClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onCellClick) return
    const canvas = canvasRef.current; if (!canvas) return
    const rect = canvas.getBoundingClientRect(); const x = e.clientX - rect.left; const y = e.clientY - rect.top
    const margin = { top: 40, right: 100, bottom: 60, left: 60 }
    const chartW = canvas.width - margin.left - margin.right
    const chartH = canvas.height - margin.top - margin.bottom
    const cols = fastPeriods.length, rows = slowPeriods.length
    const cellW = chartW / cols, cellH = chartH / rows
    if (x >= margin.left && x <= canvas.width - margin.right && y >= margin.top && y <= canvas.height - margin.bottom) {
      const col = Math.floor((x - margin.left) / cellW); const row = Math.floor((y - margin.top) / cellH)
      // On click, use actual fast/slow for downstream overlay
      const r = results.find(v => (v.displayX ?? v.fastPeriod) === fastPeriods[col] && (v.displayY ?? v.slowPeriod) === slowPeriods[row])
      if (r) onCellClick(r.fastPeriod, r.slowPeriod)
    }
  }

  return (
    <div className="heatmap">
      <canvas ref={canvasRef} onMouseMove={onMouseMove} onClick={onClick} />
      {hoveredCell && (
        <div style={{ fontSize: 12, color: '#666', marginTop: 6 }}>
          Fast: {hoveredCell.x}, Slow: {hoveredCell.y}, {String(metric)}: {hoveredCell.value.toFixed(4)}
        </div>
      )}
    </div>
  )
}
