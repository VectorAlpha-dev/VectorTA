import React, { useEffect, useMemo, useRef } from 'react'
import { createChart, ColorType, IChartApi, ISeriesApi } from 'lightweight-charts'

export interface ChartPanelProps {
  series: number[]
  fast: number | null
  slow: number | null
  offset: number
  sigma: number
  commission: number
  fastType?: string
  slowType?: string
}

function computeALMA(data: number[], period: number, offset: number, sigma: number): number[] {
  const out = new Array(data.length).fill(NaN)
  const m = offset * (period - 1)
  const s = period / sigma
  const s2 = 2 * s * s
  const w = new Array(period)
  let norm = 0
  for (let i = 0; i < period; i++) { const d = i - m; const wi = Math.exp(-(d * d) / s2); w[i] = wi; norm += wi }
  const inv = 1.0 / norm
  let first = 0; while (first < data.length && !isFinite(data[first])) first++
  const warm = first + period - 1
  for (let t = warm; t < data.length; t++) {
    let sum = 0; let start = t - period + 1; let ok = true
    for (let k = 0; k < period; k++) { const v = data[start + k]; if (!isFinite(v)) { ok = false; break } sum += v * w[k] }
    out[t] = ok ? sum * inv : NaN
  }
  return out
}

export const ChartPanel: React.FC<ChartPanelProps> = ({ series, fast, slow, offset, sigma, commission, fastType = 'alma', slowType = 'alma' }) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const priceRef = useRef<ISeriesApi<'Line'> | null>(null)
  const fastRef = useRef<ISeriesApi<'Line'> | null>(null)
  const slowRef = useRef<ISeriesApi<'Line'> | null>(null)
  const wasmRef = useRef<any>(null)

  useEffect(() => {
    if (!containerRef.current || chartRef.current) return
    const chart = createChart(containerRef.current, {
      autoSize: true,
      height: 360,
      layout: { background: { type: ColorType.Solid, color: '#ffffff' }, textColor: '#333' },
      rightPriceScale: { borderColor: '#ccc' },
      timeScale: { borderColor: '#ccc', timeVisible: true }
    })
    chartRef.current = chart
    priceRef.current = chart.addLineSeries({ color: '#888888', lineWidth: 1 })
    fastRef.current = chart.addLineSeries({ color: '#3b82f6', lineWidth: 2 })
    slowRef.current = chart.addLineSeries({ color: '#8b5cf6', lineWidth: 2 })
    return () => { chart.remove(); chartRef.current = null; priceRef.current = null; fastRef.current = null; slowRef.current = null }
  }, [])

  // Try to load WASM bindings from /wasm (served by backend)
  useEffect(() => {
    (async () => {
      const candidates = [
        { js: '/wasm/my_project.js', wasm: '/wasm/my_project_bg.wasm' },
        { js: '/wasm/ta_indicators.js', wasm: '/wasm/ta_indicators_bg.wasm' },
      ]
      for (const c of candidates) {
        try {
          const mod: any = await import(/* @vite-ignore */ c.js)
          if (mod && typeof mod.default === 'function') {
            await mod.default(c.wasm)
          }
          wasmRef.current = mod
          console.log('Loaded WASM from', c.js)
          break
        } catch (e) {
          // Try next
        }
      }
    })()
  }, [])

  const dataPoints = useMemo(() => {
    const now = Math.floor(Date.now() / 1000)
    const priceData: { time: number, value: number }[] = []
    for (let i = 0; i < series.length; i++) {
      const t = now - (series.length - i) * 60
      if (isFinite(series[i])) priceData.push({ time: t, value: series[i] })
    }
    return priceData
  }, [series])

  useEffect(() => {
    if (!priceRef.current) return
    priceRef.current.setData(dataPoints)
    chartRef.current?.timeScale().fitContent()
  }, [dataPoints])

  useEffect(() => {
    if (!fastRef.current || !slowRef.current || fast == null || slow == null) return
    const wasm = wasmRef.current
    const callMA = (typ: string, period: number): number[] => {
      const t = typ.toLowerCase()
      try {
        if (wasm) {
          if (t === 'alma' && typeof wasm.alma_js === 'function') return Array.from(wasm.alma_js(series, period, offset, sigma) as Float64Array)
          const fnName = t + '_js'
          if (typeof wasm[fnName] === 'function') {
            // Special case: FRAMA needs high/low/close; derive synthetic high/low if we only have close
            if (t === 'frama') {
              const close = series
              const high: number[] = []; const low: number[] = []
              for (let i=0;i<close.length;i++){
                const c = close[i]
                if (Number.isFinite(c)) {
                  const amp = 0.002 * ((i % 100)/100 + 0.5)
                  high.push(c*(1+amp)); low.push(c*(1-amp))
                } else { high.push(NaN); low.push(NaN) }
              }
              const sc = 300, fc = 1
              return Array.from(wasm.frama_js(high, low, close, period, sc, fc) as Float64Array)
            }
            return Array.from(wasm[fnName](series, period) as Float64Array)
          }
        }
      } catch (e) { console.warn('WASM call failed for', typ, e) }
      // Fallbacks: only ALMA JS implemented here; others would need JS versions if no WASM
      if (t === 'alma') return computeALMA(series, period, offset, sigma)
      return new Array(series.length).fill(NaN)
    }
    const f = callMA(fastType, fast)
    const s = callMA(slowType, slow)
    const now = Math.floor(Date.now() / 1000)
    const fData: { time: number, value: number }[] = []
    const sData: { time: number, value: number }[] = []
    for (let i = 0; i < series.length; i++) {
      const t = now - (series.length - i) * 60
      if (isFinite(f[i])) fData.push({ time: t, value: f[i] })
      if (isFinite(s[i])) sData.push({ time: t, value: s[i] })
    }
    fastRef.current.setData(fData)
    slowRef.current.setData(sData)
  }, [series, fast, slow, offset, sigma, commission])

  return <div ref={containerRef} style={{ width: 640, height: 360 }} />
}
