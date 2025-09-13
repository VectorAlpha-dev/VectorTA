import React, { useMemo, useState } from 'react'
import { HeatmapChart, BacktestResult } from './components/HeatmapChart'
import { ChartPanel } from './components/ChartPanel'

type Backend = 'cpu' | 'gpu'

interface OptimizeResponse {
  ok: boolean
  data?: {
    meta: { fast_periods: number[]; slow_periods: number[]; rows: number; cols: number; metrics: string[] }
    values: number[]
  }
  error?: string
}

export const App: React.FC = () => {
  const [backend, setBackend] = useState<Backend>('cpu')
  const [len, setLen] = useState(50000)
  const [fast, setFast] = useState('5,240,5')
  const [slow, setSlow] = useState('20,240,5')
  const [offset, setOffset] = useState(0.85)
  const [sigma, setSigma] = useState(6.0)
  const [fastType, setFastType] = useState('alma')
  const [slowType, setSlowType] = useState('alma')
  const [commission, setCommission] = useState(0.0)
  const [status, setStatus] = useState('')
  const [results, setResults] = useState<BacktestResult[]>([])
  const [selected, setSelected] = useState<{ fast: number; slow: number } | null>(null)

  const series = useMemo(() => {
    // Generate the same synthetic series used by the backend
    const T = len
    const s = new Array(T).fill(NaN)
    for (let i = 3; i < T; i++) { const x = i; s[i] = Math.sin(x * 0.001) + 0.0001 * x }
    return s
  }, [len])

  const run = async () => {
    setStatus('Running...')
    const parseRange = (s: string) => { const [a, b, c] = s.split(',').map(x => parseFloat(x)); return [a|0, b|0, c|0] as [number,number,number] }
    const body = {
      backend,
      synthetic_len: len,
      series: null,
      fast_period: parseRange(fast),
      slow_period: parseRange(slow),
      fast_type: fastType,
      slow_type: slowType,
      offset,
      sigma,
      commission,
      metrics: 5,
    }
    const resp = await fetch('/api/optimize', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    const json: OptimizeResponse = await resp.json()
    if (!json.ok || !json.data) { setStatus('Error: ' + (json.error || 'unknown')); return }
    const { meta, values } = json.data
    const rows = meta.rows, cols = meta.cols, M = 5
    const out: BacktestResult[] = []
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const base = (i * cols + j) * M
        out.push({
          fastPeriod: meta.fast_periods[i],
          slowPeriod: meta.slow_periods[j],
          totalReturn: values[base + 0],
          trades: values[base + 1],
          maxDrawdown: values[base + 2],
          meanRet: values[base + 3],
          stdRet: values[base + 4],
        })
      }
    }
    setResults(out)
    setStatus(`Rows=${rows} Cols=${cols}`)
  }

  return (
    <div style={{ padding: 16, fontFamily: 'system-ui, Arial, sans-serif' }}>
      <h1>ALMA Double-Crossover Optimizer Demo</h1>
      <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, minWidth: 300 }}>
          <label>Backend
            <select value={backend} onChange={e => setBackend(e.target.value as Backend)}>
              <option value="cpu">CPU (fp64)</option>
              <option value="gpu">GPU (fp32)</option>
            </select>
          </label>
          <label>Fast MA Type
            <select value={fastType} onChange={e => setFastType(e.target.value)}>
              <option value="alma">ALMA</option>
              <option value="ema">EMA</option>
              <option value="sma">SMA</option>
              <option value="wma">WMA</option>
              <option value="zlema">ZLEMA</option>
              <option value="dema">DEMA</option>
              <option value="trima">TRIMA</option>
              <option value="hma">HMA</option>
              <option value="smma">SMMA</option>
              <option value="dma">DMA</option>
              <option value="swma">SWMA</option>
              <option value="sqwma">SQWMA</option>
              <option value="srwma">SRWMA</option>
              <option value="supersmoother">SuperSmoother</option>
              <option value="supersmoother_3_pole">3-Pole SuperSmoother</option>
              <option value="sinwma">SinWMA</option>
              <option value="pwma">PWMA</option>
              <option value="vwma">VWMA</option>
              <option value="wilders">Wilders</option>
              <option value="fwma">FWMA</option>
              <option value="linreg">LinReg</option>
              <option value="tema">TEMA</option>
            </select>
          </label>
          <label>Slow MA Type
            <select value={slowType} onChange={e => setSlowType(e.target.value)}>
              <option value="alma">ALMA</option>
              <option value="ema">EMA</option>
              <option value="sma">SMA</option>
              <option value="wma">WMA</option>
              <option value="zlema">ZLEMA</option>
              <option value="dema">DEMA</option>
              <option value="trima">TRIMA</option>
              <option value="hma">HMA</option>
              <option value="smma">SMMA</option>
              <option value="dma">DMA</option>
              <option value="swma">SWMA</option>
              <option value="sqwma">SQWMA</option>
              <option value="srwma">SRWMA</option>
              <option value="supersmoother">SuperSmoother</option>
              <option value="supersmoother_3_pole">3-Pole SuperSmoother</option>
              <option value="sinwma">SinWMA</option>
              <option value="pwma">PWMA</option>
              <option value="vwma">VWMA</option>
              <option value="wilders">Wilders</option>
              <option value="fwma">FWMA</option>
              <option value="linreg">LinReg</option>
              <option value="tema">TEMA</option>
            </select>
          </label>
          <label>Series length <input type="number" value={len} onChange={e => setLen(parseInt(e.target.value))} /></label>
          <label>Fast period (start,end,step) <input value={fast} onChange={e => setFast(e.target.value)} /></label>
          <label>Slow period (start,end,step) <input value={slow} onChange={e => setSlow(e.target.value)} /></label>
          { (fastType === 'alma' || slowType === 'alma') && (
            <>
              <label>ALMA Offset <input type="number" step={0.01} value={offset} onChange={e => setOffset(parseFloat(e.target.value))} /></label>
              <label>ALMA Sigma <input type="number" step={0.1} value={sigma} onChange={e => setSigma(parseFloat(e.target.value))} /></label>
            </>
          )}
          <label>Commission <input type="number" step={0.0001} value={commission} onChange={e => setCommission(parseFloat(e.target.value))} /></label>
          <button onClick={run}>Run</button>
          <div style={{ color: '#666' }}>{status}</div>
        </div>

        <div style={{ flex: 1 }}>
          <HeatmapChart results={results} onCellClick={(f, s) => setSelected({ fast: f, slow: s })} selectedParams={selected} />
        </div>

        <div style={{ flex: 1 }}>
          <ChartPanel
            series={series}
            fast={selected?.fast ?? null}
            slow={selected?.slow ?? null}
            offset={offset}
            sigma={sigma}
            commission={commission}
            fastType={fastType}
            slowType={slowType}
          />
        </div>
      </div>
    </div>
  )
}
