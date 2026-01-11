import React, { useMemo, useState } from 'react'
import { HeatmapChart, BacktestResult } from './components/HeatmapChart'
import { ChartPanel } from './components/ChartPanel'

type Backend = 'cpu' | 'gpu'

interface OptimizeResponse {
  ok: boolean
  data?: {
    meta: { fast_periods: number[]; slow_periods: number[]; rows: number; cols: number; metrics: string[]; axes: { name: string; values: number[] }[] }
    values: number[]
    layers: number
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
  const [metric, setMetric] = useState<'totalReturn'|'trades'|'maxDrawdown'|'meanRet'|'stdRet'>('totalReturn')
  const [commission, setCommission] = useState(0.0)
  const [axes, setAxes] = useState<{ name: string; values: number[] }[]>([])
  const [slicers, setSlicers] = useState<number[]>([])
  const [status, setStatus] = useState('')
  const [results, setResults] = useState<BacktestResult[]>([])
  const [selected, setSelected] = useState<{ fast: number; slow: number } | null>(null)

  const series = useMemo(() => {
    
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
    const ax = meta.axes || []
    setAxes(ax)
    setSlicers(new Array(ax.length).fill(0))
    const rows = meta.rows, cols = meta.cols, M = 5
    const out: BacktestResult[] = []
    
    const extraAxes = (meta.axes || []).filter(a => a.name !== 'fast_period' && a.name !== 'slow_period')
    const extraLens = extraAxes.map(a => a.values.length)
    
    const layerIndex = extraLens.length === 0 ? 0 : (() => {
      const idxs = (meta.axes || []).map((a, i) => ({ name: a.name, idx: slicers[i] || 0 }))
      const extras = idxs.filter(x => x.name !== 'fast_period' && x.name !== 'slow_period').map(x => x.idx)
      
      let mul = 1, idx = 0
      for (let k = extraLens.length - 1; k >= 0; k--) { idx += (extras[k] || 0) * mul; mul *= extraLens[k] }
      return idx
    })()
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const base = (((layerIndex * rows + i) * cols + j) * M)
        const fastVal = meta.fast_periods[i]
        const slowVal = meta.slow_periods[j]
        out.push({
          fastPeriod: fastVal,
          slowPeriod: slowVal,
          displayX: slowVal,
          displayY: fastVal,
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
              <option value="buff_averages">Buff Averages</option>
              <option value="cwma">CWMA</option>
              <option value="edcf">EDCF</option>
              <option value="zlema">ZLEMA</option>
              <option value="dema">DEMA</option>
              <option value="trima">TRIMA</option>
              <option value="ehma">EHMA</option>
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
              <option value="ma">MA (dispatcher)</option>
              <option value="ma_stream">MA Stream (dispatcher)</option>
              <option value="tema">TEMA</option>
              <option value="jma">JMA</option>
              <option value="kama">KAMA</option>
              <option value="ehlers_kama">Ehlers KAMA</option>
              <option value="epma">EPMA</option>
              <option value="gaussian">Gaussian</option>
              <option value="highpass">HighPass</option>
              <option value="highpass_2_pole">HighPass 2-Pole</option>
              <option value="hwma">HWMA</option>
              <option value="jsa">JSA</option>
              <option value="nama">NAMA</option>
              <option value="nma">NMA</option>
              <option value="mwdx">MWDX</option>
              <option value="reflex">Reflex</option>
              <option value="sama">SAMA</option>
              <option value="tilson">Tilson (T3)</option>
              <option value="tradjema">TrAdjEMA</option>
              <option value="trendflex">TrendFlex</option>
              <option value="uma">UMA</option>
              <option value="vama">VAMA</option>
              <option value="volume_adjusted_ma">Volume Adjusted MA</option>
              <option value="vpwma">VPWMA</option>
              <option value="ehlers_ecema">Ehlers ECEMA</option>
              <option value="ehlers_itrend">Ehlers ITrend</option>
              <option value="ehlers_pma">Ehlers PMA</option>
              <option value="frama">FRAMA</option>
              <option value="vwap">VWAP</option>
            </select>
          </label>
          <label>Slow MA Type
            <select value={slowType} onChange={e => setSlowType(e.target.value)}>
              <option value="alma">ALMA</option>
              <option value="ema">EMA</option>
              <option value="sma">SMA</option>
              <option value="wma">WMA</option>
              <option value="buff_averages">Buff Averages</option>
              <option value="cwma">CWMA</option>
              <option value="edcf">EDCF</option>
              <option value="zlema">ZLEMA</option>
              <option value="dema">DEMA</option>
              <option value="trima">TRIMA</option>
              <option value="ehma">EHMA</option>
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
              <option value="ma">MA (dispatcher)</option>
              <option value="ma_stream">MA Stream (dispatcher)</option>
              <option value="tema">TEMA</option>
              <option value="jma">JMA</option>
              <option value="kama">KAMA</option>
              <option value="ehlers_kama">Ehlers KAMA</option>
              <option value="epma">EPMA</option>
              <option value="gaussian">Gaussian</option>
              <option value="highpass">HighPass</option>
              <option value="highpass_2_pole">HighPass 2-Pole</option>
              <option value="hwma">HWMA</option>
              <option value="jsa">JSA</option>
              <option value="nama">NAMA</option>
              <option value="nma">NMA</option>
              <option value="mwdx">MWDX</option>
              <option value="reflex">Reflex</option>
              <option value="sama">SAMA</option>
              <option value="tilson">Tilson (T3)</option>
              <option value="tradjema">TrAdjEMA</option>
              <option value="trendflex">TrendFlex</option>
              <option value="uma">UMA</option>
              <option value="vama">VAMA</option>
              <option value="volume_adjusted_ma">Volume Adjusted MA</option>
              <option value="vpwma">VPWMA</option>
              <option value="ehlers_ecema">Ehlers ECEMA</option>
              <option value="ehlers_itrend">Ehlers ITrend</option>
              <option value="ehlers_pma">Ehlers PMA</option>
              <option value="frama">FRAMA</option>
              <option value="vwap">VWAP</option>
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
          <label>Metric
            <select value={metric} onChange={e => setMetric(e.target.value as any)}>
              <option value="totalReturn">Total Return</option>
              <option value="trades">Trades</option>
              <option value="maxDrawdown">Max Drawdown</option>
              <option value="meanRet">Mean Return</option>
              <option value="stdRet">Std. Return</option>
            </select>
          </label>
          <button onClick={run}>Run</button>
          <div style={{ color: '#666' }}>{status}</div>
        </div>

        <div style={{ flex: 1 }}>
          {/* Slicers for extra axes */}
          {(axes.length > 2) && (
            <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 8, flexWrap: 'wrap' }}>
              {axes.map((a, idx) => (a.name === 'fast_period' || a.name === 'slow_period') ? null : (
                <label key={idx}>{a.name}
                  <select value={slicers[idx] || 0} onChange={e => {
                    const next = slicers.slice(); next[idx] = parseInt(e.target.value); setSlicers(next)
                  }}>
                    {a.values.map((v, vi) => <option key={vi} value={vi}>{v}</option>)}
                  </select>
                </label>
              ))}
            </div>
          )}
          <HeatmapChart results={results} metric={metric} xLabel={'slow_period'} yLabel={'fast_period'} onCellClick={(f, s) => setSelected({ fast: f, slow: s })} selectedParams={selected} />
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
          {/* Top-10 by selected metric */}
          {results.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontWeight: 600, marginBottom: 6 }}>Top 10 by {metric}</div>
              <div style={{ maxHeight: 240, overflow: 'auto', border: '1px solid #eee' }}>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      <th style={{ textAlign: 'left', padding: 4 }}>Fast</th>
                      <th style={{ textAlign: 'left', padding: 4 }}>Slow</th>
                      <th style={{ textAlign: 'left', padding: 4 }}>{metric}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {([...results]
                      .sort((a, b) => {
                        const ka = a[metric] as number; const kb = b[metric] as number
                        const asc = metric === 'maxDrawdown'
                        return asc ? (ka - kb) : (kb - ka)
                      })
                      .slice(0, 10))
                      .map((r, idx) => (
                        <tr key={idx}>
                          <td style={{ padding: 4 }}>{r.fastPeriod}</td>
                          <td style={{ padding: 4 }}>{r.slowPeriod}</td>
                          <td style={{ padding: 4 }}>{(r[metric] as number).toFixed(6)}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
