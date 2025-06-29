import { useEffect, useRef, useState } from 'react';
import { type IChartApi } from 'lightweight-charts';
import {
  createChart,
  ColorType,
  CandlestickSeries,
  HistogramSeries,
} from 'lightweight-charts';
import { loadCSVData, getDataSubset } from '../../lib/utils/csv-data-loader';

export function FinalPriceChart({ height = 500 }: { height?: number }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [status, setStatus] = useState('initialising…');

  useEffect(() => {
    let alive = true;
    (async () => {
      if (!wrapRef.current) return;

      setStatus('loading data…');
      const candles = getDataSubset(await loadCSVData('/2018-09-01-2024-Bitfinex_Spot-4h.csv'), 1000);
      if (!alive) return;

      const chart = createChart(wrapRef.current, {
        width: wrapRef.current.clientWidth,
        height,
        layout: { background: { type: ColorType.Solid, color: '#fff' }, textColor: '#333' },
        grid: { vertLines: { color: '#e5e7eb' }, horzLines: { color: '#e5e7eb' } },
        timeScale: { timeVisible: true, secondsVisible: false },
      });
      chartRef.current = chart;

      const candle = chart.addSeries(CandlestickSeries, {
        upColor: '#10b981',
        downColor: '#ef4444',
        wickUpColor: '#10b981',
        wickDownColor: '#ef4444',
        borderVisible: false,
      });
      candle.setData(candles as any);

      const volume = chart.addSeries(HistogramSeries, {
        priceFormat: { type: 'volume' },
        priceScaleId: '',
      });
      volume.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
      volume.setData(
        candles.map(c => ({
          time: c.time,
          value: c.volume ?? 0,
          color: c.close >= c.open ? '#10b98140' : '#ef444440',
        })) as any,
      );

      chart.timeScale().fitContent();
      setStatus('chart ready');

      const resize = () => {
        chart.applyOptions({ width: wrapRef.current!.clientWidth });
      };
      window.addEventListener('resize', resize);
      return () => {
        window.removeEventListener('resize', resize);
        chart.remove();
      };
    })();
    return () => {
      alive = false;
      chartRef.current?.remove();
    };
  }, [height]);

  return (
    <div className="w-full">
      <div className="text-xs text-gray-500 mb-2">{status}</div>
      <div ref={wrapRef} className="w-full bg-gray-50 rounded border border-gray-200" style={{ height }} />
    </div>
  );
}
