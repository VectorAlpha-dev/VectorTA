import { useEffect, useRef, useState } from 'react';
import { createChart, ColorType } from 'lightweight-charts';
import type { IChartApi } from 'lightweight-charts';
import { talib } from '../../lib/mock-wasm';

export function BacktestChart() {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [sampleData, setSampleData] = useState<any[]>([]);

  useEffect(() => {
    // Load sample data
    fetch('/data/sample-ohlcv.json')
      .then(res => res.json())
      .then(data => {
        setSampleData(data);
        setIsLoading(false);
      })
      .catch(err => {
        console.error('Error loading sample data:', err);
        setIsLoading(false);
      });
  }, []);

  useEffect(() => {
    if (!chartContainerRef.current || sampleData.length === 0) return;

    const isDark = document.documentElement.classList.contains('dark');
    
    // Create chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 400,
      layout: {
        background: { type: ColorType.Solid, color: isDark ? '#1a1a1a' : '#ffffff' },
        textColor: isDark ? '#d1d5db' : '#333',
      },
      grid: {
        vertLines: { color: isDark ? '#2a2a2a' : '#e0e0e0' },
        horzLines: { color: isDark ? '#2a2a2a' : '#e0e0e0' },
      },
    });

    // Add candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
    });

    candlestickSeries.setData(sampleData);

    // Calculate and add moving averages
    const closes = sampleData.map(d => d.close);
    const sma20 = talib.sma(closes, 20);
    const sma50 = talib.sma(closes, 50);

    // Add SMA lines
    const sma20Series = chart.addLineSeries({
      color: '#2962ff',
      lineWidth: 2,
      title: 'SMA 20',
    });

    const sma50Series = chart.addLineSeries({
      color: '#ff6b6b',
      lineWidth: 2,
      title: 'SMA 50',
    });

    const sma20Data = sma20
      .map((value, index) => ({
        time: sampleData[index]?.time,
        value,
      }))
      .filter(item => item.value !== null);

    const sma50Data = sma50
      .map((value, index) => ({
        time: sampleData[index]?.time,
        value,
      }))
      .filter(item => item.value !== null);

    sma20Series.setData(sma20Data as any);
    sma50Series.setData(sma50Data as any);

    // Add crossover markers
    const markers: any[] = [];
    for (let i = 1; i < sma20.length; i++) {
      if (sma20[i] !== null && sma50[i] !== null && sma20[i - 1] !== null && sma50[i - 1] !== null) {
        // Golden cross (bullish)
        if (sma20[i - 1]! <= sma50[i - 1]! && sma20[i]! > sma50[i]!) {
          markers.push({
            time: sampleData[i].time,
            position: 'belowBar',
            color: '#26a69a',
            shape: 'arrowUp',
            text: 'Buy',
          });
        }
        // Death cross (bearish)
        else if (sma20[i - 1]! >= sma50[i - 1]! && sma20[i]! < sma50[i]!) {
          markers.push({
            time: sampleData[i].time,
            position: 'aboveBar',
            color: '#ef5350',
            shape: 'arrowDown',
            text: 'Sell',
          });
        }
      }
    }

    candlestickSeries.setMarkers(markers);

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [sampleData]);

  return (
    <div className="space-y-4">
      <div className="text-sm text-gray-600 dark:text-gray-400">
        This demo shows a simple moving average crossover strategy. Golden crosses (SMA 20 &gt; SMA 50) indicate potential buy signals, 
        while death crosses indicate potential sell signals.
      </div>
      <div className="relative">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/80 dark:bg-gray-900/80 z-10">
            <div className="text-gray-600 dark:text-gray-400">Loading chart...</div>
          </div>
        )}
        <div ref={chartContainerRef} className="w-full" />
      </div>
    </div>
  );
}