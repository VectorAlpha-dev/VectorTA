import { useEffect, useRef, useState } from 'react';
import { createChart, ColorType } from 'lightweight-charts';
import type { IChartApi } from 'lightweight-charts';

interface BasicPriceChartProps {
  height?: number;
}

export function BasicPriceChart({ height = 500 }: BasicPriceChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [status, setStatus] = useState('Initializing...');

  useEffect(() => {
    if (!chartContainerRef.current) {
      setStatus('No container');
      return;
    }

    setStatus('Creating chart...');
    
    // Generate dummy data
    const data = [];
    const baseTime = Math.floor(Date.now() / 1000) - 100 * 24 * 60 * 60;
    let lastClose = 40000;

    for (let i = 0; i < 100; i++) {
      const open = lastClose;
      const close = open + (Math.random() - 0.5) * 1000;
      const high = Math.max(open, close) + Math.random() * 500;
      const low = Math.min(open, close) - Math.random() * 500;
      
      data.push({
        time: baseTime + i * 24 * 60 * 60,
        open,
        high,
        low,
        close
      });
      
      lastClose = close;
    }

    try {
      // Create chart with explicit dimensions
      const chart = createChart(chartContainerRef.current, {
        width: chartContainerRef.current.clientWidth || 800,
        height: height,
        layout: {
          background: { type: ColorType.Solid, color: '#ffffff' },
          textColor: '#333',
        },
        grid: {
          vertLines: { color: '#e0e0e0' },
          horzLines: { color: '#e0e0e0' },
        },
        timeScale: {
          borderColor: '#e0e0e0',
        },
        rightPriceScale: {
          borderColor: '#e0e0e0',
        },
      });

      chartRef.current = chart;

      // Add candlestick series
      const candlestickSeries = chart.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
      });

      candlestickSeries.setData(data);
      chart.timeScale().fitContent();

      setStatus('Chart created successfully!');

      // Handle resize
      const handleResize = () => {
        if (chartContainerRef.current && chart) {
          chart.applyOptions({ 
            width: chartContainerRef.current.clientWidth 
          });
        }
      };

      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
        chart.remove();
      };
    } catch (error) {
      console.error('BasicPriceChart error:', error);
      setStatus(`Error: ${error}`);
    }
  }, [height]);

  return (
    <div className="w-full">
      <div className="mb-2 text-sm text-gray-600">{status}</div>
      <div 
        ref={chartContainerRef} 
        className="w-full border-2 border-blue-500"
        style={{ height: `${height}px` }}
      />
    </div>
  );
}