import { useEffect, useRef, useState } from 'react';
import { createChart, ColorType } from 'lightweight-charts';
import type { IChartApi } from 'lightweight-charts';

interface SimplePriceChartProps {
  darkMode?: boolean;
  height?: number;
}

export function SimplePriceChart({ darkMode = false, height = 500 }: SimplePriceChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    console.log('SimplePriceChart mounted');
  }, []);

  useEffect(() => {
    if (!chartContainerRef.current || !mounted) return;

    console.log('Creating chart...');
    
    // Create chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: darkMode ? '#111827' : '#ffffff' },
        textColor: darkMode ? '#d1d5db' : '#333',
      },
      grid: {
        vertLines: { color: darkMode ? '#1f2937' : '#e5e7eb' },
        horzLines: { color: darkMode ? '#1f2937' : '#e5e7eb' },
      },
    });

    // Add candlestick series with dummy data
    const candlestickSeries = (chart as any).addCandlestickSeries({
      upColor: '#10b981',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
    });

    // Generate dummy data
    const dummyData = [];
    let time = Math.floor(Date.now() / 1000) - 100 * 24 * 60 * 60; // 100 days ago
    let lastClose = 40000;

    for (let i = 0; i < 100; i++) {
      const open = lastClose;
      const close = open + (Math.random() - 0.5) * 1000;
      const high = Math.max(open, close) + Math.random() * 500;
      const low = Math.min(open, close) - Math.random() * 500;
      
      dummyData.push({
        time: time + i * 24 * 60 * 60, // Daily candles
        open,
        high,
        low,
        close
      });
      
      lastClose = close;
    }

    candlestickSeries.setData(dummyData);
    chart.timeScale().fitContent();

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
  }, [mounted, darkMode, height]);

  return (
    <div className="w-full">
      <div className="text-center mb-4 text-gray-600 dark:text-gray-400">
        TradingView Lightweight Chart Test
      </div>
      <div ref={chartContainerRef} className="w-full border border-gray-300 dark:border-gray-700 rounded-lg" />
    </div>
  );
}