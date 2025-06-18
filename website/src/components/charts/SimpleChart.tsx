import { useEffect, useRef } from 'react';
import { createChart, ColorType } from 'lightweight-charts';

export function SimpleChart() {
  const chartContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: 800,
      height: 400,
      layout: {
        background: { type: ColorType.Solid, color: '#ffffff' },
        textColor: '#333',
      },
    });

    const candlestickSeries = chart.addCandlestickSeries();

    // Generate dummy data
    const data = [];
    const baseTime = new Date(2024, 0, 1).getTime() / 1000;
    
    for (let i = 0; i < 50; i++) {
      const time = baseTime + i * 86400; // Daily data
      const open = 40000 + Math.random() * 1000;
      const close = open + (Math.random() - 0.5) * 1000;
      const high = Math.max(open, close) + Math.random() * 500;
      const low = Math.min(open, close) - Math.random() * 500;
      
      data.push({ time, open, high, low, close });
    }

    candlestickSeries.setData(data);

    return () => {
      chart.remove();
    };
  }, []);

  return (
    <div 
      ref={chartContainerRef} 
      style={{ width: '100%', height: '400px' }}
    />
  );
}