import { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ColorType } from 'lightweight-charts';
import { talib } from '../../lib/mock-wasm';
import type { OHLCV } from '../../lib/utils/data-loader-client';

interface IndicatorChartProps {
  indicatorId: string;
  data: OHLCV[];
  parameters: Record<string, any>;
  darkMode?: boolean;
}

export function IndicatorChart({ indicatorId, data, parameters, darkMode = false }: IndicatorChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!chartContainerRef.current || !data || data.length === 0) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 500,
      layout: {
        background: { type: ColorType.Solid, color: darkMode ? '#1a1a1a' : '#ffffff' },
        textColor: darkMode ? '#d1d5db' : '#333',
      },
      grid: {
        vertLines: { color: darkMode ? '#2a2a2a' : '#e0e0e0' },
        horzLines: { color: darkMode ? '#2a2a2a' : '#e0e0e0' },
      },
      crosshair: {
        mode: 0,
      },
      rightPriceScale: {
        borderColor: darkMode ? '#2a2a2a' : '#e0e0e0',
      },
      timeScale: {
        borderColor: darkMode ? '#2a2a2a' : '#e0e0e0',
        timeVisible: true,
        secondsVisible: false,
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

    // Calculate and add indicator
    try {
      const indicatorData = talib.calculate(indicatorId, data, parameters);
      
      if (Array.isArray(indicatorData)) {
        // Single line indicator
        addLineSeries(chart, indicatorData, data, 'Indicator', '#2962ff');
      } else if (typeof indicatorData === 'object') {
        // Multi-line indicator (like Bollinger Bands, MACD)
        const colors = ['#2962ff', '#ff6b6b', '#4ecdc4', '#ffe66d'];
        let colorIndex = 0;
        
        Object.entries(indicatorData).forEach(([key, values]) => {
          if (Array.isArray(values)) {
            addLineSeries(
              chart, 
              values as (number | null)[], 
              data, 
              key.charAt(0).toUpperCase() + key.slice(1),
              colors[colorIndex++ % colors.length]
            );
          }
        });
      }
      
      setIsLoading(false);
    } catch (error) {
      console.error('Error calculating indicator:', error);
      setIsLoading(false);
    }

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
  }, [indicatorId, data, parameters, darkMode]);

  function addLineSeries(
    chart: IChartApi,
    values: (number | null)[],
    ohlcv: OHLCV[],
    title: string,
    color: string
  ) {
    const lineSeries = chart.addLineSeries({
      color,
      lineWidth: 2,
      title,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 5,
    });

    const lineData = values
      .map((value, index) => ({
        time: ohlcv[index]?.time,
        value,
      }))
      .filter(item => item.value !== null && item.time !== undefined);

    lineSeries.setData(lineData as any);
  }

  return (
    <div className="relative">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/80 dark:bg-gray-900/80 z-10">
          <div className="text-gray-600 dark:text-gray-400">Loading chart...</div>
        </div>
      )}
      <div ref={chartContainerRef} className="w-full" />
    </div>
  );
}