// src/components/charts/IndicatorChart.tsx
import { useEffect, useRef, useState } from 'react';
import {
  createChart,
  ColorType,
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  type IChartApi,
  type UTCTimestamp,
  type CandlestickData,
} from 'lightweight-charts';
import { loadCSVData } from '../../lib/utils/csv-data-loader'; // ← this one

interface IndicatorChartProps {
  height?: number;
  indicatorData?: number[];
  indicatorType?: 'line' | 'histogram';
  indicatorColor?: string;
}

export function IndicatorChart({
  height = 500,
  indicatorData,
  indicatorType = 'line',
  indicatorColor = '#3b82f6',
}: IndicatorChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isDark, setIsDark] = useState(false);
  const [chartData, setChartData] = useState<any[]>([]);

  // Detect theme changes
  useEffect(() => {
    const checkTheme = () => {
      setIsDark(document.documentElement.classList.contains('dark'));
    };
    
    checkTheme();
    
    // Watch for theme changes
    const observer = new MutationObserver(checkTheme);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    });
    
    return () => observer.disconnect();
  }, []);

  // Load data only once
  useEffect(() => {
    const loadData = async () => {
      try {
        setIsLoading(true);
        const candles = await loadCSVData('/2018-09-01-2024-Bitfinex_Spot-4h.csv');
        if (!candles.length) throw new Error('No valid candle data');
        setChartData(candles);
      } catch (err) {
        console.error('Failed to load data', err);
        setError(err instanceof Error ? err.message : 'Failed to load chart data');
      } finally {
        setIsLoading(false);
      }
    };
    loadData();
  }, []);

  // Update chart when theme changes
  useEffect(() => {
    if (chartRef.current && !isLoading) {
      chartRef.current.applyOptions({
        layout: {
          background: { type: ColorType.Solid, color: isDark ? '#111827' : '#ffffff' },
          textColor: isDark ? '#e5e7eb' : '#374151',
        },
        grid: {
          vertLines: { color: isDark ? '#374151' : '#e5e7eb' },
          horzLines: { color: isDark ? '#374151' : '#e5e7eb' },
        },
        rightPriceScale: { borderColor: isDark ? '#374151' : '#e5e7eb' },
        timeScale: {
          borderColor: isDark ? '#374151' : '#e5e7eb',
        },
      });
    }
  }, [isDark]);

  // Create chart
  useEffect(() => {
    if (!containerRef.current || chartData.length === 0) return;

    /* Create the chart */
    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: isDark ? '#111827' : '#ffffff' },
        textColor: isDark ? '#e5e7eb' : '#374151',
      },
      grid: {
        vertLines: { color: isDark ? '#374151' : '#e5e7eb' },
        horzLines: { color: isDark ? '#374151' : '#e5e7eb' },
      },
      crosshair: { mode: 0 },
      rightPriceScale: { borderColor: isDark ? '#374151' : '#e5e7eb' },
      timeScale: {
        borderColor: isDark ? '#374151' : '#e5e7eb',
        timeVisible: true,
        secondsVisible: false,
      },
    });
    chartRef.current = chart;

    /* Candles */
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#10b981',
      downColor: '#ef4444',
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
      borderVisible: false,
    });
    candleSeries.setData(chartData);

    /* Volume */
    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: '',
    });
    volumeSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });
    volumeSeries.setData(
      chartData.map((bar) => ({
        time: bar.time as UTCTimestamp,
        value: bar.volume ?? 0,
        color: bar.close >= bar.open ? '#10b98140' : '#ef444440',
      })),
    );

    /* Optional indicator */
    if (indicatorData?.length) {
      const len = Math.min(indicatorData.length, chartData.length);
      const base = chartData.slice(-len);

      const series =
        indicatorType === 'histogram'
          ? chart.addSeries(HistogramSeries, { priceScaleId: 'left' })
          : chart.addSeries(LineSeries, {
              color: indicatorColor,
              lineWidth: 2,
              priceScaleId: 'left',
            });

      series.setData(
        base.map((bar, i) =>
          indicatorType === 'histogram'
            ? {
                time: bar.time,
                value: indicatorData[i],
                color: indicatorData[i] >= 0 ? '#10b981' : '#ef4444',
              }
            : { time: bar.time, value: indicatorData[i] },
        ),
      );
    }

    chart.timeScale().fitContent();

    /* Resize handler */
    const handleResize = () => {
      if (containerRef.current && chart) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    // Cleanup function
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [height, indicatorData, indicatorType, indicatorColor, chartData]);

  const handleScrollToStart = () => {
    if (chartRef.current && chartData.length > 0) {
      const firstTime = chartData[0].time;
      const range = 100; // Show first 100 candles
      const endTime = chartData[Math.min(range, chartData.length - 1)].time;
      chartRef.current.timeScale().setVisibleRange({
        from: firstTime as UTCTimestamp,
        to: endTime as UTCTimestamp,
      });
    }
  };

  const handleScrollToEnd = () => {
    if (chartRef.current && chartData.length > 0) {
      const range = 100; // Show last 100 candles
      const startIndex = Math.max(0, chartData.length - range);
      const startTime = chartData[startIndex].time;
      const endTime = chartData[chartData.length - 1].time;
      chartRef.current.timeScale().setVisibleRange({
        from: startTime as UTCTimestamp,
        to: endTime as UTCTimestamp,
      });
    }
  };

  return (
    <div className="relative w-full">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/80 dark:bg-gray-900/80 z-10">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500" />
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/80 dark:bg-gray-900/80 z-10">
          <div className="text-red-500">{error}</div>
        </div>
      )}
      <div ref={containerRef} style={{ height }} />
      
      {/* Navigation controls */}
      {!isLoading && !error && chartData.length > 0 && (
        <>
          <button
            onClick={handleScrollToStart}
            className="absolute left-2 bottom-12 p-2 bg-white/90 dark:bg-gray-800/90 rounded-full shadow-md hover:shadow-lg transition-all hover:bg-white dark:hover:bg-gray-800 border border-gray-200 dark:border-gray-700 z-10"
            title="Go to start"
          >
            <svg className="w-5 h-5 text-gray-700 dark:text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <button
            onClick={handleScrollToEnd}
            className="absolute right-20 bottom-12 p-2 bg-white/90 dark:bg-gray-800/90 rounded-full shadow-md hover:shadow-lg transition-all hover:bg-white dark:hover:bg-gray-800 border border-gray-200 dark:border-gray-700 z-10"
            title="Go to end"
          >
            <svg className="w-5 h-5 text-gray-700 dark:text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </>
      )}
    </div>
  );
}
