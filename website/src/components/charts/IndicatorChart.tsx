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
import { loadCSVData, getDataSubset } from '../../lib/utils/csv-data-loader'; // ← this one
import { ExportButton } from '../export/ExportButton';

interface IndicatorChartProps {
  height?: number;
  indicatorData?: number[];
  indicatorType?: 'line' | 'histogram';
  indicatorColor?: string;
  indicatorId?: string;
  parameters?: Record<string, any>;
}

export function IndicatorChart({
  height = 500,
  indicatorData,
  indicatorType = 'line',
  indicatorColor = '#3b82f6',
  indicatorId = 'indicator',
  parameters = {},
}: IndicatorChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isDark, setIsDark] = useState(false);
  const [chartData, setChartData] = useState<any[]>([]);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const fullscreenRef = useRef<HTMLDivElement>(null);

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
        // First try to load the cached JSON data for better performance
        try {
          const base = import.meta.env.BASE_URL || '/';
          const jsonPath = `${base}data/sample-ohlcv.json`;
          console.log('Trying to load JSON from:', jsonPath);
          const response = await fetch(jsonPath);
          if (response.ok) {
            const candles = await response.json();
            setChartData(candles);
            return;
          }
        } catch (jsonError) {
          console.log('JSON load failed, trying CSV:', jsonError);
        }
        
        // Fallback to CSV if JSON is not available
        const candles = await loadCSVData('2018-09-01-2024-Bitfinex_Spot-4h.csv');
        if (!candles.length) throw new Error('No valid candle data');
        
        // Use full dataset
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

  const handleZoomIn = () => {
    if (chartRef.current) {
      const timeScale = chartRef.current.timeScale();
      const currentRange = timeScale.getVisibleRange();
      if (currentRange) {
        const rangeSize = (currentRange.to as number) - (currentRange.from as number);
        const newRangeSize = rangeSize * 0.7; // Zoom in by 30%
        const center = ((currentRange.from as number) + (currentRange.to as number)) / 2;
        timeScale.setVisibleRange({
          from: (center - newRangeSize / 2) as UTCTimestamp,
          to: (center + newRangeSize / 2) as UTCTimestamp,
        });
      }
    }
  };

  const handleZoomOut = () => {
    if (chartRef.current) {
      const timeScale = chartRef.current.timeScale();
      const currentRange = timeScale.getVisibleRange();
      if (currentRange) {
        const rangeSize = (currentRange.to as number) - (currentRange.from as number);
        const newRangeSize = rangeSize * 1.3; // Zoom out by 30%
        const center = ((currentRange.from as number) + (currentRange.to as number)) / 2;
        timeScale.setVisibleRange({
          from: (center - newRangeSize / 2) as UTCTimestamp,
          to: (center + newRangeSize / 2) as UTCTimestamp,
        });
      }
    }
  };

  const handleResetZoom = () => {
    if (chartRef.current) {
      chartRef.current.timeScale().resetTimeScale();
    }
  };

  const handleFullscreen = async () => {
    if (!fullscreenRef.current) return;

    try {
      if (!document.fullscreenElement) {
        await fullscreenRef.current.requestFullscreen();
        setIsFullscreen(true);
      } else {
        await document.exitFullscreen();
        setIsFullscreen(false);
      }
    } catch (err) {
      console.error('Error toggling fullscreen:', err);
    }
  };

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);

  return (
    <div ref={fullscreenRef} className={`relative w-full ${isFullscreen ? 'bg-white dark:bg-gray-900' : ''}`}>
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
      <div ref={containerRef} style={{ height: isFullscreen ? '100vh' : height }} />
      
      {/* Navigation controls and Export buttons */}
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
          
          {/* Zoom controls */}
          <div className="absolute left-2 top-2 flex flex-col gap-2 z-10">
            <button
              onClick={handleZoomIn}
              className="p-2 bg-white/90 dark:bg-gray-800/90 rounded-lg shadow-md hover:shadow-lg transition-all hover:bg-white dark:hover:bg-gray-800 border border-gray-200 dark:border-gray-700"
              title="Zoom in"
            >
              <svg className="w-5 h-5 text-gray-700 dark:text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
              </svg>
            </button>
            <button
              onClick={handleZoomOut}
              className="p-2 bg-white/90 dark:bg-gray-800/90 rounded-lg shadow-md hover:shadow-lg transition-all hover:bg-white dark:hover:bg-gray-800 border border-gray-200 dark:border-gray-700"
              title="Zoom out"
            >
              <svg className="w-5 h-5 text-gray-700 dark:text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM7 10h6" />
              </svg>
            </button>
            <button
              onClick={handleResetZoom}
              className="p-2 bg-white/90 dark:bg-gray-800/90 rounded-lg shadow-md hover:shadow-lg transition-all hover:bg-white dark:hover:bg-gray-800 border border-gray-200 dark:border-gray-700"
              title="Reset zoom"
            >
              <svg className="w-5 h-5 text-gray-700 dark:text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
            <button
              onClick={handleFullscreen}
              className="p-2 bg-white/90 dark:bg-gray-800/90 rounded-lg shadow-md hover:shadow-lg transition-all hover:bg-white dark:hover:bg-gray-800 border border-gray-200 dark:border-gray-700"
              title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            >
              {isFullscreen ? (
                <svg className="w-5 h-5 text-gray-700 dark:text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              ) : (
                <svg className="w-5 h-5 text-gray-700 dark:text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-5h-4m4 0v4m0-4l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5h-4m4 0v-4m0 4l-5-5" />
                </svg>
              )}
            </button>
          </div>
          
          {/* Export buttons */}
          <div className="absolute top-2 right-2 flex gap-2 z-10">
            {indicatorData && (
              <ExportButton
                data={{
                  ohlcv: chartData,
                  [indicatorId]: indicatorData,
                  parameters
                }}
                filename={`${indicatorId}-data`}
                format="json"
                className="text-sm"
              />
            )}
            <ExportButton
              data={chartData}
              filename="ohlcv-data"
              format="csv"
              className="text-sm"
            />
            <ExportButton
              data={null}
              filename={`${indicatorId}-chart`}
              format="image"
              chartRef={chartRef}
              className="text-sm"
            />
          </div>
        </>
      )}
    </div>
  );
}
