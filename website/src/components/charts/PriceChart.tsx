import { useEffect, useRef, useState } from 'react';
import { createChart, ColorType } from 'lightweight-charts';
import type { IChartApi } from 'lightweight-charts';
import { loadCSVData, getDataSubset, type CandlestickData } from '../../lib/utils/csv-data-loader';

interface PriceChartProps {
  darkMode?: boolean;
  height?: number;
}

export function PriceChart({ darkMode = false, height = 500 }: PriceChartProps) {
  console.log('PriceChart component mounted, height:', height, 'darkMode:', darkMode);
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<CandlestickData[]>([]);
  const [chartCreated, setChartCreated] = useState(false);

  // Load CSV data
  useEffect(() => {
    async function loadData() {
      try {
        console.log('PriceChart: Starting to load CSV data...');
        setIsLoading(true);
        setError(null);
        const csvData = await loadCSVData('/2018-09-01-2024-Bitfinex_Spot-4h.csv');
        console.log('PriceChart: Loaded CSV data, length:', csvData.length);
        // Get last 1000 candles for performance
        const subset = getDataSubset(csvData, 1000);
        console.log('PriceChart: Using subset of', subset.length, 'candles');
        setData(subset);
      } catch (err) {
        setError('Failed to load price data');
        console.error('PriceChart: Error loading data:', err);
      } finally {
        setIsLoading(false);
      }
    }
    
    loadData();
  }, []);

  // Create and update chart
  useEffect(() => {
    if (!chartContainerRef.current || data.length === 0) {
      console.log('PriceChart: Skipping chart creation - container:', !!chartContainerRef.current, 'data length:', data.length);
      return;
    }

    console.log('PriceChart: Creating chart with', data.length, 'candles');
    console.log('PriceChart: First candle:', data[0]);
    console.log('PriceChart: Container dimensions:', chartContainerRef.current.clientWidth, 'x', height);

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth || 800,
      height: height || 500,
      layout: {
        background: { type: ColorType.Solid, color: darkMode ? '#111827' : '#ffffff' },
        textColor: darkMode ? '#d1d5db' : '#333',
      },
      grid: {
        vertLines: { color: darkMode ? '#1f2937' : '#e5e7eb' },
        horzLines: { color: darkMode ? '#1f2937' : '#e5e7eb' },
      },
      crosshair: {
        mode: 0,
      },
      rightPriceScale: {
        borderColor: darkMode ? '#374151' : '#e5e7eb',
      },
      timeScale: {
        borderColor: darkMode ? '#374151' : '#e5e7eb',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

    try {
      console.log('PriceChart: Chart object:', chart);
      console.log('PriceChart: Available methods:', Object.getOwnPropertyNames(Object.getPrototypeOf(chart)));
      
      // Add candlestick series - checking if method exists
      if (typeof (chart as any).addCandlestickSeries !== 'function') {
        console.error('PriceChart: addCandlestickSeries is not a function. Chart methods:', Object.keys(chart));
        throw new Error('Chart API mismatch - addCandlestickSeries not found');
      }
      
      const candlestickSeries = (chart as any).addCandlestickSeries({
        upColor: '#10b981',
        downColor: '#ef4444',
        borderVisible: false,
        wickUpColor: '#10b981',
        wickDownColor: '#ef4444',
      });

      console.log('PriceChart: Added candlestick series');

      // Set the candlestick data
      candlestickSeries.setData(data);
      console.log('PriceChart: Set candlestick data');

      // Add volume series
      const volumeSeries = (chart as any).addHistogramSeries({
        color: '#6366f1',
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: '',
      });

      // Set volume data
      const volumeData = data.map(candle => ({
        time: candle.time,
        value: candle.volume || 0,
        color: candle.close >= candle.open ? '#10b98180' : '#ef444480'
      }));
      volumeSeries.setData(volumeData);

      // Configure volume series scale
      chart.priceScale('').applyOptions({
        scaleMargins: {
          top: 0.8,
          bottom: 0,
        },
      });

      // Fit content
      chart.timeScale().fitContent();
      console.log('PriceChart: Chart created successfully');
      setChartCreated(true);
    } catch (error) {
      console.error('PriceChart: Error creating chart:', error);
      setError('Failed to create chart');
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
  }, [data, darkMode, height]);

  if (error) {
    return (
      <div className="flex items-center justify-center h-[500px] bg-gray-50 dark:bg-gray-900 rounded-lg">
        <div className="text-red-500 dark:text-red-400">{error}</div>
      </div>
    );
  }

  return (
    <div className="relative">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/80 dark:bg-gray-900/80 z-10 rounded-lg">
          <div className="flex flex-col items-center space-y-3">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
            <div className="text-gray-600 dark:text-gray-400">Loading price data...</div>
          </div>
        </div>
      )}
      <div 
        ref={chartContainerRef} 
        className="w-full rounded-lg overflow-hidden bg-white dark:bg-gray-800" 
        style={{ height: `${height}px`, minHeight: `${height}px` }}
      >
        {!isLoading && data.length === 0 && !error && (
          <div className="flex items-center justify-center h-full text-gray-500">
            No data to display
          </div>
        )}
      </div>
      {!isLoading && data.length > 0 && (
        <div className="mt-2 text-sm text-gray-500 dark:text-gray-400 text-center">
          Showing {data.length} candles (4-hour timeframe)
        </div>
      )}
    </div>
  );
}