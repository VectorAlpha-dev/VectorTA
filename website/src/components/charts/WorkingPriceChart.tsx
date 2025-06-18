import { useEffect, useRef, useState } from 'react';
import { loadCSVData, getDataSubset, type CandlestickData } from '../../lib/utils/csv-data-loader';

export function WorkingPriceChart({ height = 500 }: { height?: number }) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<CandlestickData[]>([]);

  // Load CSV data
  useEffect(() => {
    async function loadData() {
      try {
        console.log('WorkingPriceChart: Loading CSV data...');
        const csvData = await loadCSVData('/2018-09-01-2024-Bitfinex_Spot-4h.csv');
        const subset = getDataSubset(csvData, 1000);
        console.log('WorkingPriceChart: Loaded', subset.length, 'candles');
        setData(subset);
        setIsLoading(false);
      } catch (err) {
        console.error('WorkingPriceChart: Error loading data:', err);
        setError('Failed to load price data');
        setIsLoading(false);
      }
    }
    loadData();
  }, []);

  // Create chart after data loads
  useEffect(() => {
    if (!chartContainerRef.current || data.length === 0) return;

    const loadChart = async () => {
      try {
        // Dynamically import to ensure it's loaded on client side
        const LightweightCharts = await import('lightweight-charts');
        
        console.log('WorkingPriceChart: Creating chart...');
        const chart = LightweightCharts.createChart(chartContainerRef.current, {
          width: chartContainerRef.current.clientWidth,
          height: height,
          layout: {
            background: { type: LightweightCharts.ColorType.Solid, color: '#ffffff' },
            textColor: '#333',
          },
          grid: {
            vertLines: { color: '#e0e0e0' },
            horzLines: { color: '#e0e0e0' },
          },
        });

        // Create candlestick series
        const candlestickSeries = chart.addCandlestickSeries({
          upColor: '#26a69a',
          downColor: '#ef5350',
          borderVisible: false,
          wickUpColor: '#26a69a',
          wickDownColor: '#ef5350',
        });

        // Set data
        candlestickSeries.setData(data);
        chart.timeScale().fitContent();

        console.log('WorkingPriceChart: Chart created successfully!');

        // Handle resize
        const handleResize = () => {
          if (chartContainerRef.current) {
            chart.applyOptions({ width: chartContainerRef.current.clientWidth });
          }
        };
        window.addEventListener('resize', handleResize);

        // Cleanup
        return () => {
          window.removeEventListener('resize', handleResize);
          chart.remove();
        };
      } catch (err) {
        console.error('WorkingPriceChart: Chart creation error:', err);
        setError('Failed to create chart');
      }
    };

    const cleanup = loadChart();
    return () => {
      cleanup.then(fn => fn?.());
    };
  }, [data, height]);

  if (error) {
    return (
      <div className="flex items-center justify-center h-[500px] bg-red-50 rounded-lg">
        <div className="text-red-500">{error}</div>
      </div>
    );
  }

  return (
    <div className="relative">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/80 z-10">
          <div className="text-gray-600">Loading chart data...</div>
        </div>
      )}
      <div 
        ref={chartContainerRef} 
        className="w-full bg-gray-50"
        style={{ height: `${height}px` }}
      />
      {!isLoading && data.length > 0 && (
        <div className="mt-2 text-sm text-gray-500 text-center">
          Loaded {data.length} candles
        </div>
      )}
    </div>
  );
}