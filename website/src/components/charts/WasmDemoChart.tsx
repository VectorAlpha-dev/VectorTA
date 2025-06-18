import { useEffect, useRef, useState } from 'react';
import {
  createChart,
  ColorType,
  CandlestickSeries,
  LineSeries,
} from 'lightweight-charts';
import type { IChartApi, ISeriesApi } from 'lightweight-charts';

interface WasmDemoChartProps {
  height?: number;
}

// Mock WASM interface for demo purposes
interface WasmIndicator {
  calculate: (data: number[]) => Promise<number[]>;
  name: string;
  color: string;
}

// Mock WASM indicators
const mockIndicators: WasmIndicator[] = [
  {
    name: 'SMA 20',
    color: '#3b82f6',
    calculate: async (closes: number[]) => {
      // Simple moving average calculation
      const period = 20;
      const result: number[] = [];
      for (let i = 0; i < closes.length; i++) {
        if (i < period - 1) {
          result.push(NaN);
        } else {
          const sum = closes.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
          result.push(sum / period);
        }
      }
      return result;
    },
  },
  {
    name: 'EMA 12',
    color: '#10b981',
    calculate: async (closes: number[]) => {
      // Exponential moving average calculation
      const period = 12;
      const multiplier = 2 / (period + 1);
      const result: number[] = [];
      
      // Start with SMA for first value
      let ema = closes.slice(0, period).reduce((a, b) => a + b, 0) / period;
      
      for (let i = 0; i < closes.length; i++) {
        if (i < period - 1) {
          result.push(NaN);
        } else if (i === period - 1) {
          result.push(ema);
        } else {
          ema = (closes[i] - ema) * multiplier + ema;
          result.push(ema);
        }
      }
      return result;
    },
  },
];

export function WasmDemoChart({ height = 400 }: WasmDemoChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>(['SMA 20']);

  useEffect(() => {
    if (!containerRef.current) return;

    const initChart = async () => {
      try {
        setIsLoading(true);
        
        // Generate demo price data
        const basePrice = 50000;
        const volatility = 0.02;
        const numCandles = 200;
        const candles = [];
        
        let currentPrice = basePrice;
        const now = Math.floor(Date.now() / 1000);
        
        for (let i = 0; i < numCandles; i++) {
          const change = (Math.random() - 0.5) * volatility;
          currentPrice *= (1 + change);
          
          const high = currentPrice * (1 + Math.random() * 0.01);
          const low = currentPrice * (1 - Math.random() * 0.01);
          const open = low + Math.random() * (high - low);
          const close = low + Math.random() * (high - low);
          
          candles.push({
            time: (now - (numCandles - i) * 14400) as any, // 4 hour candles
            open,
            high,
            low,
            close,
          });
        }
        
        // Create chart
        const chart = createChart(containerRef.current!, {
          width: containerRef.current!.clientWidth,
          height,
          layout: {
            background: { type: ColorType.Solid, color: '#ffffff' },
            textColor: '#333333',
          },
          grid: {
            vertLines: { color: '#e5e7eb' },
            horzLines: { color: '#e5e7eb' },
          },
          crosshair: {
            mode: 0,
          },
          rightPriceScale: {
            borderColor: '#e5e7eb',
          },
          timeScale: {
            borderColor: '#e5e7eb',
            timeVisible: true,
            secondsVisible: false,
          },
        });
        
        chartRef.current = chart;

        // Add candlestick series
        const candleSeries = chart.addSeries(CandlestickSeries, {
          upColor: '#10b981',
          downColor: '#ef4444',
          wickUpColor: '#10b981',
          wickDownColor: '#ef4444',
          borderVisible: false,
        });
        candleSeries.setData(candles);

        // Calculate and add selected indicators
        const closes = candles.map(c => c.close);
        
        for (const indicator of mockIndicators) {
          if (selectedIndicators.includes(indicator.name)) {
            const values = await indicator.calculate(closes);
            const indicatorSeries = chart.addSeries(LineSeries, {
              color: indicator.color,
              lineWidth: 2,
              title: indicator.name,
            });
            
            const indicatorData = candles.map((candle, i) => ({
              time: candle.time,
              value: values[i],
            })).filter(d => !isNaN(d.value));
            
            indicatorSeries.setData(indicatorData);
          }
        }

        // Fit content
        chart.timeScale().fitContent();
        
        // Handle resize
        const handleResize = () => {
          if (containerRef.current && chart) {
            chart.applyOptions({ width: containerRef.current.clientWidth });
          }
        };
        window.addEventListener('resize', handleResize);
        
        return () => {
          window.removeEventListener('resize', handleResize);
          chart.remove();
        };
      } catch (err) {
        console.error('Failed to initialize chart:', err);
      } finally {
        setIsLoading(false);
      }
    };

    initChart();
  }, [height, selectedIndicators]);

  const toggleIndicator = (name: string) => {
    setSelectedIndicators(prev => 
      prev.includes(name) 
        ? prev.filter(n => n !== name)
        : [...prev, name]
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center space-x-4">
        <span className="text-sm font-medium text-gray-700">Indicators:</span>
        {mockIndicators.map(indicator => (
          <label key={indicator.name} className="flex items-center space-x-2 cursor-pointer">
            <input
              type="checkbox"
              checked={selectedIndicators.includes(indicator.name)}
              onChange={() => toggleIndicator(indicator.name)}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-sm text-gray-600">{indicator.name}</span>
          </label>
        ))}
      </div>
      
      <div className="relative w-full bg-white rounded-lg shadow-sm border border-gray-200">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/80 z-10">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          </div>
        )}
        <div ref={containerRef} style={{ height }} />
      </div>
      
      <div className="text-xs text-gray-500 text-center">
        Demo chart with simulated data - WASM indicators coming soon!
      </div>
    </div>
  );
}