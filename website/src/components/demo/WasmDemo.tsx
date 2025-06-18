import { useState, useEffect, useCallback } from 'react';
import { createChart, IChartApi } from 'lightweight-charts';
import { wasmLoader } from '../../lib/wasm/loader';
import { PerformanceMonitor } from './PerformanceMonitor';
import type { WasmModule } from '../../lib/wasm/types';

interface WasmDemoProps {
  initialIndicator?: string;
}

export function WasmDemo({ initialIndicator = 'sma' }: WasmDemoProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [wasmModule, setWasmModule] = useState<WasmModule | null>(null);
  const [selectedIndicator, setSelectedIndicator] = useState(initialIndicator);
  const [parameters, setParameters] = useState({
    period: 20,
    stdDev: 2,
    fastPeriod: 12,
    slowPeriod: 26,
    signalPeriod: 9
  });
  const [isCalculating, setIsCalculating] = useState(false);
  const [performanceMetrics, setPerformanceMetrics] = useState(null);
  const [chart, setChart] = useState<IChartApi | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Generate sample data
  const generateSampleData = useCallback(() => {
    const data = new Float64Array(1000);
    let price = 100;
    for (let i = 0; i < data.length; i++) {
      price += (Math.random() - 0.5) * 2;
      data[i] = Math.max(price, 1);
    }
    return data;
  }, []);

  // Load WASM module
  useEffect(() => {
    const loadWasm = async () => {
      try {
        setIsLoading(true);
        const module = await wasmLoader.loadModule();
        setWasmModule(module);
        setError(null);
      } catch (err) {
        setError('Failed to load WebAssembly module. Please ensure your browser supports WASM.');
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };

    loadWasm();

    return () => {
      wasmLoader.dispose();
    };
  }, []);

  // Calculate indicator
  const calculateIndicator = useCallback(async () => {
    if (!wasmModule) return;

    setIsCalculating(true);
    const startTime = performance.now();
    const data = generateSampleData();

    try {
      let result;
      switch (selectedIndicator) {
        case 'sma':
          result = wasmModule.calculate_sma(data, parameters.period);
          break;
        case 'ema':
          result = wasmModule.calculate_ema(data, parameters.period);
          break;
        case 'rsi':
          result = wasmModule.calculate_rsi(data, parameters.period);
          break;
        case 'bollinger':
          result = wasmModule.calculate_bollinger_bands(data, parameters.period, parameters.stdDev);
          break;
        case 'macd':
          result = wasmModule.calculate_macd(
            data, 
            parameters.fastPeriod, 
            parameters.slowPeriod, 
            parameters.signalPeriod
          );
          break;
      }

      const endTime = performance.now();
      const executionTime = endTime - startTime;

      setPerformanceMetrics({
        executionTime,
        operationsPerSecond: wasmModule.get_operations_per_second(),
        dataPointsProcessed: data.length
      });

      // Update chart (implementation would go here)
      // For now, just log the result
      console.log('Calculation complete:', result);

    } catch (err) {
      console.error('Calculation error:', err);
      setError('Failed to calculate indicator');
    } finally {
      setIsCalculating(false);
    }
  }, [wasmModule, selectedIndicator, parameters, generateSampleData]);

  // Handle parameter changes
  const handleParameterChange = (param: string, value: number) => {
    setParameters(prev => ({ ...prev, [param]: value }));
  };

  if (error) {
    return (
      <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-6 text-center">
        <svg className="w-12 h-12 mx-auto mb-4 text-destructive" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
            d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <h3 className="text-lg font-semibold mb-2">WebAssembly Not Available</h3>
        <p className="text-sm text-muted-foreground">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Demo Controls */}
      <div className="bg-card rounded-lg border border-border p-6">
        <h3 className="text-lg font-semibold mb-4">Interactive WASM Demo</h3>
        
        {isLoading ? (
          <div className="text-center py-8">
            <div className="inline-flex items-center gap-3">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
              <span>Loading WebAssembly module...</span>
            </div>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium mb-2">Indicator</label>
                <select
                  value={selectedIndicator}
                  onChange={(e) => setSelectedIndicator(e.target.value)}
                  className="w-full px-3 py-2 bg-muted border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
                >
                  <option value="sma">Simple Moving Average (SMA)</option>
                  <option value="ema">Exponential Moving Average (EMA)</option>
                  <option value="rsi">Relative Strength Index (RSI)</option>
                  <option value="bollinger">Bollinger Bands</option>
                  <option value="macd">MACD</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2">Period</label>
                <input
                  type="range"
                  min="5"
                  max="50"
                  value={parameters.period}
                  onChange={(e) => handleParameterChange('period', Number(e.target.value))}
                  className="w-full"
                />
                <div className="text-center text-sm text-muted-foreground mt-1">
                  {parameters.period}
                </div>
              </div>
            </div>

            <button
              onClick={calculateIndicator}
              disabled={isCalculating}
              className="w-full py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium"
            >
              {isCalculating ? 'Calculating...' : 'Run Calculation'}
            </button>
          </>
        )}
      </div>

      {/* Performance Monitor */}
      <PerformanceMonitor
        metrics={performanceMetrics}
        isCalculating={isCalculating}
        comparison={{ name: 'Pure JS', executionTime: 50 }}
      />

      {/* Chart Placeholder */}
      <div className="bg-card rounded-lg border border-border p-6">
        <h3 className="text-lg font-semibold mb-4">Visualization</h3>
        <div className="h-64 bg-muted/20 rounded flex items-center justify-center text-muted-foreground">
          Chart visualization will appear here
        </div>
      </div>

      {/* Info Box */}
      <div className="bg-muted/50 rounded-lg p-4 text-sm">
        <p className="font-medium mb-1">About This Demo</p>
        <p className="text-muted-foreground">
          This interactive demo showcases VectorTA's WebAssembly performance. The calculations run entirely 
          in your browser at near-native speed. Try adjusting the parameters to see real-time updates.
        </p>
      </div>
    </div>
  );
}