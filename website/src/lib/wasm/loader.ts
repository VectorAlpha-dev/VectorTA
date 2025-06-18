import type { WasmModule } from './types';

class WasmLoader {
  private module: WasmModule | null = null;
  private loadingPromise: Promise<WasmModule> | null = null;
  private worker: Worker | null = null;

  async loadModule(): Promise<WasmModule> {
    // Return existing module if already loaded
    if (this.module) {
      return this.module;
    }

    // Return existing loading promise if already loading
    if (this.loadingPromise) {
      return this.loadingPromise;
    }

    // Start loading
    this.loadingPromise = this.initializeWasm();
    this.module = await this.loadingPromise;
    return this.module;
  }

  private async initializeWasm(): Promise<WasmModule> {
    try {
      // Check for WebAssembly support
      if (!('WebAssembly' in window)) {
        throw new Error('WebAssembly is not supported in this browser');
      }

      // In production, this would load the actual WASM file
      // For now, return a mock implementation
      console.log('Loading WASM module...');
      
      // Simulate loading delay
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Return mock implementation
      return this.createMockModule();
    } catch (error) {
      console.error('Failed to load WASM module:', error);
      throw error;
    }
  }

  private createMockModule(): WasmModule {
    return {
      calculate_sma: (data: Float64Array, period: number) => {
        const result = new Float64Array(data.length);
        for (let i = 0; i < data.length; i++) {
          if (i < period - 1) {
            result[i] = NaN;
          } else {
            let sum = 0;
            for (let j = 0; j < period; j++) {
              sum += data[i - j];
            }
            result[i] = sum / period;
          }
        }
        return result;
      },

      calculate_ema: (data: Float64Array, period: number) => {
        const result = new Float64Array(data.length);
        const multiplier = 2 / (period + 1);
        
        // Calculate initial SMA
        let sum = 0;
        for (let i = 0; i < period; i++) {
          sum += data[i];
        }
        result[period - 1] = sum / period;
        
        // Calculate EMA
        for (let i = period; i < data.length; i++) {
          result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
        }
        
        return result;
      },

      calculate_rsi: (data: Float64Array, period: number) => {
        const result = new Float64Array(data.length);
        // Simplified RSI calculation for demo
        for (let i = 0; i < data.length; i++) {
          result[i] = 50 + Math.sin(i / 10) * 30; // Mock oscillating RSI
        }
        return result;
      },

      calculate_macd: (data: Float64Array, fastPeriod: number, slowPeriod: number, signalPeriod: number) => {
        const macd = new Float64Array(data.length);
        const signal = new Float64Array(data.length);
        const histogram = new Float64Array(data.length);
        
        // Mock MACD calculation
        for (let i = 0; i < data.length; i++) {
          macd[i] = Math.sin(i / 20) * 10;
          signal[i] = Math.sin(i / 25) * 8;
          histogram[i] = macd[i] - signal[i];
        }
        
        return { macd, signal, histogram };
      },

      calculate_bollinger_bands: (data: Float64Array, period: number, stdDev: number) => {
        const upper = new Float64Array(data.length);
        const middle = new Float64Array(data.length);
        const lower = new Float64Array(data.length);
        
        // Mock Bollinger Bands
        for (let i = 0; i < data.length; i++) {
          middle[i] = data[i];
          const deviation = Math.abs(Math.sin(i / 15)) * 100;
          upper[i] = middle[i] + deviation * stdDev;
          lower[i] = middle[i] - deviation * stdDev;
        }
        
        return { upper, middle, lower };
      },

      run_backtest: (config) => {
        // Mock backtest results
        const days = 252; // One year of trading days
        const equity = new Float64Array(days);
        let currentEquity = config.initialCapital;
        
        for (let i = 0; i < days; i++) {
          // Simulate daily returns
          const dailyReturn = (Math.random() - 0.48) * 0.02; // Slight positive bias
          currentEquity *= (1 + dailyReturn);
          equity[i] = currentEquity;
        }
        
        return {
          equity,
          trades: this.generateMockTrades(),
          metrics: {
            totalReturn: ((currentEquity - config.initialCapital) / config.initialCapital) * 100,
            annualizedReturn: 12.5,
            sharpeRatio: 1.25,
            maxDrawdown: -15.3,
            winRate: 0.55,
            profitFactor: 1.35,
            totalTrades: 156
          },
          executionTime: Math.random() * 50 + 10 // 10-60ms
        };
      },

      allocate_array: (size: number) => {
        return 0; // Mock pointer
      },

      free_array: (ptr: number) => {
        // No-op in mock
      },

      get_last_calculation_time: () => {
        return Math.random() * 5 + 1; // 1-6ms
      },

      get_operations_per_second: () => {
        return Math.floor(Math.random() * 500000) + 1000000; // 1M - 1.5M ops/sec
      }
    };
  }

  private generateMockTrades(): any[] {
    const trades = [];
    const numTrades = 20;
    
    for (let i = 0; i < numTrades; i++) {
      const entryPrice = 100 + Math.random() * 20;
      const exitPrice = entryPrice * (1 + (Math.random() - 0.45) * 0.1);
      trades.push({
        entryTime: Date.now() - (numTrades - i) * 86400000,
        exitTime: Date.now() - (numTrades - i - 1) * 86400000,
        entryPrice,
        exitPrice,
        quantity: 100,
        profit: (exitPrice - entryPrice) * 100,
        type: 'long' as const
      });
    }
    
    return trades;
  }

  async loadInWorker(): Promise<void> {
    if (!this.worker && typeof Worker !== 'undefined') {
      // In production, create a worker for heavy computations
      // this.worker = new Worker(new URL('./wasm.worker.ts', import.meta.url));
    }
  }

  dispose(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.module = null;
    this.loadingPromise = null;
  }
}

// Export singleton instance
export const wasmLoader = new WasmLoader();