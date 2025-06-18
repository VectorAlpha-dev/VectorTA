// WASM module types for VectorTA
export interface WasmModule {
  // Indicator calculation functions
  calculate_sma: (data: Float64Array, period: number) => Float64Array;
  calculate_ema: (data: Float64Array, period: number) => Float64Array;
  calculate_rsi: (data: Float64Array, period: number) => Float64Array;
  calculate_macd: (data: Float64Array, fastPeriod: number, slowPeriod: number, signalPeriod: number) => MacdResult;
  calculate_bollinger_bands: (data: Float64Array, period: number, stdDev: number) => BollingerBandsResult;
  
  // Backtesting functions
  run_backtest: (config: BacktestConfig) => BacktestResult;
  
  // Memory management
  allocate_array: (size: number) => number;
  free_array: (ptr: number) => void;
  
  // Performance tracking
  get_last_calculation_time: () => number;
  get_operations_per_second: () => number;
}

export interface MacdResult {
  macd: Float64Array;
  signal: Float64Array;
  histogram: Float64Array;
}

export interface BollingerBandsResult {
  upper: Float64Array;
  middle: Float64Array;
  lower: Float64Array;
}

export interface BacktestConfig {
  strategy: 'sma_crossover' | 'rsi_threshold' | 'macd_signal';
  data: Float64Array;
  parameters: Record<string, number>;
  initialCapital: number;
  positionSize: number;
  commission: number;
}

export interface BacktestResult {
  equity: Float64Array;
  trades: Trade[];
  metrics: PerformanceMetrics;
  executionTime: number;
}

export interface Trade {
  entryTime: number;
  exitTime: number;
  entryPrice: number;
  exitPrice: number;
  quantity: number;
  profit: number;
  type: 'long' | 'short';
}

export interface PerformanceMetrics {
  totalReturn: number;
  annualizedReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
}