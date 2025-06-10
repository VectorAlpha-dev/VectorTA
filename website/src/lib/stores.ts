import { writable, derived } from 'svelte/store';

// WASM module store
export const wasmModule = writable<any>(null);

// Loading state for WASM
export const wasmLoading = writable<boolean>(false);

// Sample market data
export interface CandleData {
	time: string;
	open: number;
	high: number;
	low: number;
	close: number;
	volume: number;
}

// Generate realistic sample market data
export function generateSampleData(days: number = 100): CandleData[] {
	const data: CandleData[] = [];
	const basePrice = 50000; // Starting price like BTC
	let currentPrice = basePrice;
	
	const startDate = new Date();
	startDate.setDate(startDate.getDate() - days);
	
	for (let i = 0; i < days; i++) {
		const date = new Date(startDate);
		date.setDate(date.getDate() + i);
		
		// Generate realistic OHLC data
		const volatility = 0.02; // 2% daily volatility
		const change = (Math.random() - 0.5) * volatility;
		
		const open = currentPrice;
		const close = open * (1 + change);
		const high = Math.max(open, close) * (1 + Math.random() * 0.01);
		const low = Math.min(open, close) * (1 - Math.random() * 0.01);
		const volume = Math.random() * 1000000;
		
		data.push({
			time: date.toISOString().split('T')[0], // YYYY-MM-DD format
			open: parseFloat(open.toFixed(2)),
			high: parseFloat(high.toFixed(2)),
			low: parseFloat(low.toFixed(2)),
			close: parseFloat(close.toFixed(2)),
			volume: Math.floor(volume)
		});
		
		currentPrice = close;
	}
	
	return data;
}

// Indicator categories and metadata
export interface IndicatorInfo {
	id: string;
	name: string;
	category: string;
	description: string;
	parameters: Array<{
		name: string;
		type: string;
		default: any;
		description: string;
	}>;
}

export const indicatorCategories = writable<Record<string, IndicatorInfo[]>>({
	'moving-averages': [
		{ id: 'sma', name: 'Simple Moving Average (SMA)', category: 'moving-averages', description: 'Arithmetic mean of prices over a specified period', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'ema', name: 'Exponential Moving Average (EMA)', category: 'moving-averages', description: 'Weighted average giving more importance to recent prices', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'wma', name: 'Weighted Moving Average (WMA)', category: 'moving-averages', description: 'Weighted average with linearly decreasing weights', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'dema', name: 'Double Exponential Moving Average (DEMA)', category: 'moving-averages', description: 'Double smoothed EMA for reduced lag', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'tema', name: 'Triple Exponential Moving Average (TEMA)', category: 'moving-averages', description: 'Triple smoothed EMA for even less lag', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'hma', name: 'Hull Moving Average (HMA)', category: 'moving-averages', description: 'Fast and smooth moving average by Alan Hull', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'alma', name: 'Adaptive Linear Moving Average (ALMA)', category: 'moving-averages', description: 'Adaptive moving average with configurable responsiveness', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }, { name: 'sigma', type: 'number', default: 6, description: 'Smoothness factor' }, { name: 'offset', type: 'number', default: 0.85, description: 'Phase offset' }] },
		{ id: 'vwma', name: 'Volume Weighted Moving Average (VWMA)', category: 'moving-averages', description: 'Moving average weighted by volume', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'vwap', name: 'Volume Weighted Average Price (VWAP)', category: 'moving-averages', description: 'Average price weighted by volume from session start', parameters: [] },
		{ id: 'smma', name: 'Smoothed Moving Average (SMMA)', category: 'moving-averages', description: 'Smoothed moving average (Wilder\'s smoothing)', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'kama', name: 'Kaufman Adaptive Moving Average (KAMA)', category: 'moving-averages', description: 'Adaptive moving average that adjusts to market volatility', parameters: [{ name: 'period', type: 'number', default: 10, description: 'Efficiency ratio period' }, { name: 'fast_sc', type: 'number', default: 2, description: 'Fast smoothing constant' }, { name: 'slow_sc', type: 'number', default: 30, description: 'Slow smoothing constant' }] },
		{ id: 'jma', name: 'Jurik Moving Average (JMA)', category: 'moving-averages', description: 'Low-lag adaptive moving average', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }, { name: 'phase', type: 'number', default: 0, description: 'Phase parameter' }] },
		{ id: 'tilson', name: 'Tilson T3 Moving Average', category: 'moving-averages', description: 'Triple exponential smoothing with volume factor', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }, { name: 'vfactor', type: 'number', default: 0.7, description: 'Volume factor' }] },
		{ id: 'frama', name: 'Fractal Adaptive Moving Average (FRAMA)', category: 'moving-averages', description: 'Adaptive MA based on fractal geometry', parameters: [{ name: 'period', type: 'number', default: 16, description: 'Number of periods' }] },
		{ id: 'mama', name: 'MESA Adaptive Moving Average (MAMA)', category: 'moving-averages', description: 'Adaptive MA using MESA algorithm', parameters: [{ name: 'fast_limit', type: 'number', default: 0.5, description: 'Fast limit' }, { name: 'slow_limit', type: 'number', default: 0.05, description: 'Slow limit' }] },
		{ id: 'trima', name: 'Triangular Moving Average (TRIMA)', category: 'moving-averages', description: 'Double-smoothed moving average', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'zlema', name: 'Zero Lag Exponential Moving Average (ZLEMA)', category: 'moving-averages', description: 'EMA with zero lag using error correction', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'wilders', name: 'Wilder\'s Smoothing', category: 'moving-averages', description: 'Wilder\'s exponential smoothing method', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'sinwma', name: 'Sine Weighted Moving Average', category: 'moving-averages', description: 'Moving average weighted with sine function', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'linreg', name: 'Linear Regression Moving Average', category: 'moving-averages', description: 'Linear regression forecast as moving average', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'hwma', name: 'Holt-Winter Moving Average', category: 'moving-averages', description: 'Exponential smoothing with trend', parameters: [{ name: 'na', type: 'number', default: 20, description: 'Number of periods' }, { name: 'nb', type: 'number', default: 20, description: 'Trend periods' }] },
		{ id: 'pwma', name: 'Pascal Weighted Moving Average', category: 'moving-averages', description: 'Weighted MA using Pascal triangle weights', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'swma', name: 'Symmetric Weighted Moving Average', category: 'moving-averages', description: 'Symmetrically weighted moving average', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'supersmoother', name: 'Ehlers SuperSmoother', category: 'moving-averages', description: '2-pole SuperSmoother filter', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'supersmoother_3_pole', name: 'Ehlers 3-Pole SuperSmoother', category: 'moving-averages', description: '3-pole SuperSmoother filter', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'gaussian', name: 'Gaussian Filter', category: 'moving-averages', description: 'Gaussian smoothing filter', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }, { name: 'poles', type: 'number', default: 4, description: 'Number of poles' }] },
		{ id: 'highpass', name: 'Ehlers High Pass Filter', category: 'moving-averages', description: 'High-pass filter for cycle extraction', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Cutoff period' }] },
		{ id: 'highpass_2_pole', name: 'Ehlers 2-Pole High Pass Filter', category: 'moving-averages', description: '2-pole high-pass filter', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Cutoff period' }] },
		{ id: 'reflex', name: 'Ehlers Reflex', category: 'moving-averages', description: 'Reflex indicator for trend identification', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'trendflex', name: 'Ehlers TrendFlex', category: 'moving-averages', description: 'Flexible trend indicator', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'ehlers_itrend', name: 'Ehlers Instantaneous Trend', category: 'moving-averages', description: 'Instantaneous trend line', parameters: [{ name: 'alpha', type: 'number', default: 0.07, description: 'Alpha parameter' }] },
		{ id: 'vpwma', name: 'Variable Period Weighted Moving Average', category: 'moving-averages', description: 'WMA with variable period', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Base period' }] },
		{ id: 'cwma', name: 'Centered Weighted Moving Average', category: 'moving-averages', description: 'Centered WMA for better smoothing', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'sqwma', name: 'Square Weighted Moving Average', category: 'moving-averages', description: 'MA weighted by square of position', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'fwma', name: 'Fibonacci Weighted Moving Average', category: 'moving-averages', description: 'MA weighted by Fibonacci sequence', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'maaq', name: 'Moving Average Adaptive Q', category: 'moving-averages', description: 'Adaptive moving average with Q factor', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }, { name: 'fast_sc', type: 'number', default: 0.2, description: 'Fast smoothing' }, { name: 'slow_sc', type: 'number', default: 0.04, description: 'Slow smoothing' }] },
		{ id: 'epma', name: 'End Point Moving Average', category: 'moving-averages', description: 'Endpoint moving average', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'edcf', name: 'Ehlers Distance Coefficient Filter', category: 'moving-averages', description: 'Distance coefficient filter', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'jsa', name: 'Jurik Smoothing Average', category: 'moving-averages', description: 'Jurik smoothing algorithm', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }, { name: 'phase', type: 'number', default: 0, description: 'Phase parameter' }] },
		{ id: 'mwdx', name: 'Mesa Window Discrete Transform', category: 'moving-averages', description: 'MESA windowed DX transform', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'nma', name: 'Natural Moving Average', category: 'moving-averages', description: 'Natural logarithm based moving average', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'srwma', name: 'Square Root Weighted Moving Average', category: 'moving-averages', description: 'MA weighted by square root', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] }
	],
	'momentum': [
		{ id: 'rsi', name: 'Relative Strength Index (RSI)', category: 'momentum', description: 'Momentum oscillator measuring speed and change of price movements', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'macd', name: 'Moving Average Convergence Divergence (MACD)', category: 'momentum', description: 'Trend-following momentum indicator', parameters: [{ name: 'fast_period', type: 'number', default: 12, description: 'Fast EMA period' }, { name: 'slow_period', type: 'number', default: 26, description: 'Slow EMA period' }, { name: 'signal_period', type: 'number', default: 9, description: 'Signal line period' }] },
		{ id: 'stoch', name: 'Stochastic Oscillator', category: 'momentum', description: 'Compares closing price to price range over given time period', parameters: [{ name: 'k_period', type: 'number', default: 14, description: '%K period' }, { name: 'd_period', type: 'number', default: 3, description: '%D smoothing period' }] },
		{ id: 'stochf', name: 'Fast Stochastic', category: 'momentum', description: 'Fast version of stochastic oscillator', parameters: [{ name: 'k_period', type: 'number', default: 14, description: '%K period' }, { name: 'd_period', type: 'number', default: 3, description: '%D period' }] },
		{ id: 'roc', name: 'Rate of Change (ROC)', category: 'momentum', description: 'Measures percentage change in price from n periods ago', parameters: [{ name: 'period', type: 'number', default: 10, description: 'Number of periods' }] },
		{ id: 'rocp', name: 'Rate of Change Percentage', category: 'momentum', description: 'Rate of change expressed as percentage', parameters: [{ name: 'period', type: 'number', default: 10, description: 'Number of periods' }] },
		{ id: 'rocr', name: 'Rate of Change Ratio', category: 'momentum', description: 'Rate of change as ratio', parameters: [{ name: 'period', type: 'number', default: 10, description: 'Number of periods' }] },
		{ id: 'mom', name: 'Momentum', category: 'momentum', description: 'Difference between current and n-period ago price', parameters: [{ name: 'period', type: 'number', default: 10, description: 'Number of periods' }] },
		{ id: 'cci', name: 'Commodity Channel Index (CCI)', category: 'momentum', description: 'Measures deviation from statistical mean', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'willr', name: 'Williams %R', category: 'momentum', description: 'Momentum indicator showing overbought/oversold levels', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'srsi', name: 'Stochastic RSI', category: 'momentum', description: 'Stochastic oscillator applied to RSI', parameters: [{ name: 'rsi_period', type: 'number', default: 14, description: 'RSI period' }, { name: 'stoch_period', type: 'number', default: 14, description: 'Stochastic period' }, { name: 'k_period', type: 'number', default: 3, description: '%K smoothing' }, { name: 'd_period', type: 'number', default: 3, description: '%D smoothing' }] },
		{ id: 'cmo', name: 'Chande Momentum Oscillator (CMO)', category: 'momentum', description: 'Momentum oscillator using up/down price sums', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'ppo', name: 'Percentage Price Oscillator (PPO)', category: 'momentum', description: 'MACD in percentage terms', parameters: [{ name: 'fast_period', type: 'number', default: 12, description: 'Fast period' }, { name: 'slow_period', type: 'number', default: 26, description: 'Slow period' }, { name: 'signal_period', type: 'number', default: 9, description: 'Signal period' }] },
		{ id: 'apo', name: 'Absolute Price Oscillator (APO)', category: 'momentum', description: 'Difference between fast and slow moving averages', parameters: [{ name: 'fast_period', type: 'number', default: 12, description: 'Fast period' }, { name: 'slow_period', type: 'number', default: 26, description: 'Slow period' }] },
		{ id: 'aroonosc', name: 'Aroon Oscillator', category: 'momentum', description: 'Difference between Aroon Up and Aroon Down', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'fosc', name: 'Forecast Oscillator', category: 'momentum', description: 'Oscillator based on linear regression forecast', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'tsi', name: 'True Strength Index (TSI)', category: 'momentum', description: 'Momentum oscillator using double smoothed momentum', parameters: [{ name: 'first_smooth', type: 'number', default: 25, description: 'First smoothing period' }, { name: 'second_smooth', type: 'number', default: 13, description: 'Second smoothing period' }] },
		{ id: 'ultosc', name: 'Ultimate Oscillator', category: 'momentum', description: 'Momentum oscillator using multiple timeframes', parameters: [{ name: 'period1', type: 'number', default: 7, description: 'Short period' }, { name: 'period2', type: 'number', default: 14, description: 'Medium period' }, { name: 'period3', type: 'number', default: 28, description: 'Long period' }] },
		{ id: 'kdj', name: 'KDJ Indicator', category: 'momentum', description: 'Enhanced stochastic oscillator with J line', parameters: [{ name: 'k_period', type: 'number', default: 9, description: 'K period' }, { name: 'd_period', type: 'number', default: 3, description: 'D period' }] },
		{ id: 'stc', name: 'Schaff Trend Cycle (STC)', category: 'momentum', description: 'Cyclical indicator combining MACD and stochastic', parameters: [{ name: 'fast_period', type: 'number', default: 23, description: 'Fast MACD period' }, { name: 'slow_period', type: 'number', default: 50, description: 'Slow MACD period' }, { name: 'cycle_period', type: 'number', default: 10, description: 'Cycle period' }] },
		{ id: 'squeeze_momentum', name: 'Squeeze Momentum', category: 'momentum', description: 'Momentum during squeeze conditions', parameters: [{ name: 'bb_period', type: 'number', default: 20, description: 'Bollinger Bands period' }, { name: 'kc_period', type: 'number', default: 20, description: 'Keltner Channel period' }] },
		{ id: 'lrsi', name: 'Laguerre RSI', category: 'momentum', description: 'RSI using Laguerre filter', parameters: [{ name: 'gamma', type: 'number', default: 0.7, description: 'Gamma parameter' }] },
		{ id: 'rsx', name: 'Relative Strength Xtra (RSX)', category: 'momentum', description: 'Smoothed version of RSI', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'ift_rsi', name: 'Inverse Fisher Transform RSI', category: 'momentum', description: 'Fisher transform applied to RSI', parameters: [{ name: 'period', type: 'number', default: 9, description: 'RSI period' }] },
		{ id: 'cg', name: 'Center of Gravity', category: 'momentum', description: 'Center of gravity oscillator', parameters: [{ name: 'period', type: 'number', default: 10, description: 'Number of periods' }] },
		{ id: 'ao', name: 'Awesome Oscillator', category: 'momentum', description: 'Bill Williams Awesome Oscillator', parameters: [{ name: 'fast_period', type: 'number', default: 5, description: 'Fast period' }, { name: 'slow_period', type: 'number', default: 34, description: 'Slow period' }] },
		{ id: 'gatorosc', name: 'Gator Oscillator', category: 'momentum', description: 'Extension of Alligator indicator', parameters: [{ name: 'jaw_period', type: 'number', default: 13, description: 'Jaw period' }, { name: 'teeth_period', type: 'number', default: 8, description: 'Teeth period' }, { name: 'lips_period', type: 'number', default: 5, description: 'Lips period' }] },
		{ id: 'fisher', name: 'Fisher Transform', category: 'momentum', description: 'Transform prices to Gaussian distribution', parameters: [{ name: 'period', type: 'number', default: 10, description: 'Number of periods' }] },
		{ id: 'chande', name: 'Chande Forecast Oscillator', category: 'momentum', description: 'Forecast oscillator by Tushar Chande', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'dti', name: 'Directional Trend Index', category: 'momentum', description: 'Trend strength and direction indicator', parameters: [{ name: 'r_period', type: 'number', default: 14, description: 'R period' }, { name: 's_period', type: 'number', default: 10, description: 'S period' }] },
		{ id: 'er', name: 'Efficiency Ratio', category: 'momentum', description: 'Measures trend efficiency', parameters: [{ name: 'period', type: 'number', default: 10, description: 'Number of periods' }] },
		{ id: 'eri', name: 'Elder Ray Index', category: 'momentum', description: 'Elder Ray bull and bear power', parameters: [{ name: 'period', type: 'number', default: 13, description: 'EMA period' }] },
		{ id: 'kst', name: 'Know Sure Thing (KST)', category: 'momentum', description: 'Momentum oscillator by Martin Pring', parameters: [{ name: 'roc1', type: 'number', default: 10, description: 'First ROC period' }, { name: 'roc2', type: 'number', default: 15, description: 'Second ROC period' }, { name: 'roc3', type: 'number', default: 20, description: 'Third ROC period' }, { name: 'roc4', type: 'number', default: 30, description: 'Fourth ROC period' }] },
		{ id: 'pfe', name: 'Polarized Fractal Efficiency (PFE)', category: 'momentum', description: 'Efficiency of price movement', parameters: [{ name: 'period', type: 'number', default: 10, description: 'Number of periods' }] },
		{ id: 'qstick', name: 'Qstick Indicator', category: 'momentum', description: 'Candlestick momentum indicator', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'rsmk', name: 'Relative Strength Mansfield', category: 'momentum', description: 'Mansfield relative strength', parameters: [{ name: 'period', type: 'number', default: 10, description: 'Number of periods' }] },
		{ id: 'coppock', name: 'Coppock Curve', category: 'momentum', description: 'Long-term momentum indicator', parameters: [{ name: 'wma_period', type: 'number', default: 10, description: 'WMA period' }, { name: 'roc1_period', type: 'number', default: 14, description: 'First ROC period' }, { name: 'roc2_period', type: 'number', default: 11, description: 'Second ROC period' }] },
		{ id: 'wavetrend', name: 'WaveTrend', category: 'momentum', description: 'Wave trend oscillator', parameters: [{ name: 'channel_length', type: 'number', default: 10, description: 'Channel length' }, { name: 'average_length', type: 'number', default: 21, description: 'Average length' }] },
		{ id: 'msw', name: 'Mesa Sine Wave', category: 'momentum', description: 'MESA sine wave indicator', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'dec_osc', name: 'Detrended Ehlers Cycle Oscillator', category: 'momentum', description: 'Ehlers detrended cycle oscillator', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'acosc', name: 'Acceleration Oscillator', category: 'momentum', description: 'Acceleration/deceleration oscillator', parameters: [{ name: 'fast_period', type: 'number', default: 5, description: 'Fast period' }, { name: 'slow_period', type: 'number', default: 34, description: 'Slow period' }, { name: 'signal_period', type: 'number', default: 5, description: 'Signal period' }] },
		{ id: 'bop', name: 'Balance of Power', category: 'momentum', description: 'Measures buying vs selling pressure', parameters: [] },
		{ id: 'cfo', name: 'Chande Forecast Oscillator', category: 'momentum', description: 'Forecast oscillator implementation', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'correlation_cycle', name: 'Correlation Cycle', category: 'momentum', description: 'Correlation-based cycle indicator', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'voss', name: 'VOSS Predictor', category: 'momentum', description: 'VOSS predictive oscillator', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }, { name: 'predict', type: 'number', default: 3, description: 'Prediction length' }] },
		{ id: 'ttm_trend', name: 'TTM Trend', category: 'momentum', description: 'TTM trend indicator', parameters: [{ name: 'period', type: 'number', default: 6, description: 'Number of periods' }] },
		{ id: 'cksp', name: 'Chande Kroll Stop', category: 'momentum', description: 'Volatility stop by Chande and Kroll', parameters: [{ name: 'period', type: 'number', default: 10, description: 'Number of periods' }, { name: 'mult', type: 'number', default: 3, description: 'Multiplier' }] }
	],
	'volatility': [
		{ id: 'atr', name: 'Average True Range (ATR)', category: 'volatility', description: 'Measures market volatility using true range', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'natr', name: 'Normalized Average True Range (NATR)', category: 'volatility', description: 'ATR normalized by closing price', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'bollinger_bands', name: 'Bollinger Bands', category: 'volatility', description: 'Moving average with standard deviation bands', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Period for moving average' }, { name: 'std_dev', type: 'number', default: 2, description: 'Standard deviation multiplier' }] },
		{ id: 'bollinger_bands_width', name: 'Bollinger Bands Width', category: 'volatility', description: 'Width of Bollinger Bands', parameters: [{ name: 'period', type: 'number', default: 20, description: 'BB period' }, { name: 'std_dev', type: 'number', default: 2, description: 'Standard deviation' }] },
		{ id: 'keltner', name: 'Keltner Channels', category: 'volatility', description: 'Moving average with ATR-based bands', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Period' }, { name: 'multiplier', type: 'number', default: 2, description: 'ATR multiplier' }] },
		{ id: 'donchian', name: 'Donchian Channels', category: 'volatility', description: 'Highest high and lowest low channels', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'stddev', name: 'Standard Deviation', category: 'volatility', description: 'Statistical measure of price volatility', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'var', name: 'Variance', category: 'volatility', description: 'Statistical variance of price', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'chop', name: 'Choppiness Index', category: 'volatility', description: 'Measures market choppiness vs trending', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'ui', name: 'Ulcer Index', category: 'volatility', description: 'Measures downside volatility', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'damiani_volatmeter', name: 'Damiani Volatmeter', category: 'volatility', description: 'Advanced volatility measure', parameters: [{ name: 'vis_atr', type: 'number', default: 13, description: 'Viscosity ATR period' }, { name: 'vis_std', type: 'number', default: 20, description: 'Viscosity StdDev period' }, { name: 'sed_atr', type: 'number', default: 40, description: 'Sedimentation ATR period' }, { name: 'sed_std', type: 'number', default: 100, description: 'Sedimentation StdDev period' }] },
		{ id: 'mass', name: 'Mass Index', category: 'volatility', description: 'Identifies reversal points using high-low range', parameters: [{ name: 'fast_period', type: 'number', default: 9, description: 'Fast EMA period' }, { name: 'slow_period', type: 'number', default: 25, description: 'Slow EMA period' }] },
		{ id: 'rvi', name: 'Relative Volatility Index', category: 'volatility', description: 'RSI applied to standard deviation', parameters: [{ name: 'period', type: 'number', default: 10, description: 'Number of periods' }] },
		{ id: 'kurtosis', name: 'Kurtosis', category: 'volatility', description: 'Measures tail risk in price distribution', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'cvi', name: 'ChaikinVolatility Index', category: 'volatility', description: 'Volatility based on high-low range', parameters: [{ name: 'ema_period', type: 'number', default: 10, description: 'EMA period' }, { name: 'roc_period', type: 'number', default: 10, description: 'ROC period' }] },
		{ id: 'deviation', name: 'Mean Deviation', category: 'volatility', description: 'Average deviation from mean', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'bandpass', name: 'Ehlers Bandpass Filter', category: 'volatility', description: 'Bandpass filter for cycle extraction', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Period' }, { name: 'bandwidth', type: 'number', default: 0.3, description: 'Bandwidth' }] },
		{ id: 'decycler', name: 'Ehlers Decycler', category: 'volatility', description: 'High-pass filter removing cycles', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] }
	],
	'volume': [
		{ id: 'obv', name: 'On-Balance Volume (OBV)', category: 'volume', description: 'Cumulative volume based on price direction', parameters: [] },
		{ id: 'mfi', name: 'Money Flow Index (MFI)', category: 'volume', description: 'Volume-weighted RSI', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'ad', name: 'Accumulation/Distribution Line', category: 'volume', description: 'Cumulative flow of money into/out of security', parameters: [] },
		{ id: 'adosc', name: 'Chaikin A/D Oscillator', category: 'volume', description: 'MACD applied to A/D line', parameters: [{ name: 'fast_period', type: 'number', default: 3, description: 'Fast EMA period' }, { name: 'slow_period', type: 'number', default: 10, description: 'Slow EMA period' }] },
		{ id: 'emv', name: 'Ease of Movement', category: 'volume', description: 'Price movement relative to volume', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'vpt', name: 'Volume Price Trend (VPT)', category: 'volume', description: 'Cumulative volume adjusted by price change', parameters: [] },
		{ id: 'nvi', name: 'Negative Volume Index (NVI)', category: 'volume', description: 'Tracks price changes on low volume days', parameters: [] },
		{ id: 'pvi', name: 'Positive Volume Index (PVI)', category: 'volume', description: 'Tracks price changes on high volume days', parameters: [] },
		{ id: 'efi', name: 'Elder Force Index', category: 'volume', description: 'Volume and price change force indicator', parameters: [{ name: 'period', type: 'number', default: 13, description: 'Number of periods' }] },
		{ id: 'kvo', name: 'Klinger Volume Oscillator', category: 'volume', description: 'Volume-based momentum oscillator', parameters: [{ name: 'fast_period', type: 'number', default: 34, description: 'Fast EMA period' }, { name: 'slow_period', type: 'number', default: 55, description: 'Slow EMA period' }, { name: 'signal_period', type: 'number', default: 13, description: 'Signal line period' }] },
		{ id: 'marketefi', name: 'Market Facilitation Index', category: 'volume', description: 'Price movement per unit of volume', parameters: [] },
		{ id: 'vosc', name: 'Volume Oscillator', category: 'volume', description: 'Difference between two volume averages', parameters: [{ name: 'fast_period', type: 'number', default: 12, description: 'Fast period' }, { name: 'slow_period', type: 'number', default: 26, description: 'Slow period' }] },
		{ id: 'vwmacd', name: 'Volume Weighted MACD', category: 'volume', description: 'MACD weighted by volume', parameters: [{ name: 'fast_period', type: 'number', default: 12, description: 'Fast period' }, { name: 'slow_period', type: 'number', default: 26, description: 'Slow period' }, { name: 'signal_period', type: 'number', default: 9, description: 'Signal period' }] },
		{ id: 'wad', name: 'Williams Accumulation/Distribution', category: 'volume', description: 'Williams A/D line', parameters: [] },
		{ id: 'vpci', name: 'Volume Price Confirmation Indicator', category: 'volume', description: 'Confirms price moves with volume', parameters: [{ name: 'short_period', type: 'number', default: 5, description: 'Short period' }, { name: 'long_period', type: 'number', default: 25, description: 'Long period' }] },
		{ id: 'vlma', name: 'Volume Adjusted Moving Average', category: 'volume', description: 'Moving average adjusted by volume', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] }
	],
	'trend': [
		{ id: 'adx', name: 'Average Directional Index (ADX)', category: 'trend', description: 'Measures trend strength regardless of direction', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'adxr', name: 'Average Directional Index Rating (ADXR)', category: 'trend', description: 'Smoothed version of ADX', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'dx', name: 'Directional Movement Index (DX)', category: 'trend', description: 'Measures directional movement', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'di', name: 'Directional Indicator (+DI/-DI)', category: 'trend', description: 'Positive and negative directional indicators', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'dm', name: 'Directional Movement (+DM/-DM)', category: 'trend', description: 'Directional movement calculation', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'aroon', name: 'Aroon Indicator', category: 'trend', description: 'Identifies trend changes and strength', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'sar', name: 'Parabolic SAR', category: 'trend', description: 'Stop and reverse trend following system', parameters: [{ name: 'start', type: 'number', default: 0.02, description: 'Starting acceleration' }, { name: 'increment', type: 'number', default: 0.02, description: 'Acceleration increment' }, { name: 'maximum', type: 'number', default: 0.2, description: 'Maximum acceleration' }] },
		{ id: 'supertrend', name: 'SuperTrend', category: 'trend', description: 'Trend-following indicator using ATR', parameters: [{ name: 'period', type: 'number', default: 10, description: 'ATR period' }, { name: 'multiplier', type: 'number', default: 3, description: 'ATR multiplier' }] },
		{ id: 'trix', name: 'TRIX', category: 'trend', description: 'Triple exponentially smoothed moving average', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'dpo', name: 'Detrended Price Oscillator (DPO)', category: 'trend', description: 'Removes trend to highlight cycles', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'pma', name: 'Pinpoint Moving Average', category: 'trend', description: 'Precise moving average calculation', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'linearreg_slope', name: 'Linear Regression Slope', category: 'trend', description: 'Slope of linear regression line', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'linearreg_angle', name: 'Linear Regression Angle', category: 'trend', description: 'Angle of linear regression line', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'linearreg_intercept', name: 'Linear Regression Intercept', category: 'trend', description: 'Y-intercept of linear regression', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'tsf', name: 'Time Series Forecast (TSF)', category: 'trend', description: 'Linear regression forecast', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'alligator', name: 'Williams Alligator', category: 'trend', description: 'Three smoothed moving averages', parameters: [{ name: 'jaw_period', type: 'number', default: 13, description: 'Jaw period' }, { name: 'teeth_period', type: 'number', default: 8, description: 'Teeth period' }, { name: 'lips_period', type: 'number', default: 5, description: 'Lips period' }] },
		{ id: 'vi', name: 'Vortex Indicator', category: 'trend', description: 'Identifies trend reversals', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'vidya', name: 'Variable Index Dynamic Average (VIDYA)', category: 'trend', description: 'Adaptive moving average using volatility', parameters: [{ name: 'short_period', type: 'number', default: 9, description: 'Short period' }, { name: 'long_period', type: 'number', default: 20, description: 'Long period' }, { name: 'alpha', type: 'number', default: 0.2, description: 'Alpha parameter' }] },
		{ id: 'mab', name: 'Moving Average Bands', category: 'trend', description: 'Moving average with percentage bands', parameters: [{ name: 'period', type: 'number', default: 20, description: 'MA period' }, { name: 'percent', type: 'number', default: 5, description: 'Band percentage' }] },
		{ id: 'devstop', name: 'Deviation Stop', category: 'trend', description: 'Stop loss based on standard deviation', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }, { name: 'factor', type: 'number', default: 2, description: 'Standard deviation factor' }] },
		{ id: 'safezonestop', name: 'SafeZone Stop', category: 'trend', description: 'Volatility-based stop loss', parameters: [{ name: 'period', type: 'number', default: 22, description: 'Number of periods' }, { name: 'mult', type: 'number', default: 2.5, description: 'Multiplier' }] },
		{ id: 'kaufmanstop', name: 'Kaufman Stop', category: 'trend', description: 'Adaptive stop loss by Perry Kaufman', parameters: [{ name: 'period', type: 'number', default: 22, description: 'Number of periods' }, { name: 'mult', type: 'number', default: 2, description: 'Multiplier' }] },
		{ id: 'ht_trendline', name: 'Hilbert Transform Trendline', category: 'trend', description: 'Instantaneous trendline using Hilbert Transform', parameters: [] },
		{ id: 'ht_trendmode', name: 'Hilbert Transform Trend Mode', category: 'trend', description: 'Trend vs cycle mode detection', parameters: [] },
		{ id: 'ht_dcperiod', name: 'Hilbert Transform Dominant Cycle Period', category: 'trend', description: 'Dominant cycle period detection', parameters: [] },
		{ id: 'ht_dcphase', name: 'Hilbert Transform Dominant Cycle Phase', category: 'trend', description: 'Dominant cycle phase', parameters: [] },
		{ id: 'ht_phasor', name: 'Hilbert Transform Phasor Components', category: 'trend', description: 'In-phase and quadrature components', parameters: [] },
		{ id: 'ht_sine', name: 'Hilbert Transform Sine Wave', category: 'trend', description: 'Sine wave indicator', parameters: [] },
		{ id: 'pivot', name: 'Pivot Points', category: 'trend', description: 'Support and resistance levels', parameters: [{ name: 'high', type: 'number', default: 0, description: 'Previous high' }, { name: 'low', type: 'number', default: 0, description: 'Previous low' }, { name: 'close', type: 'number', default: 0, description: 'Previous close' }] },
		{ id: 'correl_hl', name: 'High-Low Correlation', category: 'trend', description: 'Correlation between high and low prices', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'emd', name: 'Empirical Mode Decomposition', category: 'trend', description: 'Decomposes signal into intrinsic modes', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'heikin_ashi_candles', name: 'Heikin Ashi Candles', category: 'trend', description: 'Modified candlestick chart for trend analysis', parameters: [] },
		{ id: 'zscore', name: 'Z-Score', category: 'trend', description: 'Standard score for trend analysis', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] }
	],
	'statistical': [
		{ id: 'avgprice', name: 'Average Price', category: 'statistical', description: 'Average of OHLC prices', parameters: [] },
		{ id: 'medprice', name: 'Median Price', category: 'statistical', description: 'Median of high and low', parameters: [] },
		{ id: 'wclprice', name: 'Weighted Close Price', category: 'statistical', description: 'Weighted close price (HLC/3)', parameters: [] },
		{ id: 'midpoint', name: 'Midpoint', category: 'statistical', description: 'Midpoint of price over period', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'midprice', name: 'Midprice', category: 'statistical', description: 'Midpoint of high-low range', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'minmax', name: 'Min-Max', category: 'statistical', description: 'Minimum and maximum values', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'mean_ad', name: 'Mean Absolute Deviation', category: 'statistical', description: 'Average absolute deviation from mean', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'medium_ad', name: 'Median Absolute Deviation', category: 'statistical', description: 'Median absolute deviation', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] }
	]
});