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

export const indicatorCategories: Record<string, IndicatorInfo[]> = {
	'moving-averages': [
		{ id: 'sma', name: 'Simple Moving Average (SMA)', category: 'moving-averages', description: 'Arithmetic mean of prices over a specified period', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'ema', name: 'Exponential Moving Average (EMA)', category: 'moving-averages', description: 'Weighted average giving more importance to recent prices', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		{ id: 'wma', name: 'Weighted Moving Average (WMA)', category: 'moving-averages', description: 'Weighted average with linearly decreasing weights', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Number of periods' }] },
		// Additional moving averages...
	],
	'momentum': [
		{ id: 'rsi', name: 'Relative Strength Index (RSI)', category: 'momentum', description: 'Momentum oscillator measuring speed and change of price movements', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'macd', name: 'Moving Average Convergence Divergence (MACD)', category: 'momentum', description: 'Trend-following momentum indicator', parameters: [{ name: 'fast_period', type: 'number', default: 12, description: 'Fast EMA period' }, { name: 'slow_period', type: 'number', default: 26, description: 'Slow EMA period' }, { name: 'signal_period', type: 'number', default: 9, description: 'Signal line period' }] },
		// Additional momentum indicators...
	],
	'volatility': [
		{ id: 'atr', name: 'Average True Range (ATR)', category: 'volatility', description: 'Measures market volatility using true range', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'bollinger_bands', name: 'Bollinger Bands', category: 'volatility', description: 'Moving average with standard deviation bands', parameters: [{ name: 'period', type: 'number', default: 20, description: 'Period for moving average' }, { name: 'std_dev', type: 'number', default: 2, description: 'Standard deviation multiplier' }] },
		// Additional volatility indicators...
	],
	'volume': [
		{ id: 'obv', name: 'On-Balance Volume (OBV)', category: 'volume', description: 'Cumulative volume based on price direction', parameters: [] },
		{ id: 'mfi', name: 'Money Flow Index (MFI)', category: 'volume', description: 'Volume-weighted RSI', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		// Additional volume indicators...
	],
	'trend': [
		{ id: 'adx', name: 'Average Directional Index (ADX)', category: 'trend', description: 'Measures trend strength regardless of direction', parameters: [{ name: 'period', type: 'number', default: 14, description: 'Number of periods' }] },
		{ id: 'sar', name: 'Parabolic SAR', category: 'trend', description: 'Stop and reverse trend following system', parameters: [{ name: 'start', type: 'number', default: 0.02, description: 'Starting acceleration' }, { name: 'increment', type: 'number', default: 0.02, description: 'Acceleration increment' }, { name: 'maximum', type: 'number', default: 0.2, description: 'Maximum acceleration' }] },
		// Additional trend indicators...
	]
};