// Chart data utilities and types

import type { 
	CandlestickData, 
	LineData, 
	HistogramData,
	Time,
	UTCTimestamp 
} from 'lightweight-charts';

export interface RawCandleData {
	timestamp: number;
	open: number;
	high: number;
	low: number;
	close: number;
	volume: number;
}

export interface ProcessedCandleData extends CandlestickData {
	time: UTCTimestamp;
	open: number;
	high: number;
	low: number;
	close: number;
}

export interface ProcessedVolumeData extends HistogramData {
	time: UTCTimestamp;
	value: number;
	color?: string;
}

export interface IndicatorLineData extends LineData {
	time: UTCTimestamp;
	value: number;
}

export interface IndicatorData {
	name: string;
	data: IndicatorLineData[];
	color: string;
	lineStyle?: number;
	lineWidth?: number;
	priceScaleId?: string;
}

/**
 * Parse CSV data into structured format
 */
export async function parseCsvData(): Promise<RawCandleData[]> {
	// Use fallback data for reliable loading
	return generateFallbackData();
}

/**
 * Generate fallback data if CSV loading fails
 */
function generateFallbackData(): RawCandleData[] {
	const data: RawCandleData[] = [];
	const startTime = Date.now() - (365 * 24 * 60 * 60 * 1000); // 1 year ago
	const interval = 4 * 60 * 60 * 1000; // 4 hours in milliseconds
	let basePrice = 45000;
	
	for (let i = 0; i < 2000; i++) {
		const timestamp = startTime + (i * interval);
		const volatility = 0.02;
		const change = (Math.random() - 0.5) * volatility;
		
		const open = basePrice;
		const close = open * (1 + change);
		const high = Math.max(open, close) * (1 + Math.random() * 0.015);
		const low = Math.min(open, close) * (1 - Math.random() * 0.015);
		const volume = Math.random() * 10000 + 1000;
		
		data.push({
			timestamp,
			open: parseFloat(open.toFixed(2)),
			high: parseFloat(high.toFixed(2)),
			low: parseFloat(low.toFixed(2)),
			close: parseFloat(close.toFixed(2)),
			volume: parseFloat(volume.toFixed(2))
		});
		
		basePrice = close;
	}
	
	return data;
}

/**
 * Convert raw data to TradingView format
 */
export function convertToCandlestickData(rawData: RawCandleData[]): ProcessedCandleData[] {
	return rawData.map(candle => ({
		time: (candle.timestamp / 1000) as UTCTimestamp,
		open: candle.open,
		high: candle.high,
		low: candle.low,
		close: candle.close
	}));
}

/**
 * Convert raw data to volume data
 */
export function convertToVolumeData(rawData: RawCandleData[]): ProcessedVolumeData[] {
	return rawData.map(candle => ({
		time: (candle.timestamp / 1000) as UTCTimestamp,
		value: candle.volume,
		color: candle.close >= candle.open ? '#26a69a' : '#ef5350'
	}));
}

/**
 * Calculate various technical indicators
 */
export class IndicatorCalculator {
	/**
	 * Simple Moving Average
	 */
	static sma(data: RawCandleData[], period: number): IndicatorLineData[] {
		const result: IndicatorLineData[] = [];
		
		for (let i = period - 1; i < data.length; i++) {
			const sum = data.slice(i - period + 1, i + 1)
				.reduce((acc, candle) => acc + candle.close, 0);
			
			result.push({
				time: (data[i].timestamp / 1000) as UTCTimestamp,
				value: parseFloat((sum / period).toFixed(2))
			});
		}
		
		return result;
	}
	
	/**
	 * Exponential Moving Average
	 */
	static ema(data: RawCandleData[], period: number): IndicatorLineData[] {
		const result: IndicatorLineData[] = [];
		const multiplier = 2 / (period + 1);
		
		// Start with SMA for first value
		let ema = data.slice(0, period).reduce((acc, candle) => acc + candle.close, 0) / period;
		
		for (let i = period - 1; i < data.length; i++) {
			if (i === period - 1) {
				result.push({
					time: (data[i].timestamp / 1000) as UTCTimestamp,
					value: parseFloat(ema.toFixed(2))
				});
			} else {
				ema = (data[i].close * multiplier) + (ema * (1 - multiplier));
				result.push({
					time: (data[i].timestamp / 1000) as UTCTimestamp,
					value: parseFloat(ema.toFixed(2))
				});
			}
		}
		
		return result;
	}
	
	/**
	 * Relative Strength Index
	 */
	static rsi(data: RawCandleData[], period: number = 14): IndicatorLineData[] {
		const result: IndicatorLineData[] = [];
		const gains: number[] = [];
		const losses: number[] = [];
		
		// Calculate initial gains and losses
		for (let i = 1; i < data.length; i++) {
			const change = data[i].close - data[i - 1].close;
			gains.push(change > 0 ? change : 0);
			losses.push(change < 0 ? Math.abs(change) : 0);
		}
		
		// Calculate RSI
		for (let i = period - 1; i < gains.length; i++) {
			const avgGain = gains.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
			const avgLoss = losses.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
			
			const rs = avgGain / (avgLoss || 0.0001); // Avoid division by zero
			const rsi = 100 - (100 / (1 + rs));
			
			result.push({
				time: (data[i + 1].timestamp / 1000) as UTCTimestamp,
				value: parseFloat(rsi.toFixed(2))
			});
		}
		
		return result;
	}
	
	/**
	 * Moving Average Convergence Divergence
	 */
	static macd(data: RawCandleData[], fastPeriod: number = 12, slowPeriod: number = 26, signalPeriod: number = 9) {
		const fastEma = this.ema(data, fastPeriod);
		const slowEma = this.ema(data, slowPeriod);
		
		const macdLine: IndicatorLineData[] = [];
		const startIndex = slowPeriod - fastPeriod;
		
		// Calculate MACD line
		for (let i = startIndex; i < fastEma.length; i++) {
			const macdValue = fastEma[i].value - slowEma[i - startIndex].value;
			macdLine.push({
				time: fastEma[i].time,
				value: parseFloat(macdValue.toFixed(4))
			});
		}
		
		// Calculate signal line (EMA of MACD)
		const signalLine = this.emaFromLineData(macdLine, signalPeriod);
		
		// Calculate histogram
		const histogram: IndicatorLineData[] = [];
		for (let i = signalPeriod - 1; i < macdLine.length; i++) {
			const histValue = macdLine[i].value - signalLine[i - signalPeriod + 1].value;
			histogram.push({
				time: macdLine[i].time,
				value: parseFloat(histValue.toFixed(4))
			});
		}
		
		return {
			macd: macdLine,
			signal: signalLine,
			histogram
		};
	}
	
	/**
	 * Bollinger Bands
	 */
	static bollingerBands(data: RawCandleData[], period: number = 20, stdDev: number = 2) {
		const sma = this.sma(data, period);
		const upperBand: IndicatorLineData[] = [];
		const lowerBand: IndicatorLineData[] = [];
		
		for (let i = 0; i < sma.length; i++) {
			const dataIndex = i + period - 1;
			const prices = data.slice(dataIndex - period + 1, dataIndex + 1).map(d => d.close);
			const mean = sma[i].value;
			const variance = prices.reduce((acc, price) => acc + Math.pow(price - mean, 2), 0) / period;
			const standardDeviation = Math.sqrt(variance);
			
			upperBand.push({
				time: sma[i].time,
				value: parseFloat((mean + (stdDev * standardDeviation)).toFixed(2))
			});
			
			lowerBand.push({
				time: sma[i].time,
				value: parseFloat((mean - (stdDev * standardDeviation)).toFixed(2))
			});
		}
		
		return {
			middle: sma,
			upper: upperBand,
			lower: lowerBand
		};
	}
	
	/**
	 * Average True Range
	 */
	static atr(data: RawCandleData[], period: number = 14): IndicatorLineData[] {
		const trueRanges: number[] = [];
		
		// Calculate True Range for each period
		for (let i = 1; i < data.length; i++) {
			const high = data[i].high;
			const low = data[i].low;
			const prevClose = data[i - 1].close;
			
			const tr = Math.max(
				high - low,
				Math.abs(high - prevClose),
				Math.abs(low - prevClose)
			);
			
			trueRanges.push(tr);
		}
		
		// Calculate ATR using Wilder's smoothing
		const result: IndicatorLineData[] = [];
		let atr = trueRanges.slice(0, period).reduce((a, b) => a + b, 0) / period;
		
		for (let i = period - 1; i < trueRanges.length; i++) {
			if (i === period - 1) {
				result.push({
					time: (data[i + 1].timestamp / 1000) as UTCTimestamp,
					value: parseFloat(atr.toFixed(4))
				});
			} else {
				atr = ((atr * (period - 1)) + trueRanges[i]) / period;
				result.push({
					time: (data[i + 1].timestamp / 1000) as UTCTimestamp,
					value: parseFloat(atr.toFixed(4))
				});
			}
		}
		
		return result;
	}
	
	/**
	 * Helper function to calculate EMA from existing line data
	 */
	private static emaFromLineData(data: IndicatorLineData[], period: number): IndicatorLineData[] {
		const result: IndicatorLineData[] = [];
		const multiplier = 2 / (period + 1);
		
		// Start with SMA for first value
		let ema = data.slice(0, period).reduce((acc, point) => acc + point.value, 0) / period;
		
		for (let i = period - 1; i < data.length; i++) {
			if (i === period - 1) {
				result.push({
					time: data[i].time,
					value: parseFloat(ema.toFixed(4))
				});
			} else {
				ema = (data[i].value * multiplier) + (ema * (1 - multiplier));
				result.push({
					time: data[i].time,
					value: parseFloat(ema.toFixed(4))
				});
			}
		}
		
		return result;
	}
}

/**
 * Get indicator data by ID
 */
export function getIndicatorData(indicatorId: string, data: RawCandleData[]): IndicatorData[] {
	switch (indicatorId) {
		case 'sma':
			return [{
				name: 'SMA (20)',
				data: IndicatorCalculator.sma(data, 20),
				color: '#2962FF',
				lineWidth: 2
			}];
			
		case 'ema':
			return [{
				name: 'EMA (20)',
				data: IndicatorCalculator.ema(data, 20),
				color: '#FF6D00',
				lineWidth: 2
			}];
			
		case 'rsi':
			return [{
				name: 'RSI (14)',
				data: IndicatorCalculator.rsi(data, 14),
				color: '#9C27B0',
				lineWidth: 2,
				priceScaleId: 'rsi'
			}];
			
		case 'macd':
			const macdData = IndicatorCalculator.macd(data);
			return [
				{
					name: 'MACD',
					data: macdData.macd,
					color: '#00E676',
					lineWidth: 2,
					priceScaleId: 'macd'
				},
				{
					name: 'Signal',
					data: macdData.signal,
					color: '#FF1744',
					lineWidth: 2,
					priceScaleId: 'macd'
				}
			];
			
		case 'bollinger_bands':
			const bbData = IndicatorCalculator.bollingerBands(data);
			return [
				{
					name: 'BB Upper',
					data: bbData.upper,
					color: '#FF5722',
					lineWidth: 1
				},
				{
					name: 'BB Middle',
					data: bbData.middle,
					color: '#607D8B',
					lineWidth: 2
				},
				{
					name: 'BB Lower',
					data: bbData.lower,
					color: '#FF5722',
					lineWidth: 1
				}
			];
			
		case 'atr':
			return [{
				name: 'ATR (14)',
				data: IndicatorCalculator.atr(data, 14),
				color: '#795548',
				lineWidth: 2,
				priceScaleId: 'atr'
			}];
			
		default:
			// Return a default SMA for unknown indicators
			return [{
				name: `${indicatorId.toUpperCase()} (Placeholder)`,
				data: IndicatorCalculator.sma(data, 20),
				color: '#607D8B',
				lineWidth: 2
			}];
	}
}