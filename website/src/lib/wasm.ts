// WASM integration utilities for Rust-Backtester

import { wasmModule, wasmLoading } from './stores';

export interface WasmModule {
	// Placeholder for actual WASM exports
	// These will be replaced with real functions when WASM is built
	calculate_rsi?: (data: Float64Array, period: number) => Float64Array;
	calculate_sma?: (data: Float64Array, period: number) => Float64Array;
	calculate_ema?: (data: Float64Array, period: number) => Float64Array;
	calculate_macd?: (data: Float64Array, fast: number, slow: number, signal: number) => {
		macd: Float64Array;
		signal: Float64Array;
		histogram: Float64Array;
	};
	calculate_bollinger_bands?: (data: Float64Array, period: number, std_dev: number) => {
		upper: Float64Array;
		middle: Float64Array;
		lower: Float64Array;
	};
	calculate_atr?: (high: Float64Array, low: Float64Array, close: Float64Array, period: number) => Float64Array;
	calculate_stoch?: (high: Float64Array, low: Float64Array, close: Float64Array, k_period: number, d_period: number) => {
		k: Float64Array;
		d: Float64Array;
	};
	calculate_adx?: (high: Float64Array, low: Float64Array, close: Float64Array, period: number) => Float64Array;
	calculate_obv?: (close: Float64Array, volume: Float64Array) => Float64Array;
	calculate_mfi?: (high: Float64Array, low: Float64Array, close: Float64Array, volume: Float64Array, period: number) => Float64Array;
}

let wasmModuleCache: WasmModule | null = null;

/**
 * Load the WASM module
 * This function will be updated when the actual WASM build is available
 */
export async function loadWasmModule(): Promise<WasmModule | null> {
	try {
		wasmLoading.set(true);
		
		// For now, we'll simulate loading
		// In the future, this will be:
		// const wasm = await import('../../../pkg/rust_backtester.js');
		// await wasm.default();
		// wasmModuleCache = wasm;
		
		console.log('WASM module loading simulation...');
		
		// Simulate async loading
		await new Promise(resolve => setTimeout(resolve, 1000));
		
		// Create mock module for development
		wasmModuleCache = createMockWasmModule();
		
		wasmModule.set(wasmModuleCache);
		console.log('WASM module loaded successfully (mock)');
		
		return wasmModuleCache;
	} catch (error) {
		console.error('Failed to load WASM module:', error);
		return null;
	} finally {
		wasmLoading.set(false);
	}
}

/**
 * Get the cached WASM module
 */
export function getWasmModule(): WasmModule | null {
	return wasmModuleCache;
}

/**
 * Create a mock WASM module for development
 * This will be removed when real WASM is available
 */
function createMockWasmModule(): WasmModule {
	return {
		calculate_rsi: (data: Float64Array, period: number = 14): Float64Array => {
			const result = new Float64Array(data.length);
			
			// Simple RSI calculation simulation
			for (let i = period; i < data.length; i++) {
				// Mock RSI calculation - replace with actual algorithm
				const gain = Math.max(0, data[i] - data[i - 1]);
				const loss = Math.max(0, data[i - 1] - data[i]);
				const avgGain = gain; // Simplified
				const avgLoss = loss || 0.01; // Avoid division by zero
				const rs = avgGain / avgLoss;
				result[i] = 100 - (100 / (1 + rs));
			}
			
			return result;
		},
		
		calculate_sma: (data: Float64Array, period: number = 20): Float64Array => {
			const result = new Float64Array(data.length);
			
			for (let i = period - 1; i < data.length; i++) {
				let sum = 0;
				for (let j = 0; j < period; j++) {
					sum += data[i - j];
				}
				result[i] = sum / period;
			}
			
			return result;
		},
		
		calculate_ema: (data: Float64Array, period: number = 20): Float64Array => {
			const result = new Float64Array(data.length);
			const multiplier = 2 / (period + 1);
			
			// Start with SMA for first value
			let sum = 0;
			for (let i = 0; i < period; i++) {
				sum += data[i];
			}
			result[period - 1] = sum / period;
			
			// Calculate EMA for remaining values
			for (let i = period; i < data.length; i++) {
				result[i] = (data[i] * multiplier) + (result[i - 1] * (1 - multiplier));
			}
			
			return result;
		},
		
		calculate_macd: (data: Float64Array, fast: number = 12, slow: number = 26, signal: number = 9) => {
			const fastEma = createMockWasmModule().calculate_ema!(data, fast);
			const slowEma = createMockWasmModule().calculate_ema!(data, slow);
			
			const macd = new Float64Array(data.length);
			for (let i = 0; i < data.length; i++) {
				macd[i] = fastEma[i] - slowEma[i];
			}
			
			const signalLine = createMockWasmModule().calculate_ema!(macd, signal);
			const histogram = new Float64Array(data.length);
			
			for (let i = 0; i < data.length; i++) {
				histogram[i] = macd[i] - signalLine[i];
			}
			
			return { macd, signal: signalLine, histogram };
		},
		
		calculate_bollinger_bands: (data: Float64Array, period: number = 20, std_dev: number = 2) => {
			const middle = createMockWasmModule().calculate_sma!(data, period);
			const upper = new Float64Array(data.length);
			const lower = new Float64Array(data.length);
			
			for (let i = period - 1; i < data.length; i++) {
				// Calculate standard deviation
				let sum = 0;
				for (let j = 0; j < period; j++) {
					sum += Math.pow(data[i - j] - middle[i], 2);
				}
				const stdDev = Math.sqrt(sum / period);
				
				upper[i] = middle[i] + (std_dev * stdDev);
				lower[i] = middle[i] - (std_dev * stdDev);
			}
			
			return { upper, middle, lower };
		},
		
		calculate_atr: (high: Float64Array, low: Float64Array, close: Float64Array, period: number = 14): Float64Array => {
			const result = new Float64Array(close.length);
			const trueRanges = new Float64Array(close.length);
			
			// Calculate True Range
			for (let i = 1; i < close.length; i++) {
				const hl = high[i] - low[i];
				const hc = Math.abs(high[i] - close[i - 1]);
				const lc = Math.abs(low[i] - close[i - 1]);
				trueRanges[i] = Math.max(hl, hc, lc);
			}
			
			// Calculate ATR as SMA of True Range
			for (let i = period; i < close.length; i++) {
				let sum = 0;
				for (let j = 0; j < period; j++) {
					sum += trueRanges[i - j];
				}
				result[i] = sum / period;
			}
			
			return result;
		}
	};
}

/**
 * Helper function to convert array to Float64Array
 */
export function toFloat64Array(data: number[]): Float64Array {
	return new Float64Array(data);
}

/**
 * Helper function to convert Float64Array to regular array
 */
export function fromFloat64Array(data: Float64Array): number[] {
	return Array.from(data);
}

/**
 * Extract price arrays from candlestick data
 */
export function extractPriceArrays(data: Array<{open: number, high: number, low: number, close: number, volume: number}>) {
	return {
		open: toFloat64Array(data.map(d => d.open)),
		high: toFloat64Array(data.map(d => d.high)),
		low: toFloat64Array(data.map(d => d.low)),
		close: toFloat64Array(data.map(d => d.close)),
		volume: toFloat64Array(data.map(d => d.volume))
	};
}