// WASM integration utilities for Rust-Backtester

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
	calculate_atr?: (highs: Float64Array, lows: Float64Array, closes: Float64Array, period: number) => Float64Array;
	calculate_stochastic?: (highs: Float64Array, lows: Float64Array, closes: Float64Array, k_period: number, d_period: number) => {
		k: Float64Array;
		d: Float64Array;
	};
	// Add more indicator functions as needed
}

let wasmModuleCache: WasmModule | null = null;
let isLoading = false;

export async function loadWasmModule(): Promise<WasmModule | null> {
	if (wasmModuleCache) {
		return wasmModuleCache;
	}

	if (isLoading) {
		// Wait for ongoing load
		while (isLoading) {
			await new Promise(resolve => setTimeout(resolve, 10));
		}
		return wasmModuleCache;
	}

	try {
		isLoading = true;
		
		// Note: This assumes the WASM package is built and available
		// In a real implementation, this would import from the generated WASM package
		// const wasmModule = await import('../../pkg');
		// await wasmModule.default(); // Initialize WASM
		
		// For now, return a mock module
		const mockModule: WasmModule = {
			calculate_rsi: (data: Float64Array, period: number) => {
				// Mock RSI calculation
				return new Float64Array(data.length).fill(50);
			},
			calculate_sma: (data: Float64Array, period: number) => {
				// Mock SMA calculation
				return new Float64Array(data.length).fill(data[data.length - 1]);
			}
		};
		
		wasmModuleCache = mockModule;
		return wasmModuleCache;
	} catch (error) {
		console.error('Failed to load WASM module:', error);
		return null;
	} finally {
		isLoading = false;
	}
}

export function getWasmModule(): WasmModule | null {
	return wasmModuleCache;
}

export function isWasmLoaded(): boolean {
	return wasmModuleCache !== null;
}