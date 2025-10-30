#!/usr/bin/env node
import { describe, it } from 'node:test';
import assert from 'node:assert';
import * as wasm from '../../pkg/my_project.js';
import { loadTestData, assertArrayClose } from './test_utils.js';

describe('VWMACD WASM', () => {
    // Test data
    const close = [
        42.15, 42.25, 42.35, 42.45, 42.55, 42.65, 42.75, 42.85, 42.95, 43.05,
        43.15, 43.25, 43.35, 43.45, 43.55, 43.65, 43.75, 43.85, 43.95, 44.05,
        44.15, 44.25, 44.35, 44.45, 44.55, 44.65, 44.75, 44.85, 44.95, 45.05,
        45.15, 45.25, 45.35, 45.45, 45.55
    ];
    
    const volume = [
        1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
        2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900,
        3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900,
        4000, 4100, 4200, 4300, 4400
    ];

    describe('Safe API', () => {
        it('matches Rust reference last5 on CSV (tol=2e-4)', () => {
            const candles = loadTestData();
            const close = candles.close;
            const volume = candles.volume;

            const out = wasm.vwmacd_js(close, volume, 12, 26, 9, "sma", "sma", "ema");

            const len = close.length;
            const macd = out.slice(0, len);
            const signal = out.slice(len, len * 2);
            const hist = out.slice(len * 2);

            const last5 = (arr) => arr.slice(arr.length - 5);

            const expected_macd_last5 = [
                -394.95161155,
                -508.29106210,
                -490.70190723,
                -388.94996199,
                -341.13720646,
            ];
            const expected_signal_last5 = [
                -539.48861567,
                -533.24910496,
                -524.73966541,
                -497.58172247,
                -466.29282108,
            ];
            const expected_hist_last5 = [
                144.53700412,
                24.95804286,
                34.03775818,
                108.63176274,
                125.15561462,
            ];

            const tol = 2e-4; // must not exceed Rust tolerance
            assertArrayClose(last5(macd), expected_macd_last5, tol, 'macd last5');
            assertArrayClose(last5(signal), expected_signal_last5, tol, 'signal last5');
            assertArrayClose(last5(hist), expected_hist_last5, tol, 'hist last5');
        });
        it('should calculate VWMACD with default parameters', () => {
            const result = wasm.vwmacd_js(close, volume, 12, 26, 9, "sma", "sma", "ema");
            
            // Result should be 3x the input length (macd, signal, hist)
            assert.strictEqual(result.length, close.length * 3);
            
            // Extract the three components
            const macd = result.slice(0, close.length);
            const signal = result.slice(close.length, close.length * 2);
            const hist = result.slice(close.length * 2);
            
            // Check first values should be NaN for warmup period
            assert(isNaN(macd[0]));
            assert(isNaN(signal[0]));
            assert(isNaN(hist[0]));
            
            // Check later values are numbers
            // With default parameters (12, 26, 9), total warmup is 33
            // So only indices 33 and 34 have valid signal/hist values
            assert(!isNaN(macd[30]));
            assert(!isNaN(signal[34]));
            assert(!isNaN(hist[34]));
            
            // Histogram should be macd - signal
            const tolerance = 1e-10;
            assert(Math.abs(hist[34] - (macd[34] - signal[34])) < tolerance);
        });

        it('should handle different MA types', () => {
            const result1 = wasm.vwmacd_js(close, volume, 12, 26, 9, "sma", "sma", "ema");
            const result2 = wasm.vwmacd_js(close, volume, 12, 26, 9, "ema", "ema", "sma");
            
            // Results should be different with different MA types
            assert.notDeepStrictEqual(result1, result2);
        });

        it('should handle edge cases', () => {
            // Empty arrays
            assert.throws(() => wasm.vwmacd_js([], [], 12, 26, 9, "sma", "sma", "ema"));
            
            // Mismatched lengths
            assert.throws(() => wasm.vwmacd_js(close, volume.slice(0, 10), 12, 26, 9, "sma", "sma", "ema"));
            
            // Invalid period
            assert.throws(() => wasm.vwmacd_js(close, volume, 0, 26, 9, "sma", "sma", "ema"));
            assert.throws(() => wasm.vwmacd_js(close, volume, 12, 0, 9, "sma", "sma", "ema"));
            assert.throws(() => wasm.vwmacd_js(close, volume, 12, 26, 0, "sma", "sma", "ema"));
        });
    });

    describe('Fast API', () => {
        it('should calculate VWMACD using pointers', () => {
            const len = close.length;
            
            // Allocate memory
            const closePtr = wasm.vwmacd_alloc(len);
            const volumePtr = wasm.vwmacd_alloc(len);
            const macdPtr = wasm.vwmacd_alloc(len);
            const signalPtr = wasm.vwmacd_alloc(len);
            const histPtr = wasm.vwmacd_alloc(len);
            
            try {
                // Copy data to WASM memory
                const memory = new Float64Array(wasm.__wasm.memory.buffer);
                memory.set(close, closePtr / 8);
                memory.set(volume, volumePtr / 8);
                
                // Calculate
                wasm.vwmacd_into(closePtr, volumePtr, macdPtr, signalPtr, histPtr, len, 12, 26, 9, "sma", "sma", "ema");
                
                // Re-create memory view after potential growth
                const memoryAfter = new Float64Array(wasm.__wasm.memory.buffer);
                
                // Read results
                const macd = Array.from(memoryAfter.slice(macdPtr / 8, macdPtr / 8 + len));
                const signal = Array.from(memoryAfter.slice(signalPtr / 8, signalPtr / 8 + len));
                const hist = Array.from(memoryAfter.slice(histPtr / 8, histPtr / 8 + len));
                
                // Verify results match safe API
                const safeResult = wasm.vwmacd_js(close, volume, 12, 26, 9, "sma", "sma", "ema");
                const safeMacd = Array.from(safeResult.slice(0, len));
                const safeSignal = Array.from(safeResult.slice(len, len * 2));
                const safeHist = Array.from(safeResult.slice(len * 2));
                
                assert.deepStrictEqual(macd, safeMacd);
                assert.deepStrictEqual(signal, safeSignal);
                assert.deepStrictEqual(hist, safeHist);
            } finally {
                // Clean up
                wasm.vwmacd_free(closePtr, len);
                wasm.vwmacd_free(volumePtr, len);
                wasm.vwmacd_free(macdPtr, len);
                wasm.vwmacd_free(signalPtr, len);
                wasm.vwmacd_free(histPtr, len);
            }
        });

        it('should handle aliasing correctly', () => {
            const len = close.length;

            // Allocate memory
            const closePtr = wasm.vwmacd_alloc(len);
            const volumePtr = wasm.vwmacd_alloc(len);
            const outPtr = wasm.vwmacd_alloc(len); // alias outputs only

            try {
                // Copy data to WASM memory
                const memory = new Float64Array(wasm.__wasm.memory.buffer);
                memory.set(close, closePtr / 8);
                memory.set(volume, volumePtr / 8);

                // Calculate with aliasing across outputs (safe): macd/signal/hist -> same buffer
                wasm.vwmacd_into(closePtr, volumePtr, outPtr, outPtr, outPtr, len, 12, 26, 9, "sma", "sma", "ema");

                // Re-create memory view after potential growth
                const memoryAfter = new Float64Array(wasm.__wasm.memory.buffer);

                // Read result (should be histogram values since it's written last)
                const result = Array.from(memoryAfter.slice(outPtr / 8, outPtr / 8 + len));

                // Should not crash and should produce valid histogram at last index after warmup
                assert(!isNaN(result[34]));
            } finally {
                // Clean up
                wasm.vwmacd_free(closePtr, len);
                wasm.vwmacd_free(volumePtr, len);
                wasm.vwmacd_free(outPtr, len);
            }
        });
    });

    describe('Batch API', () => {
        it.skip('should calculate batch VWMACD', () => {
            const config = {
                fast_range: [10, 14, 2],    // 3 values: 10, 12, 14
                slow_range: [24, 28, 2],    // 3 values: 24, 26, 28
                signal_range: [8, 10, 1],   // 3 values: 8, 9, 10
                fast_ma_type: "sma",
                slow_ma_type: "sma",
                signal_ma_type: "ema"
            };
            
            const result = wasm.vwmacd_batch(close, volume, config);
            
            // Should have 3 * 3 * 3 = 27 combinations
            assert.strictEqual(result.rows, 27);
            assert.strictEqual(result.cols, close.length);
            assert.strictEqual(result.values.length, 27 * close.length);
            assert.strictEqual(result.combos.length, 27);
            
            // Verify first combo parameters
            assert.strictEqual(result.combos[0].fast_period, 10);
            assert.strictEqual(result.combos[0].slow_period, 24);
            assert.strictEqual(result.combos[0].signal_period, 8);
        });

        it.skip('should handle edge cases in batch API', () => {
            // Empty range
            const emptyConfig = {
                fast_range: [12, 12, 1],    // Only one value
                slow_range: [26, 26, 1],    // Only one value
                signal_range: [9, 9, 1],    // Only one value
            };
            
            const result = wasm.vwmacd_batch(close, volume, emptyConfig);
            assert.strictEqual(result.rows, 1);
            
            // Invalid config
            assert.throws(() => wasm.vwmacd_batch(close, volume, {}));
        });
    });
});
