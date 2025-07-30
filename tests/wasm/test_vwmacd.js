<<<<<<< HEAD
#!/usr/bin/env node
import { describe, it } from 'node:test';
import assert from 'node:assert';
import * as wasm from '../../pkg/my_project.js';

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
                
                // Read results
                const macd = Array.from(memory.slice(macdPtr / 8, macdPtr / 8 + len));
                const signal = Array.from(memory.slice(signalPtr / 8, signalPtr / 8 + len));
                const hist = Array.from(memory.slice(histPtr / 8, histPtr / 8 + len));
                
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
            const dataPtr = wasm.vwmacd_alloc(len);
            const volumePtr = wasm.vwmacd_alloc(len);
            
            try {
                // Copy data to WASM memory
                const memory = new Float64Array(wasm.__wasm.memory.buffer);
                memory.set(close, dataPtr / 8);
                memory.set(volume, volumePtr / 8);
                
                // Calculate with aliasing (all outputs to same buffer)
                // Note: When all outputs point to the same buffer, only the last output (histogram) will be preserved
                wasm.vwmacd_into(dataPtr, volumePtr, dataPtr, dataPtr, dataPtr, len, 12, 26, 9, "sma", "sma", "ema");
                
                // Read result (should be histogram values since it's written last)
                const result = Array.from(memory.slice(dataPtr / 8, dataPtr / 8 + len));
                
                // Should not crash and should produce valid values
                // Check histogram values (which should be available after total warmup)
                assert(!isNaN(result[34]));
            } finally {
                // Clean up
                wasm.vwmacd_free(dataPtr, len);
                wasm.vwmacd_free(volumePtr, len);
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

        it('should handle edge cases in batch API', () => {
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
=======
/**
 * WASM binding tests for VWMACD indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
const test = require('node:test');
const assert = require('node:assert');
const path = require('path');
const { 
    loadTestData, 
    assertArrayClose, 
    assertClose,
    isNaN,
    assertAllNaN,
    assertNoNaN,
    EXPECTED_OUTPUTS 
} = require('./test_utils');

let wasm;
let testData;

test.before(async () => {
    // Load WASM module
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        wasm = await import(wasmPath);
        await wasm.default();
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('VWMACD accuracy', () => {
    // Test VWMACD matches expected values from Rust tests
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // Test with default parameters
    const result = wasm.vwmacd_js(close, volume, 12, 26, 9, 'sma', 'sma', 'ema');
    
    assert.ok(result.macd, 'Result should have macd array');
    assert.ok(result.signal, 'Result should have signal array');
    assert.ok(result.hist, 'Result should have hist array');
    assert.strictEqual(result.macd.length, close.length);
    assert.strictEqual(result.signal.length, close.length);
    assert.strictEqual(result.hist.length, close.length);
    
    // Expected values from Rust tests
    const expectedMacd = [
        -394.95161155,
        -508.29106210,
        -490.70190723,
        -388.94996199,
        -341.13720646,
    ];
    
    const expectedSignal = [
        -539.48861567,
        -533.24910496,
        -524.73966541,
        -497.58172247,
        -466.29282108,
    ];
    
    const expectedHistogram = [
        144.53700412,
        24.95804286,
        34.03775818,
        108.63176274,
        125.15561462,
    ];
    
    // Check last 5 values
    const macdLast5 = result.macd.slice(-5);
    const signalLast5 = result.signal.slice(-5);
    const histLast5 = result.hist.slice(-5);
    
    assertArrayClose(macdLast5, expectedMacd, 1e-3, "MACD last 5 values mismatch");
    assertArrayClose(signalLast5, expectedSignal, 1e-3, "Signal last 5 values mismatch");
    assertArrayClose(histLast5, expectedHistogram, 1e-3, "Histogram last 5 values mismatch");
});

test('VWMACD custom MA types', () => {
    // Test VWMACD with custom MA types
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // Test with custom MA types
    const result = wasm.vwmacd_js(
        close, volume, 
        12, 26, 9,
        'ema', 'wma', 'sma'
    );
    
    assert.ok(result.macd);
    assert.ok(result.signal);
    assert.ok(result.hist);
    assert.strictEqual(result.macd.length, close.length);
});

test('VWMACD error handling - NaN data', () => {
    // Test with all NaN data
    const nanData = new Float64Array(10).fill(NaN);
    
    assert.throws(() => {
        wasm.vwmacd_js(nanData, nanData, 12, 26, 9, 'sma', 'sma', 'ema');
    }, /All values are NaN/);
});

test('VWMACD error handling - zero period', () => {
    // Test with zero period
    const data = new Float64Array([1.0, 2.0, 3.0]);
    const volume = new Float64Array([100.0, 200.0, 300.0]);
    
    assert.throws(() => {
        wasm.vwmacd_js(data, volume, 0, 26, 9, 'sma', 'sma', 'ema');
    }, /Invalid period/);
});

test('VWMACD error handling - period exceeds data', () => {
    // Test with period exceeding data length
    const data = new Float64Array([1.0, 2.0, 3.0]);
    const volume = new Float64Array([100.0, 200.0, 300.0]);
    
    assert.throws(() => {
        wasm.vwmacd_js(data, volume, 12, 10, 9, 'sma', 'sma', 'ema');
    }, /Invalid period/);
});

test('VWMACD batch processing', () => {
    // Test batch processing
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    const config = {
        fast_range: [10, 14, 2],
        slow_range: [20, 26, 3],
        signal_range: [5, 9, 2]
    };
    
    const result = wasm.vwmacd_batch(close, volume, config);
    
    assert.ok(result.values, 'Result should have values array');
    assert.ok(result.combos, 'Result should have combos array');
    assert.ok(result.rows, 'Result should have rows');
    assert.ok(result.cols, 'Result should have cols');
    
    // Should have 3 * 3 * 3 = 27 combinations
    const expectedCombos = 3 * 3 * 3;
    assert.strictEqual(result.combos.length, expectedCombos);
    assert.strictEqual(result.rows, expectedCombos);
    assert.strictEqual(result.cols, close.length);
    
    // Values should contain MACD results only (batch output doesn't include signal/hist)
    assert.strictEqual(result.values.length, expectedCombos * close.length);
});

test('VWMACD fast API - basic', () => {
    // Test fast API without aliasing
    const close = new Float64Array(testData.close.slice(0, 100));
    const volume = new Float64Array(testData.volume.slice(0, 100));
    const len = close.length;
    
    // Allocate output buffers
    const macdPtr = wasm.vwmacd_alloc(len);
    const signalPtr = wasm.vwmacd_alloc(len);
    const histPtr = wasm.vwmacd_alloc(len);
    
    try {
        // Compute VWMACD
        wasm.vwmacd_into(
            close, volume,
            macdPtr, signalPtr, histPtr,
            len,
            12, 26, 9,
            'sma', 'sma', 'ema'
        );
        
        // Read results
        const macd = new Float64Array(wasm.memory.buffer, macdPtr, len);
        const signal = new Float64Array(wasm.memory.buffer, signalPtr, len);
        const hist = new Float64Array(wasm.memory.buffer, histPtr, len);
        
        // Verify some values are not NaN after warmup
        assert.ok(macd.slice(30).some(v => !isNaN(v)), 'MACD should have valid values');
        assert.ok(signal.slice(30).some(v => !isNaN(v)), 'Signal should have valid values');
        assert.ok(hist.slice(30).some(v => !isNaN(v)), 'Histogram should have valid values');
        
    } finally {
        // Clean up
        wasm.vwmacd_free(macdPtr, len);
        wasm.vwmacd_free(signalPtr, len);
        wasm.vwmacd_free(histPtr, len);
    }
});

test('VWMACD fast API - aliasing', () => {
    // Test fast API with aliasing (output same as input)
    const data = new Float64Array(testData.close.slice(0, 100));
    const volume = new Float64Array(testData.volume.slice(0, 100));
    const len = data.length;
    
    // Create copies for comparison
    const originalData = new Float64Array(data);
    
    // Use data buffer as output (aliasing)
    const dataPtr = data.byteOffset;
    const volumePtr = volume.byteOffset;
    
    // This should handle aliasing correctly
    wasm.vwmacd_into(
        data, volume,
        dataPtr, volumePtr, dataPtr,  // Reuse input pointers
        len,
        12, 26, 9,
        'sma', 'sma', 'ema'
    );
    
    // Verify data was modified (not equal to original)
    assert.ok(
        !data.every((v, i) => v === originalData[i]),
        'Data should be modified when used as output'
    );
});

test.after(() => {
    console.log('VWMACD WASM tests completed');
});
>>>>>>> trip-6
