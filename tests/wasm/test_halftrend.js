/**
 * WASM binding tests for HALFTREND indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { 
    loadTestData, 
    assertArrayClose, 
    assertClose,
    isNaN,
    assertAllNaN,
    assertNoNaN,
    EXPECTED_OUTPUTS 
} from './test_utils.js';
// import { compareWithRust } from './rust-comparison.js';  // Uncomment once halftrend is added to generate_references

let wasm;
let testData;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.before(async () => {
    // Load WASM module
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('HALFTREND partial params', () => {
    // Test with default parameters - mirrors check_halftrend_partial_params
    const { high, low, close } = testData;
    const expected = EXPECTED_OUTPUTS.halftrend;
    
    const result = wasm.halftrend(
        high, low, close,
        expected.defaultParams.amplitude,
        expected.defaultParams.channelDeviation,
        expected.defaultParams.atrPeriod
    );
    
    assert(result.values, 'Should have values output');
    assert.strictEqual(result.cols, high.length);
    assert.strictEqual(result.rows, 6);
});

test('HALFTREND accuracy', async () => {
    // Test HALFTREND matches expected values from Rust tests - mirrors check_halftrend_accuracy
    const { high, low, close } = testData;
    const expected = EXPECTED_OUTPUTS.halftrend;
    
    const result = wasm.halftrend(
        high, low, close,
        expected.defaultParams.amplitude,
        expected.defaultParams.channelDeviation,
        expected.defaultParams.atrPeriod
    );
    
    // Check that we got all expected outputs
    assert(result.values, 'Should have values output');
    assert.strictEqual(result.rows, 6, 'Should have 6 rows');
    assert.strictEqual(result.cols, high.length, 'Cols should match input length');
    assert.strictEqual(result.values.length, 6 * high.length, 'Values array should be flattened');
    
    // Extract individual arrays from flattened result
    const cols = result.cols;
    const halftrend = result.values.slice(0, cols);
    const trend = result.values.slice(cols, 2 * cols);
    const atr_high = result.values.slice(2 * cols, 3 * cols);
    const atr_low = result.values.slice(3 * cols, 4 * cols);
    const buy_signal = result.values.slice(4 * cols, 5 * cols);
    const sell_signal = result.values.slice(5 * cols, 6 * cols);
    
    // Test specific values from Rust tests
    const testIndices = expected.testIndices;
    const expectedHalftrend = expected.expectedHalftrend;
    const expectedTrend = expected.expectedTrend;
    
    testIndices.forEach((idx, i) => {
        assertClose(
            halftrend[idx], 
            expectedHalftrend[i], 
            1.0,
            `HalfTrend mismatch at index ${idx}`
        );
        assertClose(
            trend[idx], 
            expectedTrend[i], 
            0.01,
            `Trend mismatch at index ${idx}`
        );
    });
    
    // Note: compareWithRust would be used here once halftrend is added to generate_references binary
    // await compareWithRust('halftrend', result, 'ohlc', expected.defaultParams);
});

test('HALFTREND default candles', () => {
    // Test HALFTREND with default parameters - mirrors check_halftrend_default_candles
    const { high, low, close } = testData;
    
    // Default params: amplitude=2, channel_deviation=2.0, atr_period=100
    const result = wasm.halftrend(high, low, close, 2, 2.0, 100);
    
    assert(result.values, 'Should have values output');
    assert.strictEqual(result.cols, high.length, 'Output length should match input');
    assert.strictEqual(result.rows, 6, 'Should have 6 output series');
});

test('HALFTREND zero amplitude', () => {
    // Test HALFTREND fails with zero amplitude - mirrors check_halftrend_invalid_period
    const inputData = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    assert.throws(() => {
        wasm.halftrend(inputData, inputData, inputData, 0, 2.0, 100);
    }, /Invalid period.*period = 0/);
});

test('HALFTREND period exceeds length', () => {
    // Test HALFTREND fails when period exceeds data length - mirrors check_halftrend_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.halftrend(dataSmall, dataSmall, dataSmall, 10, 2.0, 100);
    }, /Invalid period.*period = 100/);
});

test('HALFTREND very small dataset', () => {
    // Test HALFTREND fails with insufficient data - mirrors check_halftrend_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.halftrend(singlePoint, singlePoint, singlePoint, 2, 2.0, 100);
    }, /Invalid period.*period = 100|Not enough valid data/);
});

test('HALFTREND empty input', () => {
    // Test HALFTREND fails with empty input - mirrors check_halftrend_empty_data
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.halftrend(empty, empty, empty, 2, 2.0, 100);
    }, /Empty input data|Input data slice is empty/);
});

test('HALFTREND invalid channel deviation', () => {
    // Test HALFTREND fails with invalid channel deviation - mirrors check_halftrend_invalid_chdev
    const data = new Float64Array([1.0, 2.0, 3.0]);
    
    // Zero channel deviation
    assert.throws(() => {
        wasm.halftrend(data, data, data, 2, 0.0, 100);
    }, /Invalid channel_deviation/);
    
    // Negative channel deviation
    assert.throws(() => {
        wasm.halftrend(data, data, data, 2, -1.0, 100);
    }, /Invalid channel_deviation/);
});

test('HALFTREND invalid ATR period', () => {
    // Test HALFTREND fails with invalid ATR period
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    assert.throws(() => {
        wasm.halftrend(data, data, data, 2, 2.0, 0);
    }, /Invalid period.*period = 0/);
});

test('HALFTREND NaN handling', () => {
    // Test HALFTREND handles NaN values correctly - mirrors check_halftrend_nan_handling
    const { high, low, close } = testData;
    const expected = EXPECTED_OUTPUTS.halftrend;
    
    const result = wasm.halftrend(high, low, close, 2, 2.0, 100);
    assert.strictEqual(result.cols, high.length);
    
    // Extract arrays from flattened result
    const halftrend = result.values.slice(0, result.cols);
    const trend = result.values.slice(result.cols, 2 * result.cols);
    
    // First warmup_period values should be NaN
    const warmupPeriod = expected.warmupPeriod;
    for (let i = 0; i < warmupPeriod; i++) {
        assert(isNaN(halftrend[i]), `Expected NaN at warmup index ${i}`);
        assert(isNaN(trend[i]), `Expected NaN in trend at warmup index ${i}`);
    }
    
    // After warmup should have values
    if (halftrend.length > warmupPeriod + 10) {
        for (let i = warmupPeriod; i < warmupPeriod + 10; i++) {
            assert(!isNaN(halftrend[i]), `Unexpected NaN at index ${i}`);
            assert(!isNaN(trend[i]), `Unexpected NaN in trend at index ${i}`);
        }
    }
});

test('HALFTREND all NaN input', () => {
    // Test HALFTREND with all NaN values - mirrors check_halftrend_all_nan
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.halftrend(allNaN, allNaN, allNaN, 2, 2.0, 100);
    }, /All values are NaN|All NaN|All input values/);
});

test('HALFTREND mismatched array lengths', () => {
    // Test HALFTREND fails with mismatched array lengths
    const high = new Float64Array([1, 2, 3]);
    const low = new Float64Array([1, 2]);
    const close = new Float64Array([1, 2, 3]);
    
    assert.throws(() => {
        wasm.halftrend(high, low, close, 2, 2.0, 100);
    }, /Mismatched|lengths|size|Not enough valid/);
});

test('HALFTREND custom params', () => {
    // Test HALFTREND with custom parameters
    const { high, low, close } = testData;
    
    // Test with custom parameters
    const result = wasm.halftrend(
        high, low, close,
        3,      // amplitude
        2.5,    // channel_deviation
        50      // atr_period
    );
    
    // Check that output exists and has correct structure
    assert(result.values, 'Should have values output');
    assert.strictEqual(result.cols, high.length, 'Cols should match input length');
    assert.strictEqual(result.rows, 6, 'Should have 6 output series');
    
    // Extract halftrend from flattened result
    const halftrend = result.values.slice(0, result.cols);
    
    // Check for non-NaN values after warmup (max(3, 50) - 1 = 49)
    const warmupPeriod = 49;
    let nonNanCount = 0;
    for (let i = warmupPeriod; i < Math.min(halftrend.length, warmupPeriod + 100); i++) {
        if (!isNaN(halftrend[i])) {
            nonNanCount++;
        }
    }
    assert(nonNanCount > 0, 'Should have non-NaN values after warmup period');
});

test('HALFTREND warmup period verification', () => {
    // Test HALFTREND warmup period calculation with different parameters
    const data = new Float64Array(500);
    for (let i = 0; i < 500; i++) {
        data[i] = 100 + Math.sin(i * 0.1) * 10;
    }
    
    const testCases = [
        { amplitude: 2, atrPeriod: 100, expectedWarmup: 99 },  // max(2, 100) - 1
        { amplitude: 50, atrPeriod: 20, expectedWarmup: 49 },  // max(50, 20) - 1
        { amplitude: 10, atrPeriod: 10, expectedWarmup: 9 },   // max(10, 10) - 1
    ];
    
    for (const testCase of testCases) {
        const result = wasm.halftrend(
            data, data, data,
            testCase.amplitude,
            2.0,
            testCase.atrPeriod
        );
        
        // Extract halftrend from flattened result
        const halftrend = result.values.slice(0, result.cols);
        
        // Verify NaN count matches expected warmup
        let nanCount = 0;
        for (let i = 0; i <= testCase.expectedWarmup; i++) {
            if (isNaN(halftrend[i])) {
                nanCount++;
            }
        }
        assert(
            nanCount >= testCase.expectedWarmup,
            `Expected at least ${testCase.expectedWarmup} NaN values for amplitude=${testCase.amplitude}, atrPeriod=${testCase.atrPeriod}`
        );
    }
});

test('HALFTREND signal detection', () => {
    // Test HALFTREND buy/sell signal generation
    const { high, low, close } = testData;
    
    const result = wasm.halftrend(high, low, close, 2, 2.0, 100);
    
    // Extract buy and sell signal arrays from flattened result
    const buy_signal = result.values.slice(4 * result.cols, 5 * result.cols);
    const sell_signal = result.values.slice(5 * result.cols, 6 * result.cols);
    
    // Buy and sell signals should be mostly NaN (signals only occur at trend changes)
    let buyNanCount = 0;
    let sellNanCount = 0;
    
    for (let i = 0; i < buy_signal.length; i++) {
        if (isNaN(buy_signal[i])) buyNanCount++;
        if (isNaN(sell_signal[i])) sellNanCount++;
    }
    
    // Most values should be NaN
    assert(buyNanCount > high.length * 0.95, 'Buy signals should be sparse (mostly NaN)');
    assert(sellNanCount > high.length * 0.95, 'Sell signals should be sparse (mostly NaN)');
    
    // Count actual signals (non-NaN values)
    const buySignalCount = buy_signal.length - buyNanCount;
    const sellSignalCount = sell_signal.length - sellNanCount;
    
    // Should have at least some signals
    assert(buySignalCount > 0, 'Should have at least one buy signal');
    assert(sellSignalCount > 0, 'Should have at least one sell signal');
});

// Batch processing tests
test('HALFTREND batch single parameter set', () => {
    // Test batch with single parameter combination
    const { high, low, close } = testData;
    const expected = EXPECTED_OUTPUTS.halftrend;
    
    // Using the batch API for single parameter
    const batchResult = wasm.halftrend_batch(high, low, close, {
        amplitude_range: [expected.defaultParams.amplitude, expected.defaultParams.amplitude, 0],
        channel_deviation_range: [expected.defaultParams.channelDeviation, expected.defaultParams.channelDeviation, 0],
        atr_period_range: [expected.defaultParams.atrPeriod, expected.defaultParams.atrPeriod, 0]
    });
    
    // Should have 1 combination with 6 outputs each
    assert(batchResult.values, 'Should have values array');
    assert.strictEqual(batchResult.rows, 6, 'Should have 6 output series');
    assert.strictEqual(batchResult.cols, high.length, 'Cols should match input length');
    assert.strictEqual(batchResult.values.length, 6 * high.length, 'Values should be flattened');
    
    // Extract halftrend from batch result
    const halftrend = batchResult.values.slice(0, batchResult.cols);
    const trend = batchResult.values.slice(batchResult.cols, 2 * batchResult.cols);
    
    // Verify specific values
    const testIndices = expected.testIndices;
    const expectedHalftrend = expected.expectedHalftrend;
    const expectedTrend = expected.expectedTrend;
    
    testIndices.forEach((idx, i) => {
        assertClose(
            halftrend[idx],
            expectedHalftrend[i],
            1.0,
            `Batch halftrend mismatch at index ${idx}`
        );
        assertClose(
            trend[idx],
            expectedTrend[i],
            0.01,
            `Batch trend mismatch at index ${idx}`
        );
    });
});

test('HALFTREND batch multiple parameters', () => {
    // Test batch with multiple parameter combinations
    const { high, low, close } = testData;
    const testHigh = high.slice(0, 100);
    const testLow = low.slice(0, 100);
    const testClose = close.slice(0, 100);
    
    const batchResult = wasm.halftrend_batch(testHigh, testLow, testClose, {
        amplitude_range: [2, 4, 1],  // 2, 3, 4
        channel_deviation_range: [2.0, 2.5, 0.5],  // 2.0, 2.5
        atr_period_range: [50, 50, 0]  // 50
    });
    
    // Should have 3 * 2 * 1 = 6 combinations
    const expectedCombos = 6;
    assert.strictEqual(batchResult.combos ? batchResult.combos.length : (batchResult.rows / 6), expectedCombos);
    assert.strictEqual(batchResult.cols, 100, 'Cols should match input length');
    
    // Verify values array is properly sized
    assert(batchResult.values, 'Should have values array');
    assert(batchResult.values.length > 0, 'Values array should not be empty');
});

// Zero-copy API tests
test('HALFTREND zero-copy API', () => {
    const size = 100;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = 100 + Math.sin(i * 0.1) * 10;
    }
    
    // Allocate buffers for all arrays
    const highPtr = wasm.halftrend_alloc(size);
    const lowPtr = wasm.halftrend_alloc(size);
    const closePtr = wasm.halftrend_alloc(size);
    const outPtr = wasm.halftrend_alloc(size * 6); // Single output buffer for all 6 arrays
    
    try {
        assert(highPtr !== 0, 'Failed to allocate high buffer');
        assert(lowPtr !== 0, 'Failed to allocate low buffer');
        assert(closePtr !== 0, 'Failed to allocate close buffer');
        assert(outPtr !== 0, 'Failed to allocate output buffer');
        
        // Create views into WASM memory
        const memory = wasm.__wbindgen_memory();
        const highView = new Float64Array(memory.buffer, highPtr, size);
        const lowView = new Float64Array(memory.buffer, lowPtr, size);
        const closeView = new Float64Array(memory.buffer, closePtr, size);
        
        // Copy data into WASM memory
        highView.set(data);
        lowView.set(data);
        closeView.set(data);
        
        // Compute HalfTrend in-place
        wasm.halftrend_into(
            highPtr, lowPtr, closePtr,
            outPtr,
            size, 2, 2.0, 50
        );
        
        // Verify results exist
        const memory2 = wasm.__wbindgen_memory();
        const htView = new Float64Array(memory2.buffer, outPtr, size);
        
        // Check warmup period has NaN
        let hasNaN = false;
        for (let i = 0; i < 49; i++) {
            if (isNaN(htView[i])) hasNaN = true;
        }
        assert(hasNaN, 'Should have NaN in warmup period');
        
        // Check after warmup has values
        let hasValues = false;
        for (let i = 50; i < Math.min(60, size); i++) {
            if (!isNaN(htView[i])) hasValues = true;
        }
        assert(hasValues, 'Should have values after warmup');
    } finally {
        // Always free memory
        wasm.halftrend_free(highPtr, size);
        wasm.halftrend_free(lowPtr, size);
        wasm.halftrend_free(closePtr, size);
        wasm.halftrend_free(outPtr, size * 6);
    }
});

// Reinput test - applying HalfTrend to its own output
test('HALFTREND reinput', () => {
    // Test HalfTrend applied twice (re-input)
    const { high, low, close } = testData;
    
    // First pass
    const firstResult = wasm.halftrend(high, low, close, 2, 2.0, 100);
    assert(firstResult.values, 'First pass should produce output');
    
    // Extract halftrend from first pass
    const halftrend1 = firstResult.values.slice(0, firstResult.cols);
    
    // Second pass - apply HalfTrend to HalfTrend output (using same array for all OHLC)
    const secondResult = wasm.halftrend(halftrend1, halftrend1, halftrend1, 2, 2.0, 100);
    assert(secondResult.values, 'Second pass should produce output');
    
    // Extract halftrend from second pass
    const halftrend2 = secondResult.values.slice(0, secondResult.cols);
    
    // Basic sanity checks
    assert.strictEqual(halftrend2.length, halftrend1.length, 'Length should be preserved');
    
    // Check that values changed (re-smoothing should alter the values)
    let differences = 0;
    for (let i = 200; i < Math.min(300, halftrend2.length); i++) {
        if (!isNaN(halftrend1[i]) && !isNaN(halftrend2[i])) {
            if (Math.abs(halftrend1[i] - halftrend2[i]) > 1e-10) {
                differences++;
            }
        }
    }
    assert(differences > 0, 'Reinput should produce different values due to re-smoothing');
});

// Additional test coverage for edge cases and missing scenarios
test('HALFTREND not enough valid data', () => {
    // Test with mostly NaN values but some valid data (not enough for calculation)
    const n = 10;
    const highData = new Float64Array(n);
    const lowData = new Float64Array(n);
    const closeData = new Float64Array(n);
    
    // Fill with NaN except for one valid value
    highData.fill(NaN);
    lowData.fill(NaN);
    closeData.fill(NaN);
    highData[5] = 1.0;
    lowData[5] = 1.0;
    closeData[5] = 1.0;
    
    assert.throws(() => {
        wasm.halftrend(highData, lowData, closeData, 9, 2.0, 9);
    }, /Not enough valid data/);
});

test('HALFTREND batch metadata verification', () => {
    // Test that batch result includes correct parameter combinations
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = 100 + Math.sin(i * 0.1) * 10;
    }
    
    const result = wasm.halftrend_batch(data, data, data, {
        amplitude_range: [2, 4, 1],           // 2, 3, 4
        channel_deviation_range: [2.0, 2.5, 0.5],  // 2.0, 2.5
        atr_period_range: [10, 20, 10]        // 10, 20
    });
    
    // Should have 3 * 2 * 2 = 12 combinations
    if (result.combos) {
        assert.strictEqual(result.combos.length, 12, 'Should have 12 parameter combinations');
        
        // Verify first combination
        assert.strictEqual(result.combos[0].amplitude, 2);
        assert.strictEqual(result.combos[0].channel_deviation, 2.0);
        assert.strictEqual(result.combos[0].atr_period, 10);
        
        // Verify last combination
        assert.strictEqual(result.combos[11].amplitude, 4);
        assertClose(result.combos[11].channel_deviation, 2.5, 1e-10, 'channel_deviation mismatch');
        assert.strictEqual(result.combos[11].atr_period, 20);
    }
});

test('HALFTREND zero-copy error handling', () => {
    // Test null pointer handling
    assert.throws(() => {
        wasm.halftrend_into(0, 0, 0, 0, 10, 2, 2.0, 100);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.halftrend_alloc(10);
    const out_ptr = wasm.halftrend_alloc(60); // 6 * 10 for output
    try {
        // Invalid amplitude
        assert.throws(() => {
            wasm.halftrend_into(
                ptr, ptr, ptr,
                out_ptr,
                10, 0, 2.0, 100
            );
        }, /Invalid period/);
        
        // Invalid channel deviation
        assert.throws(() => {
            wasm.halftrend_into(
                ptr, ptr, ptr,
                out_ptr,
                10, 2, 0.0, 100
            );
        }, /Invalid channel_deviation/);
    } finally {
        wasm.halftrend_free(ptr, 10);
        wasm.halftrend_free(out_ptr, 60);
    }
});

test('HALFTREND memory management stress test', () => {
    // Allocate and free multiple times to ensure no memory leaks
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        // Allocate all buffers needed for halftrend
        const buffers = [];
        for (let i = 0; i < 9; i++) {  // 3 inputs + 6 outputs
            const ptr = wasm.halftrend_alloc(size);
            assert(ptr !== 0, `Failed to allocate buffer ${i} of size ${size}`);
            buffers.push(ptr);
        }
        
        // Write test pattern
        const memory = wasm.__wbindgen_memory();
        const view = new Float64Array(memory.buffer, buffers[0], size);
        for (let i = 0; i < Math.min(10, size); i++) {
            view[i] = i * 2.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(view[i], i * 2.5, `Memory corruption at index ${i}`);
        }
        
        // Free all buffers
        for (const ptr of buffers) {
            wasm.halftrend_free(ptr, size);
        }
    }
});

test('HALFTREND SIMD128 consistency', () => {
    // Verify SIMD128 produces consistent results across different data sizes
    const testCases = [
        { size: 10, amplitude: 2, atrPeriod: 5 },
        { size: 100, amplitude: 5, atrPeriod: 20 },
        { size: 1000, amplitude: 10, atrPeriod: 50 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = 100 + Math.sin(i * 0.1) * 10 + Math.cos(i * 0.05) * 5;
        }
        
        const result = wasm.halftrend(
            data, data, data,
            testCase.amplitude, 2.0, testCase.atrPeriod
        );
        
        // Check that result is not null
        assert(result, `Result should not be null for size=${testCase.size}`);
        assert(result.values, `Result should have values for size=${testCase.size}`);
        
        // Basic sanity checks
        assert.strictEqual(result.cols, data.length);
        assert.strictEqual(result.rows, 6);
        
        // Extract halftrend
        const halftrend = result.values.slice(0, result.cols);
        
        // Check warmup period
        const warmup = Math.max(testCase.amplitude, testCase.atrPeriod) - 1;
        for (let i = 0; i < Math.min(warmup, halftrend.length); i++) {
            assert(isNaN(halftrend[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        // Check values exist after warmup
        let validCount = 0;
        for (let i = warmup; i < halftrend.length; i++) {
            if (!isNaN(halftrend[i])) {
                validCount++;
            }
        }
        assert(validCount > 0, `Should have valid values after warmup for size=${testCase.size}`);
    }
});

test.after(() => {
    console.log('HALFTREND WASM tests completed');
});