/**
 * WASM binding tests for TRIMA indicator.
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
    assertNoNaN 
} from './test_utils.js';
import { compareWithRust } from './rust-comparison.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

// Expected outputs for TRIMA
const EXPECTED_OUTPUTS = {
    trima: {
        default_params: { period: 30 },
        last_5_values: [
            59957.916666666664,
            59846.770833333336,
            59750.620833333334,
            59665.2125,
            59581.612499999996,
        ]
    }
};

test.before(async () => {
    // Load WASM module
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        wasm = await import(wasmPath);
    } catch (error) {
        console.error('Failed to load WASM module:', error);
        throw error;
    }
    
    // Load test data
    testData = loadTestData();
});

test('trima_partial_params', () => {
    const close = testData.close;
    
    // Test with default period of 30
    const result = wasm.trima_js(close, 30);
    assert.equal(result.length, close.length);
});

test('trima_accuracy', () => {
    const close = testData.close;
    const expected = EXPECTED_OUTPUTS.trima;
    
    const result = wasm.trima_js(close, expected.default_params.period);
    
    assert.equal(result.length, close.length);
    
    // Check last 5 values
    const last5 = result.slice(-5);
    expected.last_5_values.forEach((expectedVal, i) => {
        assertClose(last5[i], expectedVal, 1e-6, `TRIMA mismatch at index ${i}`);
    });
    
    // Compare with Rust
    compareWithRust('trima', result, 'close', expected.default_params);
});

test('trima_default_candles', () => {
    const close = testData.close;
    
    // Default params: period=30
    const result = wasm.trima_js(close, 30);
    assert.equal(result.length, close.length);
});

test('trima_zero_period', () => {
    const inputData = [10.0, 20.0, 30.0];
    
    assert.throws(() => {
        wasm.trima_js(inputData, 0);
    }, /Invalid period/);
});

test('trima_period_exceeds_length', () => {
    const dataSmall = [10.0, 20.0, 30.0];
    
    assert.throws(() => {
        wasm.trima_js(dataSmall, 10);
    }, /Invalid period/);
});

test('trima_very_small_dataset', () => {
    const singlePoint = [42.0];
    
    assert.throws(() => {
        wasm.trima_js(singlePoint, 9);
    }, /Invalid period/);
});

test('trima_empty_input', () => {
    const empty = [];
    
    assert.throws(() => {
        wasm.trima_js(empty, 9);
    }, /No data provided/);
});

test('trima_period_too_small', () => {
    const data = [1.0, 2.0, 3.0, 4.0, 5.0];
    
    // Period = 3
    assert.throws(() => {
        wasm.trima_js(data, 3);
    }, /Period too small/);
    
    // Period = 2
    assert.throws(() => {
        wasm.trima_js(data, 2);
    }, /Period too small/);
    
    // Period = 1
    assert.throws(() => {
        wasm.trima_js(data, 1);
    }, /Period too small/);
});

test('trima_all_nan_input', () => {
    const allNan = new Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.trima_js(allNan, 20);
    }, /All values are NaN/);
});

test('trima_reinput', () => {
    const close = testData.close;
    
    // First pass
    const firstResult = wasm.trima_js(close, 30);
    assert.equal(firstResult.length, close.length);
    
    // Second pass - apply TRIMA to TRIMA output
    const secondResult = wasm.trima_js(firstResult, 10);
    assert.equal(secondResult.length, firstResult.length);
    
    // Check for NaN handling after warmup
    if (secondResult.length > 240) {
        for (let i = 240; i < secondResult.length; i++) {
            assert.ok(!isNaN(secondResult[i]), `Unexpected NaN at index ${i}`);
        }
    }
});

test('trima_nan_handling', () => {
    const close = testData.close;
    
    const result = wasm.trima_js(close, 30);
    assert.equal(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert.ok(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
        }
    }
    
    // First period-1 values should be NaN
    for (let i = 0; i < 29; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
});

test('trima_batch', () => {
    const close = testData.close;
    
    const result = wasm.trima_batch_js(close, 30, 30, 0);
    const metadata = wasm.trima_batch_metadata_js(30, 30, 0);
    
    // Should have 1 period value
    assert.equal(metadata.length, 1);
    assert.equal(metadata[0], 30);
    
    // Result should be flattened array (1 row × data length)
    assert.equal(result.length, close.length);
    
    // Check last 5 values
    const expected = EXPECTED_OUTPUTS.trima.last_5_values;
    const last5 = result.slice(-5);
    expected.forEach((expectedVal, i) => {
        assertClose(last5[i], expectedVal, 1e-6, `TRIMA batch mismatch at index ${i}`);
    });
});

test('trima_batch_multiple_periods', () => {
    const close = testData.close;
    
    const result = wasm.trima_batch_js(close, 10, 30, 10);
    const metadata = wasm.trima_batch_metadata_js(10, 30, 10);
    
    // Should have 3 periods: 10, 20, 30
    assert.equal(metadata.length, 3);
    assert.deepEqual(Array.from(metadata), [10, 20, 30]);
    
    // Result should be flattened array (3 rows × data length)
    assert.equal(result.length, 3 * close.length);
});

test('trima_candles_with_nan', () => {
    // Test with some NaN values in input
    const dataWithNaN = testData.close.slice();
    dataWithNaN[0] = NaN;
    dataWithNaN[1] = NaN;
    dataWithNaN[2] = NaN;
    
    const result = wasm.trima_js(dataWithNaN, 30);
    assert.equal(result.length, dataWithNaN.length);
    
    // First few values plus warmup should be NaN
    // Warmup starts after first valid value
    let firstValidIdx = 3; // Since we set first 3 to NaN
    let warmup = firstValidIdx + 29; // period-1
    
    for (let i = 0; i < warmup; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
});

test('trima_consistency_check', () => {
    const close = testData.close;
    
    // Run multiple times to ensure consistency
    const result1 = wasm.trima_js(close, 20);
    const result2 = wasm.trima_js(close, 20);
    
    assert.equal(result1.length, result2.length);
    
    for (let i = 0; i < result1.length; i++) {
        if (isNaN(result1[i]) && isNaN(result2[i])) {
            continue;
        }
        assert.equal(result1[i], result2[i], `Inconsistent results at index ${i}`);
    }
});

test('trima_edge_cases', () => {
    // Test with minimum valid period (4)
    const data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    const result = wasm.trima_js(data, 4);
    assert.equal(result.length, data.length);
    
    // First period-1 values should be NaN
    for (let i = 0; i < 3; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    // Values after warmup should be valid
    for (let i = 4; i < result.length; i++) {
        assert.ok(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
    }
});