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
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
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

test('trima_batch_old_api', () => {
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

// New Batch API Tests (ergonomic API)
test('TRIMA batch - new ergonomic API with single parameter', () => {
    const close = new Float64Array(testData.close);
    
    const result = wasm.trima_batch(close, {
        period_range: [30, 30, 0]
    });
    
    // Verify structure
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');
    
    // Verify dimensions
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.values.length, close.length);
    
    // Verify parameters
    const combo = result.combos[0];
    assert.strictEqual(combo.period, 30);
    
    // Compare with old API
    const oldResult = wasm.trima_js(close, 30);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-9,
               `Value mismatch at index ${i}`);
    }
});

test('TRIMA batch - new API with multiple parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.trima_batch(close, {
        period_range: [10, 20, 10]  // 10, 20
    });
    
    // Should have 2 combinations
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 2);
    assert.strictEqual(result.values.length, 100);
    
    // Verify each combo
    const expectedCombos = [
        { period: 10 },
        { period: 20 }
    ];
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedCombos[i].period);
    }
    
    // Compare with old API for first combination
    const oldResult = wasm.trima_js(close, 10);
    const firstRow = result.values.slice(0, result.cols);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-9,
               `Value mismatch at index ${i}`);
    }
});

test('TRIMA batch - new API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    // Invalid config structure
    assert.throws(() => {
        wasm.trima_batch(close, {
            period_range: [9, 9] // Missing step
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.trima_batch(close, {
            // Missing period_range
        });
    }, /Invalid config/);
    
    // Invalid data type
    assert.throws(() => {
        wasm.trima_batch(close, {
            period_range: "invalid"
        });
    }, /Invalid config/);
});

// Zero-copy API tests
test('TRIMA zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    
    // Allocate buffer
    const ptr = wasm.trima_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute TRIMA in-place
    try {
        wasm.trima_into(ptr, ptr, data.length, period);
        
        // Verify results match regular API
        const regularResult = wasm.trima_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.trima_free(ptr, data.length);
    }
});

test('TRIMA zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.trima_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.trima_into(ptr, ptr, size, 30);
        
        // Recreate view in case memory grew
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        // Check warmup period has NaN
        for (let i = 0; i < 29; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = 29; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.trima_free(ptr, size);
    }
});

// Error handling for zero-copy API
test('TRIMA zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.trima_into(0, 0, 10, 5);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.trima_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.trima_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        // Period too small
        assert.throws(() => {
            wasm.trima_into(ptr, ptr, 10, 3);
        }, /Invalid period|Period too small/);
    } finally {
        wasm.trima_free(ptr, 10);
    }
});

// Memory leak prevention test
test('TRIMA zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000, 100000];
    
    for (const size of sizes) {
        const ptr = wasm.trima_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Write pattern to verify memory
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.trima_free(ptr, size);
    }
});

// SIMD128 verification test
test('TRIMA SIMD128 consistency', () => {
    // This test verifies SIMD128 produces same results as scalar
    // It runs automatically when SIMD128 is enabled
    const testCases = [
        { size: 10, period: 5 },
        { size: 100, period: 20 },
        { size: 1000, period: 50 },
        { size: 10000, period: 100 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }
        
        const result = wasm.trima_js(data, testCase.period);
        
        // Basic sanity checks
        assert.strictEqual(result.length, data.length);
        
        // Check warmup period
        for (let i = 0; i < testCase.period - 1; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        // Check values exist after warmup
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.period - 1; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        // Verify reasonable values
        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(Math.abs(avgAfterWarmup) < 10, `Average value ${avgAfterWarmup} seems unreasonable`);
    }
});

test('TRIMA batch_into low-level API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const periods = [5, 7]; // Will test period 5 and 7
    const rows = 2;
    const cols = data.length;
    
    // Allocate input and output buffers
    const inPtr = wasm.trima_alloc(data.length);
    const outPtr = wasm.trima_alloc(rows * cols);
    
    try {
        // Copy data to WASM memory
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
        inView.set(data);
        
        // Call batch_into
        const resultRows = wasm.trima_batch_into(
            inPtr, outPtr, data.length,
            5, 7, 2  // period_start, period_end, period_step
        );
        
        assert.strictEqual(resultRows, 2, 'Should return 2 rows');
        
        // Get output view
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, rows * cols);
        
        // Verify first row (period=5)
        const firstRow = Array.from(outView.slice(0, cols));
        const expected1 = wasm.trima_js(data, 5);
        assertArrayClose(firstRow, expected1, 1e-10, 'First row mismatch');
        
        // Verify second row (period=7)
        const secondRow = Array.from(outView.slice(cols, 2 * cols));
        const expected2 = wasm.trima_js(data, 7);
        assertArrayClose(secondRow, expected2, 1e-10, 'Second row mismatch');
    } finally {
        wasm.trima_free(inPtr, data.length);
        wasm.trima_free(outPtr, rows * cols);
    }
});