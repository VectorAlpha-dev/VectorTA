/**
 * WASM binding tests for TrendFlex indicator.
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

// Expected outputs for TrendFlex
const EXPECTED_OUTPUTS = {
    trendflex: {
        default_params: { period: 20 },
        last_5_values: [
            -0.19724678008015128,
            -0.1238001236481444,
            -0.10515389737087717,
            -0.1149541079904878,
            -0.16006869484450567,
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

test('trendflex_partial_params', () => {
    const close = testData.close;
    
    // Test with default period of 20
    const result = wasm.trendflex_js(close, 20);
    assert.equal(result.length, close.length);
});

test('trendflex_accuracy', () => {
    const close = testData.close;
    const expected = EXPECTED_OUTPUTS.trendflex;
    
    const result = wasm.trendflex_js(close, expected.default_params.period);
    
    assert.equal(result.length, close.length);
    
    // Check last 5 values
    const last5 = result.slice(-5);
    expected.last_5_values.forEach((expectedVal, i) => {
        assertClose(last5[i], expectedVal, 1e-8, `TrendFlex mismatch at index ${i}`);
    });
    
    // Compare with Rust
    compareWithRust('trendflex', result, 'close', expected.default_params);
});

test('trendflex_default_candles', () => {
    const close = testData.close;
    
    // Default params: period=20
    const result = wasm.trendflex_js(close, 20);
    assert.equal(result.length, close.length);
});

test('trendflex_zero_period', () => {
    const inputData = [10.0, 20.0, 30.0];
    
    assert.throws(() => {
        wasm.trendflex_js(inputData, 0);
    }, /period = 0/);
});

test('trendflex_period_exceeds_length', () => {
    const dataSmall = [10.0, 20.0, 30.0];
    
    assert.throws(() => {
        wasm.trendflex_js(dataSmall, 10);
    }, /period > data len/);
});

test('trendflex_very_small_dataset', () => {
    const singlePoint = [42.0];
    
    assert.throws(() => {
        wasm.trendflex_js(singlePoint, 9);
    }, /period > data len/);
});

test('trendflex_empty_input', () => {
    const empty = [];
    
    assert.throws(() => {
        wasm.trendflex_js(empty, 20);
    }, /No data provided/);
});

test('trendflex_reinput', () => {
    const close = testData.close;
    
    // First pass
    const firstResult = wasm.trendflex_js(close, 20);
    assert.equal(firstResult.length, close.length);
    
    // Second pass - apply TrendFlex to TrendFlex output
    const secondResult = wasm.trendflex_js(firstResult, 10);
    assert.equal(secondResult.length, firstResult.length);
    
    // Check for NaN handling after warmup
    if (secondResult.length > 240) {
        for (let i = 240; i < secondResult.length; i++) {
            assert.ok(!isNaN(secondResult[i]), `Unexpected NaN at index ${i}`);
        }
    }
});

test('trendflex_nan_handling', () => {
    const close = testData.close;
    
    const result = wasm.trendflex_js(close, 20);
    assert.equal(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert.ok(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
        }
    }
    
    // First 19 values should be NaN (period-1)
    for (let i = 0; i < 19; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
});

test('trendflex_batch', () => {
    const close = testData.close;
    
    const result = wasm.trendflex_batch_js(close, 20, 20, 0);
    const metadata = wasm.trendflex_batch_metadata_js(20, 20, 0);
    
    // Should have 1 period value
    assert.equal(metadata.length, 1);
    assert.equal(metadata[0], 20);
    
    // Result should be flattened array (1 row × data length)
    assert.equal(result.length, close.length);
    
    // Check last 5 values
    const expected = EXPECTED_OUTPUTS.trendflex.last_5_values;
    const last5 = result.slice(-5);
    expected.forEach((expectedVal, i) => {
        assertClose(last5[i], expectedVal, 1e-8, `TrendFlex batch mismatch at index ${i}`);
    });
});

test('trendflex_batch_multiple_periods', () => {
    const close = testData.close;
    
    const result = wasm.trendflex_batch_js(close, 10, 30, 10);
    const metadata = wasm.trendflex_batch_metadata_js(10, 30, 10);
    
    // Should have 3 periods: 10, 20, 30
    assert.equal(metadata.length, 3);
    assert.deepEqual(Array.from(metadata), [10, 20, 30]);
    
    // Result should be flattened array (3 rows × data length)
    assert.equal(result.length, 3 * close.length);
});

test('trendflex_all_nan_input', () => {
    const allNan = new Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.trendflex_js(allNan, 20);
    }, /All values are NaN/);
});

// New ergonomic batch API tests
test('trendflex_batch_ergonomic_single_parameter', () => {
    const close = testData.close;
    
    // Using the new ergonomic batch API for single parameter
    const batchResult = wasm.trendflex_batch(close, {
        period_range: [20, 20, 0]
    });
    
    // Should match single calculation
    const singleResult = wasm.trendflex_js(close, 20);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('trendflex_batch_ergonomic_multiple_periods', () => {
    const close = testData.close.slice(0, 100);
    
    // Multiple periods: 10, 20, 30 using ergonomic API
    const batchResult = wasm.trendflex_batch(close, {
        period_range: [10, 30, 10]      // period range
    });
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    // Check combos
    assert.strictEqual(batchResult.combos.length, 3);
    assert.strictEqual(batchResult.combos[0].period, 10);
    assert.strictEqual(batchResult.combos[1].period, 20);
    assert.strictEqual(batchResult.combos[2].period, 30);
});

test('trendflex_batch_edge_cases', () => {
    const close = new Float64Array(10);
    close.fill(100);
    
    // Single value sweep
    const singleBatch = wasm.trendflex_batch(close, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.trendflex_batch(close, {
        period_range: [5, 7, 10] // Step larger than range
    });
    
    // Should only have period=5
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.trendflex_batch(new Float64Array([]), {
            period_range: [20, 20, 0]
        });
    }, /All values are NaN/);
});

// Zero-copy API tests
test('trendflex_zero_copy_basic', () => {
    const data = new Float64Array(testData.close);
    const period = 20;
    
    // Allocate buffer
    const ptr = wasm.trendflex_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute TrendFlex in-place
    try {
        wasm.trendflex_into(ptr, ptr, data.length, period);
        
        // Verify results match regular API
        const regularResult = wasm.trendflex_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.trendflex_free(ptr, data.length);
    }
});

test('trendflex_zero_copy_error_handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.trendflex_into(0, 0, 10, 20);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.trendflex_alloc(10);
    try {
        // Invalid period (0)
        assert.throws(() => {
            wasm.trendflex_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        // Period exceeds length
        assert.throws(() => {
            wasm.trendflex_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.trendflex_free(ptr, 10);
    }
});

test('trendflex_batch_into', () => {
    const data = new Float64Array(testData.close.slice(0, 100));
    const period_start = 10;
    const period_end = 30;
    const period_step = 10;
    
    // Calculate expected size (3 periods × 100 data points)
    const expected_combos = 3;
    const total_size = expected_combos * data.length;
    
    // Allocate input and output buffers
    const in_ptr = wasm.trendflex_alloc(data.length);
    const out_ptr = wasm.trendflex_alloc(total_size);
    
    try {
        // Copy data to input buffer
        const inView = new Float64Array(wasm.__wasm.memory.buffer, in_ptr, data.length);
        inView.set(data);
        
        // Call batch_into
        const n_combos = wasm.trendflex_batch_into(
            in_ptr, out_ptr, data.length,
            period_start, period_end, period_step
        );
        
        assert.strictEqual(n_combos, expected_combos);
        
        // Verify output
        const outView = new Float64Array(wasm.__wasm.memory.buffer, out_ptr, total_size);
        
        // Compare with regular batch API
        const regularBatch = wasm.trendflex_batch(data, {
            period_range: [period_start, period_end, period_step]
        });
        
        assertArrayClose(
            Array.from(outView),
            regularBatch.values,
            1e-10,
            "Batch into vs regular batch mismatch"
        );
    } finally {
        wasm.trendflex_free(in_ptr, data.length);
        wasm.trendflex_free(out_ptr, total_size);
    }
});