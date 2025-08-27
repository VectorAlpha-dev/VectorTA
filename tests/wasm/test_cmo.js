/**
 * WASM binding tests for CMO indicator.
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
import { compareWithRust } from './rust-comparison.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

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

test('CMO partial params', () => {
    // Test with default parameters - mirrors check_cmo_partial_params
    const close = new Float64Array(testData.close);
    
    // Test with default period (undefined = None in Rust)
    const result = wasm.cmo_js(close);
    assert.strictEqual(result.length, close.length);
    
    // Test with explicit period
    const result2 = wasm.cmo_js(close, 10);
    assert.strictEqual(result2.length, close.length);
});

test('CMO accuracy', async () => {
    // Test CMO matches expected values from Rust tests - mirrors check_cmo_accuracy
    const close = new Float64Array(testData.close);
    
    // Expected values from Rust test
    const expectedLastFive = [
        -13.152504931406101,
        -14.649876201213106,
        -16.760170709240303,
        -14.274505732779227,
        -21.984038127126716,
    ];
    
    const result = wasm.cmo_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-6,
        "CMO last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('cmo', result, 'close', { period: 14 });
});

test('CMO default candles', () => {
    // Test CMO with default parameters - mirrors check_cmo_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.cmo_js(close);
    assert.strictEqual(result.length, close.length);
});

test('CMO zero period', () => {
    // Test CMO fails with zero period - mirrors check_cmo_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cmo_js(inputData, 0);
    }, /Invalid period/);
});

test('CMO period exceeds length', () => {
    // Test CMO fails when period exceeds data length - mirrors check_cmo_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cmo_js(dataSmall, 10);
    }, /Invalid period/);
});

test('CMO very small dataset', () => {
    // Test CMO fails with insufficient data - mirrors check_cmo_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.cmo_js(singlePoint, 14);
    }, /Invalid period/);
});

test('CMO empty input', () => {
    // Test CMO fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.cmo_js(empty, 14);
    }, /Empty data provided/);
});

test('CMO reinput', () => {
    // Test CMO applied twice (re-input) - mirrors check_cmo_reinput
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.cmo_js(close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply CMO to CMO output
    const secondResult = wasm.cmo_js(firstResult, 14);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // After double warmup period (28), no NaN values should exist
    if (secondResult.length > 28) {
        for (let i = 28; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Expected no NaN after index 28, found NaN at ${i}`);
        }
    }
});

test('CMO NaN handling', () => {
    // Test CMO handles NaN values correctly - mirrors check_cmo_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.cmo_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First period values should be NaN (period=14, so first 14 values)
    assertAllNaN(result.slice(0, 14), "Expected NaN in warmup period");
});

test('CMO all NaN input', () => {
    // Test CMO with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.cmo_js(allNaN, 14);
    }, /All values are NaN/);
});

// Fast API tests
test('CMO fast API - basic operation', () => {
    // Test fast/unsafe API basic functionality
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate input and output buffers
    const inPtr = wasm.cmo_alloc(len);
    const outPtr = wasm.cmo_alloc(len);
    
    // Copy data into WASM memory
    const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
    inView.set(close);
    
    // Compute CMO using fast API
    wasm.cmo_into(inPtr, outPtr, len, 14);
    
    // Create output array view from pointer (after computation, in case memory grew)
    const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
    
    // Convert to regular array to avoid detached buffer issues
    const resultArray = Array.from(result);
    
    // Compare with safe API
    const safeResult = wasm.cmo_js(close, 14);
    
    assertArrayClose(resultArray, safeResult, 1e-10, "Fast API vs Safe API mismatch");
    
    // Free memory
    wasm.cmo_free(inPtr, len);
    wasm.cmo_free(outPtr, len);
});

test('CMO fast API - in-place operation (aliasing)', () => {
    // Test fast API with same input/output buffer
    const data = new Float64Array(testData.close.slice(0, 100)); // Smaller dataset
    const len = data.length;
    
    // Allocate buffer and copy data
    const bufPtr = wasm.cmo_alloc(len);
    const buffer = new Float64Array(wasm.__wasm.memory.buffer, bufPtr, len);
    buffer.set(data);
    
    // Save expected result from safe API
    const expected = wasm.cmo_js(data, 14);
    
    // Compute in-place (same buffer for input and output)
    wasm.cmo_into(bufPtr, bufPtr, len, 14);
    
    // Verify result
    assertArrayClose(buffer, expected, 1e-10, "In-place operation mismatch");
    
    // Free memory
    wasm.cmo_free(bufPtr, len);
});

test('CMO fast API - null pointer error', () => {
    // Test error handling for null pointers
    const len = 100;
    
    // Null input pointer
    assert.throws(() => {
        wasm.cmo_into(0, wasm.cmo_alloc(len), len, 14);
    }, /Null pointer/);
    
    // Null output pointer
    const inPtr = wasm.cmo_alloc(len);
    assert.throws(() => {
        wasm.cmo_into(inPtr, 0, len, 14);
    }, /Null pointer/);
    wasm.cmo_free(inPtr, len);
});

// Batch API tests
test('CMO batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    const batchResult = wasm.cmo_batch(close, {
        period_range: [14, 14, 0]
    });
    
    // Should match single calculation
    const singleResult = wasm.cmo_js(close, 14);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('CMO batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 10, 12, 14, 16, 18, 20
    const batchResult = wasm.cmo_batch(close, {
        period_range: [10, 20, 2]
    });
    
    // Should have 6 rows * 100 cols = 600 values
    assert.strictEqual(batchResult.values.length, 6 * 100);
    assert.strictEqual(batchResult.rows, 6);
    assert.strictEqual(batchResult.cols, 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 12, 14, 16, 18, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.cmo_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('CMO batch metadata', () => {
    // Test that batch result includes correct parameter combinations
    const close = new Float64Array(50);
    close.fill(100);
    
    const result = wasm.cmo_batch(close, {
        period_range: [10, 20, 5] // 10, 15, 20
    });
    
    // Should have 3 combinations
    assert.strictEqual(result.combos.length, 3);
    
    // Check combinations
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[1].period, 15);
    assert.strictEqual(result.combos[2].period, 20);
});

test('CMO batch warmup verification', () => {
    // Verify proper warmup handling in batch mode
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.cmo_batch(close, {
        period_range: [10, 20, 10] // periods 10 and 20
    });
    
    assert.strictEqual(batchResult.rows, 2);
    assert.strictEqual(batchResult.cols, 50);
    
    // Check warmup for each row
    for (let row = 0; row < 2; row++) {
        const period = batchResult.combos[row].period;
        const rowStart = row * 50;
        const rowData = batchResult.values.slice(rowStart, rowStart + 50);
        
        // First 'period' values should be NaN
        for (let i = 0; i < period; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        // After warmup should have values
        for (let i = period; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('CMO batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    
    // Single value sweep
    const singleBatch = wasm.cmo_batch(close, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 15);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.cmo_batch(close, {
        period_range: [5, 7, 10] // Step larger than range
    });
    
    // Should only have period=5
    assert.strictEqual(largeBatch.values.length, 15);
    assert.strictEqual(largeBatch.combos.length, 1);
    assert.strictEqual(largeBatch.combos[0].period, 5);
});

// Fast batch API tests
test('CMO fast batch API - basic operation', () => {
    // Test fast batch API functionality
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;
    
    // Parameters for 3 periods: 10, 15, 20
    const periodStart = 10;
    const periodEnd = 20;
    const periodStep = 5;
    const expectedRows = 3;
    
    // Allocate input buffer and copy data
    const inPtr = wasm.cmo_alloc(len);
    const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
    inView.set(close);
    
    // Allocate output buffer for batch
    const outPtr = wasm.cmo_alloc(len * expectedRows);
    
    // Compute batch using fast API
    const rows = wasm.cmo_batch_into(inPtr, outPtr, len, periodStart, periodEnd, periodStep);
    assert.strictEqual(rows, expectedRows);
    
    // Create output array view
    const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len * rows);
    
    // Compare with safe batch API
    const safeBatch = wasm.cmo_batch(close, {
        period_range: [periodStart, periodEnd, periodStep]
    });
    
    assertArrayClose(result, safeBatch.values, 1e-10, "Fast batch vs safe batch mismatch");
    
    // Free memory
    wasm.cmo_free(inPtr, len);
    wasm.cmo_free(outPtr, len * rows);
});

test('CMO fast batch API - null pointer error', () => {
    // Test error handling for null pointers in batch
    const len = 100;
    
    assert.throws(() => {
        wasm.cmo_batch_into(0, wasm.cmo_alloc(len * 3), len, 10, 20, 5);
    }, /null pointer/);
});

test.after(() => {
    console.log('CMO WASM tests completed');
});