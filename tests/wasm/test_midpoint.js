/**
 * WASM binding tests for Midpoint indicator.
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
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('Midpoint partial params', () => {
    // Test with default parameters - mirrors check_midpoint_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.midpoint_js(close, 14);
    assert(result instanceof Float64Array);
    assert.strictEqual(result.length, close.length);
});

test('Midpoint accuracy', async () => {
    // Test Midpoint matches expected values from Rust tests - mirrors check_midpoint_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.midpoint_js(close, 14);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected values from Rust test
    const expectedLastFive = [59578.5, 59578.5, 59578.5, 58886.0, 58886.0];
    
    const actualLastFive = result.slice(-5);
    
    for (let i = 0; i < 5; i++) {
        assertClose(actualLastFive[i], expectedLastFive[i], 1e-1, 
            `Midpoint mismatch at index ${i}`);
    }
    
    // Compare with Rust implementation
    await compareWithRust('midpoint', result, 'close', { period: 14 });
});

test('Midpoint invalid period', () => {
    // Test Midpoint fails with invalid period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    // Period = 0 should fail
    assert.throws(() => {
        wasm.midpoint_js(inputData, 0);
    }, /Invalid period/);
});

test('Midpoint period exceeds length', () => {
    // Test Midpoint fails with period exceeding length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.midpoint_js(dataSmall, 10);
    }, /Invalid period/);
});

test('Midpoint very small dataset', () => {
    // Test Midpoint fails with insufficient data
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.midpoint_js(singlePoint, 9);
    }, /Invalid period|Not enough valid data/);
});

test('Midpoint empty input', () => {
    // Test Midpoint fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.midpoint_js(empty, 14);
    }, /All values are NaN/);
});

test('Midpoint reinput', () => {
    // Test Midpoint applied twice (re-input) - mirrors check_midpoint_reinput
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.midpoint_js(close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply midpoint to midpoint output
    const secondResult = wasm.midpoint_js(firstResult, 14);
    assert.strictEqual(secondResult.length, firstResult.length);
});

test('Midpoint NaN handling', () => {
    // Test Midpoint handles NaN values correctly
    const close = new Float64Array(testData.close);
    
    const result = wasm.midpoint_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    // First period-1 values should be NaN
    for (let i = 0; i < 13; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('Midpoint all NaN input', () => {
    // Test Midpoint with all NaN values
    const allNan = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.midpoint_js(allNan, 14);
    }, /All values are NaN/);
});

test('Midpoint simple case', () => {
    // Test Midpoint with simple known values
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    const period = 3;
    
    const result = wasm.midpoint_js(data, period);
    
    // First period-1 values should be NaN
    assert(isNaN(result[0]));
    assert(isNaN(result[1]));
    
    // At index 2: window [1, 2, 3], min=1, max=3, midpoint=2
    assertClose(result[2], 2.0, 1e-10);
    
    // At index 3: window [2, 3, 4], min=2, max=4, midpoint=3
    assertClose(result[3], 3.0, 1e-10);
    
    // At index 4: window [3, 4, 5], min=3, max=5, midpoint=4
    assertClose(result[4], 4.0, 1e-10);
});

test('Midpoint fast API basic', () => {
    // Test the fast/unsafe API
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate input buffer
    const inPtr = wasm.midpoint_alloc(len);
    // Allocate output buffer
    const outPtr = wasm.midpoint_alloc(len);
    
    try {
        // Create memory views
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(close);
        
        // Compute midpoint using fast API
        wasm.midpoint_into(inPtr, outPtr, len, 14);
        
        // Read result back (recreate view in case memory grew)
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        // Verify last 5 values
        const expectedLastFive = [59578.5, 59578.5, 59578.5, 58886.0, 58886.0];
        const actualLastFive = result.slice(-5);
        
        for (let i = 0; i < 5; i++) {
            assertClose(actualLastFive[i], expectedLastFive[i], 1e-1);
        }
    } finally {
        // Always free memory
        wasm.midpoint_free(inPtr, len);
        wasm.midpoint_free(outPtr, len);
    }
});

test('Midpoint fast API in-place', () => {
    // Test the fast API with in-place operation (aliasing)
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    const len = data.length;
    
    // Allocate memory
    const ptr = wasm.midpoint_alloc(len);
    
    try {
        // Create memory view and copy data
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        memView.set(data);
        
        // Use the same buffer for input and output
        wasm.midpoint_into(ptr, ptr, len, 3);
        
        // Read result back (recreate view in case memory grew)
        const result = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        
        // Verify results
        assert(isNaN(result[0]));
        assert(isNaN(result[1]));
        assertClose(result[2], 2.0, 1e-10);
        assertClose(result[3], 3.0, 1e-10);
        assertClose(result[4], 4.0, 1e-10);
    } finally {
        wasm.midpoint_free(ptr, len);
    }
});

test('Midpoint batch API', async () => {
    // Test batch processing
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    const config = {
        period_range: [10, 20, 2]  // periods: 10, 12, 14, 16, 18, 20
    };
    
    const result = wasm.midpoint_batch(close, config);
    
    assert(result.values);
    assert(result.combos);
    assert.strictEqual(result.rows, 6); // 6 different periods
    assert.strictEqual(result.cols, 100);
    assert.strictEqual(result.values.length, 600); // 6 * 100
    
    // Verify first row matches single calculation with period=10
    const singleResult = wasm.midpoint_js(close, 10);
    const firstRow = result.values.slice(0, 100);
    
    for (let i = 0; i < 100; i++) {
        if (!isNaN(singleResult[i]) && !isNaN(firstRow[i])) {
            assertClose(firstRow[i], singleResult[i], 1e-10);
        }
    }
});

test('Midpoint batch single parameter', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const config = {
        period_range: [14, 14, 0]  // Single period
    };
    
    const result = wasm.midpoint_batch(close, config);
    
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, 100);
    
    // Should match single calculation
    const singleResult = wasm.midpoint_js(close, 14);
    const batchValues = result.values;
    
    for (let i = 0; i < 100; i++) {
        if (!isNaN(singleResult[i]) && !isNaN(batchValues[i])) {
            assertClose(batchValues[i], singleResult[i], 1e-10);
        }
    }
});

test('Midpoint batch zero-copy API', () => {
    // Test the zero-copy batch API
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;
    
    // Test with multiple periods: 10, 14, 18
    const periodStart = 10;
    const periodEnd = 18;
    const periodStep = 4;
    const expectedRows = 3; // [10, 14, 18]
    
    // Allocate input buffer
    const inPtr = wasm.midpoint_alloc(len);
    // Allocate output buffer for batch results
    const outPtr = wasm.midpoint_alloc(len * expectedRows);
    
    try {
        // Copy data to input buffer
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(close);
        
        // Run batch computation
        const rows = wasm.midpoint_batch_into(
            inPtr, outPtr, len,
            periodStart, periodEnd, periodStep
        );
        
        assert.strictEqual(rows, expectedRows, 'Should return correct number of rows');
        
        // Read results (recreate view in case memory grew)
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len * rows);
        
        // Verify each row matches individual calculations
        const periods = [10, 14, 18];
        for (let row = 0; row < rows; row++) {
            const period = periods[row];
            const singleResult = wasm.midpoint_js(close, period);
            const rowStart = row * len;
            
            for (let col = 0; col < len; col++) {
                const batchVal = outView[rowStart + col];
                const singleVal = singleResult[col];
                
                if (!isNaN(singleVal) && !isNaN(batchVal)) {
                    assertClose(batchVal, singleVal, 1e-10,
                        `Row ${row} (period ${period}), col ${col} mismatch`);
                }
            }
        }
    } finally {
        // Always free memory
        wasm.midpoint_free(inPtr, len);
        wasm.midpoint_free(outPtr, len * expectedRows);
    }
});

test('Midpoint batch zero-copy with large dataset', () => {
    // Test zero-copy batch with larger dataset
    const size = 1000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    // Test with 3 periods
    const periodStart = 10;
    const periodEnd = 30;
    const periodStep = 10;
    const expectedRows = 3; // [10, 20, 30]
    
    // Allocate buffers
    const inPtr = wasm.midpoint_alloc(size);
    const outPtr = wasm.midpoint_alloc(size * expectedRows);
    
    try {
        // Copy data to input buffer
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, size);
        inView.set(data);
        
        // Run batch computation
        const rows = wasm.midpoint_batch_into(
            inPtr, outPtr, size,
            periodStart, periodEnd, periodStep
        );
        
        assert.strictEqual(rows, expectedRows);
        
        // Read results
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, size * rows);
        
        // Check each row has proper warmup period
        const periods = [10, 20, 30];
        for (let row = 0; row < rows; row++) {
            const period = periods[row];
            const rowStart = row * size;
            
            // Check warmup period has NaN
            for (let i = 0; i < period - 1; i++) {
                assert(isNaN(outView[rowStart + i]), 
                    `Row ${row} (period ${period}): Expected NaN at warmup index ${i}`);
            }
            
            // Check after warmup has values
            for (let i = period - 1; i < Math.min(period + 10, size); i++) {
                assert(!isNaN(outView[rowStart + i]), 
                    `Row ${row} (period ${period}): Expected value at index ${i}, got NaN`);
            }
        }
    } finally {
        wasm.midpoint_free(inPtr, size);
        wasm.midpoint_free(outPtr, size * expectedRows);
    }
});