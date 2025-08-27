/**
 * WASM binding tests for MOM indicator.
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

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
    // Load WASM module - dynamically import CommonJS module
    try {
        // Use createRequire to load CommonJS module from ES module
        const { createRequire } = await import('module');
        const require = createRequire(import.meta.url);
        wasm = require('../../pkg/my_project.js');
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        console.error('Error:', error);
        throw error;
    }
    
    testData = loadTestData();
});

test('MOM accuracy', () => {
    const period = EXPECTED_OUTPUTS.mom.defaultParams.period;
    const result = wasm.mom_js(testData.close, period);
    
    // Result can be either Array or Float64Array
    assert(result && (Array.isArray(result) || result instanceof Float64Array), 'MOM should return an array or typed array');
    assert.strictEqual(result.length, testData.close.length, 'Result length should match input length');
    
    // Check warmup period
    for (let i = 0; i < period; i++) {
        assert(isNaN(result[i]), `Value at index ${i} should be NaN (warmup period)`);
    }
    
    // Check last 5 values match expected
    const expectedLast5 = EXPECTED_OUTPUTS.mom.last5Values;
    const resultLast5 = result.slice(-5);
    
    for (let i = 0; i < 5; i++) {
        assertClose(resultLast5[i], expectedLast5[i], 0.1, `MOM last-5[${i}]`);
    }
});

test('MOM error handling', () => {
    // Test empty data - should now return EmptyInputData error
    assert.throws(() => {
        wasm.mom_js([], 10);
    }, /Input data slice is empty|Invalid period/);
    
    // Test all NaN values
    const nanData = new Array(10).fill(NaN);
    assert.throws(() => {
        wasm.mom_js(nanData, 5);
    }, /All values are NaN/);
    
    // Test period exceeds data length
    const shortData = [10.0, 20.0, 30.0];
    assert.throws(() => {
        wasm.mom_js(shortData, 10);
    }, /Invalid period/);
    
    // Test zero period
    assert.throws(() => {
        wasm.mom_js([10.0, 20.0, 30.0], 0);
    }, /Invalid period/);
});

test('MOM fast API (in-place)', async () => {
    const period = 10;
    
    // Allocate memory for input/output
    const ptr = wasm.mom_alloc(testData.close.length);
    
    try {
        // Copy data to WASM memory - wasm.memory might not be exposed, use the wasm instance
        const memory = wasm.memory || wasm.__wasm.memory || wasm.wasm?.memory;
        if (!memory) {
            console.log('Skipping fast API test - memory not exposed');
            return;
        }
        const wasmMemory = new Float64Array(memory.buffer, ptr, testData.close.length);
        wasmMemory.set(testData.close);
        
        // Compute in-place (same pointer for input and output)
        wasm.mom_into(ptr, ptr, testData.close.length, period);
        
        // Read results
        const result = Array.from(wasmMemory);
        
        // Verify against safe API
        const expected = wasm.mom_js(testData.close, period);
        assertArrayClose(result, Array.from(expected), 1e-10);
        
    } finally {
        // Clean up
        wasm.mom_free(ptr, testData.close.length);
    }
});

test('MOM batch API', () => {
    const config = {
        period_range: [5, 15, 5]  // 5, 10, 15
    };
    
    const result = wasm.mom_batch(testData.close, config);
    
    assert(result.values, 'Batch result should have values');
    assert(result.combos, 'Batch result should have combos');
    assert.strictEqual(result.rows, 3, 'Should have 3 rows for periods 5, 10, 15');
    assert.strictEqual(result.cols, testData.close.length, 'Columns should match data length');
    assert.strictEqual(result.values.length, result.rows * result.cols, 'Values array size should be rows * cols');
    
    // Verify one of the results matches single computation
    const period10Result = wasm.mom_js(testData.close, 10);
    const row1 = result.values.slice(result.cols, 2 * result.cols); // Second row (period=10)
    assertArrayClose(row1, period10Result, 1e-10);
});

test('MOM partial params', () => {
    // Test with default parameters - mirrors check_mom_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.mom_js(close, 10); // Default period
    assert.strictEqual(result.length, close.length, 'Result length should match input');
});

test('MOM very small dataset', () => {
    // Test MOM fails with insufficient data - mirrors check_mom_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.mom_js(singlePoint, 9);
    }, /Invalid period|Not enough valid data/);
});

test('MOM empty input', () => {
    // Test MOM fails with empty input - mirrors check_mom_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.mom_js(empty, 10);
    }, /Input data slice is empty/);
});


test('MOM NaN handling', () => {
    // Test MOM handles NaN values correctly - mirrors check_mom_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.mom_js(close, 10);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First period values should be NaN
    assertAllNaN(result.slice(0, 10), "Expected NaN in warmup period");
});

test('MOM all NaN input', () => {
    // Test MOM with all NaN values
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.mom_js(allNaN, 10);
    }, /All values are NaN/);
});

test('MOM warmup period', () => {
    // Test that MOM correctly handles warmup period
    const close = new Float64Array(testData.close.slice(0, 100));
    const period = 10;
    
    const result = wasm.mom_js(close, period);
    
    // First `period` values should be NaN
    for (let i = 0; i < period; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} in warmup period`);
    }
    
    // After warmup, values should be defined
    for (let i = period; i < result.length; i++) {
        assert(!isNaN(result[i]), `Unexpected NaN at index ${i} after warmup`);
    }
    
    // Verify the calculation for a specific point
    // MOM(i) = close[i] - close[i-period]
    for (let i = period; i < Math.min(period + 5, close.length); i++) {
        const expected = close[i] - close[i - period];
        assertClose(result[i], expected, 1e-10, `MOM calculation mismatch at index ${i}`);
    }
});

test('MOM batch into API', () => {
    // Test batch_into for direct memory operations
    const periods = [5, 10, 15];
    const rows = periods.length;
    const cols = testData.close.length;
    
    // Allocate memory for input and output
    const inPtr = wasm.mom_alloc(cols);
    const outPtr = wasm.mom_alloc(rows * cols);
    
    try {
        // Copy data to WASM memory - check if memory is exposed
        const memory = wasm.memory || wasm.__wasm.memory || wasm.wasm?.memory;
        if (!memory) {
            console.log('Skipping batch into API test - memory not exposed');
            wasm.mom_free(inPtr, cols);
            wasm.mom_free(outPtr, rows * cols);
            return;
        }
        const wasmMemory = new Float64Array(memory.buffer, inPtr, cols);
        wasmMemory.set(testData.close);
        
        // Compute batch into output buffer
        const resultRows = wasm.mom_batch_into(
            inPtr, outPtr, cols,
            5, 15, 5  // period range
        );
        
        assert.strictEqual(resultRows, rows, 'Should return correct number of rows');
        
        // Read results
        const outMemory = new Float64Array(memory.buffer, outPtr, rows * cols);
        const result = Array.from(outMemory);
        
        // Verify middle row (period=10) matches single computation
        const period10Result = wasm.mom_js(testData.close, 10);
        const row1 = result.slice(cols, 2 * cols);
        assertArrayClose(row1, Array.from(period10Result), 1e-10, 'Batch into row should match single computation');
        
    } finally {
        // Clean up
        wasm.mom_free(inPtr, cols);
        wasm.mom_free(outPtr, rows * cols);
    }
});

test.after(() => {
    console.log('MOM WASM tests completed');
});
