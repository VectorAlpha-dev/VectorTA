/**
 * WASM binding tests for QSTICK indicator.
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

test('QSTICK accuracy', () => {
    const open_data = testData.open;
    const close_data = testData.close;
    const period = 5;
    
    // Calculate QSTICK
    const result = wasm.qstick_js(open_data, close_data, period);
    
    // Expected values from Rust tests
    const expected_last_five = [219.4, 61.6, -51.8, -53.4, -123.2];
    
    // Check last 5 values
    for (let i = 0; i < expected_last_five.length; i++) {
        const actual = result[result.length - (5 - i)];
        assertClose(actual, expected_last_five[i], 1e-1, 
            `QSTICK mismatch at index ${i}: expected ${expected_last_five[i]}, got ${actual}`);
    }
});

test('QSTICK error handling', () => {
    // Test mismatched lengths
    assert.throws(() => {
        wasm.qstick_js(new Float64Array([10.0, 20.0]), new Float64Array([15.0, 25.0, 30.0]), 5);
    }, /must have the same length/);
    
    // Test zero period
    assert.throws(() => {
        wasm.qstick_js(new Float64Array([10.0, 20.0]), new Float64Array([15.0, 25.0]), 0);
    }, /Invalid period/);
    
    // Test period exceeds length
    assert.throws(() => {
        wasm.qstick_js(new Float64Array([10.0, 20.0]), new Float64Array([15.0, 25.0]), 10);
    }, /Invalid period/);
    
    // Test empty data
    assert.throws(() => {
        wasm.qstick_js(new Float64Array([]), new Float64Array([]), 5);
    }, /Invalid period/);
});

test('QSTICK fast API', () => {
    const open_data = testData.open;
    const close_data = testData.close;
    const period = 5;
    const len = open_data.length;
    
    // Allocate output buffer
    const out_ptr = wasm.qstick_alloc(len);
    
    try {
        // Compute using fast API
        wasm.qstick_into(open_data, close_data, out_ptr, len, period);
        
        // Get results from memory
        const memory = wasm.memory.buffer;
        const result = new Float64Array(memory, out_ptr, len);
        
        // Compare with safe API
        const expected = wasm.qstick_js(open_data, close_data, period);
        assertArrayClose(result, expected, 1e-10, 'Fast API should match safe API');
    } finally {
        // Clean up
        wasm.qstick_free(out_ptr, len);
    }
});

test('QSTICK batch API', () => {
    const open_data = testData.open;
    const close_data = testData.close;
    
    // Test batch with period range
    const config = {
        period_range: [5, 20, 5]  // 4 values: 5, 10, 15, 20
    };
    
    const result = wasm.qstick_batch(open_data, close_data, config);
    
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert.equal(result.rows, 4, 'Should have 4 rows');
    assert.equal(result.cols, open_data.length, 'Should have correct columns');
    assert.equal(result.combos.length, 4, 'Should have 4 parameter combinations');
    
    // First row should match single calculation with period=5
    const single_result = wasm.qstick_js(open_data, close_data, 5);
    const first_row = result.values.slice(0, open_data.length);
    assertArrayClose(first_row, single_result, 1e-10, 'First batch row should match single calculation');
});

test.after(() => {
    console.log('QSTICK WASM tests completed');
});
