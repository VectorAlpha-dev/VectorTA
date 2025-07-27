/**
 * WASM binding tests for MOM indicator.
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

test('MOM accuracy', () => {
    const period = EXPECTED_OUTPUTS.mom.defaultParams.period;
    const result = wasm.mom_js(testData.close, period);
    
    assert(Array.isArray(result), 'MOM should return an array');
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
    // Test empty data
    assert.throws(() => {
        wasm.mom_js([], 10);
    }, /All values are NaN/);
    
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
        // Copy data to WASM memory
        const wasmMemory = new Float64Array(wasm.memory.buffer, ptr, testData.close.length);
        wasmMemory.set(testData.close);
        
        // Compute in-place (same pointer for input and output)
        wasm.mom_into(ptr, ptr, testData.close.length, period);
        
        // Read results
        const result = Array.from(wasmMemory);
        
        // Verify against safe API
        const expected = wasm.mom_js(testData.close, period);
        assertArrayClose(result, expected, 1e-10);
        
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

test.after(() => {
    console.log('MOM WASM tests completed');
});
