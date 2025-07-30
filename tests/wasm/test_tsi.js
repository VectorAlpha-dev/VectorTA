/**
 * WASM binding tests for TSI indicator.
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

test('TSI accuracy', () => {
    const tsiResult = wasm.tsi_js(testData.close, 25, 13);
    assert(Array.isArray(tsiResult), 'TSI should return an array');
    assert.strictEqual(tsiResult.length, testData.close.length, 'Output length should match input');
    
    // Check expected values from Rust tests
    const expectedLastFive = [
        -17.757654061849838,
        -17.367527062626184,
        -17.305577681249513,
        -16.937565646991143,
        -17.61825617316731,
    ];
    
    const start = tsiResult.length - 5;
    for (let i = 0; i < 5; i++) {
        assertClose(tsiResult[start + i], expectedLastFive[i], 1e-7, `TSI mismatch at position ${i}`);
    }
});

test('TSI error handling', () => {
    // Test null/empty data
    assert.throws(() => wasm.tsi_js([], 25, 13), 'Should throw on empty data');
    
    // Test invalid periods
    assert.throws(() => wasm.tsi_js(testData.close, 0, 13), 'Should throw on zero long period');
    assert.throws(() => wasm.tsi_js(testData.close, 25, 0), 'Should throw on zero short period');
    
    // Test period exceeds data length
    const smallData = [1, 2, 3, 4, 5];
    assert.throws(() => wasm.tsi_js(smallData, 10, 5), 'Should throw when period exceeds data length');
    
    // Test all NaN data
    const nanData = new Array(100).fill(NaN);
    assert.throws(() => wasm.tsi_js(nanData, 25, 13), 'Should throw on all NaN data');
});

test('TSI fast API', () => {
    const len = testData.close.length;
    const inPtr = wasm.tsi_alloc(len);
    const outPtr = wasm.tsi_alloc(len);
    
    try {
        // Copy data to WASM memory
        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        memory.set(testData.close, inPtr / 8);
        
        // Execute TSI
        wasm.tsi_into(inPtr, outPtr, len, 25, 13);
        
        // Read result
        const result = Array.from(memory.subarray(outPtr / 8, outPtr / 8 + len));
        
        // Verify against safe API
        const safeResult = wasm.tsi_js(testData.close, 25, 13);
        assertArrayClose(result, safeResult, 1e-10, 'Fast API should match safe API');
        
    } finally {
        wasm.tsi_free(inPtr, len);
        wasm.tsi_free(outPtr, len);
    }
});

test('TSI batch API', () => {
    const config = {
        long_period_range: [20, 30, 5],
        short_period_range: [10, 15, 5]
    };
    
    const result = wasm.tsi_batch(testData.close, config);
    
    assert(result.values, 'Batch result should have values');
    assert(result.combos, 'Batch result should have combos');
    assert(result.rows > 0, 'Batch result should have rows');
    assert(result.cols === testData.close.length, 'Batch result cols should match input length');
    
    // Verify batch size
    const expectedCombos = 3 * 2; // (20,25,30) x (10,15)
    assert.strictEqual(result.rows, expectedCombos, 'Should have correct number of combinations');
    assert.strictEqual(result.values.length, result.rows * result.cols, 'Values array should have correct size');
});

test.after(() => {
    console.log('TSI WASM tests completed');
});
