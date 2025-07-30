/**
 * WASM binding tests for DX indicator.
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

test('DX basic functionality', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    
    // Test with default parameters
    const result = wasm.dx_js(high, low, close, 14);
    assert.strictEqual(result.length, high.length, 'Output length should match input length');
    
    // Verify warmup period (should be at least period - 1)
    const warmup = 14 - 1;
    for (let i = 0; i < warmup; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup period`);
    }
    
    // Verify we have some non-NaN values after warmup
    let hasValidValues = false;
    for (let i = warmup + 10; i < result.length; i++) {
        if (!isNaN(result[i])) {
            hasValidValues = true;
            break;
        }
    }
    assert(hasValidValues, 'Should have valid values after warmup period');
    
    // DX values should be between 0 and 100
    for (let i = warmup; i < result.length; i++) {
        if (!isNaN(result[i])) {
            assert(result[i] >= 0 && result[i] <= 100, 
                `DX value at index ${i} should be between 0 and 100, got ${result[i]}`);
        }
    }
});

test('DX fast API', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low); 
    const close = new Float64Array(testData.close);
    const len = high.length;
    
    // Allocate output buffer
    const outPtr = wasm.dx_alloc(len);
    
    try {
        // Call fast API
        wasm.dx_into(
            high.buffer,
            low.buffer,
            close.buffer,
            outPtr,
            len,
            14
        );
        
        // Read results
        const result = new Float64Array(wasm.memory.buffer, outPtr, len);
        const resultCopy = Float64Array.from(result);
        
        // Compare with safe API
        const safeResult = wasm.dx_js(high, low, close, 14);
        assertArrayClose(resultCopy, safeResult, 1e-10, 'Fast API should match safe API');
        
    } finally {
        wasm.dx_free(outPtr, len);
    }
});

test('DX fast API with aliasing', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const len = high.length;
    
    // Test aliasing with high array
    const highCopy = Float64Array.from(high);
    wasm.dx_into(
        high.buffer,
        low.buffer,
        close.buffer,
        high.buffer,  // Output to high buffer (aliasing)
        len,
        14
    );
    
    // The function should handle aliasing correctly
    const expected = wasm.dx_js(highCopy, low, close, 14);
    assertArrayClose(high, expected, 1e-10, 'Should handle aliasing correctly');
});

test('DX batch API', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    
    const config = {
        period_range: [10, 20, 5]  // 10, 15, 20
    };
    
    const result = wasm.dx_batch(high, low, close, config);
    
    // Verify structure
    assert(result.values, 'Result should have values array');
    assert(result.combos, 'Result should have combos array');
    assert.strictEqual(result.rows, 3, 'Should have 3 parameter combinations');
    assert.strictEqual(result.cols, high.length, 'Cols should match input length');
    assert.strictEqual(result.values.length, 3 * high.length, 'Values should be flattened matrix');
    
    // Verify each batch result matches individual computation
    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const batchRow = result.values.slice(i * high.length, (i + 1) * high.length);
        const singleResult = wasm.dx_js(high, low, close, periods[i]);
        assertArrayClose(batchRow, singleResult, 1e-10, 
            `Batch result for period ${periods[i]} should match single computation`);
    }
});

test('DX error handling', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    
    // Test with empty arrays
    assert.throws(() => {
        wasm.dx_js([], [], [], 14);
    }, 'Should throw on empty input');
    
    // Test with mismatched lengths
    assert.throws(() => {
        wasm.dx_js(high.slice(0, 10), low.slice(0, 5), close, 14);
    }, 'Should throw on mismatched input lengths');
    
    // Test with period too large
    assert.throws(() => {
        wasm.dx_js(high.slice(0, 10), low.slice(0, 10), close.slice(0, 10), 20);
    }, 'Should throw when period exceeds data length');
    
    // Test with period = 0
    assert.throws(() => {
        wasm.dx_js(high, low, close, 0);
    }, 'Should throw on zero period');
});

test.after(() => {
    console.log('DX WASM tests completed');
});
