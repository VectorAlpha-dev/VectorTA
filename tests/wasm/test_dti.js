/**
 * WASM binding tests for DTI indicator.
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

test('DTI basic functionality', () => {
    const high = testData.high;
    const low = testData.low;
    
    // Test with default parameters
    const result = wasm.dti_js(high, low, 14, 10, 5);
    assert.strictEqual(result.length, high.length, 'Output length should match input length');
    
    // Verify warmup period
    const warmup = Math.max(14, 10, 5);
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
});

test('DTI fast API (in-place)', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const len = high.length;
    
    // Allocate output buffer
    const outPtr = wasm.dti_alloc(len);
    
    try {
        // Call fast API
        wasm.dti_into(
            high.buffer, 
            low.buffer, 
            outPtr, 
            len, 
            14, 10, 5
        );
        
        // Read results
        const result = new Float64Array(wasm.memory.buffer, outPtr, len);
        const resultCopy = Float64Array.from(result);
        
        // Compare with safe API
        const safeResult = wasm.dti_js(high, low, 14, 10, 5);
        assertArrayClose(resultCopy, safeResult, 1e-10, 'Fast API should match safe API');
        
    } finally {
        wasm.dti_free(outPtr, len);
    }
});

test('DTI fast API with aliasing', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const len = high.length;
    
    // Test aliasing with high array
    const highCopy = Float64Array.from(high);
    wasm.dti_into(
        high.buffer,
        low.buffer,
        high.buffer, // Output to same as high input
        len,
        14, 10, 5
    );
    
    // Verify result is correct despite aliasing
    const expected = wasm.dti_js(highCopy, low, 14, 10, 5);
    assertArrayClose(high, expected, 1e-10, 'Aliasing with high array should work');
    
    // Test aliasing with low array
    const lowCopy = Float64Array.from(low);
    wasm.dti_into(
        highCopy.buffer,
        low.buffer,
        low.buffer, // Output to same as low input
        len,
        14, 10, 5
    );
    
    assertArrayClose(low, expected, 1e-10, 'Aliasing with low array should work');
});

test('DTI batch processing', () => {
    const high = testData.high;
    const low = testData.low;
    
    const config = {
        r_range: [14, 14, 1],
        s_range: [10, 10, 1],
        u_range: [5, 5, 1]
    };
    
    const result = wasm.dti_batch(high, low, config);
    
    assert(result.values, 'Batch result should have values');
    assert(result.combos, 'Batch result should have combos');
    assert.strictEqual(result.rows, 1, 'Should have 1 row for single combination');
    assert.strictEqual(result.cols, high.length, 'Columns should match input length');
    assert.strictEqual(result.values.length, high.length, 'Values should be flattened array');
    
    // Compare with single calculation
    const singleResult = wasm.dti_js(high, low, 14, 10, 5);
    assertArrayClose(result.values, singleResult, 1e-10, 'Batch should match single calculation');
});

test('DTI error handling - empty data', () => {
    assert.throws(() => {
        wasm.dti_js([], [], 14, 10, 5);
    }, /Empty data/, 'Should throw on empty data');
});

test('DTI error handling - zero period', () => {
    const high = [10.0, 11.0, 12.0];
    const low = [9.0, 10.0, 11.0];
    
    assert.throws(() => {
        wasm.dti_js(high, low, 0, 10, 5);
    }, /Invalid period/, 'Should throw on zero r period');
    
    assert.throws(() => {
        wasm.dti_js(high, low, 14, 0, 5);
    }, /Invalid period/, 'Should throw on zero s period');
    
    assert.throws(() => {
        wasm.dti_js(high, low, 14, 10, 0);
    }, /Invalid period/, 'Should throw on zero u period');
});

test('DTI error handling - period exceeds length', () => {
    const high = [10.0, 11.0];
    const low = [9.0, 10.0];
    
    assert.throws(() => {
        wasm.dti_js(high, low, 14, 10, 5);
    }, /Invalid period/, 'Should throw when period exceeds data length');
});

test('DTI error handling - all NaN values', () => {
    const high = [NaN, NaN, NaN];
    const low = [NaN, NaN, NaN];
    
    assert.throws(() => {
        wasm.dti_js(high, low, 1, 1, 1);
    }, /All.*values are NaN/, 'Should throw on all NaN values');
});

test('DTI error handling - null pointers', () => {
    assert.throws(() => {
        wasm.dti_into(null, null, null, 100, 14, 10, 5);
    }, /Null pointer/, 'Should throw on null pointers');
});

test('DTI memory allocation/deallocation', () => {
    const len = 1000;
    
    // Test allocation
    const ptr = wasm.dti_alloc(len);
    assert(ptr !== 0, 'Should return non-zero pointer');
    
    // Test we can write to allocated memory
    const arr = new Float64Array(wasm.memory.buffer, ptr, len);
    arr[0] = 42.0;
    assert.strictEqual(arr[0], 42.0, 'Should be able to write to allocated memory');
    
    // Test deallocation (should not throw)
    assert.doesNotThrow(() => {
        wasm.dti_free(ptr, len);
    }, 'Deallocation should not throw');
    
    // Test free with null pointer (should not throw)
    assert.doesNotThrow(() => {
        wasm.dti_free(0, len);
    }, 'Free with null pointer should not throw');
});

test('DTI batch with multiple parameter combinations', () => {
    const high = testData.high.slice(0, 100); // Use smaller dataset for speed
    const low = testData.low.slice(0, 100);
    
    const config = {
        r_range: [10, 20, 5],    // 10, 15, 20
        s_range: [8, 12, 2],     // 8, 10, 12
        u_range: [4, 6, 1]       // 4, 5, 6
    };
    
    const result = wasm.dti_batch(high, low, config);
    
    // Should have 3 * 3 * 3 = 27 combinations
    assert.strictEqual(result.rows, 27, 'Should have 27 parameter combinations');
    assert.strictEqual(result.cols, high.length, 'Columns should match input length');
    assert.strictEqual(result.values.length, 27 * high.length, 'Values should be flattened matrix');
    assert.strictEqual(result.combos.length, 27, 'Should have 27 combo objects');
    
    // Verify first combo matches expected parameters
    const firstCombo = result.combos[0];
    assert.strictEqual(firstCombo.r, 10, 'First r should be 10');
    assert.strictEqual(firstCombo.s, 8, 'First s should be 8');
    assert.strictEqual(firstCombo.u, 4, 'First u should be 4');
});

test('DTI batch fast API (dti_batch_into)', () => {
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const len = high.length;
    
    // Parameters for 3 * 2 * 2 = 12 combinations
    const r_start = 10, r_end = 20, r_step = 5;  // 10, 15, 20
    const s_start = 8, s_end = 10, s_step = 2;   // 8, 10
    const u_start = 4, u_end = 5, u_step = 1;    // 4, 5
    
    const expectedRows = 3 * 2 * 2;
    const totalSize = expectedRows * len;
    
    // Allocate output buffer
    const outPtr = wasm.dti_alloc(totalSize);
    
    try {
        // Call fast batch API
        const rows = wasm.dti_batch_into(
            high.buffer,
            low.buffer,
            outPtr,
            len,
            r_start, r_end, r_step,
            s_start, s_end, s_step,
            u_start, u_end, u_step
        );
        
        assert.strictEqual(rows, expectedRows, 'Should return correct number of rows');
        
        // Read results
        const result = new Float64Array(wasm.memory.buffer, outPtr, totalSize);
        const resultCopy = Float64Array.from(result);
        
        // Verify first row matches single calculation with same params
        const firstRow = resultCopy.slice(0, len);
        const expected = wasm.dti_js(high, low, r_start, s_start, u_start);
        assertArrayClose(firstRow, expected, 1e-10, 'First row should match single calculation');
        
    } finally {
        wasm.dti_free(outPtr, totalSize);
    }
});

test.after(() => {
    console.log('DTI WASM tests completed');
});
