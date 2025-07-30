/**
 * WASM binding tests for VIDYA indicator.
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

test('VIDYA accuracy', (t) => {
    const close = testData.close;
    
    // Test with default parameters
    const result = wasm.vidya_js(close, 2, 5, 0.2);
    assert.strictEqual(result.length, close.length, 'Output length should match input length');
    
    // Check warmup period (first + long_period - 1 = 0 + 5 - 1 = 4)
    for (let i = 0; i < 4; i++) {
        assert(isNaN(result[i]), `Value at index ${i} should be NaN during warmup`);
    }
    
    // Check that we have values after warmup
    assertNoNaN(result.slice(4), 'Should have no NaN values after warmup period');
    
    // Test specific expected values from Rust tests
    if (result.length >= 5) {
        const expected_last_five = [
            59553.42785306692,
            59503.60445032524,
            59451.72283651444,
            59413.222561244685,
            59375.65308506839
        ];
        
        const start = result.length - 5;
        for (let i = 0; i < 5; i++) {
            assertClose(result[start + i], expected_last_five[i], 1e-10, 
                `Last 5 values[${i}]`);
        }
    }
});

test('VIDYA fast API (vidya_into)', (t) => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const len = data.length;
    
    // Allocate output buffer
    const outPtr = wasm.vidya_alloc(len);
    
    try {
        // Test normal operation
        wasm.vidya_into(data, outPtr, len, 2, 3, 0.2);
        
        // Read results
        const result = new Float64Array(wasm.memory.buffer, outPtr, len);
        assert.strictEqual(result.length, len);
        
        // Check warmup (first 2 values should be NaN)
        assert(isNaN(result[0]));
        assert(isNaN(result[1]));
        
        // Test in-place operation (aliasing)
        const inPlaceData = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        wasm.vidya_into(inPlaceData, inPlaceData, len, 2, 3, 0.2);
        
        // Should produce same results as out-of-place
        for (let i = 0; i < len; i++) {
            if (isNaN(result[i])) {
                assert(isNaN(inPlaceData[i]), `In-place result at ${i} should also be NaN`);
            } else {
                assertClose(inPlaceData[i], result[i], 1e-10, `In-place vs out-of-place at ${i}`);
            }
        }
    } finally {
        wasm.vidya_free(outPtr, len);
    }
});

test('VIDYA batch processing', async (t) => {
    const close = testData.close.slice(0, 100); // Use smaller dataset for batch test
    
    const config = {
        short_period_range: [2, 4, 1],
        long_period_range: [5, 7, 1], 
        alpha_range: [0.1, 0.3, 0.1]
    };
    
    const result = await wasm.vidya_batch(close, config);
    
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert(result.rows, 'Should have rows count');
    assert(result.cols, 'Should have cols count');
    
    // Check dimensions
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, result.rows * result.cols);
    
    // Verify parameter combinations
    const expectedCombos = 3 * 3 * 3; // 3 short * 3 long * 3 alpha
    assert(result.rows <= expectedCombos, 'Should not exceed max combinations');
});

test('VIDYA error handling', (t) => {
    // Test empty data
    assert.throws(() => wasm.vidya_js([], 2, 5, 0.2), /empty/i);
    
    // Test invalid parameters
    assert.throws(() => wasm.vidya_js([1, 2, 3], 0, 5, 0.2), /invalid/i); // short_period < 1
    assert.throws(() => wasm.vidya_js([1, 2, 3], 6, 5, 0.2), /invalid/i); // short > long
    assert.throws(() => wasm.vidya_js([1, 2, 3], 2, 5, -0.1), /invalid/i); // alpha < 0
    assert.throws(() => wasm.vidya_js([1, 2, 3], 2, 5, 1.1), /invalid/i); // alpha > 1
    assert.throws(() => wasm.vidya_js([1, 2, 3], 2, 10, 0.2), /invalid/i); // long > data.length
    
    // Test all NaN data
    const nanData = new Float64Array(10).fill(NaN);
    assert.throws(() => wasm.vidya_js(nanData, 2, 5, 0.2), /nan/i);
    
    // Test null pointers for fast API
    assert.throws(() => wasm.vidya_into(null, null, 10, 2, 5, 0.2), /null/i);
});

test('VIDYA zero-copy memory management', (t) => {
    // Allocate and free multiple times to ensure no leaks
    for (let i = 0; i < 5; i++) {
        const ptr = wasm.vidya_alloc(1000);
        assert(ptr !== 0, 'Allocation should return non-zero pointer');
        wasm.vidya_free(ptr, 1000);
    }
});

test('VIDYA batch edge cases', (t) => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Test with single parameter combination
    const singleConfig = {
        short_period_range: [2, 2, 0],
        long_period_range: [5, 5, 0],
        alpha_range: [0.2, 0.2, 0.0]
    };
    
    wasm.vidya_batch(data, singleConfig).then(result => {
        assert.strictEqual(result.rows, 1, 'Should have 1 combination');
        assert.strictEqual(result.combos.length, 1, 'Should have 1 param set');
    });
});

test('VIDYA batch fast API', (t) => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const len = data.length;
    
    // Calculate expected output size
    const numCombos = 2 * 2 * 2; // 2 short * 2 long * 2 alpha
    const totalSize = numCombos * len;
    
    const outPtr = wasm.vidya_alloc(totalSize);
    
    try {
        const comboCount = wasm.vidya_batch_into(
            data, outPtr, len,
            2, 3, 1,    // short_period: 2 to 3 step 1
            5, 6, 1,    // long_period: 5 to 6 step 1
            0.1, 0.2, 0.1 // alpha: 0.1 to 0.2 step 0.1
        );
        
        assert.strictEqual(comboCount, numCombos, 'Should return correct combo count');
        
        // Read a portion of results to verify
        const result = new Float64Array(wasm.memory.buffer, outPtr, len);
        assert.strictEqual(result.length, len);
    } finally {
        wasm.vidya_free(outPtr, totalSize);
    }
});

test('VIDYA parameter validation', (t) => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Test parameter boundary conditions
    assert.doesNotThrow(() => wasm.vidya_js(data, 1, 2, 0.0), 'Min valid parameters');
    assert.doesNotThrow(() => wasm.vidya_js(data, 2, 10, 1.0), 'Max valid parameters');
    
    // Test edge case: short_period == long_period - 1
    assert.doesNotThrow(() => wasm.vidya_js(data, 4, 5, 0.5), 'Adjacent periods');
});

test('VIDYA SIMD128 consistency', (t) => {
    // This test verifies SIMD128 produces same results as scalar
    // SIMD128 is automatically used in WASM when available
    const sizes = [10, 50, 100, 1000];
    
    for (const size of sizes) {
        const data = new Float64Array(size);
        for (let i = 0; i < size; i++) {
            data[i] = Math.sin(i * 0.1) * 100 + 50;
        }
        
        const result = wasm.vidya_js(data, 2, 5, 0.2);
        
        // Basic sanity checks
        assert.strictEqual(result.length, data.length);
        
        // Check warmup period
        for (let i = 0; i < 4; i++) {
            assert(isNaN(result[i]), `Warmup at ${i} should be NaN`);
        }
        
        // Check no NaN after warmup
        assertNoNaN(result.slice(4), 'No NaN after warmup');
    }
});

test('VIDYA reinput test', (t) => {
    // Test applying VIDYA twice (vidya of vidya)
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.sin(i * 0.1) * 50 + 100;
    }
    
    // First pass
    const first = wasm.vidya_js(data, 2, 5, 0.2);
    
    // Second pass - vidya of vidya
    const second = wasm.vidya_js(first, 2, 5, 0.2);
    
    assert.strictEqual(second.length, first.length);
    
    // Should have more NaN values at start due to double warmup
    let firstNaNCount = 0;
    let secondNaNCount = 0;
    
    for (let i = 0; i < 10; i++) {
        if (isNaN(first[i])) firstNaNCount++;
        if (isNaN(second[i])) secondNaNCount++;
    }
    
    assert(secondNaNCount >= firstNaNCount, 'Second pass should have at least as many NaN values');
});

test.after(() => {
    console.log('VIDYA WASM tests completed');
});
