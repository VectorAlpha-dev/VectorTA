/**
 * WASM binding tests for KVO indicator.
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

test('KVO accuracy', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    const volume = testData.volume;
    
    // Using default parameters: short_period=2, long_period=5
    const result = wasm.kvo_js(high, low, close, volume, 2, 5);
    
    assert.strictEqual(result.length, close.length, 'Result length should match input length');
    
    // Check last 5 values match expected from Rust tests
    const expectedLastFive = [
        -246.42698280402647,
        530.8651474164992,
        237.2148311016648,
        608.8044103976362,
        -6339.615516805162,
    ];
    
    const actualLastFive = result.slice(-5);
    assertArrayClose(actualLastFive, expectedLastFive, 1e-1, 'KVO last 5 values mismatch');
});

test('KVO with default parameters', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    const volume = testData.volume;
    
    // Test with default values (short=2, long=5)
    const result = wasm.kvo_js(high, low, close, volume, 2, 5);
    assert(Array.isArray(result), 'Result should be an array');
    assert.strictEqual(result.length, close.length, 'Result length should match input length');
});

test('KVO error handling - zero period', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    const volume = testData.volume;
    
    assert.throws(
        () => wasm.kvo_js(high, low, close, volume, 0, 5),
        /Invalid/,
        'Should throw error for zero short period'
    );
});

test('KVO error handling - invalid period', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    const volume = testData.volume;
    
    assert.throws(
        () => wasm.kvo_js(high, low, close, volume, 5, 2),
        /Invalid/,
        'Should throw error when long_period < short_period'
    );
});

test('KVO error handling - insufficient data', () => {
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(
        () => wasm.kvo_js(singlePoint, singlePoint, singlePoint, singlePoint, 2, 5),
        /Not enough valid data/,
        'Should throw error for insufficient data'
    );
});

test('KVO error handling - empty input', () => {
    const empty = new Float64Array([]);
    
    assert.throws(
        () => wasm.kvo_js(empty, empty, empty, empty, 2, 5),
        /Empty/,
        'Should throw error for empty input'
    );
});

test('KVO error handling - all NaN input', () => {
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(
        () => wasm.kvo_js(allNaN, allNaN, allNaN, allNaN, 2, 5),
        /All values are NaN/,
        'Should throw error for all NaN values'
    );
});

test('KVO NaN handling', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    const volume = testData.volume;
    
    const result = wasm.kvo_js(high, low, close, volume, 2, 5);
    assert.strictEqual(result.length, close.length, 'Result length should match input length');
    
    // Check that we have expected number of NaN values at the beginning
    let nanCount = 0;
    for (let i = 0; i < result.length; i++) {
        if (isNaN(result[i])) {
            nanCount++;
        } else {
            break;
        }
    }
    
    // KVO has a warmup period based on first valid data
    assert(nanCount > 0, 'Should have NaN values during warmup period');
});

test('KVO batch processing', () => {
    const high = testData.high.slice(0, 100); // Use smaller dataset for speed
    const low = testData.low.slice(0, 100);
    const close = testData.close.slice(0, 100);
    const volume = testData.volume.slice(0, 100);
    
    // Test batch with single parameter set
    const config = {
        short_period_range: [2, 2, 0],
        long_period_range: [5, 5, 0]
    };
    
    const result = wasm.kvo_batch(high, low, close, volume, config);
    
    assert(result.values, 'Result should have values array');
    assert(result.combos, 'Result should have combos array');
    assert.strictEqual(result.rows, 1, 'Should have 1 row for single parameter set');
    assert.strictEqual(result.cols, 100, 'Should have 100 columns');
    
    // Compare with single calculation
    const singleResult = wasm.kvo_js(high, low, close, volume, 2, 5);
    assertArrayClose(
        result.values, 
        singleResult, 
        1e-10, 
        'Batch result should match single calculation'
    );
});

test('KVO batch with multiple parameters', () => {
    const high = testData.high.slice(0, 50); // Use smaller dataset for speed
    const low = testData.low.slice(0, 50);
    const close = testData.close.slice(0, 50);
    const volume = testData.volume.slice(0, 50);
    
    // Multiple parameter combinations
    const config = {
        short_period_range: [2, 3, 1],    // 2, 3
        long_period_range: [5, 6, 1]      // 5, 6
    };
    
    const result = wasm.kvo_batch(high, low, close, volume, config);
    
    // Should have 2 * 2 = 4 combinations
    assert.strictEqual(result.rows, 4, 'Should have 4 parameter combinations');
    assert.strictEqual(result.cols, 50, 'Should have 50 columns');
    assert.strictEqual(result.values.length, 200, 'Should have 4 * 50 = 200 values');
    
    // Check first combination matches single calculation
    const firstRow = result.values.slice(0, 50);
    const singleResult = wasm.kvo_js(high, low, close, volume, 2, 5);
    
    assertArrayClose(firstRow, singleResult, 1e-10, 'First batch row should match single calculation');
});

test('KVO memory allocation/deallocation', () => {
    const len = 1000;
    const ptr = wasm.kvo_alloc(len);
    
    assert(ptr !== 0, 'Allocated pointer should not be null');
    
    // Free the memory
    wasm.kvo_free(ptr, len);
    
    // Test multiple allocations
    const ptrs = [];
    for (let i = 0; i < 10; i++) {
        ptrs.push(wasm.kvo_alloc(100));
    }
    
    // Free all
    ptrs.forEach(p => wasm.kvo_free(p, 100));
});

test('KVO fast API (kvo_into)', () => {
    const high = testData.high.slice(0, 100);
    const low = testData.low.slice(0, 100);
    const close = testData.close.slice(0, 100);
    const volume = testData.volume.slice(0, 100);
    const len = high.length;
    
    // Allocate output buffer
    const outPtr = wasm.kvo_alloc(len);
    
    try {
        // Create typed arrays from test data
        const highBytes = new Float64Array(high);
        const lowBytes = new Float64Array(low);
        const closeBytes = new Float64Array(close);
        const volumeBytes = new Float64Array(volume);
        
        // Call fast API
        wasm.kvo_into(
            highBytes.buffer,
            lowBytes.buffer,
            closeBytes.buffer,
            volumeBytes.buffer,
            outPtr,
            len,
            2,
            5
        );
        
        // Read results from WASM memory
        const memory = new Float64Array(wasm.memory.buffer, outPtr, len);
        const result = Array.from(memory);
        
        // Compare with safe API
        const expected = wasm.kvo_js(high, low, close, volume, 2, 5);
        assertArrayClose(result, expected, 1e-14, 'Fast API should match safe API');
        
    } finally {
        wasm.kvo_free(outPtr, len);
    }
});

test.after(() => {
    console.log('KVO WASM tests completed');
});
