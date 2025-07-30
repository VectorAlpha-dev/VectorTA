/**
 * WASM binding tests for MEAN_AD indicator.
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

test('MEAN_AD accuracy', () => {
    // Test with hl2 source like in Rust
    const hl2 = testData.high.map((h, i) => (h + testData.low[i]) / 2);
    const result = wasm.mean_ad_js(hl2, 5);
    
    const expected = EXPECTED_OUTPUTS.meanAd.last5Values;
    const last5 = result.slice(-5);
    
    for (let i = 0; i < 5; i++) {
        assertClose(last5[i], expected[i], 1e-1, `mean_ad last value ${i} mismatch`);
    }
});

test('MEAN_AD default params', () => {
    const result = wasm.mean_ad_js(testData.close, 5);
    assert.strictEqual(result.length, testData.close.length, 'Output length mismatch');
    
    // Check warmup period (first + 2 * period - 2 values should be NaN)
    // For period=5, warmup should be 8 values (0 + 2*5 - 2 = 8)
    for (let i = 0; i < 8; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    // Check no NaN after warmup
    for (let i = 240; i < result.length; i++) {
        assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
    }
});

test('MEAN_AD error handling', () => {
    // Test with empty data
    assert.throws(() => {
        wasm.mean_ad_js(new Float64Array(0), 5);
    }, 'Empty data should throw error');
    
    // Test with zero period
    assert.throws(() => {
        wasm.mean_ad_js(new Float64Array([1, 2, 3]), 0);
    }, 'Zero period should throw error');
    
    // Test with period exceeding data length
    assert.throws(() => {
        wasm.mean_ad_js(new Float64Array([1, 2, 3]), 10);
    }, 'Period > data length should throw error');
});

test('MEAN_AD fast API', () => {
    const data = testData.close;
    const period = 5;
    
    // Allocate output buffer
    const outPtr = wasm.mean_ad_alloc(data.length);
    
    try {
        // Test normal operation
        wasm.mean_ad_into(data, outPtr, data.length, period);
        const result = new Float64Array(wasm.memory.buffer, outPtr, data.length);
        
        // Compare with safe API
        const expected = wasm.mean_ad_js(data, period);
        assertArrayClose(result, expected, 1e-10, 'Fast API mismatch');
        
        // Test in-place operation (aliasing)
        const inPlaceData = new Float64Array(data);
        wasm.mean_ad_into(inPlaceData, inPlaceData, data.length, period);
        assertArrayClose(inPlaceData, expected, 1e-10, 'In-place operation mismatch');
    } finally {
        wasm.mean_ad_free(outPtr, data.length);
    }
});

test('MEAN_AD batch API', () => {
    const config = {
        period_range: [5, 50, 5] // 5 to 50, step 5
    };
    
    const result = wasm.mean_ad_batch(testData.close.slice(0, 100), config);
    
    assert(result.values, 'Batch result should have values');
    assert(result.periods, 'Batch result should have periods');
    assert.strictEqual(result.rows, 10, 'Should have 10 period combinations');
    assert.strictEqual(result.cols, 100, 'Should have 100 data points');
    assert.strictEqual(result.values.length, 1000, 'Values array size mismatch');
    assert.strictEqual(result.periods.length, 10, 'Periods array size mismatch');
    
    // Check periods are correct
    const expectedPeriods = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
    assert.deepStrictEqual(result.periods, expectedPeriods, 'Period values mismatch');
});

test('MEAN_AD NaN handling', () => {
    const dataWithNaN = new Float64Array([NaN, NaN, 1, 2, 3, 4, 5, NaN, 6, 7]);
    const result = wasm.mean_ad_js(dataWithNaN, 3);
    
    // First non-NaN is at index 2, warmup = 2 + 2*3 - 2 = 6
    for (let i = 0; i < 6; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    // Should have valid values after warmup
    assert(!isNaN(result[6]), 'Expected valid value at index 6');
    assert(!isNaN(result[9]), 'Expected valid value at index 9');
});

test('MEAN_AD all NaN input', () => {
    const allNaN = new Float64Array([NaN, NaN, NaN, NaN, NaN]);
    assert.throws(() => {
        wasm.mean_ad_js(allNaN, 3);
    }, 'All NaN values should throw error');
});

test('MEAN_AD partial params', () => {
    // mean_ad only has period parameter, so this is simpler than alma
    const result = wasm.mean_ad_js(testData.close, 5);
    assert(result.length === testData.close.length, 'Should handle default parameters');
});

test('MEAN_AD zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i / 100) * 100 + 50;
    }
    
    const outPtr = wasm.mean_ad_alloc(size);
    
    try {
        const start = performance.now();
        wasm.mean_ad_into(data, outPtr, size, 50);
        const fastTime = performance.now() - start;
        
        const start2 = performance.now();
        const safeResult = wasm.mean_ad_js(data, 50);
        const safeTime = performance.now() - start2;
        
        // Fast API should be faster (or at least not significantly slower)
        console.log(`Fast API: ${fastTime.toFixed(2)}ms, Safe API: ${safeTime.toFixed(2)}ms`);
        
        // Compare results
        const result = new Float64Array(wasm.memory.buffer, outPtr, size);
        assertArrayClose(result, safeResult, 1e-10, 'Large dataset mismatch');
    } finally {
        wasm.mean_ad_free(outPtr, size);
    }
});

test('MEAN_AD zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    for (let i = 0; i < 10; i++) {
        const ptr = wasm.mean_ad_alloc(1000);
        assert(ptr !== 0, 'Allocation should return non-zero pointer');
        wasm.mean_ad_free(ptr, 1000);
    }
});

test('MEAN_AD batch edge cases', () => {
    // Test with minimum data
    const config = {
        period_range: [2, 3, 1] // Just 2 periods
    };
    
    const minData = new Float64Array([1, 2, 3, 4, 5]);
    const result = wasm.mean_ad_batch(minData, config);
    
    assert.strictEqual(result.rows, 2, 'Should have 2 rows');
    assert.strictEqual(result.cols, 5, 'Should have 5 columns');
    assert.deepStrictEqual(result.periods, [2, 3], 'Should have periods 2 and 3');
});

test('MEAN_AD batch metadata', () => {
    const config = {
        period_range: [10, 50, 10] // 5 periods: 10, 20, 30, 40, 50
    };
    
    const result = wasm.mean_ad_batch(testData.close.slice(0, 200), config);
    
    // Check metadata structure
    assert(result.values, 'Should have values array');
    assert(result.periods, 'Should have periods array');
    assert(result.rows === 5, 'Should have 5 rows');
    assert(result.cols === 200, 'Should have 200 columns');
    
    // Verify periods are correct
    assert.deepStrictEqual(result.periods, [10, 20, 30, 40, 50], 'Periods mismatch');
});

test('MEAN_AD batch fast API', () => {
    const data = testData.close.slice(0, 100);
    const periods = { start: 5, end: 15, step: 5 }; // 3 periods
    
    const outSize = 3 * data.length;
    const outPtr = wasm.mean_ad_alloc(outSize);
    
    try {
        const rows = wasm.mean_ad_batch_into(
            data, outPtr, data.length,
            periods.start, periods.end, periods.step
        );
        
        assert.strictEqual(rows, 3, 'Should return 3 rows');
        
        // Compare with safe API
        const config = {
            period_range: [periods.start, periods.end, periods.step]
        };
        const safeResult = wasm.mean_ad_batch(data, config);
        
        const fastResult = new Float64Array(wasm.memory.buffer, outPtr, outSize);
        assertArrayClose(fastResult, safeResult.values, 1e-10, 'Batch fast API mismatch');
    } finally {
        wasm.mean_ad_free(outPtr, outSize);
    }
});

test.after(() => {
    console.log('MEAN_AD WASM tests completed');
});
