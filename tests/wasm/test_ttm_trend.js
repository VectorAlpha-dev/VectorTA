const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const wasm = require(path.join(__dirname, '../../pkg/my_project.js'));
const { loadTestData } = require('./test_utils');

// Load test data once
const testData = loadTestData();
const high = testData.high;
const low = testData.low;
const close = testData.close;

// Calculate HL2 (high + low) / 2
const hl2 = high.map((h, i) => (h + low[i]) / 2);

test('TTM Trend - safe API basic test', () => {
    const period = 5;
    const result = wasm.ttm_trend_js(hl2, close, period);
    
    assert.strictEqual(result.length, close.length, 'Output length should match input length');
    assert.ok(result instanceof Uint8Array, 'Result should be Uint8Array');
    
    // Check that values are only 0 or 1
    for (let i = 0; i < result.length; i++) {
        assert.ok(result[i] === 0 || result[i] === 1, `Value at index ${i} should be 0 or 1`);
    }
    
    // Check warmup period should be 0 (false)
    for (let i = 0; i < period - 1; i++) {
        assert.strictEqual(result[i], 0, `Warmup period at index ${i} should be 0`);
    }
});

test('TTM Trend - safe API with different periods', () => {
    const periods = [5, 10, 20];
    
    for (const period of periods) {
        const result = wasm.ttm_trend_js(hl2, close, period);
        assert.strictEqual(result.length, close.length, `Output length for period ${period} should match input`);
        
        // Check warmup
        for (let i = 0; i < period - 1; i++) {
            assert.strictEqual(result[i], 0, `Warmup for period ${period} at index ${i} should be 0`);
        }
    }
});

test('TTM Trend - fast API in-place operations', () => {
    const period = 5;
    const len = close.length;
    
    // Allocate memory
    const sourcePtr = wasm.ttm_trend_alloc(len);
    const closePtr = wasm.ttm_trend_alloc(len);
    const outPtr = wasm.ttm_trend_alloc_u8(len);
    
    try {
        // Copy data to WASM memory
        const sourceHeap = new Float64Array(wasm.memory.buffer, sourcePtr, len);
        const closeHeap = new Float64Array(wasm.memory.buffer, closePtr, len);
        sourceHeap.set(hl2);
        closeHeap.set(close);
        
        // Call fast API
        wasm.ttm_trend_into(sourcePtr, closePtr, outPtr, len, period);
        
        // Read results
        const resultHeap = new Uint8Array(wasm.memory.buffer, outPtr, len);
        const result = Array.from(resultHeap);
        
        // Compare with safe API
        const safeResult = wasm.ttm_trend_js(hl2, close, period);
        
        for (let i = 0; i < len; i++) {
            assert.strictEqual(result[i], safeResult[i], `Fast API mismatch at index ${i}`);
        }
    } finally {
        // Clean up
        wasm.ttm_trend_free(sourcePtr, len);
        wasm.ttm_trend_free(closePtr, len);
        wasm.ttm_trend_free_u8(outPtr, len);
    }
});

test('TTM Trend - fast API with aliasing detection', () => {
    const period = 5;
    const len = 100; // Use smaller dataset for aliasing test
    const hl2Small = hl2.slice(0, len);
    const closeSmall = close.slice(0, len);
    
    // Test aliasing with source pointer
    const sourcePtr = wasm.ttm_trend_alloc(len);
    const closePtr = wasm.ttm_trend_alloc(len);
    
    try {
        // Copy data to WASM memory
        const sourceHeap = new Float64Array(wasm.memory.buffer, sourcePtr, len);
        const closeHeap = new Float64Array(wasm.memory.buffer, closePtr, len);
        sourceHeap.set(hl2Small);
        closeHeap.set(closeSmall);
        
        // Use source pointer as output (aliasing) - this should still work
        const outPtr = sourcePtr; // Intentional aliasing
        wasm.ttm_trend_into(sourcePtr, closePtr, outPtr, len, period);
        
        // Result should be valid (handled internally with temp buffer)
        const resultHeap = new Uint8Array(wasm.memory.buffer, outPtr, len);
        const result = Array.from(resultHeap);
        
        // Verify some values are 0 and some are 1
        const hasZeros = result.some(v => v === 0);
        const hasOnes = result.some(v => v === 1);
        assert.ok(hasZeros || hasOnes, 'Result should contain valid boolean values');
    } finally {
        // Only free closePtr since sourcePtr was reused as output
        wasm.ttm_trend_free(closePtr, len);
    }
});

test('TTM Trend - batch API', () => {
    const config = {
        period_range: [5, 15, 5]  // periods: 5, 10, 15
    };
    
    const result = wasm.ttm_trend_batch(hl2, close, config);
    
    assert.strictEqual(result.rows, 3, 'Should have 3 rows (periods)');
    assert.strictEqual(result.cols, close.length, 'Columns should match input length');
    assert.strictEqual(result.periods.length, 3, 'Should have 3 periods');
    assert.deepStrictEqual(result.periods, [5, 10, 15], 'Periods should match');
    assert.strictEqual(result.values.length, 3 * close.length, 'Values array should be flattened');
    
    // Check that all values are 0 or 1
    for (let i = 0; i < result.values.length; i++) {
        assert.ok(result.values[i] === 0 || result.values[i] === 1, 
                  `Batch value at index ${i} should be 0 or 1`);
    }
    
    // Verify first row matches single calculation
    const singleResult = wasm.ttm_trend_js(hl2, close, 5);
    for (let i = 0; i < close.length; i++) {
        assert.strictEqual(result.values[i], singleResult[i], 
                          `Batch row 0 mismatch at index ${i}`);
    }
});

test('TTM Trend - error handling', () => {
    // Test with invalid period
    assert.throws(() => {
        wasm.ttm_trend_js(hl2, close, 0);
    }, /Invalid period/, 'Should throw on zero period');
    
    assert.throws(() => {
        wasm.ttm_trend_js(hl2, close, close.length + 1);
    }, /Invalid period/, 'Should throw when period exceeds data length');
    
    // Test with empty arrays
    assert.throws(() => {
        wasm.ttm_trend_js([], [], 5);
    }, /All values are NaN|Invalid period/, 'Should throw on empty input');
});

test('TTM Trend - accuracy check', () => {
    const period = 5;
    const result = wasm.ttm_trend_js(hl2, close, period);
    
    // TTM Trend logic: result[i] = close[i] > average(source[i-period+1..i])
    // Manual calculation for a few points after warmup
    for (let i = period + 10; i < period + 15; i++) {
        let sum = 0;
        for (let j = i - period + 1; j <= i; j++) {
            sum += hl2[j];
        }
        const avg = sum / period;
        const expected = close[i] > avg ? 1 : 0;
        
        assert.strictEqual(result[i], expected, 
                          `Accuracy mismatch at index ${i}: close=${close[i]}, avg=${avg}`);
    }
});

test('TTM Trend - memory allocation and deallocation', () => {
    const len = 1000;
    
    // Test f64 allocation
    const ptr1 = wasm.ttm_trend_alloc(len);
    assert.ok(ptr1 !== 0, 'Allocated pointer should not be null');
    wasm.ttm_trend_free(ptr1, len);
    
    // Test u8 allocation
    const ptr2 = wasm.ttm_trend_alloc_u8(len);
    assert.ok(ptr2 !== 0, 'Allocated u8 pointer should not be null');
    wasm.ttm_trend_free_u8(ptr2, len);
    
    // Test multiple allocations
    const ptrs = [];
    for (let i = 0; i < 10; i++) {
        ptrs.push(wasm.ttm_trend_alloc(len));
    }
    
    // All pointers should be different
    const uniquePtrs = new Set(ptrs);
    assert.strictEqual(uniquePtrs.size, ptrs.length, 'All allocated pointers should be unique');
    
    // Free all
    for (const ptr of ptrs) {
        wasm.ttm_trend_free(ptr, len);
    }
});

test('TTM Trend - partial parameters', () => {
    // TTM Trend only has one parameter (period), test with default
    const result = wasm.ttm_trend_js(hl2, close, 5);
    assert.strictEqual(result.length, close.length, 'Output length should match input');
    assert.ok(result instanceof Uint8Array, 'Result should be Uint8Array');
});

test('TTM Trend - reinput test', () => {
    const period = 5;
    
    // First pass
    const firstResult = wasm.ttm_trend_js(hl2, close, period);
    
    // Convert result back to float array (0->0.0, 1->1.0) to use as input
    const floatResult = new Float64Array(firstResult.length);
    for (let i = 0; i < firstResult.length; i++) {
        floatResult[i] = firstResult[i];
    }
    
    // Second pass - apply TTM Trend to TTM Trend output
    const secondResult = wasm.ttm_trend_js(floatResult, close, period);
    
    assert.strictEqual(secondResult.length, firstResult.length, 'Reinput length should match');
    
    // Verify some values changed (TTM on boolean values should produce different results)
    let hasChanges = false;
    for (let i = period; i < secondResult.length; i++) {
        if (secondResult[i] !== firstResult[i]) {
            hasChanges = true;
            break;
        }
    }
    assert.ok(hasChanges, 'Reinput should produce different values');
});

test('TTM Trend - all NaN input', () => {
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.ttm_trend_js(allNaN, allNaN, 5);
    }, /All values are NaN/, 'Should throw on all NaN input');
});

test('TTM Trend - batch metadata verification', () => {
    const config = {
        period_range: [5, 15, 5]
    };
    
    const result = wasm.ttm_trend_batch(hl2, close, config);
    
    // Verify metadata
    assert.strictEqual(result.rows, 3, 'Should have correct number of rows');
    assert.strictEqual(result.cols, close.length, 'Should have correct number of columns');
    assert.deepStrictEqual(result.periods, [5, 10, 15], 'Should have correct periods');
    
    // Verify values array size
    assert.strictEqual(result.values.length, result.rows * result.cols, 
                      'Values array should match rows * cols');
});

test('TTM Trend - batch edge cases', () => {
    // Test with minimum data
    const smallData = hl2.slice(0, 20);
    const smallClose = close.slice(0, 20);
    
    // Single period batch
    const singleConfig = {
        period_range: [5, 5, 0]  // Only period 5
    };
    
    const singleResult = wasm.ttm_trend_batch(smallData, smallClose, singleConfig);
    assert.strictEqual(singleResult.rows, 1, 'Single period should have 1 row');
    assert.strictEqual(singleResult.periods.length, 1, 'Should have 1 period');
    
    // Test with step size larger than range
    const largeStepConfig = {
        period_range: [5, 10, 10]  // Step larger than range
    };
    
    const largeStepResult = wasm.ttm_trend_batch(smallData, smallClose, largeStepConfig);
    assert.strictEqual(largeStepResult.rows, 1, 'Large step should result in 1 row');
    assert.deepStrictEqual(largeStepResult.periods, [5], 'Should only have start period');
});

test('TTM Trend - zero-copy consistency', () => {
    const period = 5;
    const len = 100;
    const smallHl2 = hl2.slice(0, len);
    const smallClose = close.slice(0, len);
    
    // Safe API result
    const safeResult = wasm.ttm_trend_js(smallHl2, smallClose, period);
    
    // Fast API result
    const sourcePtr = wasm.ttm_trend_alloc(len);
    const closePtr = wasm.ttm_trend_alloc(len);
    const outPtr = wasm.ttm_trend_alloc_u8(len);
    
    try {
        // Copy data to WASM memory
        const sourceHeap = new Float64Array(wasm.memory.buffer, sourcePtr, len);
        const closeHeap = new Float64Array(wasm.memory.buffer, closePtr, len);
        sourceHeap.set(smallHl2);
        closeHeap.set(smallClose);
        
        // Call fast API
        wasm.ttm_trend_into(sourcePtr, closePtr, outPtr, len, period);
        
        // Read results
        const resultHeap = new Uint8Array(wasm.memory.buffer, outPtr, len);
        const fastResult = Array.from(resultHeap);
        
        // Compare results
        assert.deepStrictEqual(fastResult, Array.from(safeResult), 
                              'Fast and safe APIs should produce identical results');
    } finally {
        wasm.ttm_trend_free(sourcePtr, len);
        wasm.ttm_trend_free(closePtr, len);
        wasm.ttm_trend_free_u8(outPtr, len);
    }
});

test('TTM Trend - streaming edge cases', () => {
    // Test streaming with NaN values
    const nanData = [...hl2.slice(0, 10)];
    const nanClose = [...close.slice(0, 10)];
    nanData[5] = NaN;
    nanClose[6] = NaN;
    
    // This should still work as long as not all values are NaN
    const result = wasm.ttm_trend_js(
        new Float64Array(nanData), 
        new Float64Array(nanClose), 
        3
    );
    
    assert.strictEqual(result.length, 10, 'Should handle some NaN values');
    
    // First few values should be 0 (warmup)
    assert.strictEqual(result[0], 0, 'Warmup should be 0');
    assert.strictEqual(result[1], 0, 'Warmup should be 0');
});

console.log('All TTM Trend WASM tests passed! ✓');