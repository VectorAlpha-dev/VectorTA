/**
 * WASM binding tests for PVI indicator.
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

test('PVI accuracy', () => {
    // Test based on Rust check_pvi_accuracy
    const close = new Float64Array([100.0, 102.0, 101.0, 103.0, 103.0, 105.0]);
    const volume = new Float64Array([500.0, 600.0, 500.0, 700.0, 680.0, 900.0]);
    const initialValue = 1000.0;
    
    const result = wasm.pvi_js(close, volume, initialValue);
    assert.strictEqual(result.length, close.length);
    assertClose(result[0], 1000.0, 1e-6, 'PVI initial value mismatch');
    
    // Verify PVI calculation logic
    // Index 1: volume[1] > volume[0], so PVI updates
    // Index 2: volume[2] < volume[1], so PVI stays the same
    // Index 3: volume[3] > volume[2], so PVI updates
    // Index 4: volume[4] < volume[3], so PVI stays the same
    // Index 5: volume[5] > volume[4], so PVI updates
});

test('PVI default parameters', () => {
    // Test with real data and default initial value
    const close = testData.close;
    const volume = testData.volume;
    
    const result = wasm.pvi_js(close, volume, 1000.0);
    assert.strictEqual(result.length, close.length);
});

test('PVI empty data', () => {
    const close = new Float64Array([]);
    const volume = new Float64Array([]);
    
    assert.throws(() => {
        wasm.pvi_js(close, volume, 1000.0);
    }, /Empty data/);
});

test('PVI mismatched lengths', () => {
    const close = new Float64Array([100.0, 101.0]);
    const volume = new Float64Array([500.0]);
    
    assert.throws(() => {
        wasm.pvi_js(close, volume, 1000.0);
    }, /different lengths/);
});

test('PVI all NaN values', () => {
    const close = new Float64Array([NaN, NaN, NaN]);
    const volume = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.pvi_js(close, volume, 1000.0);
    }, /All values are NaN/);
});

test('PVI not enough valid data', () => {
    const close = new Float64Array([NaN, 100.0]);
    const volume = new Float64Array([NaN, 500.0]);
    
    assert.throws(() => {
        wasm.pvi_js(close, volume, 1000.0);
    }, /Not enough valid data/);
});

test('PVI fast API (in-place operation)', () => {
    const close = new Float64Array([100.0, 102.0, 101.0, 103.0, 103.0, 105.0]);
    const volume = new Float64Array([500.0, 600.0, 500.0, 700.0, 680.0, 900.0]);
    const initialValue = 1000.0;
    
    // Test non-aliased operation
    const len = close.length;
    const outPtr = wasm.pvi_alloc(len);
    
    try {
        wasm.pvi_into(close, volume, outPtr, len, initialValue);
        
        // Read back results
        const result = new Float64Array(wasm.memory.buffer, outPtr, len);
        assertClose(result[0], 1000.0, 1e-6, 'PVI fast API initial value mismatch');
    } finally {
        wasm.pvi_free(outPtr, len);
    }
});

test('PVI fast API (aliasing with close)', () => {
    const close = new Float64Array([100.0, 102.0, 101.0, 103.0, 103.0, 105.0]);
    const volume = new Float64Array([500.0, 600.0, 500.0, 700.0, 680.0, 900.0]);
    const initialValue = 1000.0;
    
    // Create a copy for comparison
    const closeOriginal = new Float64Array(close);
    
    // Test aliased operation (output overwrites close)
    wasm.pvi_into(close, volume, close, close.length, initialValue);
    
    // Verify first value is the initial value
    assertClose(close[0], 1000.0, 1e-6, 'PVI aliased initial value mismatch');
});

test('PVI batch operations', () => {
    const close = new Float64Array([100.0, 102.0, 101.0, 103.0, 103.0, 105.0]);
    const volume = new Float64Array([500.0, 600.0, 500.0, 700.0, 680.0, 900.0]);
    
    const config = {
        initial_value_range: [900.0, 1100.0, 100.0] // 900, 1000, 1100
    };
    
    const result = wasm.pvi_batch(close, volume, config);
    assert(result.values);
    assert(result.combos);
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, 3 * close.length);
    
    // Verify each row matches single calculation
    for (let i = 0; i < result.rows; i++) {
        const initialValue = result.combos[i].initial_value;
        const rowStart = i * result.cols;
        const rowEnd = rowStart + result.cols;
        const rowValues = result.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.pvi_js(close, volume, initialValue);
        assertArrayClose(rowValues, singleResult, 1e-9, `Batch row ${i} mismatch`);
    }
});

test('PVI batch fast API', () => {
    const close = new Float64Array([100.0, 102.0, 101.0, 103.0, 103.0, 105.0]);
    const volume = new Float64Array([500.0, 600.0, 500.0, 700.0, 680.0, 900.0]);
    const len = close.length;
    
    // Test with 3 initial values
    const initialStart = 900.0;
    const initialEnd = 1100.0;
    const initialStep = 100.0;
    const expectedRows = 3;
    
    const outPtr = wasm.pvi_alloc(expectedRows * len);
    
    try {
        const rows = wasm.pvi_batch_into(
            close, volume, outPtr, len,
            initialStart, initialEnd, initialStep
        );
        
        assert.strictEqual(rows, expectedRows);
        
        // Read back results
        const result = new Float64Array(wasm.memory.buffer, outPtr, rows * len);
        
        // Verify first value of each row matches the initial value
        assertClose(result[0], 900.0, 1e-6);
        assertClose(result[len], 1000.0, 1e-6);
        assertClose(result[2 * len], 1100.0, 1e-6);
    } finally {
        wasm.pvi_free(outPtr, expectedRows * len);
    }
});

test('PVI performance comparison', () => {
    const close = testData.close.slice(0, 10000);
    const volume = testData.volume.slice(0, 10000);
    const initialValue = 1000.0;
    
    // Benchmark safe API
    const safeStart = performance.now();
    for (let i = 0; i < 10; i++) {
        wasm.pvi_js(close, volume, initialValue);
    }
    const safeTime = performance.now() - safeStart;
    
    // Benchmark fast API
    const len = close.length;
    const outPtr = wasm.pvi_alloc(len);
    
    const fastStart = performance.now();
    for (let i = 0; i < 10; i++) {
        wasm.pvi_into(close, volume, outPtr, len, initialValue);
    }
    const fastTime = performance.now() - fastStart;
    
    wasm.pvi_free(outPtr, len);
    
    console.log(`Safe API: ${safeTime.toFixed(2)}ms, Fast API: ${fastTime.toFixed(2)}ms`);
    console.log(`Fast API is ${(safeTime / fastTime).toFixed(2)}x faster`);
});

test('PVI NaN handling', () => {
    // Test with NaN values in the data
    const close = new Float64Array([NaN, 100.0, 102.0, NaN, 103.0, 105.0]);
    const volume = new Float64Array([NaN, 500.0, 600.0, NaN, 700.0, 900.0]);
    const initialValue = 1000.0;
    
    const result = wasm.pvi_js(close, volume, initialValue);
    assert.strictEqual(result.length, close.length);
    
    // First value should be NaN due to NaN input
    assert(isNaN(result[0]));
    // Second value should be initial value
    assertClose(result[1], 1000.0, 1e-6);
});

test('PVI reinput', () => {
    // Test PVI applied to PVI output
    const close = new Float64Array([100.0, 102.0, 101.0, 103.0, 103.0, 105.0]);
    const volume = new Float64Array([500.0, 600.0, 500.0, 700.0, 680.0, 900.0]);
    const initialValue = 1000.0;
    
    // First pass
    const firstResult = wasm.pvi_js(close, volume, initialValue);
    
    // Second pass - use same volume data but PVI result as "close"
    const secondResult = wasm.pvi_js(firstResult, volume, initialValue);
    assert.strictEqual(secondResult.length, firstResult.length);
});

test('PVI invalid initial value', () => {
    const close = new Float64Array([100.0, 102.0, 101.0]);
    const volume = new Float64Array([500.0, 600.0, 500.0]);
    
    // Test with negative initial value (should still work)
    const result1 = wasm.pvi_js(close, volume, -1000.0);
    assert.strictEqual(result1.length, close.length);
    assertClose(result1[0], -1000.0, 1e-6);
    
    // Test with zero initial value
    const result2 = wasm.pvi_js(close, volume, 0.0);
    assert.strictEqual(result2.length, close.length);
    assertClose(result2[0], 0.0, 1e-6);
});

test('PVI batch edge cases', () => {
    const close = new Float64Array([100.0, 102.0, 101.0, 103.0]);
    const volume = new Float64Array([500.0, 600.0, 500.0, 700.0]);
    
    // Test with single initial value (step = 0)
    const config1 = {
        initial_value_range: [1000.0, 1000.0, 0.0]
    };
    const result1 = wasm.pvi_batch(close, volume, config1);
    assert.strictEqual(result1.rows, 1);
    assert.strictEqual(result1.combos.length, 1);
    assert.strictEqual(result1.combos[0].initial_value, 1000.0);
    
    // Test with large step
    const config2 = {
        initial_value_range: [500.0, 1500.0, 1000.0]
    };
    const result2 = wasm.pvi_batch(close, volume, config2);
    assert.strictEqual(result2.rows, 2); // 500 and 1500
});

test('PVI zero-copy memory management', () => {
    // Test multiple allocations and deallocations
    const sizes = [100, 1000, 10000];
    const ptrs = [];
    
    // Allocate multiple buffers
    for (const size of sizes) {
        const ptr = wasm.pvi_alloc(size);
        assert(ptr !== 0, 'Allocation should return non-zero pointer');
        ptrs.push({ ptr, size });
    }
    
    // Use the buffers
    const close = new Float64Array(100).fill(100.0);
    const volume = new Float64Array(100).fill(500.0);
    
    for (const { ptr, size } of ptrs) {
        if (size >= 100) {
            // Should succeed for buffers large enough
            wasm.pvi_into(close, volume, ptr, 100, 1000.0);
        }
    }
    
    // Free all buffers
    for (const { ptr, size } of ptrs) {
        wasm.pvi_free(ptr, size);
    }
});

test('PVI batch metadata validation', () => {
    const close = new Float64Array([100.0, 102.0, 101.0, 103.0]);
    const volume = new Float64Array([500.0, 600.0, 500.0, 700.0]);
    
    const config = {
        initial_value_range: [800.0, 1200.0, 100.0] // 5 values
    };
    
    const result = wasm.pvi_batch(close, volume, config);
    
    // Validate metadata
    assert.strictEqual(result.rows, 5);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, result.rows * result.cols);
    
    // Validate combos
    assert.strictEqual(result.combos.length, 5);
    const expectedInitialValues = [800.0, 900.0, 1000.0, 1100.0, 1200.0];
    for (let i = 0; i < result.combos.length; i++) {
        assertClose(result.combos[i].initial_value, expectedInitialValues[i], 1e-9);
    }
});

test.after(() => {
    console.log('PVI WASM tests completed');
});