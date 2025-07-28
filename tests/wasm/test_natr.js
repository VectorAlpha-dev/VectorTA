/**
 * WASM binding tests for NATR indicator.
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

test('NATR accuracy', () => {
    const period = EXPECTED_OUTPUTS['natr']['default_params']['period'];
    const expected = EXPECTED_OUTPUTS['natr']['last_5_values'];
    
    // Run NATR with default parameters
    const result = wasm.natr_js(testData.high, testData.low, testData.close, period);
    
    assert.strictEqual(result.length, testData.close.length, 'Output length should match input length');
    
    // Check last 5 values
    const actual = result.slice(-5);
    assertArrayClose(actual, expected, 1e-8, 'NATR last 5 values should match expected');
});

test('NATR error handling', () => {
    // Test with zero period
    assert.throws(
        () => wasm.natr_js([10, 20], [5, 10], [7, 15], 0),
        /Invalid period/,
        'Should throw error for zero period'
    );
    
    // Test with period exceeding data length
    assert.throws(
        () => wasm.natr_js([10, 20], [5, 10], [7, 15], 10),
        /Invalid period/,
        'Should throw error when period exceeds data length'
    );
    
    // Test with empty input
    assert.throws(
        () => wasm.natr_js([], [], [], 14),
        /Empty data/,
        'Should throw error for empty input'
    );
    
    // Test with all NaN values
    assert.throws(
        () => wasm.natr_js([NaN, NaN], [NaN, NaN], [NaN, NaN], 2),
        /All values are NaN/,
        'Should throw error when all values are NaN'
    );
});

test('NATR with NaN handling', () => {
    // Create data with some NaN values
    const high = testData.high.slice(0, 100);
    const low = testData.low.slice(0, 100);
    const close = testData.close.slice(0, 100);
    
    // Insert some NaNs
    high[10] = NaN;
    high[11] = NaN;
    low[10] = NaN;
    low[11] = NaN;
    close[10] = NaN;
    close[11] = NaN;
    
    const result = wasm.natr_js(high, low, close, 14);
    assert.strictEqual(result.length, 100, 'Output length should match input length');
    
    // Check that we have valid values after the NaN region
    const validCount = result.slice(20).filter(v => !isNaN(v)).length;
    assert(validCount > 0, 'Should have valid values after NaN region');
});

test('NATR fast API', () => {
    const period = 14;
    const len = testData.close.length;
    
    // Allocate memory for all inputs and output
    const highPtr = wasm.natr_alloc(len);
    const lowPtr = wasm.natr_alloc(len);
    const closePtr = wasm.natr_alloc(len);
    const outPtr = wasm.natr_alloc(len);
    
    try {
        // Copy data into WASM memory
        const highMem = new Float64Array(wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.memory.buffer, lowPtr, len);
        const closeMem = new Float64Array(wasm.memory.buffer, closePtr, len);
        
        highMem.set(testData.high);
        lowMem.set(testData.low);
        closeMem.set(testData.close);
        
        // Compute NATR
        wasm.natr_into(
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            len,
            period
        );
        
        // Read results
        const memory = new Float64Array(wasm.memory.buffer, outPtr, len);
        const result = Array.from(memory);
        
        // Compare with safe API
        const expected = wasm.natr_js(testData.high, testData.low, testData.close, period);
        assertArrayClose(result, expected, 1e-10, 'Fast API should match safe API');
        
    } finally {
        // Free all memory
        wasm.natr_free(highPtr, len);
        wasm.natr_free(lowPtr, len);
        wasm.natr_free(closePtr, len);
        wasm.natr_free(outPtr, len);
    }
});

test('NATR batch API', () => {
    const config = {
        period_range: [10, 20, 5]  // periods: 10, 15, 20
    };
    
    const result = wasm.natr_batch(testData.high, testData.low, testData.close, config);
    
    assert(result.values, 'Batch result should have values');
    assert(result.combos, 'Batch result should have combos');
    assert(result.rows, 'Batch result should have rows');
    assert(result.cols, 'Batch result should have cols');
    
    assert.strictEqual(result.rows, 3, 'Should have 3 rows (periods)');
    assert.strictEqual(result.cols, testData.close.length, 'Columns should match input length');
    assert.strictEqual(result.combos.length, 3, 'Should have 3 parameter combinations');
    
    // Verify each row matches single calculation
    for (let i = 0; i < result.rows; i++) {
        const period = result.combos[i].period;
        const rowStart = i * result.cols;
        const rowEnd = (i + 1) * result.cols;
        const batchRow = result.values.slice(rowStart, rowEnd);
        
        const single = wasm.natr_js(testData.high, testData.low, testData.close, period);
        assertArrayClose(batchRow, single, 1e-9, `Batch row ${i} (period ${period}) should match single calculation`);
    }
});

test('NATR fast API in-place (aliasing)', () => {
    const period = 14;
    const len = testData.close.length;
    
    // Test aliasing with high array
    const highPtr = wasm.natr_alloc(len);
    const lowPtr = wasm.natr_alloc(len);
    const closePtr = wasm.natr_alloc(len);
    
    try {
        // Copy data into WASM memory
        const highMem = new Float64Array(wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.memory.buffer, lowPtr, len);
        const closeMem = new Float64Array(wasm.memory.buffer, closePtr, len);
        
        highMem.set(testData.high);
        lowMem.set(testData.low);
        closeMem.set(testData.close);
        
        // Compute NATR in-place (output overwrites high)
        wasm.natr_into(
            highPtr,
            lowPtr,
            closePtr,
            highPtr,  // Output to same location as high input
            len,
            period
        );
        
        // Read results
        const memory = new Float64Array(wasm.memory.buffer, highPtr, len);
        const result = Array.from(memory);
        
        // Compare with safe API
        const expected = wasm.natr_js(testData.high, testData.low, testData.close, period);
        assertArrayClose(result, expected, 1e-10, 'In-place operation should match safe API');
        
    } finally {
        // Free all memory
        wasm.natr_free(highPtr, len);
        wasm.natr_free(lowPtr, len);
        wasm.natr_free(closePtr, len);
    }
});

test('NATR zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.natr_into(0, 0, 0, 0, 10, 14);
    }, /Null pointer|null pointer/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.natr_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.natr_into(ptr, ptr, ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        // Period exceeds data length
        assert.throws(() => {
            wasm.natr_into(ptr, ptr, ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.natr_free(ptr, 10);
    }
});

test.after(() => {
    console.log('NATR WASM tests completed');
});
