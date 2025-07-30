/**
 * WASM binding tests for TRIX indicator.
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

test('TRIX accuracy', () => {
    const closePrices = testData.close;
    const period = 18;
    
    // Test default parameters
    const result = wasm.trix_js(closePrices, period);
    
    assert.equal(result.length, closePrices.length, 'TRIX length mismatch');
    
    // Expected values from Rust tests
    const expectedLastFive = [-16.03736447, -15.92084231, -15.76171478, -15.53571033, -15.34967155];
    
    // Check last 5 values
    assert(result.length >= 5, 'TRIX length too short');
    const resultLastFive = result.slice(-5);
    
    for (let i = 0; i < expectedLastFive.length; i++) {
        assertClose(
            resultLastFive[i], 
            expectedLastFive[i], 
            1e-6, 
            `TRIX mismatch at index ${i}`
        );
    }
});

test('TRIX error handling', () => {
    // Test zero period
    assert.throws(
        () => wasm.trix_js([10.0, 20.0, 30.0], 0),
        /Invalid period/,
        'TRIX should fail with zero period'
    );
    
    // Test period exceeds length
    assert.throws(
        () => wasm.trix_js([10.0, 20.0, 30.0], 10),
        /Invalid period/,
        'TRIX should fail with period exceeding length'
    );
    
    // Test very small dataset
    assert.throws(
        () => wasm.trix_js([42.0], 18),
        /Invalid period|Not enough valid data/,
        'TRIX should fail with insufficient data'
    );
    
    // Test empty data
    assert.throws(
        () => wasm.trix_js([], 18),
        /Empty/,
        'TRIX should fail with empty data'
    );
    
    // Test all NaN  
    assert.throws(
        () => wasm.trix_js([NaN, NaN, NaN], 18),
        /All values are NaN/,
        'TRIX should fail with all NaN values'
    );
});

test('TRIX partial params', () => {
    const closePrices = testData.close;
    
    // Test with different periods
    const result14 = wasm.trix_js(closePrices, 14);
    assert.equal(result14.length, closePrices.length);
    
    const result20 = wasm.trix_js(closePrices, 20);
    assert.equal(result20.length, closePrices.length);
});

test('TRIX fast API (unsafe)', async () => {
    const closePrices = testData.close;
    const len = closePrices.length;
    const period = 18;
    
    // Allocate output buffer
    const outPtr = wasm.trix_alloc(len);
    
    try {
        // Create input array in WASM memory
        const inPtr = wasm.trix_alloc(len);
        const wasmMemory = new Float64Array(wasm.memory.buffer);
        wasmMemory.set(closePrices, inPtr / 8);
        
        // Compute TRIX
        wasm.trix_into(inPtr, outPtr, len, period);
        
        // Read results
        const result = new Float64Array(wasm.memory.buffer, outPtr, len);
        
        // Verify results match safe API
        const safeResult = wasm.trix_js(closePrices, period);
        assertArrayClose(Array.from(result), safeResult, 1e-10, 'Fast API results should match safe API');
        
        // Test in-place operation (aliasing)
        wasm.trix_into(inPtr, inPtr, len, period);
        const inPlaceResult = new Float64Array(wasm.memory.buffer, inPtr, len);
        assertArrayClose(Array.from(inPlaceResult), safeResult, 1e-10, 'In-place results should match safe API');
        
        // Clean up input
        wasm.trix_free(inPtr, len);
    } finally {
        // Clean up output
        wasm.trix_free(outPtr, len);
    }
});

test('TRIX batch processing', () => {
    const closePrices = testData.close;
    
    // Test batch with single parameter
    const singleConfig = {
        period_range: [18, 18, 0]
    };
    const singleResult = wasm.trix_batch(closePrices, singleConfig);
    
    assert(singleResult.values, 'Batch result should have values');
    assert(singleResult.periods, 'Batch result should have periods');
    assert.equal(singleResult.rows, 1, 'Single batch should have 1 row');
    assert.equal(singleResult.cols, closePrices.length, 'Columns should match input length');
    assert.equal(singleResult.periods.length, 1, 'Should have 1 period');
    assert.equal(singleResult.periods[0], 18, 'Period should be 18');
    
    // Test batch with range
    const rangeConfig = {
        period_range: [10, 20, 5]  // 10, 15, 20
    };
    const rangeResult = wasm.trix_batch(closePrices, rangeConfig);
    
    assert.equal(rangeResult.rows, 3, 'Range batch should have 3 rows');
    assert.equal(rangeResult.cols, closePrices.length);
    assert.equal(rangeResult.values.length, 3 * closePrices.length);
    assert.deepEqual(rangeResult.periods, [10, 15, 20]);
});

test('TRIX batch fast API', async () => {
    const closePrices = testData.close;
    const len = closePrices.length;
    
    // Batch parameters
    const periodStart = 10;
    const periodEnd = 20; 
    const periodStep = 5;  // Will generate 10, 15, 20
    const numCombos = 3;
    const totalSize = numCombos * len;
    
    // Allocate buffers
    const inPtr = wasm.trix_alloc(len);
    const outPtr = wasm.trix_alloc(totalSize);
    
    try {
        // Copy input data
        const wasmMemory = new Float64Array(wasm.memory.buffer);
        wasmMemory.set(closePrices, inPtr / 8);
        
        // Run batch computation
        const resultRows = wasm.trix_batch_into(
            inPtr, outPtr, len,
            periodStart, periodEnd, periodStep
        );
        
        assert.equal(resultRows, numCombos, 'Should return correct number of combinations');
        
        // Read results
        const results = new Float64Array(wasm.memory.buffer, outPtr, totalSize);
        
        // Verify first row matches single computation
        const firstRow = Array.from(results.slice(0, len));
        const singleResult = wasm.trix_js(closePrices, 10);
        assertArrayClose(firstRow, singleResult, 1e-10, 'First batch row should match single computation');
    } finally {
        wasm.trix_free(inPtr, len);
        wasm.trix_free(outPtr, totalSize);
    }
});

test('TRIX with NaN handling', () => {
    const closePrices = testData.close.slice();
    
    // Insert some NaN values
    closePrices[100] = NaN;
    closePrices[101] = NaN;
    closePrices[102] = NaN;
    closePrices[200] = NaN;
    closePrices[300] = NaN;
    closePrices[301] = NaN;
    
    // Should not throw error
    const result = wasm.trix_js(closePrices, 18);
    assert.equal(result.length, closePrices.length);
    
    // Check that we have some valid values after the NaN regions
    const validAfterNans = result.slice(350).filter(v => !isNaN(v));
    assert(validAfterNans.length > 0, 'Should have valid values after NaN regions');
});

test('TRIX reinput', () => {
    const closePrices = testData.close;
    const period = 10;
    
    // First TRIX calculation
    const firstResult = wasm.trix_js(closePrices, period);
    
    // Apply TRIX to its own output
    const secondResult = wasm.trix_js(firstResult, period);
    
    assert.equal(firstResult.length, secondResult.length);
    
    // The second result should have more NaN values at the beginning
    const firstValidIdx = firstResult.findIndex(v => !isNaN(v));
    const secondValidIdx = secondResult.findIndex(v => !isNaN(v));
    
    assert(secondValidIdx > firstValidIdx, 'Second TRIX should have more warmup period');
});

test.after(() => {
    console.log('TRIX WASM tests completed');
});
