/**
 * WASM binding tests for FOSC indicator.
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

test('FOSC accuracy', () => {
    const period = EXPECTED_OUTPUTS.fosc.defaultParams.period;
    
    // Test safe API
    const result = wasm.fosc_js(testData.close, period);
    assert.equal(result.length, testData.close.length);
    
    // Check last 5 values match expected
    const lastN = 5;
    const startIndex = result.length - lastN;
    const actualLast5 = result.slice(startIndex);
    assertArrayClose(actualLast5, EXPECTED_OUTPUTS.fosc.last5Values, 1e-7, 'FOSC last 5 values');
    
    // Check warmup period
    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
});

test('FOSC fast API', () => {
    const period = EXPECTED_OUTPUTS.fosc.defaultParams.period;
    const len = testData.close.length;
    
    // Allocate output buffer
    const outPtr = wasm.fosc_alloc(len);
    
    try {
        // Convert JS array to WASM memory
        const memory = new Float64Array(wasm.memory.buffer);
        const inPtr = wasm.fosc_alloc(len);
        const inArray = new Float64Array(wasm.memory.buffer, inPtr, len);
        inArray.set(testData.close);
        
        // Test computation
        wasm.fosc_into(inPtr, outPtr, len, period);
        
        // Read results
        const result = new Float64Array(wasm.memory.buffer, outPtr, len);
        const resultCopy = Array.from(result);
        
        // Verify results match safe API
        const safeResult = wasm.fosc_js(testData.close, period);
        assertArrayClose(resultCopy, safeResult, 1e-10, 'Fast API matches safe API');
        
        // Test in-place operation (aliasing)
        wasm.fosc_into(inPtr, inPtr, len, period);
        const inPlaceResult = new Float64Array(wasm.memory.buffer, inPtr, len);
        assertArrayClose(Array.from(inPlaceResult), safeResult, 1e-10, 'In-place operation');
        
        // Clean up input
        wasm.fosc_free(inPtr, len);
    } finally {
        // Always free output
        wasm.fosc_free(outPtr, len);
    }
});

test('FOSC error handling', () => {
    // Test zero period
    assert.throws(() => {
        wasm.fosc_js(testData.close, 0);
    }, /period/, 'Should fail with zero period');
    
    // Test period exceeds length
    const smallData = [10.0, 20.0, 30.0];
    assert.throws(() => {
        wasm.fosc_js(smallData, 10);
    }, /period|length/, 'Should fail when period exceeds length');
    
    // Test empty data
    assert.throws(() => {
        wasm.fosc_js([], 5);
    }, /NaN|empty/, 'Should fail with empty data');
    
    // Test all NaN data
    const nanData = new Array(10).fill(NaN);
    assert.throws(() => {
        wasm.fosc_js(nanData, 5);
    }, /NaN/, 'Should fail with all NaN values');
});

test('FOSC batch API', () => {
    const config = {
        period_range: [3, 10, 1]  // periods from 3 to 10, step 1
    };
    
    const result = wasm.fosc_batch(testData.close, config);
    
    // Check structure
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert(result.rows, 'Should have rows count');
    assert(result.cols, 'Should have cols count');
    
    // Verify dimensions
    const expectedRows = 8; // periods 3,4,5,6,7,8,9,10
    assert.equal(result.rows, expectedRows);
    assert.equal(result.cols, testData.close.length);
    assert.equal(result.values.length, result.rows * result.cols);
    assert.equal(result.combos.length, expectedRows);
    
    // Check that default params (period=5) produces expected values
    const defaultRow = result.combos.findIndex(c => c.period === 5);
    assert(defaultRow >= 0, 'Should find period=5 row');
    
    const rowStart = defaultRow * result.cols;
    const rowValues = result.values.slice(rowStart, rowStart + result.cols);
    const lastN = 5;
    const actualLast5 = rowValues.slice(-lastN);
    assertArrayClose(actualLast5, EXPECTED_OUTPUTS.fosc.last5Values, 1e-7, 'Batch default row matches');
});

test('FOSC batch fast API', () => {
    const periodStart = 3, periodEnd = 7, periodStep = 2; // 3,5,7
    const expectedRows = 3;
    const len = testData.close.length;
    const totalSize = expectedRows * len;
    
    // Allocate buffers
    const inPtr = wasm.fosc_alloc(len);
    const outPtr = wasm.fosc_alloc(totalSize);
    
    try {
        // Copy input data
        const memory = new Float64Array(wasm.memory.buffer);
        const inArray = new Float64Array(wasm.memory.buffer, inPtr, len);
        inArray.set(testData.close);
        
        // Run batch computation
        const rows = wasm.fosc_batch_into(
            inPtr, outPtr, len,
            periodStart, periodEnd, periodStep
        );
        
        assert.equal(rows, expectedRows, 'Correct number of rows');
        
        // Read results
        const results = new Float64Array(wasm.memory.buffer, outPtr, totalSize);
        
        // Verify period=5 row if present
        if (periodStart <= 5 && periodEnd >= 5 && (5 - periodStart) % periodStep === 0) {
            const rowIndex = Math.floor((5 - periodStart) / periodStep);
            const rowStart = rowIndex * len;
            const rowValues = Array.from(results.slice(rowStart, rowStart + len));
            const actualLast5 = rowValues.slice(-5);
            assertArrayClose(actualLast5, EXPECTED_OUTPUTS.fosc.last5Values, 1e-7, 
                'Batch fast API period=5 matches expected');
        }
    } finally {
        wasm.fosc_free(inPtr, len);
        wasm.fosc_free(outPtr, totalSize);
    }
});

test.after(() => {
    console.log('FOSC WASM tests completed');
});
