/**
 * WASM binding tests for ROCR indicator.
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

test('ROCR accuracy', async () => {
    // Test ROCR matches expected values from Rust tests - mirrors check_rocr_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.rocr;
    
    const result = wasm.rocr_js(close, expected.defaultParams.period);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "ROCR last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('rocr', result, 'close', expected.defaultParams);
});

test('ROCR default period', () => {
    // Test ROCR with default period
    const close = new Float64Array(testData.close);
    
    const result = wasm.rocr_js(close, 10);
    assert.strictEqual(result.length, close.length);
});

test('ROCR zero period', () => {
    // Test ROCR fails with zero period - mirrors check_rocr_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.rocr_js(inputData, 0);
    }, /Invalid period/);
});

test('ROCR period exceeds length', () => {
    // Test ROCR fails when period exceeds data length - mirrors check_rocr_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.rocr_js(dataSmall, 10);
    }, /Invalid period/);
});

test('ROCR very small dataset', () => {
    // Test ROCR fails with insufficient data - mirrors check_rocr_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.rocr_js(singlePoint, 9);
    }, /Invalid period|Not enough/);
});

test('ROCR empty input', () => {
    // Test ROCR fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.rocr_js(empty, 9);
    }, /Empty data/);
});

test('ROCR all NaN input', () => {
    // Test ROCR with all NaN values
    const allNan = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.rocr_js(allNan, 9);
    }, /All values are NaN|All NaN/);
});

test('ROCR fast/unsafe API', () => {
    // Test fast/unsafe API
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate output buffer
    const outPtr = wasm.rocr_alloc(len);
    
    try {
        // Get pointer to input data
        const inPtr = close.byteOffset;
        
        // Compute ROCR
        wasm.rocr_into(close, outPtr, len, 10);
        
        // Create view of output
        const result = new Float64Array(wasm.memory.buffer, outPtr, len);
        
        // Verify results match safe API
        const safeResult = wasm.rocr_js(close, 10);
        assertArrayClose(result, safeResult, 1e-9, "Fast API mismatch");
        
    } finally {
        // Free output buffer
        wasm.rocr_free(outPtr, len);
    }
});

test('ROCR batch API', () => {
    // Test batch processing
    const close = new Float64Array(testData.close);
    
    const config = {
        period_range: [5, 15, 5]  // 3 values: 5, 10, 15
    };
    
    const result = wasm.rocr_batch(close, config);
    
    assert.ok(result.values);
    assert.ok(result.combos);
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, 3 * close.length);
    
    // Verify the middle row (period=10) matches single computation
    const singleResult = wasm.rocr_js(close, 10);
    const middleRow = result.values.slice(close.length, 2 * close.length);
    assertArrayClose(middleRow, singleResult, 1e-9, "Batch row mismatch");
});

test.after(() => {
    console.log('ROCR WASM tests completed');
});
