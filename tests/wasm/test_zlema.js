/**
 * WASM binding tests for ZLEMA indicator.
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

test('ZLEMA partial params', () => {
    const data = testData.close;
    
    // Test with period 14
    const result = wasm.zlema_js(data, 14);
    assert.strictEqual(result.length, data.length, 'Output length should match input');
});

test('ZLEMA accuracy', () => {
    const data = testData.close;
    const expected = EXPECTED_OUTPUTS.zlema;
    
    const result = wasm.zlema_js(data, expected.defaultParams.period);
    
    assert.strictEqual(result.length, data.length, 'Output length should match input');
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(last5, expected.last5Values, 1e-1, 'ZLEMA last 5 values mismatch');
});

test('ZLEMA zero period', () => {
    const data = [10.0, 20.0, 30.0];
    
    assert.throws(() => {
        wasm.zlema_js(data, 0);
    }, /Invalid period/, 'Should throw error for zero period');
});

test('ZLEMA period exceeds length', () => {
    const data = [10.0, 20.0, 30.0];
    
    assert.throws(() => {
        wasm.zlema_js(data, 10);
    }, /Invalid period/, 'Should throw error when period exceeds data length');
});

test('ZLEMA very small dataset', () => {
    const data = [42.0];
    
    assert.throws(() => {
        wasm.zlema_js(data, 14);
    }, /Invalid period/, 'Should throw error for insufficient data');
});

test('ZLEMA reinput', () => {
    const data = testData.close;
    
    // First pass with period 21
    const firstResult = wasm.zlema_js(data, 21);
    
    // Second pass with period 14 on first result
    const secondResult = wasm.zlema_js(firstResult, 14);
    
    assert.strictEqual(secondResult.length, firstResult.length, 'Output length should match input');
    
    // Check that values after warmup period are finite
    for (let idx = 14; idx < secondResult.length; idx++) {
        assert(isFinite(secondResult[idx]), `NaN found at index ${idx}`);
    }
});

test('ZLEMA nan handling', () => {
    const data = testData.close;
    
    const result = wasm.zlema_js(data, 14);
    assert.strictEqual(result.length, data.length, 'Output length should match input');
    
    // Check that values after warmup period don't have NaN
    if (result.length > 20) {
        for (let i = 20; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('ZLEMA batch processing', () => {
    const data = testData.close;
    
    const result = wasm.zlema_batch_js(
        data,
        14,  // period_start
        40,  // period_end
        1    // period_step
    );
    
    // Result should be flattened 2D array (27 rows * data length)
    assert.strictEqual(result.length, 27 * data.length, 
        'Batch result should have 27 rows × data length values');
    
    // Get metadata to verify periods
    const metadata = wasm.zlema_batch_metadata_js(14, 40, 1);
    assert.strictEqual(metadata.length, 27, 'Metadata should have 27 periods');
    assert.strictEqual(metadata[0], 14, 'First period should be 14');
    assert.strictEqual(metadata[26], 40, 'Last period should be 40');
    
    // Check that first row (period 14) matches single ZLEMA calculation
    const single_zlema = wasm.zlema_js(data, 14);
    const first_row = result.slice(0, data.length);
    assertArrayClose(first_row, single_zlema, 1e-9, 'Batch period 14 row should match single calculation');
});

test('ZLEMA all nan input', () => {
    const data = [NaN, NaN, NaN];
    
    assert.throws(() => {
        wasm.zlema_js(data, 2);
    }, /All values are NaN/, 'Should throw error for all NaN input');
});

test('ZLEMA batch metadata', () => {
    // Test metadata generation
    const metadata = wasm.zlema_batch_metadata_js(10, 20, 2);
    
    // Should have periods: 10, 12, 14, 16, 18, 20
    assert.strictEqual(metadata.length, 6, 'Metadata should have 6 periods');
    assert.deepStrictEqual(metadata, [10, 12, 14, 16, 18, 20], 
        'Metadata should contain correct periods');
});

test('ZLEMA batch single period', () => {
    const data = testData.close;
    
    // Test batch with single period (start == end)
    const result = wasm.zlema_batch_js(
        data,
        14,  // period_start
        14,  // period_end
        0    // period_step (ignored when start == end)
    );
    
    // Should have only 1 row
    assert.strictEqual(result.length, data.length, 
        'Single period batch should have 1 row × data length values');
    
    // Should match single ZLEMA calculation
    const single_zlema = wasm.zlema_js(data, 14);
    assertArrayClose(result, single_zlema, 1e-9, 'Single period batch should match single calculation');
});

test('ZLEMA empty input', () => {
    const data = [];
    
    assert.throws(() => {
        wasm.zlema_js(data, 14);
    }, /Invalid period/, 'Should throw error for empty input');
});

test.after(() => {
    console.log('ZLEMA WASM tests completed');
});