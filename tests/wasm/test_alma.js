/**
 * WASM binding tests for ALMA indicator.
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

test('ALMA partial params', () => {
    // Test with default parameters - mirrors check_alma_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.alma_js(close, 9, 0.85, 6.0);
    assert.strictEqual(result.length, close.length);
});

test('ALMA accuracy', () => {
    // Test ALMA matches expected values from Rust tests - mirrors check_alma_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.alma;
    
    const result = wasm.alma_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.offset,
        expected.defaultParams.sigma
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "ALMA last 5 values mismatch"
    );
});

test('ALMA default candles', () => {
    // Test ALMA with default parameters - mirrors check_alma_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.alma_js(close, 9, 0.85, 6.0);
    assert.strictEqual(result.length, close.length);
});

test('ALMA zero period', () => {
    // Test ALMA fails with zero period - mirrors check_alma_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.alma_js(inputData, 0, 0.85, 6.0);
    }, /Invalid period/);
});

test('ALMA period exceeds length', () => {
    // Test ALMA fails when period exceeds data length - mirrors check_alma_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.alma_js(dataSmall, 10, 0.85, 6.0);
    }, /Invalid period/);
});

test('ALMA very small dataset', () => {
    // Test ALMA fails with insufficient data - mirrors check_alma_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.alma_js(singlePoint, 9, 0.85, 6.0);
    }, /Invalid period|Not enough valid data/);
});

test('ALMA empty input', () => {
    // Test ALMA fails with empty input - mirrors check_alma_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.alma_js(empty, 9, 0.85, 6.0);
    }, /Input data slice is empty/);
});

test('ALMA invalid sigma', () => {
    // Test ALMA fails with invalid sigma - mirrors check_alma_invalid_sigma
    const data = new Float64Array([1.0, 2.0, 3.0]);
    
    // Sigma = 0
    assert.throws(() => {
        wasm.alma_js(data, 2, 0.85, 0.0);
    }, /Invalid sigma/);
    
    // Negative sigma
    assert.throws(() => {
        wasm.alma_js(data, 2, 0.85, -1.0);
    }, /Invalid sigma/);
});

test('ALMA invalid offset', () => {
    // Test ALMA fails with invalid offset - mirrors check_alma_invalid_offset
    const data = new Float64Array([1.0, 2.0, 3.0]);
    
    // NaN offset
    assert.throws(() => {
        wasm.alma_js(data, 2, NaN, 6.0);
    }, /Invalid offset/);
    
    // Offset > 1
    assert.throws(() => {
        wasm.alma_js(data, 2, 1.5, 6.0);
    }, /Invalid offset/);
    
    // Offset < 0
    assert.throws(() => {
        wasm.alma_js(data, 2, -0.1, 6.0);
    }, /Invalid offset/);
});

test('ALMA reinput', () => {
    // Test ALMA applied twice (re-input) - mirrors check_alma_reinput
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.alma;
    
    // First pass
    const firstResult = wasm.alma_js(close, 9, 0.85, 6.0);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply ALMA to ALMA output
    const secondResult = wasm.alma_js(firstResult, 9, 0.85, 6.0);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Check last 5 values match expected
    const last5 = secondResult.slice(-5);
    assertArrayClose(
        last5,
        expected.reinputLast5,
        1e-8,
        "ALMA re-input last 5 values mismatch"
    );
});

test('ALMA NaN handling', () => {
    // Test ALMA handles NaN values correctly - mirrors check_alma_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.alma_js(close, 9, 0.85, 6.0);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First period-1 values should be NaN
    assertAllNaN(result.slice(0, 8), "Expected NaN in warmup period");
});

test('ALMA all NaN input', () => {
    // Test ALMA with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.alma_js(allNaN, 9, 0.85, 6.0);
    }, /All values are NaN/);
});

// Note: Streaming and batch tests would require those functions to be exposed in WASM bindings
// Currently testing only the basic alma_js function that's exposed

test.after(() => {
    console.log('ALMA WASM tests completed');
});