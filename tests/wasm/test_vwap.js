/**
 * WASM binding tests for VWAP indicator.
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

test('VWAP partial params', () => {
    const timestamps = testData.timestamp;
    const volumes = testData.volume;
    // Calculate hlc3 prices
    const prices = testData.high.map((h, i) => 
        (h + testData.low[i] + testData.close[i]) / 3.0
    );
    
    // Test with default anchor (undefined) and default kernel
    const result = wasm.vwap_js(timestamps, volumes, prices);
    assert.strictEqual(result.length, prices.length, 'Output length should match input');
    
    // Test with explicit kernel parameter
    const result2 = wasm.vwap_js(timestamps, volumes, prices, undefined, "scalar");
    assert.strictEqual(result2.length, prices.length, 'Output length should match input with scalar kernel');
});

test('VWAP accuracy', () => {
    const timestamps = testData.timestamp;
    const volumes = testData.volume;
    // Calculate hlc3 prices
    const prices = testData.high.map((h, i) => 
        (h + testData.low[i] + testData.close[i]) / 3.0
    );
    const expected = EXPECTED_OUTPUTS.vwap;
    
    const result = wasm.vwap_js(timestamps, volumes, prices, "1D", undefined);
    
    assert.strictEqual(result.length, prices.length, 'Output length should match input');
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(last5, expected.anchor1D, 1e-5, 'VWAP last 5 values mismatch');
});

test('VWAP anchor parsing error', () => {
    const timestamps = testData.timestamp;
    const volumes = testData.volume;
    const prices = testData.close;
    
    assert.throws(() => {
        wasm.vwap_js(timestamps, volumes, prices, "xyz");
    }, /Error parsing anchor/, 'Should throw error for invalid anchor');
});

test('VWAP kernel parameter', () => {
    const timestamps = testData.timestamp;
    const volumes = testData.volume;
    const prices = testData.close;
    
    // Test valid kernels
    const result1 = wasm.vwap_js(timestamps, volumes, prices, "1d", "scalar");
    assert.strictEqual(result1.length, prices.length, 'Scalar kernel should work');
    
    const result2 = wasm.vwap_js(timestamps, volumes, prices, "1d", "scalar_batch");
    assert.strictEqual(result2.length, prices.length, 'Scalar batch kernel should work');
    
    // Test invalid kernel
    assert.throws(() => {
        wasm.vwap_js(timestamps, volumes, prices, "1d", "invalid_kernel");
    }, /Unknown kernel/, 'Should throw error for invalid kernel');
});

test('VWAP mismatch lengths', () => {
    const timestamps = [1000, 2000, 3000];
    const volumes = [100.0, 200.0];  // Mismatched length
    const prices = [10.0, 20.0, 30.0];
    
    assert.throws(() => {
        wasm.vwap_js(timestamps, volumes, prices);
    }, /Mismatch in length/, 'Should throw error for mismatched array lengths');
});

test('VWAP empty data', () => {
    const empty = [];
    
    assert.throws(() => {
        wasm.vwap_js(empty, empty, empty);
    }, /No data/, 'Should throw error for empty input');
});

test('VWAP batch processing', () => {
    const timestamps = testData.timestamp;
    const volumes = testData.volume;
    // Calculate hlc3 prices
    const prices = testData.high.map((h, i) => 
        (h + testData.low[i] + testData.close[i]) / 3.0
    );
    
    const result = wasm.vwap_batch_js(
        timestamps,
        volumes,
        prices,
        "1d",
        "3d",
        1
    );
    
    // Result should be flattened 2D array (3 rows * data length)
    assert.strictEqual(result.length, 3 * prices.length, 
        'Batch result should have 3 rows × data length values');
    
    // Get metadata to verify anchors
    const metadata = wasm.vwap_batch_metadata_js("1d", "3d", 1);
    assert.deepStrictEqual(metadata, ["1d", "2d", "3d"], 
        'Metadata should contain expected anchors');
    
    // Check that first row (1d) matches single VWAP calculation
    const single_vwap = wasm.vwap_js(timestamps, volumes, prices, "1d");
    const first_row = result.slice(0, prices.length);
    assertArrayClose(first_row, single_vwap, 1e-9, 'Batch 1d row should match single calculation');
});

test('VWAP default params', () => {
    const timestamps = testData.timestamp;
    const volumes = testData.volume;
    const prices = testData.close;
    
    // Should use default anchor "1d"
    const result = wasm.vwap_js(timestamps, volumes, prices);
    assert.strictEqual(result.length, prices.length, 'Output length should match input');
});

test('VWAP nan handling', () => {
    const timestamps = testData.timestamp;
    const volumes = testData.volume;
    // Calculate hlc3 prices
    const prices = testData.high.map((h, i) => 
        (h + testData.low[i] + testData.close[i]) / 3.0
    );
    
    const result = wasm.vwap_js(timestamps, volumes, prices);
    assert.strictEqual(result.length, prices.length, 'Output length should match input');
    
    // Check all non-NaN values are finite
    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i])) {
            assert(isFinite(result[i]), `Found non-finite value at index ${i}`);
        }
    }
});

test.after(() => {
    console.log('VWAP WASM tests completed');
});
