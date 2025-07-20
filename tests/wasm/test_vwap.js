/**
 * WASM binding tests for VWAP indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { 
    loadTestData, 
    assertArrayClose, 
    assertClose,
    isNaN,
    assertAllNaN,
    assertNoNaN,
    EXPECTED_OUTPUTS 
} from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
    // Load WASM module
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
        // No need to call default() for ES modules
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
    
    // Use the new batch API with config object
    const result_old = wasm.vwap_batch(
        timestamps,
        volumes,
        prices,
        { anchor_range: ["1d", "3d", 1] }
    );
    
    // Result should be an object with values, anchors, rows, cols
    assert(result_old.values, 'Result should have values array');
    assert(result_old.anchors, 'Result should have anchors array');
    assert.strictEqual(result_old.rows, 3, 'Should have 3 rows (1d, 2d, 3d)');
    assert.strictEqual(result_old.cols, prices.length, 'Cols should match input length');
    assert.deepStrictEqual(result_old.anchors, ["1d", "2d", "3d"], 'Anchors should match expected');
    
    // Check that values array has correct length
    assert.strictEqual(result_old.values.length, result_old.rows * result_old.cols, 
        'Values array should have rows × cols elements');
    
    // Get metadata to verify anchors
    const metadata = wasm.vwap_batch_metadata_js("1d", "3d", 1);
    assert.deepStrictEqual(metadata, ["1d", "2d", "3d"], 
        'Metadata should contain expected anchors');
    
    // Check that first row (1d) matches single VWAP calculation
    const single_vwap = wasm.vwap_js(timestamps, volumes, prices, "1d");
    const first_row = result_old.values.slice(0, prices.length);
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

test('VWAP fast API (vwap_into)', () => {
    const timestamps = testData.timestamp;
    const volumes = testData.volume;
    const prices = testData.close;
    const len = prices.length;
    
    // Allocate memory
    const outPtr = wasm.vwap_alloc(len);
    
    try {
        // Copy input data to WASM memory
        const timestampsPtr = wasm.vwap_alloc(len);
        const volumesPtr = wasm.vwap_alloc(len);
        const pricesPtr = wasm.vwap_alloc(len);
        
        const timestampsView = new Float64Array(wasm.__wasm.memory.buffer, timestampsPtr, len);
        const volumesView = new Float64Array(wasm.__wasm.memory.buffer, volumesPtr, len);
        const pricesView = new Float64Array(wasm.__wasm.memory.buffer, pricesPtr, len);
        
        timestampsView.set(timestamps);
        volumesView.set(volumes);
        pricesView.set(prices);
        
        // Call fast API
        wasm.vwap_into(timestampsPtr, volumesPtr, pricesPtr, outPtr, len, "1d");
        
        // Read result
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const resultCopy = Array.from(result);
        
        // Compare with safe API
        const expected = wasm.vwap_js(timestamps, volumes, prices, "1d");
        assertArrayClose(resultCopy, expected, 1e-9, 'Fast API should match safe API');
        
        // Clean up input buffers
        wasm.vwap_free(timestampsPtr, len);
        wasm.vwap_free(volumesPtr, len);
        wasm.vwap_free(pricesPtr, len);
    } finally {
        // Always free allocated memory
        wasm.vwap_free(outPtr, len);
    }
});

test('VWAP fast API aliasing', () => {
    const timestamps = testData.timestamp.slice(0, 100);
    const volumes = testData.volume.slice(0, 100);
    const prices = testData.close.slice(0, 100);
    const len = prices.length;
    
    // Test in-place operation (aliasing)
    const pricesPtr = wasm.vwap_alloc(len);
    const timestampsPtr = wasm.vwap_alloc(len);
    const volumesPtr = wasm.vwap_alloc(len);
    
    try {
        // Copy data to WASM memory
        const timestampsView = new Float64Array(wasm.__wasm.memory.buffer, timestampsPtr, len);
        const volumesView = new Float64Array(wasm.__wasm.memory.buffer, volumesPtr, len);
        const pricesView = new Float64Array(wasm.__wasm.memory.buffer, pricesPtr, len);
        
        timestampsView.set(timestamps);
        volumesView.set(volumes);
        pricesView.set(prices);
        
        // Save original prices for comparison
        const originalPrices = Array.from(pricesView);
        
        // Call with aliasing (output = prices input)
        wasm.vwap_into(timestampsPtr, volumesPtr, pricesPtr, pricesPtr, len, "1d");
        
        // Result should now be in pricesView
        const result = Array.from(pricesView);
        
        // Compare with safe API
        const expected = wasm.vwap_js(timestamps, volumes, originalPrices, "1d");
        assertArrayClose(result, expected, 1e-9, 'Aliased fast API should match safe API');
    } finally {
        wasm.vwap_free(timestampsPtr, len);
        wasm.vwap_free(volumesPtr, len);
        wasm.vwap_free(pricesPtr, len);
    }
});

test('VWAP batch with serde config', () => {
    const timestamps = testData.timestamp;
    const volumes = testData.volume;
    const prices = testData.close;
    
    // Use new batch API with config object
    const config = {
        anchor_range: ["1d", "3d", 1]
    };
    
    const result = wasm.vwap_batch(timestamps, volumes, prices, config);
    
    // Result should be an object with values, anchors, rows, cols
    assert(result.values, 'Result should have values array');
    assert(result.anchors, 'Result should have anchors array');
    assert.strictEqual(result.rows, 3, 'Should have 3 rows (1d, 2d, 3d)');
    assert.strictEqual(result.cols, prices.length, 'Cols should match input length');
    assert.deepStrictEqual(result.anchors, ["1d", "2d", "3d"], 'Anchors should match expected');
    
    // Check that values array has correct length
    assert.strictEqual(result.values.length, result.rows * result.cols, 
        'Values array should have rows × cols elements');
});

test('VWAP batch_into API', () => {
    const timestamps = testData.timestamp.slice(0, 100);
    const volumes = testData.volume.slice(0, 100);
    const prices = testData.close.slice(0, 100);
    const len = prices.length;
    
    // Calculate expected output dimensions
    const rows = 3; // 1d, 2d, 3d
    const totalSize = rows * len;
    
    // Allocate memory
    const timestampsPtr = wasm.vwap_alloc(len);
    const volumesPtr = wasm.vwap_alloc(len);
    const pricesPtr = wasm.vwap_alloc(len);
    const outPtr = wasm.vwap_alloc(totalSize);
    
    try {
        // Copy input data
        const timestampsView = new Float64Array(wasm.__wasm.memory.buffer, timestampsPtr, len);
        const volumesView = new Float64Array(wasm.__wasm.memory.buffer, volumesPtr, len);
        const pricesView = new Float64Array(wasm.__wasm.memory.buffer, pricesPtr, len);
        
        timestampsView.set(timestamps);
        volumesView.set(volumes);
        pricesView.set(prices);
        
        // Call batch_into
        const actualRows = wasm.vwap_batch_into(
            timestampsPtr, volumesPtr, pricesPtr, outPtr, len,
            "1d", "3d", 1
        );
        
        assert.strictEqual(actualRows, rows, 'Should return correct number of rows');
        
        // Read and verify result
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
        const resultCopy = Array.from(result);
        
        // First row should match single VWAP with 1d anchor
        const expected1d = wasm.vwap_js(timestamps, volumes, prices, "1d");
        const firstRow = resultCopy.slice(0, len);
        assertArrayClose(firstRow, expected1d, 1e-9, 'First row should match 1d VWAP');
    } finally {
        wasm.vwap_free(timestampsPtr, len);
        wasm.vwap_free(volumesPtr, len);
        wasm.vwap_free(pricesPtr, len);
        wasm.vwap_free(outPtr, totalSize);
    }
});

test.after(() => {
    console.log('VWAP WASM tests completed');
});
