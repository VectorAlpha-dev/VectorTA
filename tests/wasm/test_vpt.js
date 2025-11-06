/**
 * WASM binding tests for VPT (Volume Price Trend).
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
import { compareWithRust } from './rust-comparison.js';

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

test('VPT basic candles', () => {
    // Test VPT with candle data - mirrors check_vpt_basic_candles
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.vpt_js(close, volume);
    assert.strictEqual(result.length, close.length);
});

test('VPT basic slices', () => {
    // Test VPT with basic slice data - mirrors check_vpt_basic_slices
    const price = new Float64Array([1.0, 1.1, 1.05, 1.2, 1.3]);
    const volume = new Float64Array([1000.0, 1100.0, 1200.0, 1300.0, 1400.0]);
    
    const result = wasm.vpt_js(price, volume);
    assert.strictEqual(result.length, price.length);
    
    // First two values should be NaN (warmup through first_valid)
    assert(isNaN(result[0]), 'First value should be NaN');
    assert(isNaN(result[1]), 'Second value should be NaN');
    // Rest should have values
    assert(!isNaN(result[2]), 'Third value should not be NaN');
});

test('VPT accuracy from CSV', async () => {
    // Test VPT matches expected values from Rust tests - mirrors check_vpt_accuracy_from_csv
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.vpt_js(close, volume);
    
    const expected_last_five = [
        -18292.323972247592,
        -18292.510374716476,
        -18292.803266539282,
        -18292.62919783763,
        -18296.152568643138,
    ];
    
    assert(result.length >= 5);
    const last5 = result.slice(-5);
    
    // Match Rust tolerance (abs diff < 1e-9)
    assertArrayClose(
        last5,
        expected_last_five,
        1e-9,
        "VPT last 5 values mismatch"
    );
    
    // Compare with Rust
    await compareWithRust('vpt', result);
});

test('VPT not enough data', () => {
    // Test VPT fails with insufficient data - mirrors check_vpt_not_enough_data
    const price = new Float64Array([100.0]);
    const volume = new Float64Array([500.0]);
    
    assert.throws(
        () => wasm.vpt_js(price, volume),
        /Not enough valid data/,
        'Should throw with insufficient data'
    );
});

test('VPT empty data', () => {
    // Test VPT fails with empty data - mirrors check_vpt_empty_data
    const price = new Float64Array([]);
    const volume = new Float64Array([]);
    
    assert.throws(
        () => wasm.vpt_js(price, volume),
        /Empty data/,
        'Should throw with empty data'
    );
});

test('VPT all NaN', () => {
    // Test VPT fails with all NaN values - mirrors check_vpt_all_nan
    const price = new Float64Array([NaN, NaN, NaN]);
    const volume = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(
        () => wasm.vpt_js(price, volume),
        /All values are NaN/,
        'Should throw with all NaN values'
    );
});

test('VPT mismatched lengths', () => {
    // Test VPT fails with mismatched input lengths
    const price = new Float64Array([1.0, 2.0, 3.0]);
    const volume = new Float64Array([100.0, 200.0]); // Different length
    
    assert.throws(
        () => wasm.vpt_js(price, volume),
        /Empty data/,
        'Should throw with mismatched lengths'
    );
});

test('VPT in-place operation', () => {
    // Test VPT fast API with in-place operation (aliasing)
    const price = new Float64Array([1.0, 1.1, 1.05, 1.2, 1.3]);
    const volume = new Float64Array([1000.0, 1100.0, 1200.0, 1300.0, 1400.0]);
    const len = price.length;
    
    // Allocate output
    const out_ptr = wasm.vpt_alloc(len);
    const output = new Float64Array(wasm.__wasm.memory.buffer, out_ptr, len);
    
    // Copy price to output for in-place test
    output.set(price);
    
    // Test in-place (price_ptr == out_ptr)
    // Create volume array in WASM memory
    const vol_ptr = wasm.vpt_alloc(len);
    const volumeWasm = new Float64Array(wasm.__wasm.memory.buffer, vol_ptr, len);
    volumeWasm.set(volume);
    
    wasm.vpt_into(out_ptr, vol_ptr, out_ptr, len);
    
    // Clean up volume allocation
    wasm.vpt_free(vol_ptr, len);
    
    // Should have results (first two are NaN due to warmup)
    assert(isNaN(output[0]), 'First value should be NaN');
    assert(isNaN(output[1]), 'Second value should be NaN');
    assert(!isNaN(output[2]), 'Should have computed values after in-place operation');
    
    // Clean up
    wasm.vpt_free(out_ptr, len);
});

test('VPT batch operations', () => {
    // Test VPT batch processing
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    // VPT has no parameters, so config is empty
    const config = {};
    
    const result = wasm.vpt_batch(close, volume, config);
    
    assert(result.values, 'Should have values array');
    assert.strictEqual(result.rows, 1, 'Should have single row (no parameters)');
    assert.strictEqual(result.cols, close.length, 'Columns should match data length');
    
    // Compare with single calculation
    const single = wasm.vpt_js(close, volume);
    assertArrayClose(
        result.values,
        single,
        1e-10,
        "Batch vs single VPT mismatch"
    );
});

test('VPT NaN handling', () => {
    // Test VPT handles NaN values correctly
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // Insert some NaN values
    close[10] = NaN;
    volume[20] = NaN;
    
    const result = wasm.vpt_js(close, volume);
    assert.strictEqual(result.length, close.length);
    
    // First value should always be NaN
    assert(isNaN(result[0]), 'First value should be NaN');
    
    // Values around NaN inputs should propagate NaN correctly
    assert(isNaN(result[11]), 'Value after NaN price should be NaN');
    assert(isNaN(result[21]), 'Value after NaN volume should be NaN');
});
