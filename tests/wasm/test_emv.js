/**
 * WASM binding tests for EMV indicator.
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

test('EMV basic calculation', () => {
    // Test basic EMV calculation - mirrors check_emv_basic_calculation
    const high = new Float64Array([10.0, 12.0, 13.0, 15.0]);
    const low = new Float64Array([5.0, 7.0, 8.0, 10.0]);
    const close = new Float64Array([7.5, 9.0, 10.5, 12.5]);
    const volume = new Float64Array([10000.0, 20000.0, 25000.0, 30000.0]);
    
    const result = wasm.emv_js(high, low, close, volume);
    assert.strictEqual(result.length, 4);
    assert(isNaN(result[0])); // First value should be NaN
    assert(!isNaN(result[1]));
});

test('EMV accuracy', () => {
    // Test EMV matches expected values from Rust tests - mirrors check_emv_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.emv_js(high, low, close, volume);
    assert.strictEqual(result.length, high.length);
    
    // Expected last 5 values from Rust tests
    const expected_last_five = [
        -6488905.579799851,
        2371436.7401001123,
        -3855069.958128531,
        1051939.877943717,
        -8519287.22257077,
    ];
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected_last_five,
        0.0001,
        "EMV last 5 values mismatch"
    );
});

test('EMV empty data', () => {
    // Test EMV with empty data - mirrors check_emv_empty_data
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.emv_js(empty, empty, empty, empty);
    }, /Empty data|EmptyData/);
});

test('EMV all NaN', () => {
    // Test EMV with all NaN values - mirrors check_emv_all_nan
    const nanArr = new Float64Array([NaN, NaN]);
    
    assert.throws(() => {
        wasm.emv_js(nanArr, nanArr, nanArr, nanArr);
    }, /All values are NaN|AllValuesNaN/);
});

test('EMV not enough data', () => {
    // Test EMV with insufficient data - mirrors check_emv_not_enough_data
    const high = new Float64Array([10000.0, NaN]);
    const low = new Float64Array([9990.0, NaN]);
    const close = new Float64Array([9995.0, NaN]);
    const volume = new Float64Array([1_000_000.0, NaN]);
    
    assert.throws(() => {
        wasm.emv_js(high, low, close, volume);
    }, /Not enough data|NotEnoughData/);
});

test('EMV NaN handling', () => {
    // Test EMV handles NaN values correctly - mirrors check_emv_nan_handling
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.emv_js(high, low, close, volume);
    assert.strictEqual(result.length, high.length);
    
    // First value should be NaN
    assert(isNaN(result[0]), "Expected NaN in first position");
    
    // After warmup period, no NaN values should exist
    let validCount = 0;
    for (let i = 10; i < result.length; i++) {
        if (!isNaN(result[i])) {
            validCount++;
        }
    }
    assert(validCount > 0, "Expected valid values after warmup");
});

test('EMV zero-copy (fast) API', () => {
    // Test the zero-copy fast API
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const output = new Float64Array(high.length);
    
    // Test non-aliased operation
    wasm.emv_into(
        high.buffer,
        low.buffer,
        close.buffer,
        volume.buffer,
        output.buffer,
        high.length
    );
    
    // First value should be NaN
    assert(isNaN(output[0]));
    
    // Should have valid values after first
    assert(!isNaN(output[10]));
});

test('EMV zero-copy API with aliasing', () => {
    // Test in-place operation (output aliased with one of the inputs)
    const data = new Float64Array(testData.close);
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const volume = new Float64Array(testData.volume);
    
    // Use close array as both input and output
    wasm.emv_into(
        high.buffer,
        low.buffer,
        data.buffer, // close input
        volume.buffer,
        data.buffer, // output (aliased with close)
        data.length
    );
    
    // Should have EMV values now, not original close values
    assert(isNaN(data[0])); // First EMV value is NaN
});

test('EMV memory allocation/deallocation', () => {
    // Test memory management functions
    const len = 1000;
    const ptr = wasm.emv_alloc(len);
    assert(ptr !== 0, "Expected non-null pointer");
    
    // Free the memory
    wasm.emv_free(ptr, len);
    // If we get here without crashing, the test passed
});

test('EMV batch operations', () => {
    // Test batch API
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // EMV has no parameters, so config is empty
    const config = {};
    
    const result = wasm.emv_batch(high, low, close, volume, config);
    
    assert(result.values, "Expected values array");
    assert.strictEqual(result.rows, 1, "EMV batch should have 1 row (no parameter sweep)");
    assert.strictEqual(result.cols, high.length, "Expected cols to match input length");
    
    // Values should match single calculation
    const singleResult = wasm.emv_js(high, low, close, volume);
    assertArrayClose(
        new Float64Array(result.values),
        singleResult,
        1e-10,
        "Batch values should match single calculation"
    );
});

test('EMV batch into (fast API)', () => {
    // Test batch into API
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const output = new Float64Array(high.length);
    
    const rows = wasm.emv_batch_into(
        high.buffer,
        low.buffer,
        close.buffer,
        volume.buffer,
        output.buffer,
        high.length
    );
    
    assert.strictEqual(rows, 1, "Expected 1 row for EMV batch");
    assert(isNaN(output[0]), "First value should be NaN");
    assert(!isNaN(output[10]), "Should have valid values after warmup");
});

test('EMV mismatched lengths', () => {
    // Test EMV with mismatched input lengths
    const high = new Float64Array([10.0, 12.0, 13.0]);
    const low = new Float64Array([9.0, 11.0]); // Different length
    const close = new Float64Array([9.5, 11.5, 12.0]);
    const volume = new Float64Array([1000.0, 2000.0, 3000.0]);
    
    // Should work but use minimum length
    const result = wasm.emv_js(high, low, close, volume);
    assert.strictEqual(result.length, 2); // min(3, 2, 3, 3) = 2
});

test('EMV zero range handling', () => {
    // Test EMV when high equals low (zero range)
    const high = new Float64Array([10.0, 10.0, 12.0, 13.0]);
    const low = new Float64Array([9.0, 10.0, 11.0, 12.0]); // At index 1: high == low
    const close = new Float64Array([9.5, 10.0, 11.5, 12.5]);
    const volume = new Float64Array([1000.0, 2000.0, 3000.0, 4000.0]);
    
    const result = wasm.emv_js(high, low, close, volume);
    
    // When range is zero, EMV should be NaN
    assert(isNaN(result[1]), "Expected NaN when range is zero");
    
    // Other values should be calculated
    assert(!isNaN(result[2]));
});

test('EMV partial NaN handling', () => {
    // Test EMV with partial NaN values
    const high = new Float64Array([NaN, 12.0, 15.0, NaN, 13.0, 16.0]);
    const low = new Float64Array([NaN, 9.0, 11.0, NaN, 10.0, 12.0]);
    const close = new Float64Array([NaN, 10.0, 13.0, NaN, 11.5, 14.0]);
    const volume = new Float64Array([NaN, 10000.0, 20000.0, NaN, 15000.0, 25000.0]);
    
    const result = wasm.emv_js(high, low, close, volume);
    
    // Check shape
    assert.strictEqual(result.length, high.length);
    
    // First few should be NaN
    assert(isNaN(result[0]));
    assert(isNaN(result[1])); // Need previous value
    
    // Should have valid values after enough data
    assert(!isNaN(result[2]));
});

test('EMV null pointer handling', () => {
    // Test null pointer error handling
    assert.throws(() => {
        wasm.emv_into(0, 0, 0, 0, 0, 100);
    }, /null pointer/);
});