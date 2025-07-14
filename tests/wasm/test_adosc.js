/**
 * WASM binding tests for ADOSC indicator.
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

test('ADOSC with default parameters', () => {
    const { high, low, close, volume } = testData;
    const result = wasm.adosc_js(high, low, close, volume, 3, 10);
    
    // WASM returns Float64Array, not regular Array
    assert.ok(result instanceof Float64Array || Array.isArray(result));
    assert.strictEqual(result.length, close.length);
    
    // All values should be finite
    result.forEach((val, i) => {
        assert.ok(isFinite(val), `Value at index ${i} should be finite`);
    });
});

test('ADOSC matches expected values from Rust tests', () => {
    const { high, low, close, volume } = testData;
    const expected = EXPECTED_OUTPUTS.adosc;
    
    const result = wasm.adosc_js(
        high, low, close, volume,
        expected.defaultParams.short_period,
        expected.defaultParams.long_period
    );
    
    // Check last 5 values match expected with tolerance
    const last5 = Array.from(result.slice(-5));
    assertArrayClose(last5, expected.last5Values, 1e-1, 'ADOSC last 5 values mismatch');
});

test('ADOSC fails with zero period', () => {
    const high = new Float64Array([10.0, 10.0, 10.0]);
    const low = new Float64Array([5.0, 5.0, 5.0]);
    const close = new Float64Array([7.0, 7.0, 7.0]);
    const volume = new Float64Array([1000.0, 1000.0, 1000.0]);
    
    // Zero short period
    assert.throws(
        () => wasm.adosc_js(high, low, close, volume, 0, 10),
        /Invalid period/
    );
    
    // Zero long period
    assert.throws(
        () => wasm.adosc_js(high, low, close, volume, 3, 0),
        /Invalid period/
    );
});

test('ADOSC fails when short period >= long period', () => {
    const high = new Float64Array([10.0, 11.0, 12.0, 13.0, 14.0]);
    const low = new Float64Array([5.0, 5.5, 6.0, 6.5, 7.0]);
    const close = new Float64Array([7.0, 8.0, 9.0, 10.0, 11.0]);
    const volume = new Float64Array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0]);
    
    // short = long
    assert.throws(
        () => wasm.adosc_js(high, low, close, volume, 3, 3),
        /short_period must be less than long_period/
    );
    
    // short > long
    assert.throws(
        () => wasm.adosc_js(high, low, close, volume, 5, 3),
        /short_period must be less than long_period/
    );
});

test('ADOSC fails when period exceeds data length', () => {
    const high = new Float64Array([10.0, 11.0, 12.0]);
    const low = new Float64Array([5.0, 5.5, 6.0]);
    const close = new Float64Array([7.0, 8.0, 9.0]);
    const volume = new Float64Array([1000.0, 1000.0, 1000.0]);
    
    assert.throws(
        () => wasm.adosc_js(high, low, close, volume, 3, 10),
        /Invalid period/
    );
});

test('ADOSC fails with empty input', () => {
    const empty = new Float64Array([]);
    
    assert.throws(
        () => wasm.adosc_js(empty, empty, empty, empty, 3, 10),
        /empty/
    );
});

test('ADOSC handles zero volume correctly', () => {
    const high = new Float64Array([10.0, 11.0, 12.0, 13.0, 14.0]);
    const low = new Float64Array([5.0, 5.5, 6.0, 6.5, 7.0]);
    const close = new Float64Array([7.0, 8.0, 9.0, 10.0, 11.0]);
    const volume = new Float64Array([0.0, 0.0, 0.0, 0.0, 0.0]); // All zero volume
    
    const result = wasm.adosc_js(high, low, close, volume, 2, 3);
    assert.strictEqual(result.length, close.length);
    
    // With zero volume, ADOSC should be 0
    result.forEach((val, i) => {
        assert.strictEqual(val, 0, `Value at index ${i} should be 0`);
    });
});

test('ADOSC handles constant price correctly', () => {
    const price = 10.0;
    const high = new Float64Array(10).fill(price);
    const low = new Float64Array(10).fill(price);
    const close = new Float64Array(10).fill(price);
    const volume = new Float64Array(10).fill(1000.0);
    
    const result = wasm.adosc_js(high, low, close, volume, 3, 5);
    assert.strictEqual(result.length, close.length);
    
    // With constant price (high = low), MFM is 0, so ADOSC should be 0
    result.forEach((val, i) => {
        assert.strictEqual(val, 0, `Value at index ${i} should be 0`);
    });
});

test('ADOSC calculates from the first value (no warmup period)', () => {
    const { high, low, close, volume } = testData;
    const result = wasm.adosc_js(high, low, close, volume, 3, 10);
    
    // First value should not be NaN (ADOSC calculates from the start)
    assert.ok(!isNaN(result[0]), 'First ADOSC value should not be NaN');
});

test('ADOSC batch calculation with default parameters', () => {
    const { high, low, close, volume } = testData;
    
    const result = wasm.adosc_batch_js(
        high, low, close, volume,
        3, 3, 0,   // short_period range (single value)
        10, 10, 0  // long_period range (single value)
    );
    
    // Batch returns flat array
    assert.ok(result instanceof Float64Array || Array.isArray(result));
    assert.strictEqual(result.length, close.length);
});

test('ADOSC batch calculation with multiple parameters', () => {
    const { high, low, close, volume } = testData;
    
    const result = wasm.adosc_batch_js(
        high, low, close, volume,
        2, 4, 1,   // short_period range: 2, 3, 4
        5, 7, 1    // long_period range: 5, 6, 7
    );
    
    // Should have 3 * 3 = 9 combinations
    const expected_rows = 3 * 3;
    assert.strictEqual(result.length, expected_rows * close.length);
});

test('ADOSC batch metadata', () => {
    // For short_period 2-4 step 1 and long_period 5-7 step 1
    const meta = wasm.adosc_batch_metadata_js(2, 4, 1, 5, 7, 1);
    
    assert.ok(meta instanceof Float64Array || Array.isArray(meta));
    // 3 short periods * 3 long periods = 9 combos, each has 2 values
    assert.strictEqual(meta.length, 3 * 3 * 2);
});

// Skip stream tests - WASM stream bindings not implemented for ADOSC
// test('ADOSC stream creation and update', () => {
//     const stream = wasm.adosc_stream_new(3, 10);
//     assert.ok(stream);
//     
//     // Test update with valid values
//     const val = wasm.adosc_stream_update(stream, 100.0, 90.0, 95.0, 1000.0);
//     assert.ok(isFinite(val));
//     
//     // Free the stream
//     wasm.adosc_stream_free(stream);
// });

test('ADOSC comparison with Rust', () => {
    const { high, low, close, volume } = testData;
    const expected = EXPECTED_OUTPUTS.adosc;
    
    const result = wasm.adosc_js(
        high, low, close, volume,
        expected.defaultParams.short_period,
        expected.defaultParams.long_period
    );
    
    compareWithRust('adosc', Array.from(result), 'hlcv', expected.defaultParams);
});