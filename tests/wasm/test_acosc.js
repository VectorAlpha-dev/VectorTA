/**
 * WASM binding tests for ACOSC indicator.
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

test('ACOSC partial params', () => {
    // Test with default parameters - mirrors check_acosc_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.acosc_js(high, low);
    assert.strictEqual(result.length, high.length * 2); // osc + change
});

test('ACOSC accuracy', async () => {
    // Test ACOSC matches expected values from Rust tests - mirrors check_acosc_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const expected = EXPECTED_OUTPUTS.acosc;
    
    const result = wasm.acosc_js(high, low);
    
    // Result contains flattened [osc, change] arrays
    const len = high.length;
    const osc = result.slice(0, len);
    const change = result.slice(len, len * 2);
    
    assert.strictEqual(osc.length, len);
    assert.strictEqual(change.length, len);
    
    // Check last 5 osc values match expected
    const last5Osc = osc.slice(-5);
    assertArrayClose(
        last5Osc,
        expected.last5Osc,
        0.1,  // ACOSC uses 1e-1 tolerance in Rust tests
        "ACOSC osc last 5 values mismatch"
    );
    
    // Check last 5 change values match expected
    const last5Change = change.slice(-5);
    assertArrayClose(
        last5Change,
        expected.last5Change,
        0.1,  // ACOSC uses 1e-1 tolerance in Rust tests
        "ACOSC change last 5 values mismatch"
    );
    
    // Skip Rust comparison for now - compareWithRust doesn't handle multi-output indicators yet
    // TODO: Update compareWithRust to handle indicators with multiple outputs
    // await compareWithRust('acosc', {osc, change}, 'high_low', expected.defaultParams);
});

test('ACOSC too short', () => {
    // Test ACOSC fails with insufficient data
    const high = new Float64Array([100.0, 101.0]);
    const low = new Float64Array([99.0, 98.0]);
    
    assert.throws(() => {
        wasm.acosc_js(high, low);
    }, /Not enough data/);
});

test('ACOSC length mismatch', () => {
    // Test ACOSC fails when high and low lengths don't match
    const high = new Float64Array([100.0, 101.0, 102.0]);
    const low = new Float64Array([99.0, 98.0]);  // Shorter array
    
    assert.throws(() => {
        wasm.acosc_js(high, low);
    }, /Mismatch/);
});

test('ACOSC NaN handling', () => {
    // Test ACOSC handles NaN values correctly
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.acosc_js(high, low);
    const len = high.length;
    const osc = result.slice(0, len);
    const change = result.slice(len, len * 2);
    
    // First 38 values should be NaN (warmup period = 34 + 5 - 1)
    assertAllNaN(osc.slice(0, 38), "Expected NaN in warmup period for osc");
    assertAllNaN(change.slice(0, 38), "Expected NaN in warmup period for change");
    
    // After warmup period, no NaN values should exist
    if (osc.length > 240) {
        for (let i = 240; i < osc.length; i++) {
            assert(!isNaN(osc[i]), `Found unexpected NaN in osc at index ${i}`);
            assert(!isNaN(change[i]), `Found unexpected NaN in change at index ${i}`);
        }
    }
});

test('ACOSC leading NaNs', () => {
    // Test ACOSC handles leading NaN values correctly
    // Create specific test data with known values
    const nanArray = new Float64Array(10);
    nanArray.fill(NaN);
    const dataArray = new Float64Array(200);
    for (let i = 0; i < 200; i++) {
        dataArray[i] = 100 + i;
    }
    
    const high = new Float64Array(210);
    high.set(nanArray, 0);
    high.set(dataArray, 10);
    
    const low = new Float64Array(210);
    low.set(nanArray, 0);
    const lowData = new Float64Array(200);
    for (let i = 0; i < 200; i++) {
        lowData[i] = 99 + i;
    }
    low.set(lowData, 10);
    
    const result = wasm.acosc_js(high, low);
    const len = high.length;
    const osc = result.slice(0, len);
    const change = result.slice(len, len * 2);
    
    // Warmup should be first_valid (10) + 38 = 48
    const expectedWarmup = 10 + 38;
    
    // Check warmup period
    assertAllNaN(osc.slice(0, expectedWarmup), `Expected NaN in warmup period [0:${expectedWarmup}] for osc`);
    assertAllNaN(change.slice(0, expectedWarmup), `Expected NaN in warmup period [0:${expectedWarmup}] for change`);
    
    // Should have valid values after warmup
    assert(!isNaN(osc[expectedWarmup]), `Expected valid value at index ${expectedWarmup} for osc`);
    assert(!isNaN(change[expectedWarmup]), `Expected valid value at index ${expectedWarmup} for change`);
});

test('ACOSC all NaN input', () => {
    // Test ACOSC with all NaN values - throws error due to no valid data
    const allNanHigh = new Float64Array(100);
    const allNanLow = new Float64Array(100);
    allNanHigh.fill(NaN);
    allNanLow.fill(NaN);
    
    // ACOSC throws error when all input is NaN (no valid data points)
    assert.throws(() => {
        wasm.acosc_js(allNanHigh, allNanLow);
    }, /Not enough data/);
});

test('ACOSC single point', () => {
    // Test ACOSC with single data point
    const singleHigh = new Float64Array([100.0]);
    const singleLow = new Float64Array([99.0]);
    
    assert.throws(() => {
        wasm.acosc_js(singleHigh, singleLow);
    }, /Not enough data/);
});

test('ACOSC batch single result', () => {
    // Test batch with single result (since ACOSC has no parameters)
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    // Batch should return same as single
    const batchResult = wasm.acosc_batch_js(high, low);
    const singleResult = wasm.acosc_js(high, low);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('ACOSC batch metadata', () => {
    // Test metadata function returns empty array (no parameters)
    const metadata = wasm.acosc_batch_metadata_js();
    
    // Should be empty since ACOSC has no parameters
    assert.strictEqual(metadata.length, 0);
});

test('ACOSC edge cases', () => {
    // Test with exactly minimum required data (39 points)
    const high = new Float64Array(testData.high.slice(0, 39));
    const low = new Float64Array(testData.low.slice(0, 39));
    
    const result = wasm.acosc_js(high, low);
    assert.strictEqual(result.length, 78); // 39 osc + 39 change
    
    const osc = result.slice(0, 39);
    const change = result.slice(39, 78);
    
    // First 38 should be NaN
    assertAllNaN(osc.slice(0, 38), "Expected NaN in first 38 values for osc");
    assertAllNaN(change.slice(0, 38), "Expected NaN in first 38 values for change");
    
    // Last value should be valid
    assert(!isNaN(osc[38]), "Expected valid value at index 38 for osc");
    assert(!isNaN(change[38]), "Expected valid value at index 38 for change");
    
    // Test with 38 points (one less than minimum) - should throw
    const tooSmallHigh = new Float64Array(testData.high.slice(0, 38));
    const tooSmallLow = new Float64Array(testData.low.slice(0, 38));
    
    assert.throws(() => {
        wasm.acosc_js(tooSmallHigh, tooSmallLow);
    }, /Not enough data/);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.acosc_js(new Float64Array([]), new Float64Array([]));
    }, /Empty input data|Not enough data/);
});

test('ACOSC batch returns same as single', () => {
    // Test that batch returns the same result as single API
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    
    const singleResult = wasm.acosc_js(high, low);
    const batchResult = wasm.acosc_batch_js(high, low);
    
    // Both should return the same values
    assert.strictEqual(singleResult.length, batchResult.length);
    assertArrayClose(singleResult, batchResult, 1e-10, "Batch should match single result");
});

test.after(() => {
    console.log('ACOSC WASM tests completed');
});
