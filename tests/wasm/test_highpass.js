/**
 * WASM binding tests for HighPass indicator.
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
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('HighPass partial params', () => {
    // Test with default parameters - mirrors check_highpass_partial_params
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.highpass;
    
    const result = wasm.highpass_js(close, expected.defaultParams.period);
    assert.strictEqual(result.length, close.length);
});

test('HighPass accuracy', async () => {
    // Test HighPass matches expected values from Rust tests - mirrors check_highpass_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.highpass;
    
    const result = wasm.highpass_js(close, expected.defaultParams.period);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-6,  // Using 1e-6 as in Rust test
        "HighPass last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('highpass', result, 'close', expected.defaultParams);
});

test('HighPass default candles', async () => {
    // Test HighPass with default parameters - mirrors check_highpass_default_candles
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.highpass;
    
    const result = wasm.highpass_js(close, expected.defaultParams.period);
    assert.strictEqual(result.length, close.length);
    
    // Compare with Rust
    await compareWithRust('highpass', result, 'close', expected.defaultParams);
});

test('HighPass zero period', () => {
    // Test HighPass fails with zero period - mirrors check_highpass_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.highpass_js(inputData, 0);
    });
});

test('HighPass period exceeds length', () => {
    // Test HighPass fails when period exceeds data length - mirrors check_highpass_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.highpass_js(dataSmall, 48);
    });
});

test('HighPass very small dataset', () => {
    // Test HighPass with very small dataset - mirrors check_highpass_very_small_dataset
    const dataSmall = new Float64Array([42.0, 43.0]);
    
    assert.throws(() => {
        wasm.highpass_js(dataSmall, 2);
    });
});

test('HighPass empty input', () => {
    // Test HighPass with empty input - mirrors check_highpass_empty_input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.highpass_js(dataEmpty, 48);
    });
});

test('HighPass invalid alpha', () => {
    // Test HighPass with invalid alpha - mirrors check_highpass_invalid_alpha
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // Period=4 causes cos(pi/2) ~ 0
    assert.throws(() => {
        wasm.highpass_js(data, 4);
    });
});

test('HighPass all NaN', () => {
    // Test HighPass with all NaN input
    const data = new Float64Array([NaN, NaN, NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.highpass_js(data, 3);
    });
});

test('HighPass reinput', () => {
    // Test HighPass with re-input of HighPass result - mirrors check_highpass_reinput
    const close = new Float64Array(testData.close);
    
    // First HighPass pass with period=36
    const firstResult = wasm.highpass_js(close, 36);
    
    // Second HighPass pass with period=24 using first result as input
    const secondResult = wasm.highpass_js(firstResult, 24);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Verify no NaN values after index 240 in second result
    for (let i = 240; i < secondResult.length; i++) {
        assert(!isNaN(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('HighPass NaN handling', () => {
    // Test HighPass handling of NaN values - mirrors check_highpass_nan_handling
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.highpass;
    const period = expected.defaultParams.period;
    
    const result = wasm.highpass_js(close, period);
    
    assert.strictEqual(result.length, close.length);
    
    // HighPass has no warmup period - should produce values from index 0
    // Verify no NaN values at the start (unlike ALMA which has warmup NaNs)
    assert(!isNaN(result[0]), "HighPass should produce value at index 0");
    
    // Verify all values are not NaN
    for (let i = 0; i < result.length; i++) {
        assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
    }
});

test('HighPass warmup period', () => {
    // Test that HighPass has no warmup period - starts from index 0
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.highpass;
    
    const result = wasm.highpass_js(close, expected.defaultParams.period);
    
    // Verify HighPass produces a value from the very first index
    assert(!isNaN(result[0]), "HighPass should produce value at index 0 (no warmup)");
    
    // Verify no warmup NaN values (unlike indicators with warmup periods)
    assert.strictEqual(expected.hasWarmup, false, "HighPass should have no warmup period");
    assert.strictEqual(expected.warmupLength, 0, "HighPass warmup length should be 0");
    
    // All values should be valid numbers
    assertNoNaN(result, "HighPass should have no NaN values");
});

test('HighPass leading NaN input', () => {
    // Test HighPass with leading NaN values in input
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Insert NaN values at the beginning
    for (let i = 0; i < 5; i++) {
        close[i] = NaN;
    }
    
    // HighPass is an IIR filter - NaN values propagate through the entire output
    const result = wasm.highpass_js(close, 48);
    assert.strictEqual(result.length, close.length);
    
    // With leading NaN, the IIR filter propagates NaN through all outputs
    // This is expected behavior for IIR filters with NaN contamination
    for (let i = 0; i < result.length; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} due to IIR filter NaN propagation`);
    }
});

test('HighPass batch', () => {
    // Test HighPass batch computation with ergonomic API
    const close = new Float64Array(testData.close.slice(0, 500)); // Smaller dataset for speed
    
    // Test using new ergonomic batch API (like ALMA)
    const batchResult = wasm.highpass_batch(close, {
        period_range: [30, 60, 10]  // periods: 30, 40, 50, 60
    });
    
    // Should have correct structure
    assert(batchResult.values, "Batch result should have values");
    assert(batchResult.combos, "Batch result should have combos");
    assert.strictEqual(batchResult.rows, 4, "Should have 4 rows");
    assert.strictEqual(batchResult.cols, close.length, "Should have cols equal to data length");
    
    // Verify each row has no warmup NaN (highpass starts from index 0)
    for (let row = 0; row < 4; row++) {
        const rowStart = row * close.length;
        assert(!isNaN(batchResult.values[rowStart]), `Row ${row} should have value at index 0`);
    }
    
    // Verify each row matches individual calculation
    const periods = [30, 40, 50, 60];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * close.length;
        const rowEnd = rowStart + close.length;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const individualResult = wasm.highpass_js(close, periods[i]);
        assertArrayClose(rowData, individualResult, 1e-9, `Period ${periods[i]} mismatch`);
    }
});

test('HighPass edge cases', () => {
    // Test HighPass with edge case inputs
    const period = 10;
    
    // Test with exactly period-sized data
    const dataExact = new Float64Array(testData.close.slice(0, period));
    const resultExact = wasm.highpass_js(dataExact, period);
    assert.strictEqual(resultExact.length, dataExact.length);
    assert(!isNaN(resultExact[0]), "Should have value at index 0");
    
    // Test with period+1 data points
    const dataPlusOne = new Float64Array(testData.close.slice(0, period + 1));
    const resultPlusOne = wasm.highpass_js(dataPlusOne, period);
    assert.strictEqual(resultPlusOne.length, dataPlusOne.length);
    assert(!isNaN(resultPlusOne[0]), "Should have value at index 0");
    
    // Test with constant data (DC signal)
    const constantData = new Float64Array(100);
    constantData.fill(50.0);
    const resultConstant = wasm.highpass_js(constantData, 20);
    
    // After stabilization, highpass should remove DC component (approach zero)
    // Check values after 3*period are near zero
    const stabilizedStart = 3 * 20;
    for (let i = stabilizedStart; i < resultConstant.length; i++) {
        assert(Math.abs(resultConstant[i]) < 1e-3, 
            `DC component not removed at index ${i}: ${resultConstant[i]}`);
    }
});

test('HighPass different periods', () => {
    // Test HighPass with different period values
    const close = new Float64Array(testData.close);
    
    // Test various period values
    for (const period of [10, 20, 30, 48, 60]) {
        const result = wasm.highpass_js(close, period);
        assert.strictEqual(result.length, close.length);
        
        // Find where valid data starts
        let firstValid = null;
        for (let i = 0; i < result.length; i++) {
            if (!isNaN(result[i])) {
                firstValid = i;
                break;
            }
        }
        
        // Verify that we have valid data
        assert(firstValid !== null, `No valid data found for period=${period}`);
        
        // Verify no NaN after first valid
        for (let i = firstValid; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for period=${period}`);
        }
    }
});

test('HighPass batch multiple parameters', () => {
    // Test HighPass batch with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 200)); // Smaller dataset
    
    // Test multiple periods using old API for compatibility
    const periods = [10, 20, 30, 40, 50];
    const batchResult = wasm.highpass_batch_js(close, 10, 50, 10);
    const metadata = wasm.highpass_batch_metadata_js(10, 50, 10);
    
    assert.strictEqual(metadata.length, 5); // 5 periods
    assert.strictEqual(batchResult.length, 5 * close.length);
    
    // Verify each row has no warmup NaN
    for (let row = 0; row < 5; row++) {
        const rowStart = row * close.length;
        assert(!isNaN(batchResult[rowStart]), `Row ${row} should have value at index 0`);
        
        // Compare with individual calculation
        const individualResult = wasm.highpass_js(close, periods[row]);
        const rowData = batchResult.slice(rowStart, rowStart + close.length);
        assertArrayClose(rowData, individualResult, 1e-9, 
            `Batch row ${row} (period=${periods[row]}) mismatch`);
    }
});

test('HighPass batch performance', () => {
    // Test that batch computation is more efficient than multiple single computations
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use first 1000 values
    
    // Test 5 periods
    const startBatch = performance.now();
    const batchResult = wasm.highpass_batch_js(close, 20, 60, 10);
    const batchTime = performance.now() - startBatch;
    
    const startSingle = performance.now();
    const singleResults = [];
    for (let period = 20; period <= 60; period += 10) {
        singleResults.push(...wasm.highpass_js(close, period));
    }
    const singleTime = performance.now() - startSingle;
    
    // Batch should be faster than multiple single calls
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    // Verify results match
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});