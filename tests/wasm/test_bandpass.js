/**
 * WASM binding tests for Band-Pass indicator.
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

test('BandPass partial params', () => {
    // Test with default parameters - mirrors check_bandpass_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.bandpass_js(close, 20, 0.3);
    assert(result, 'Should have result');
    assert(result.values, 'Should have values array');
    assert.strictEqual(result.rows, 4, 'Should have 4 rows (bp, bp_normalized, signal, trigger)');
    assert.strictEqual(result.cols, close.length, 'Cols should match input length');
    assert.strictEqual(result.values.length, close.length * 4, 'Values should be flattened array');
});

test('BandPass accuracy', async () => {
    // Test Band-Pass matches expected values from Rust tests - mirrors check_bandpass_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.bandpass;
    
    const result = wasm.bandpass_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.bandwidth
    );
    
    assert(result, 'Should have result');
    assert.strictEqual(result.rows, 4, 'Should have 4 rows');
    assert.strictEqual(result.cols, close.length, 'Cols should match input length');
    
    // Extract arrays from flattened result
    const bp = result.values.slice(0, close.length);
    const bp_normalized = result.values.slice(close.length, close.length * 2);
    const signal = result.values.slice(close.length * 2, close.length * 3);
    const trigger = result.values.slice(close.length * 3, close.length * 4);
    
    // Check last 5 values match expected with 1e-1 tolerance (as in Rust tests)
    const last5Bp = bp.slice(-5);
    assertArrayClose(
        last5Bp,
        expected.last5Values.bp,
        1e-1,
        "Band-Pass bp last 5 values mismatch"
    );
    
    const last5BpNormalized = bp_normalized.slice(-5);
    assertArrayClose(
        last5BpNormalized,
        expected.last5Values.bp_normalized,
        1e-1,
        "Band-Pass bp_normalized last 5 values mismatch"
    );
    
    const last5Signal = signal.slice(-5);
    assertArrayClose(
        last5Signal,
        expected.last5Values.signal,
        1e-1,
        "Band-Pass signal last 5 values mismatch"
    );
    
    const last5Trigger = trigger.slice(-5);
    assertArrayClose(
        last5Trigger,
        expected.last5Values.trigger,
        1e-1,
        "Band-Pass trigger last 5 values mismatch"
    );
});

test('BandPass default params', () => {
    // Test Band-Pass with default parameters
    const close = new Float64Array(testData.close);
    
    const result = wasm.bandpass_js(close, 20, 0.3);
    assert(result, 'Should have result');
    assert.strictEqual(result.rows, 4, 'Should have 4 rows');
    assert.strictEqual(result.cols, close.length, 'Cols should match input length');
    assert.strictEqual(result.values.length, close.length * 4, 'Values should be flattened array');
});

test('BandPass zero period', () => {
    // Test Band-Pass fails with zero period - mirrors check_bandpass_zero_period
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.bandpass_js(data, 0, 0.3);
    }, /Invalid period/);
});

test('BandPass period exceeds length', () => {
    // Test Band-Pass fails when period exceeds data length - mirrors check_bandpass_period_exceeds_length
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.bandpass_js(data, 10, 0.3);
    }, /Not enough data/);
});

test('BandPass very small dataset', () => {
    // Test Band-Pass fails with insufficient data - mirrors check_bandpass_very_small_dataset
    const single_point = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.bandpass_js(single_point, 20, 0.3);
    }, /Not enough data/);
});

test('BandPass reinput', () => {
    // Test Band-Pass applied twice (re-input) - mirrors check_bandpass_reinput
    const close = new Float64Array(testData.close);
    
    // First pass with specific parameters
    const firstResult = wasm.bandpass_js(close, 20, 0.3);
    assert.strictEqual(firstResult.cols, close.length);
    
    // Extract bp from flattened result
    const firstBp = firstResult.values.slice(0, close.length);
    
    // Second pass with different parameters using first result bp as input
    const secondResult = wasm.bandpass_js(
        new Float64Array(firstBp),
        30,
        0.5
    );
    assert.strictEqual(secondResult.cols, firstResult.cols);
});

test('BandPass NaN handling', () => {
    // Test Band-Pass handles NaN values correctly - mirrors check_bandpass_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.bandpass_js(close, 20, 0.3);
    assert.strictEqual(result.cols, close.length);
    
    // Extract arrays from flattened result
    const bp = result.values.slice(0, close.length);
    const bp_normalized = result.values.slice(close.length, close.length * 2);
    const signal = result.values.slice(close.length * 2, close.length * 3);
    const trigger = result.values.slice(close.length * 3, close.length * 4);
    
    // After index 30, no NaN values should exist in any output
    if (close.length > 30) {
        for (let i = 30; i < close.length; i++) {
            assert(!isNaN(bp[i]), `Found unexpected NaN in bp at index ${i}`);
            assert(!isNaN(bp_normalized[i]), `Found unexpected NaN in bp_normalized at index ${i}`);
            assert(!isNaN(signal[i]), `Found unexpected NaN in signal at index ${i}`);
            assert(!isNaN(trigger[i]), `Found unexpected NaN in trigger at index ${i}`);
        }
    }
});

test('BandPass batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Single parameter set: period=20, bandwidth=0.3
    const batchResult = wasm.bandpass_batch(close, {
        period_range: [20, 20, 0],
        bandwidth_range: [0.3, 0.3, 0.0]
    });
    
    assert(batchResult, 'Should have result');
    assert.strictEqual(batchResult.combos, 1, 'Should have 1 combination');
    assert.strictEqual(batchResult.outputs, 4, 'Should have 4 outputs');
    assert.strictEqual(batchResult.cols, close.length, 'Cols should match input length');
    
    // Extract first combination's bp values
    const batchBp = batchResult.values.slice(0, close.length);
    
    // Should match single calculation
    const singleResult = wasm.bandpass_js(close, 20, 0.3);
    const singleBp = singleResult.values.slice(0, close.length);
    
    assertArrayClose(batchBp, singleBp, 1e-10, "Batch vs single bp mismatch");
});

test('BandPass batch multiple parameters', () => {
    // Test batch with multiple parameter combinations
    // Need 600 data points for largest hp_period (period=30, bandwidth=0.2 => hp_period=600)
    const close = new Float64Array(testData.close.slice(0, 700));
    
    // Multiple periods and bandwidths
    const batchResult = wasm.bandpass_batch(close, {
        period_range: [10, 30, 10],     // period: 10, 20, 30
        bandwidth_range: [0.2, 0.4, 0.1]   // bandwidth: 0.2, 0.3, 0.4
    });
    
    // Should have 3 * 3 = 9 combinations
    assert.strictEqual(batchResult.combos, 9);
    assert.strictEqual(batchResult.outputs, 4);
    assert.strictEqual(batchResult.cols, 700);
    assert.strictEqual(batchResult.values.length, 9 * 4 * 700);
    
    // Verify first combination (period=10, bandwidth=0.2) bp values
    const firstBp = batchResult.values.slice(0, 700);
    const singleResult = wasm.bandpass_js(close, 10, 0.2);
    const singleBp = singleResult.values.slice(0, 700);
    assertArrayClose(firstBp, singleBp, 1e-10, "First combination mismatch");
});

test('BandPass batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.bandpass_batch_metadata(
        10, 30, 10,     // period: 10, 20, 30
        0.2, 0.4, 0.1   // bandwidth: 0.2, 0.3, 0.4
    );
    
    // Should have 9 combinations * 2 values each = 18 values
    assert.strictEqual(metadata.length, 18);
    
    // Check values (period, bandwidth pairs)
    const expected = [
        10, 0.2,   // combo 0
        10, 0.3,   // combo 1 (note: 0.3 due to floating point precision in step)
        10, 0.4,   // combo 2
        20, 0.2,   // combo 3
        20, 0.3,   // combo 4
        20, 0.4,   // combo 5
        30, 0.2,   // combo 6
        30, 0.3,   // combo 7
        30, 0.4    // combo 8
    ];
    
    for (let i = 0; i < expected.length; i++) {
        if (i % 2 === 0) {
            // Period values (integer comparison)
            assert.strictEqual(metadata[i], expected[i], `Metadata mismatch at index ${i}`);
        } else {
            // Bandwidth values (floating point comparison)
            assert(Math.abs(metadata[i] - expected[i]) < 1e-10, 
                   `Metadata mismatch at index ${i}: ${metadata[i]} vs ${expected[i]}`);
        }
    }
});

test('BandPass invalid bandwidth', () => {
    // Test Band-Pass with invalid bandwidth values
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    // Bandwidth must be in [0, 1]
    assert.throws(() => {
        wasm.bandpass_js(data, 5, -0.1);
    }, /Invalid period/);
    
    assert.throws(() => {
        wasm.bandpass_js(data, 5, 1.5);
    }, /Invalid period/);
});

// Batch API error handling
test('BandPass batch error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    // Invalid config structure
    assert.throws(() => {
        wasm.bandpass_batch(close, {
            period_range: [10, 10]  // Missing step
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.bandpass_batch(close, {
            period_range: [10, 20, 5]
            // Missing bandwidth_range
        });
    }, /Invalid config/);
    
    // Invalid data type
    assert.throws(() => {
        wasm.bandpass_batch(close, {
            period_range: "invalid",
            bandwidth_range: [0.2, 0.4, 0.1]
        });
    }, /Invalid config/);
});

test('BandPass edge cases', () => {
    // Minimum valid period - adjust bandwidth to avoid cos_val issue
    const data = new Float64Array(50).fill(1.0);
    const result = wasm.bandpass_js(data, 2, 0.5);
    assert.strictEqual(result.cols, data.length);
    assert.strictEqual(result.rows, 4);
    
    // With very narrow bandwidth - need larger data
    const largeData = new Float64Array(5000).fill(1.0);
    const result2 = wasm.bandpass_js(largeData, 10, 0.01);
    assert.strictEqual(result2.cols, largeData.length);
    
    // With very wide bandwidth
    const result3 = wasm.bandpass_js(data, 10, 0.99);
    assert.strictEqual(result3.cols, data.length);
});

test.after(() => {
    console.log('BandPass WASM tests completed');
});