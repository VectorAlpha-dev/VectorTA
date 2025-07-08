/**
 * WASM binding tests for SRWMA indicator.
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
        wasm = await import(wasmPath);
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('SRWMA partial params', () => {
    // Test with default parameters - mirrors check_srwma_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.srwma_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('SRWMA accuracy', async () => {
    // Test SRWMA matches expected values from Rust tests - mirrors check_srwma_accuracy
    const close = new Float64Array(testData.close);
    
    // Expected values from Rust test
    const expectedLastFive = [
        59344.28384704595,
        59282.09151629659,
        59192.76580529367,
        59178.04767548977,
        59110.03801260874,
    ];
    
    const result = wasm.srwma_js(close, 14);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-8,
        "SRWMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('srwma', result, 'close', { period: 14 });
});

test('SRWMA zero period', () => {
    // Test SRWMA fails with zero period - mirrors check_srwma_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.srwma_js(inputData, 0);
    }, /Invalid period/);
});

test('SRWMA period exceeds length', () => {
    // Test SRWMA fails when period exceeds data length - mirrors check_srwma_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.srwma_js(dataSmall, 10);
    }, /Invalid period/);
});

test('SRWMA very small dataset', () => {
    // Test SRWMA fails with insufficient data - mirrors check_srwma_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.srwma_js(singlePoint, 3);
    }, /Invalid period|Not enough valid data/);
});

test('SRWMA NaN handling', () => {
    // Test SRWMA handles NaN values correctly - mirrors check_srwma_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.srwma_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (50), no NaN values should exist
    if (result.length > 50) {
        for (let i = 50; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // Check warmup period - SRWMA needs period + 1 values before producing output
    // First period+1 values should be NaN
    for (let i = 0; i < 15; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} in warmup period`);
    }
});

test('SRWMA batch single period', () => {
    // Test SRWMA batch processing with single period
    const close = new Float64Array(testData.close);
    
    const result = wasm.srwma_batch_js(close, 14, 14, 0);
    
    // Should return flat array for single parameter combination
    assert.strictEqual(result.length, close.length);
    
    // Expected values from Rust test
    const expectedLastFive = [
        59344.28384704595,
        59282.09151629659,
        59192.76580529367,
        59178.04767548977,
        59110.03801260874,
    ];
    
    // Check last 5 values match
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-8,
        "SRWMA batch default row mismatch"
    );
});

test('SRWMA batch metadata', () => {
    // Test SRWMA batch metadata function
    const periods = wasm.srwma_batch_metadata_js(10, 20, 2);
    
    // Should return array of periods
    const expectedPeriods = [10, 12, 14, 16, 18, 20];
    assert.deepStrictEqual(Array.from(periods), expectedPeriods);
});

test('SRWMA all NaN input', () => {
    // Test SRWMA with all NaN values
    const allNan = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.srwma_js(allNan, 14);
    }, /All values are NaN/);
});

test('SRWMA period zero', () => {
    // Test SRWMA with period 0
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    assert.throws(() => {
        wasm.srwma_js(data, 0);
    }, /Invalid period/);
});

test('SRWMA batch multiple periods', () => {
    // Test SRWMA batch with multiple periods
    const close = new Float64Array(testData.close);
    
    const result = wasm.srwma_batch_js(close, 10, 20, 2);
    
    // Should return flat array with 6 periods worth of data
    assert.strictEqual(result.length, 6 * close.length);
    
    // Check metadata
    const periods = wasm.srwma_batch_metadata_js(10, 20, 2);
    assert.strictEqual(periods.length, 6);
    
    // Verify each period's data has appropriate warmup
    for (let i = 0; i < 6; i++) {
        const period = periods[i];
        const startIdx = i * close.length;
        const rowData = result.slice(startIdx, startIdx + close.length);
        
        // Check warmup period (period + 1)
        const warmupEnd = period + 1;
        for (let j = 0; j < warmupEnd; j++) {
            assert(isNaN(rowData[j]), `Expected NaN in warmup for period ${period} at index ${j}`);
        }
        
        // Check that we have valid values after warmup (if data is long enough)
        if (rowData.length > warmupEnd + 10) {
            let hasValidValues = false;
            for (let j = warmupEnd; j < warmupEnd + 10; j++) {
                if (!isNaN(rowData[j])) {
                    hasValidValues = true;
                    break;
                }
            }
            assert(hasValidValues, `Expected valid values after warmup for period ${period}`);
        }
    }
});

test('SRWMA edge case period 2', () => {
    // Test SRWMA with minimum valid period (2)
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    const result = wasm.srwma_js(data, 2);
    assert.strictEqual(result.length, data.length);
    
    // First 3 values (period + 1) should be NaN
    assertAllNaN(result.slice(0, 3));
    
    // Remaining values should be valid
    assertNoNaN(result.slice(3));
});

test('SRWMA batch with step 0', () => {
    // Test SRWMA batch with step 0 (single period)
    const close = new Float64Array(testData.close);
    
    const result = wasm.srwma_batch_js(close, 15, 15, 0);
    assert.strictEqual(result.length, close.length);
    
    // Check metadata returns single period
    const periods = wasm.srwma_batch_metadata_js(15, 15, 0);
    assert.deepStrictEqual(Array.from(periods), [15]);
});

test('SRWMA consistency check', () => {
    // Test that single SRWMA matches batch SRWMA for same period
    const close = new Float64Array(testData.close);
    const period = 14;
    
    const single = wasm.srwma_js(close, period);
    const batch = wasm.srwma_batch_js(close, period, period, 0);
    
    assert.strictEqual(single.length, batch.length);
    
    // Compare all values
    for (let i = 0; i < single.length; i++) {
        if (isNaN(single[i]) && isNaN(batch[i])) {
            continue;
        }
        assertClose(single[i], batch[i], 1e-12, 1e-12,
            `SRWMA single vs batch mismatch at index ${i}`);
    }
});

test('SRWMA with leading NaNs', () => {
    // Test SRWMA correctly handles data that starts with NaN values
    // Create data with 5 leading NaNs
    const data = new Float64Array(15);
    for (let i = 0; i < 5; i++) {
        data[i] = NaN;
    }
    for (let i = 5; i < 15; i++) {
        data[i] = i - 4; // 1, 2, 3, ..., 10
    }
    
    const result = wasm.srwma_js(data, 3);
    assert.strictEqual(result.length, data.length);
    
    // First non-NaN is at index 5, so warmup ends at 5 + 3 + 1 = 9
    for (let i = 0; i < 9; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} in warmup period including leading NaNs`);
    }
    // Should have valid values starting from index 9
    for (let i = 9; i < result.length; i++) {
        assert(!isNaN(result[i]), `Expected valid value at index ${i} after warmup`);
    }
});

test('SRWMA reinput', () => {
    // Test SRWMA with reinput - mirrors check_srwma_reinput
    const close = new Float64Array(testData.close);
    
    // First pass with period 14
    const firstResult = wasm.srwma_js(close, 14);
    
    // Second pass with the output of the first, using period 5
    const secondResult = wasm.srwma_js(firstResult, 5);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // After sufficient warmup, all values should be finite
    for (let i = 50; i < secondResult.length; i++) {
        assert(isFinite(secondResult[i]));
    }
});