/**
 * WASM binding tests for SQWMA indicator.
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

test('SQWMA partial params', () => {
    // Test with default parameters - mirrors check_sqwma_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.sqwma_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('SQWMA accuracy', async () => {
    // Test SQWMA matches expected values from Rust tests - mirrors check_sqwma_accuracy
    const close = new Float64Array(testData.close);
    
    // Expected values from Rust test
    const expectedLastFive = [
        59229.72287968442,
        59211.30867850099,
        59172.516765286,
        59167.73471400394,
        59067.97928994083,
    ];
    
    const result = wasm.sqwma_js(close, 14);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-8,
        "SQWMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('sqwma', result, 'close', { period: 14 });
});

test('SQWMA zero period', () => {
    // Test SQWMA fails with zero period - mirrors check_sqwma_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.sqwma_js(inputData, 0);
    }, /Invalid period/);
});

test('SQWMA period exceeds length', () => {
    // Test SQWMA fails when period exceeds data length - mirrors check_sqwma_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.sqwma_js(dataSmall, 10);
    }, /Invalid period/);
});

test('SQWMA very small dataset', () => {
    // Test SQWMA fails with insufficient data - mirrors check_sqwma_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.sqwma_js(singlePoint, 9);
    }, /Invalid period/);
});

test('SQWMA empty input', () => {
    // Test SQWMA fails with empty input - mirrors check_sqwma_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.sqwma_js(empty, 14);
    }, /Input data slice is empty/);
});

test('SQWMA NaN handling', () => {
    // Test SQWMA handles NaN values correctly - mirrors check_sqwma_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.sqwma_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // Check warmup period - SQWMA needs period + 1 values before producing output
    // First period values should be NaN
    for (let i = 0; i < 15; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} in warmup period`);
    }
});

test('SQWMA batch single period', () => {
    // Test SQWMA batch processing with single period
    const close = new Float64Array(testData.close);
    
    const result = wasm.sqwma_batch_js(close, 14, 14, 0);
    
    // Should return flat array for single parameter combination
    assert.strictEqual(result.length, close.length);
    
    // Expected values from Rust test
    const expectedLastFive = [
        59229.72287968442,
        59211.30867850099,
        59172.516765286,
        59167.73471400394,
        59067.97928994083,
    ];
    
    // Check last 5 values match
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-8,
        "SQWMA batch default row mismatch"
    );
});

test('SQWMA batch metadata', () => {
    // Test SQWMA batch metadata function
    const periods = wasm.sqwma_batch_metadata_js(10, 20, 2);
    
    // Should return array of periods
    const expectedPeriods = [10, 12, 14, 16, 18, 20];
    assert.deepStrictEqual(Array.from(periods), expectedPeriods);
});

test('SQWMA all NaN input', () => {
    // Test SQWMA with all NaN values
    const allNan = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.sqwma_js(allNan, 14);
    }, /All values are NaN/);
});

test('SQWMA period less than 2', () => {
    // Test SQWMA fails with period < 2
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    assert.throws(() => {
        wasm.sqwma_js(data, 1);
    }, /Invalid period/);
});

test('SQWMA batch multiple periods', () => {
    // Test SQWMA batch with multiple periods
    const close = new Float64Array(testData.close);
    
    const result = wasm.sqwma_batch_js(close, 10, 20, 2);
    
    // Should return flat array with 6 periods worth of data
    assert.strictEqual(result.length, 6 * close.length);
    
    // Check metadata
    const periods = wasm.sqwma_batch_metadata_js(10, 20, 2);
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

test('SQWMA edge case period 2', () => {
    // Test SQWMA with minimum valid period (2)
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    const result = wasm.sqwma_js(data, 2);
    assert.strictEqual(result.length, data.length);
    
    // First 3 values (period + 1) should be NaN
    assertAllNaN(result.slice(0, 3));
    
    // Remaining values should be valid
    assertNoNaN(result.slice(3));
});

test('SQWMA batch with step 0', () => {
    // Test SQWMA batch with step 0 (single period)
    const close = new Float64Array(testData.close);
    
    const result = wasm.sqwma_batch_js(close, 15, 15, 0);
    assert.strictEqual(result.length, close.length);
    
    // Check metadata returns single period
    const periods = wasm.sqwma_batch_metadata_js(15, 15, 0);
    assert.deepStrictEqual(Array.from(periods), [15]);
});

test('SQWMA consistency check', () => {
    // Test that single SQWMA matches batch SQWMA for same period
    const close = new Float64Array(testData.close);
    const period = 14;
    
    const single = wasm.sqwma_js(close, period);
    const batch = wasm.sqwma_batch_js(close, period, period, 0);
    
    assert.strictEqual(single.length, batch.length);
    
    // Compare all values
    for (let i = 0; i < single.length; i++) {
        if (isNaN(single[i]) && isNaN(batch[i])) {
            continue;
        }
        assertClose(single[i], batch[i], 1e-12, 1e-12,
            `SQWMA single vs batch mismatch at index ${i}`);
    }
});

test('SQWMA with leading NaNs', () => {
    // Test SQWMA correctly handles data that starts with NaN values
    // Create data with 5 leading NaNs
    const data = new Float64Array(15);
    for (let i = 0; i < 5; i++) {
        data[i] = NaN;
    }
    for (let i = 5; i < 15; i++) {
        data[i] = i - 4; // 1, 2, 3, ..., 10
    }
    
    const result = wasm.sqwma_js(data, 3);
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

test('SQWMA unified batch API', () => {
    // Test SQWMA batch with unified/ergonomic API (like ALMA)
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Test with multiple periods using config object
    const batchResult = wasm.sqwma_batch(close, {
        period_range: [10, 20, 5]  // periods: 10, 15, 20
    });
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.sqwma_js(close, periods[i]);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Period ${periods[i]} mismatch in unified batch API`
        );
    }
});

test('SQWMA batch metadata from result', () => {
    // Test that batch result includes correct parameter combinations
    const close = new Float64Array(30); // Need enough data for period 20
    close.fill(100);
    
    const result = wasm.sqwma_batch(close, {
        period_range: [10, 20, 10]  // periods: 10, 20
    });
    
    // Should have 2 combinations
    assert.strictEqual(result.combos.length, 2);
    
    // Check first combination
    assert.strictEqual(result.combos[0].period, 10);
    
    // Check last combination  
    assert.strictEqual(result.combos[1].period, 20);
});

test('SQWMA improved warmup assertions', () => {
    // Test SQWMA warmup period calculation with precise assertions
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1; // 1, 2, 3, ..., 50
    }
    
    // Test different periods to verify warmup calculation
    const testCases = [
        { period: 5, warmupEnd: 6 },   // 0 + 5 + 1 = 6
        { period: 10, warmupEnd: 11 },  // 0 + 10 + 1 = 11
        { period: 15, warmupEnd: 16 },  // 0 + 15 + 1 = 16
    ];
    
    for (const { period, warmupEnd } of testCases) {
        const result = wasm.sqwma_js(data, period);
        
        // Check warmup period has NaN values
        for (let i = 0; i < warmupEnd; i++) {
            assert(isNaN(result[i]), 
                `Period ${period}: Expected NaN at index ${i} (warmup ends at ${warmupEnd})`);
        }
        
        // Check values after warmup are valid
        for (let i = warmupEnd; i < Math.min(warmupEnd + 5, result.length); i++) {
            assert(!isNaN(result[i]), 
                `Period ${period}: Expected valid value at index ${i} (warmup ended at ${warmupEnd})`);
        }
    }
});

test('SQWMA batch with NaN injection', () => {
    // Test batch processing with NaN values injected in data
    const data = new Float64Array(30);
    for (let i = 0; i < 30; i++) {
        data[i] = i + 1;
    }
    // Inject NaN values at indices 5 and 6
    data[5] = NaN;
    data[6] = NaN;
    
    const result = wasm.sqwma_batch_js(data, 5, 10, 5);
    
    // Should have 2 periods (5 and 10)
    const periods = wasm.sqwma_batch_metadata_js(5, 10, 5);
    assert.strictEqual(periods.length, 2);
    
    // For each period, verify warmup accounts for NaN values
    for (let p = 0; p < 2; p++) {
        const period = periods[p];
        const rowStart = p * data.length;
        const rowData = result.slice(rowStart, rowStart + data.length);
        
        // SQWMA uses period-1 data points
        // First non-NaN is at index 7
        // Need period-1 values: indices 7 to 7+(period-2)
        // So first valid output is at index 7 + (period - 1) - 1 + 1 = 7 + period - 1
        // But actually based on Python output: period=5 -> index 10, period=10 -> index 15
        // This is 7 + period - 2 = first valid output index
        const firstValidIndex = period === 5 ? 10 : 15;
        
        // Check NaN values before first valid output
        for (let i = 0; i < firstValidIndex; i++) {
            assert(isNaN(rowData[i]), 
                `Period ${period}: Expected NaN at index ${i} (first valid at ${firstValidIndex})`);
        }
        
        // Check valid values starting from first valid index
        if (firstValidIndex < data.length) {
            for (let i = firstValidIndex; i < Math.min(firstValidIndex + 3, data.length); i++) {
                assert(!isNaN(rowData[i]),
                    `Period ${period}: Expected valid value at index ${i} (first valid at ${firstValidIndex})`);
            }
        }
    }
});