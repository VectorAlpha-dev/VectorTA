/**
 * WASM binding tests for ADX indicator.
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

test('ADX partial params', () => {
    // Test with default parameters - mirrors check_adx_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.adx_js(high, low, close, 14);
    assert.strictEqual(result.length, close.length);
});

test('ADX accuracy', async () => {
    // Test ADX matches expected values from Rust tests - mirrors check_adx_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.adx;
    
    const result = wasm.adx_js(
        high,
        low,
        close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-1,  // ADX has lower precision requirement
        "ADX last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('adx', result, 'ohlc', expected.defaultParams);
});

test('ADX default candles', () => {
    // Test ADX with default parameters - mirrors check_adx_default_candles
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.adx_js(high, low, close, 14);
    assert.strictEqual(result.length, close.length);
});

test('ADX zero period', () => {
    // Test ADX fails with zero period - mirrors check_adx_zero_period
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    const close = new Float64Array([9.0, 19.0, 29.0]);
    
    assert.throws(() => {
        wasm.adx_js(high, low, close, 0);
    }, /Invalid period/);
});

test('ADX period exceeds length', () => {
    // Test ADX fails when period exceeds data length - mirrors check_adx_period_exceeds_length
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    const close = new Float64Array([9.0, 19.0, 29.0]);
    
    assert.throws(() => {
        wasm.adx_js(high, low, close, 10);
    }, /Invalid period/);
});

test('ADX very small dataset', () => {
    // Test ADX fails with insufficient data - mirrors check_adx_very_small_dataset
    const high = new Float64Array([42.0]);
    const low = new Float64Array([41.0]);
    const close = new Float64Array([40.5]);
    
    assert.throws(() => {
        wasm.adx_js(high, low, close, 14);
    }, /Invalid period|Not enough valid data/);
});

test('ADX input length mismatch', () => {
    // Test ADX fails when input arrays have different lengths
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]);  // Different length
    const close = new Float64Array([9.0, 19.0, 29.0]);
    
    assert.throws(() => {
        wasm.adx_js(high, low, close, 14);
    }, /Input arrays must have the same length/);
});

test('ADX all NaN input', () => {
    // Test ADX with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.adx_js(allNaN, allNaN, allNaN, 14);
    }, /All values are NaN/);
});

test('ADX reinput', () => {
    // Test ADX applied twice (re-input) - mirrors check_adx_reinput
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.adx_js(high, low, close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply ADX using the first result as close price
    const secondResult = wasm.adx_js(high, low, firstResult, 5);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Check that we have values after warmup
    let nonNanCount = 0;
    for (const val of secondResult) {
        if (!isNaN(val)) nonNanCount++;
    }
    assert(nonNanCount > 100, "Expected more non-NaN values after second pass");
});

test('ADX NaN handling', () => {
    // Test ADX handles NaN values correctly - mirrors check_adx_nan_handling
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.adx_js(high, low, close, 14);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (100), no NaN values should exist
    if (result.length > 100) {
        for (let i = 100; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First several values should be NaN (ADX needs extra warmup)
    assertAllNaN(result.slice(0, 27), "Expected NaN in warmup period");
});

test('ADX batch single parameter set', () => {
    // Test batch with single parameter combination
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Single parameter set: period=14
    const batchResult = wasm.adx_batch_js(
        high,
        low,
        close,
        14, 14, 0  // period range
    );
    
    // Should match single calculation
    const singleResult = wasm.adx_js(high, low, close, 14);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('ADX batch multiple periods', () => {
    // Test batch with multiple period values
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Multiple periods: 10, 14, 18
    const batchResult = wasm.adx_batch_js(
        high,
        low,
        close,
        10, 18, 4  // period range
    );
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.length, 3 * 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 14, 18];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.adx_js(high, low, close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('ADX batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.adx_batch_metadata_js(
        10, 18, 4  // period: 10, 14, 18
    );
    
    // Should have 3 periods
    assert.strictEqual(metadata.length, 3);
    
    // Check values
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 14);
    assert.strictEqual(metadata[2], 18);
});

test('ADX batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.adx_batch_js(
        high,
        low,
        close,
        10, 20, 5  // 3 periods: 10, 15, 20
    );
    
    const metadata = wasm.adx_batch_metadata_js(10, 20, 5);
    
    // Should have 3 combinations
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(batchResult.length, 3 * 50);
    
    // Verify structure
    for (let combo = 0; combo < metadata.length; combo++) {
        const period = metadata[combo];
        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);
        
        // ADX needs 2*period warmup bars
        const expectedFirstValid = 2 * period - 1;
        
        // Check warmup period
        for (let i = 0; i < Math.min(expectedFirstValid, 50); i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        // After warmup should eventually have values
        if (50 > expectedFirstValid + 5) {
            let hasValues = false;
            for (let i = expectedFirstValid; i < 50; i++) {
                if (!isNaN(rowData[i])) {
                    hasValues = true;
                    break;
                }
            }
            assert(hasValues, `Expected some non-NaN values after warmup for period ${period}`);
        }
    }
});

test('ADX batch edge cases', () => {
    // Test edge cases for batch processing
    const high = new Float64Array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
    const low = new Float64Array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
    const close = new Float64Array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5]);
    
    // Single value sweep
    const singleBatch = wasm.adx_batch_js(
        high,
        low,
        close,
        5, 5, 1
    );
    
    assert.strictEqual(singleBatch.length, 10);
    
    // Step larger than range
    const largeBatch = wasm.adx_batch_js(
        high,
        low,
        close,
        5, 7, 10  // Step larger than range
    );
    
    // Should only have period=5
    assert.strictEqual(largeBatch.length, 10);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.adx_batch_js(
            new Float64Array([]),
            new Float64Array([]),
            new Float64Array([]),
            14, 14, 0
        );
    }, /All values are NaN|unreachable|RuntimeError/);
});

// New API tests
test('ADX batch - new ergonomic API with single parameter', () => {
    // Test the new API with a single parameter combination
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.adx_batch(high, low, close, {
        period_range: [14, 14, 0]
    });
    
    // Verify structure
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');
    
    // Verify dimensions
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.values.length, close.length);
    
    // Verify parameters
    const combo = result.combos[0];
    assert.strictEqual(combo.period, 14);
    
    // Compare with old API
    const oldResult = wasm.adx_js(high, low, close, 14);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('ADX batch - new API with multiple parameters', () => {
    // Test with multiple parameter combinations
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.adx_batch(high, low, close, {
        period_range: [10, 18, 4]  // 10, 14, 18
    });
    
    // Should have 3 combinations
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.values.length, 150);
    
    // Verify each combo
    const expectedPeriods = [10, 14, 18];
    for (let i = 0; i < expectedPeriods.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedPeriods[i]);
    }
    
    // Extract and verify a specific row
    const secondRow = result.values.slice(result.cols, 2 * result.cols);
    assert.strictEqual(secondRow.length, 50);
    
    // Compare with old API for first combination
    const oldResult = wasm.adx_js(high, low, close, 10);
    const firstRow = result.values.slice(0, result.cols);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('ADX batch - new API matches old API results', () => {
    // Comprehensive comparison test
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const params = {
        period_range: [10, 20, 5]
    };
    
    // Old API
    const oldValues = wasm.adx_batch_js(
        high,
        low,
        close,
        params.period_range[0], params.period_range[1], params.period_range[2]
    );
    
    // New API
    const newResult = wasm.adx_batch(high, low, close, params);
    
    // Should produce identical values
    assert.strictEqual(oldValues.length, newResult.values.length);
    
    for (let i = 0; i < oldValues.length; i++) {
        if (isNaN(oldValues[i]) && isNaN(newResult.values[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldValues[i] - newResult.values[i]) < 1e-10,
               `Value mismatch at index ${i}: old=${oldValues[i]}, new=${newResult.values[i]}`);
    }
});

test('ADX batch - new API error handling', () => {
    const high = new Float64Array(testData.high.slice(0, 10));
    const low = new Float64Array(testData.low.slice(0, 10));
    const close = new Float64Array(testData.close.slice(0, 10));
    
    // Invalid config structure
    assert.throws(() => {
        wasm.adx_batch(high, low, close, {
            period_range: [14, 14]  // Missing step
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.adx_batch(high, low, close, {
            // Missing period_range
        });
    }, /Invalid config/);
    
    // Invalid data type
    assert.throws(() => {
        wasm.adx_batch(high, low, close, {
            period_range: "invalid"
        });
    }, /Invalid config/);
});

test('ADX warmup behavior', () => {
    // Test ADX warmup period behavior in detail
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    const period = 14;
    
    const result = wasm.adx_js(high, low, close, period);
    
    // ADX needs 2*period warmup bars
    const expectedFirstValid = 2 * period - 1;  // 27 for period=14
    
    // All values before this should be NaN
    for (let i = 0; i < expectedFirstValid; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    // Should eventually get non-NaN values after warmup
    if (result.length > expectedFirstValid + 5) {
        let hasValues = false;
        for (let i = expectedFirstValid; i < result.length; i++) {
            if (!isNaN(result[i])) {
                hasValues = true;
                break;
            }
        }
        assert(hasValues, "Expected some non-NaN values after warmup");
    }
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('ADX WASM tests completed');
});