/**
 * WASM binding tests for CG (Center of Gravity) indicator.
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

test('CG partial params', () => {
    // Test with custom period - mirrors check_cg_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.cg_js(close, 12);
    assert.strictEqual(result.length, close.length);
});

test('CG accuracy', async () => {
    // Test CG matches expected values from Rust tests - mirrors check_cg_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.cg;
    
    const result = wasm.cg_js(
        close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-4,  // CG uses 1e-4 tolerance in Rust tests
        "CG last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('cg', result, 'close', expected.defaultParams);
});

test('CG default candles', () => {
    // Test CG with default parameters - mirrors check_cg_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.cg_js(close, 10);
    assert.strictEqual(result.length, close.length);
});

test('CG zero period', () => {
    // Test CG fails with zero period - mirrors check_cg_zero_period
    const data = new Float64Array([1.0, 2.0, 3.0]);
    
    assert.throws(() => {
        wasm.cg_js(data, 0);
    }, /Invalid period/);
});

test('CG period exceeds length', () => {
    // Test CG fails when period exceeds data length - mirrors check_cg_period_exceeds_length
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cg_js(data, 10);
    }, /Invalid period/);
});

test('CG very small dataset', () => {
    // Test CG fails with insufficient data - mirrors check_cg_very_small_dataset
    const data = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.cg_js(data, 10);
    }, /Invalid period|Not enough valid data/);
});

test('CG empty input', () => {
    // Test CG fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.cg_js(empty, 10);
    }, /Empty data/);
});

test('CG NaN handling', () => {
    // Test CG handles NaN values correctly - mirrors check_cg_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.cg_js(close, 10);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period, check for valid values
    const checkIdx = 240;
    if (result.length > checkIdx) {
        // Find first non-NaN value after checkIdx
        let foundValid = false;
        for (let i = checkIdx; i < result.length; i++) {
            if (!isNaN(result[i])) {
                foundValid = true;
                break;
            }
        }
        assert(foundValid, `All CG values from index ${checkIdx} onward are NaN.`);
    }
    
    // First period values should be NaN (CG starts at first + period)
    assertAllNaN(result.slice(0, 10), "Expected NaN in warmup period");
});

test('CG all NaN input', () => {
    // Test CG with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.cg_js(allNaN, 10);
    }, /All values are NaN/);
});

test('CG batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Single parameter set: period=10
    const batchResult = wasm.cg_batch_js(
        close,
        10, 10, 0      // period range
    );
    
    // Should match single calculation
    const singleResult = wasm.cg_js(close, 10);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('CG batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 10, 12, 14
    const batchResult = wasm.cg_batch_js(
        close,
        10, 14, 2      // period range
    );
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.length, 3 * 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 12, 14];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.cg_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('CG batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.cg_batch_metadata_js(
        10, 14, 2      // period: 10, 12, 14
    );
    
    // Should have 3 periods
    assert.strictEqual(metadata.length, 3);
    
    // Check values
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 12);
    assert.strictEqual(metadata[2], 14);
});

test('CG batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.cg_batch_js(
        close,
        10, 14, 2      // 3 periods
    );
    
    const metadata = wasm.cg_batch_metadata_js(
        10, 14, 2
    );
    
    // Should have 3 combinations
    const numCombos = metadata.length;
    assert.strictEqual(numCombos, 3);
    assert.strictEqual(batchResult.length, 3 * 50);
    
    // Verify structure
    for (let combo = 0; combo < numCombos; combo++) {
        const period = metadata[combo];
        
        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);
        
        // First period values should be NaN
        for (let i = 0; i < period; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        // After warmup should have values
        for (let i = period; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('CG batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    
    // Single value sweep
    const singleBatch = wasm.cg_batch_js(
        close,
        5, 5, 1
    );
    
    assert.strictEqual(singleBatch.length, 12);
    
    // Step larger than range
    const largeBatch = wasm.cg_batch_js(
        close,
        5, 7, 10 // Step larger than range
    );
    
    // Should only have period=5
    assert.strictEqual(largeBatch.length, 12);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.cg_batch_js(
            new Float64Array([]),
            10, 10, 0
        );
    }, /All values are NaN/);
});

// New API tests
test('CG batch - new ergonomic API with single parameter', () => {
    // Test the new API with a single parameter combination
    const close = new Float64Array(testData.close);
    
    const result = wasm.cg_batch(close, {
        period_range: [10, 10, 0]
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
    assert.strictEqual(combo.period, 10);
    
    // Compare with old API
    const oldResult = wasm.cg_js(close, 10);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('CG batch - new API with multiple parameters', () => {
    // Test with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.cg_batch(close, {
        period_range: [10, 14, 2]      // 10, 12, 14
    });
    
    // Should have 3 combinations
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.values.length, 150);
    
    // Verify each combo
    const expectedPeriods = [10, 12, 14];
    
    for (let i = 0; i < expectedPeriods.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedPeriods[i]);
    }
    
    // Extract and verify a specific row
    const secondRow = result.values.slice(result.cols, 2 * result.cols);
    assert.strictEqual(secondRow.length, 50);
    
    // Compare with old API for first combination
    const oldResult = wasm.cg_js(close, 10);
    const firstRow = result.values.slice(0, result.cols);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('CG batch - new API matches old API results', () => {
    // Comprehensive comparison test
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const params = {
        period_range: [8, 12, 2]
    };
    
    // Old API
    const oldValues = wasm.cg_batch_js(
        close,
        params.period_range[0], params.period_range[1], params.period_range[2]
    );
    
    // New API
    const newResult = wasm.cg_batch(close, params);
    
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

test('CG batch - new API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    // Invalid config structure
    assert.throws(() => {
        wasm.cg_batch(close, {
            period_range: [10, 10], // Missing step
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.cg_batch(close, {
            // Missing period_range
        });
    }, /Invalid config/);
    
    // Invalid data type
    assert.throws(() => {
        wasm.cg_batch(close, {
            period_range: "invalid"
        });
    }, /Invalid config/);
});

test('CG warmup period', () => {
    // Test CG respects warmup period requirement
    const close = new Float64Array(testData.close);
    const period = 10;
    
    const result = wasm.cg_js(close, period);
    
    // CG requires period + 1 valid points, outputs start at first + period
    // So with period=10, first 10 values should be NaN
    assertAllNaN(result.slice(0, period), `Expected NaN in first ${period} values`);
    
    // Value at index period should be valid (if input has enough data)
    if (close.length > period) {
        assert(!isNaN(result[period]), `Expected valid value at index ${period}`);
    }
});

test('CG edge case small period', () => {
    // Test CG with very small period
    const close = new Float64Array(testData.close.slice(0, 20));
    
    // Period of 2 (minimum sensible value)
    const result = wasm.cg_js(close, 2);
    assert.strictEqual(result.length, close.length);
    
    // First 2 values should be NaN
    assertAllNaN(result.slice(0, 2));
    // Third value onwards should be valid
    assert(!isNaN(result[2]));
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('CG WASM tests completed');
});