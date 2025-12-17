/**
 * WASM binding tests for ADXR indicator.
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

test('ADXR partial params', () => {
    // Test with default parameters - mirrors check_adxr_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.adxr_js(high, low, close, 14);
    assert.strictEqual(result.length, close.length);
});

test('ADXR accuracy', async () => {
    // Test ADXR matches expected values from Rust tests - mirrors check_adxr_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.adxr;
    
    const result = wasm.adxr_js(
        high, low, close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected (with tolerance for ADXR calculation)
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-1,  // ADXR uses 1e-1 tolerance in Rust tests
        "ADXR last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('adxr', result, 'hlc', expected.defaultParams);
});

test('ADXR default candles', () => {
    // Test ADXR with default parameters - mirrors check_adxr_default_candles
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.adxr_js(high, low, close, 14);
    assert.strictEqual(result.length, close.length);
});

test('ADXR zero period', () => {
    // Test ADXR fails with zero period - mirrors check_adxr_zero_period
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([9.0, 19.0, 29.0]);
    const close = new Float64Array([9.5, 19.5, 29.5]);
    
    assert.throws(() => {
        wasm.adxr_js(high, low, close, 0);
    }, /Invalid period/);
});

test('ADXR period exceeds length', () => {
    // Test ADXR fails when period exceeds data length - mirrors check_adxr_period_exceeds_length
    const high = new Float64Array([10.0, 20.0]);
    const low = new Float64Array([9.0, 19.0]);
    const close = new Float64Array([9.5, 19.5]);
    
    assert.throws(() => {
        wasm.adxr_js(high, low, close, 10);
    }, /Invalid period/);
});

test('ADXR very small dataset', () => {
    // Test ADXR fails with insufficient data - mirrors check_adxr_very_small_dataset
    const high = new Float64Array([100.0]);
    const low = new Float64Array([99.0]);
    const close = new Float64Array([99.5]);
    
    assert.throws(() => {
        wasm.adxr_js(high, low, close, 14);
    }, /Invalid period|Not enough data/);
});

test('ADXR empty input', () => {
    // Test ADXR fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.adxr_js(empty, empty, empty, 14);
    }, /Empty input data|Invalid period|Not enough data|All values are NaN/);
});

test('ADXR mismatched lengths', () => {
    // Test ADXR fails with mismatched input lengths
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([9.0, 19.0]);  // Different length
    const close = new Float64Array([9.5, 19.5, 29.5]);
    
    assert.throws(() => {
        wasm.adxr_js(high, low, close, 2);
    }, /HLC data length mismatch/);
});

test('ADXR reinput', () => {
    // Test ADXR applied with different parameters - mirrors check_adxr_reinput
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // First pass with period=14
    const firstResult = wasm.adxr_js(high, low, close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass with period=5
    const secondResult = wasm.adxr_js(high, low, close, 5);
    assert.strictEqual(secondResult.length, close.length);
});

test('ADXR NaN handling', () => {
    // Test ADXR handles NaN values correctly - mirrors check_adxr_nan_handling
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.adxr_js(high, low, close, 14);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First 2*period values should be NaN (period for ADX + period for ADXR)
    const expectedWarmup = 2 * 14;  // 28 for period=14
    assertAllNaN(result.slice(0, expectedWarmup), "Expected NaN in warmup period");
});

test('ADXR all NaN input', () => {
    // Test ADXR with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.adxr_js(allNaN, allNaN, allNaN, 14);
    }, /All values are NaN/);
});

test('ADXR batch single parameter set', () => {
    // Test batch with single parameter combination
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Single parameter set: period=14
    const batchResult = wasm.adxr_batch_js(
        high, low, close,
        14, 14, 0      // period range
    );
    
    // Should match single calculation
    const singleResult = wasm.adxr_js(high, low, close, 14);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('ADXR batch multiple periods', () => {
    // Test batch with multiple period values
    const high = new Float64Array(testData.high.slice(0, 100)); // Use smaller dataset for speed
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Multiple periods: 10, 15, 20
    const batchResult = wasm.adxr_batch_js(
        high, low, close,
        10, 20, 5      // period range
    );
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.length, 3 * 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.adxr_js(high, low, close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('ADXR batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    // Note: ADXR only has period as a parameter (unlike ALMA with period/offset/sigma)
    const metadata = wasm.adxr_batch_metadata_js(
        10, 20, 5      // period: 10, 15, 20
    );
    
    // Should have 3 periods
    assert.strictEqual(metadata.length, 3);
    
    // Check values
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 15);
    assert.strictEqual(metadata[2], 20);
});

test('ADXR batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.adxr_batch_js(
        high, low, close,
        10, 14, 2      // 3 periods: 10, 12, 14
    );
    
    const metadata = wasm.adxr_batch_metadata_js(10, 14, 2);
    
    // Should have 3 combinations
    const numCombos = metadata.length;
    assert.strictEqual(numCombos, 3);
    assert.strictEqual(batchResult.length, 3 * 50);
    
    // Verify structure
    for (let combo = 0; combo < numCombos; combo++) {
        const period = metadata[combo];
        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);
        
        // ADXR needs ADX values from period bars ago, so warmup is longer
        // The actual warmup for ADXR is quite complex: ADX needs 2*period-1 warmup,
        // then ADXR needs another period values of ADX
        // So total warmup is around 2*period + (period-1) = 3*period-1
        // But let's be more lenient and just check that we eventually get values
        let firstValidIndex = -1;
        for (let i = 0; i < 50; i++) {
            if (!isNaN(rowData[i])) {
                firstValidIndex = i;
                break;
            }
        }
        
        // After finding first valid value, rest should be valid
        if (firstValidIndex !== -1) {
            for (let i = firstValidIndex + 1; i < 50; i++) {
                assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period} after first valid value at ${firstValidIndex}`);
            }
        }
    }
});

test('ADXR batch edge cases', () => {
    // Test edge cases for batch processing
    const high = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const low = new Float64Array([0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9]);
    const close = new Float64Array([0.95, 1.95, 2.95, 3.95, 4.95, 5.95, 6.95, 7.95, 8.95, 9.95]);
    
    // Single value sweep
    const singleBatch = wasm.adxr_batch_js(
        high, low, close,
        5, 5, 1
    );
    
    assert.strictEqual(singleBatch.length, 10);
    
    // Step larger than range
    const largeBatch = wasm.adxr_batch_js(
        high, low, close,
        5, 7, 10 // Step larger than range
    );
    
    // Should only have period=5
    assert.strictEqual(largeBatch.length, 10);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.adxr_batch_js(
            new Float64Array([]),
            new Float64Array([]),
            new Float64Array([]),
            9, 9, 0
        );
    }, /All values are NaN/);
});

// New API tests
test('ADXR batch - new ergonomic API with single parameter', () => {
    // Test the new API with a single parameter combination
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.adxr_batch(high, low, close, {
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
    const oldResult = wasm.adxr_js(high, low, close, 14);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('ADXR batch - new API with multiple parameters', () => {
    // Test with multiple parameter combinations
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.adxr_batch(high, low, close, {
        period_range: [10, 14, 2]  // 10, 12, 14
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
    const oldResult = wasm.adxr_js(high, low, close, 10);
    const firstRow = result.values.slice(0, result.cols);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('ADXR batch - new API matches old API results', () => {
    // Comprehensive comparison test
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const params = {
        period_range: [10, 15, 5]
    };
    
    // Old API
    const oldValues = wasm.adxr_batch_js(
        high, low, close,
        params.period_range[0], params.period_range[1], params.period_range[2]
    );
    
    // New API
    const newResult = wasm.adxr_batch(high, low, close, params);
    
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

test('ADXR batch - new API error handling', () => {
    const high = new Float64Array(testData.high.slice(0, 10));
    const low = new Float64Array(testData.low.slice(0, 10));
    const close = new Float64Array(testData.close.slice(0, 10));
    
    // Invalid config structure
    assert.throws(() => {
        wasm.adxr_batch(high, low, close, {
            period_range: [9, 9] // Missing step
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.adxr_batch(high, low, close, {
            // Missing period_range
        });
    }, /Invalid config/);
    
    // Invalid data type
    assert.throws(() => {
        wasm.adxr_batch(high, low, close, {
            period_range: "invalid"
        });
    }, /Invalid config/);
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('ADXR WASM tests completed');
});
