/**
 * WASM binding tests for CCI indicator.
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

test('CCI partial params', () => {
    // Test with default parameters - mirrors check_cci_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.cci_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    // Test with different sources
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    // Create hl2 source
    const hl2 = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        hl2[i] = (high[i] + low[i]) / 2;
    }
    
    const resultHl2 = wasm.cci_js(hl2, 20);
    assert.strictEqual(resultHl2.length, hl2.length);
    
    // Create hlc3 source (default in Rust)
    const hlc3 = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        hlc3[i] = (high[i] + low[i] + close[i]) / 3;
    }
    
    const resultHlc3 = wasm.cci_js(hlc3, 9);
    assert.strictEqual(resultHlc3.length, hlc3.length);
});

test('CCI accuracy', async () => {
    // Test CCI matches expected values from Rust tests - mirrors check_cci_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.cci;
    
    // Create hlc3 source (default in Rust)
    const hlc3 = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        hlc3[i] = (high[i] + low[i] + close[i]) / 3;
    }
    
    const result = wasm.cci_js(
        hlc3,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, hlc3.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-6,
        "CCI last 5 values mismatch"
    );
    
    // Verify warmup period
    const period = expected.defaultParams.period;
    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} for initial period warm-up`);
    }
    
    // Compare full output with Rust
    await compareWithRust('cci', result, 'hlc3', expected.defaultParams);
});

test('CCI default candles', () => {
    // Test CCI with default parameters - mirrors check_cci_default_candles
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Create hlc3 source (default in Rust)
    const hlc3 = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        hlc3[i] = (high[i] + low[i] + close[i]) / 3;
    }
    
    const result = wasm.cci_js(hlc3, 14);
    assert.strictEqual(result.length, hlc3.length);
});

test('CCI zero period', () => {
    // Test CCI fails with zero period - mirrors check_cci_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cci_js(inputData, 0);
    }, /Invalid period/);
});

test('CCI period exceeds length', () => {
    // Test CCI fails when period exceeds data length - mirrors check_cci_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cci_js(dataSmall, 10);
    }, /Invalid period/);
});

test('CCI very small dataset', () => {
    // Test CCI fails with insufficient data - mirrors check_cci_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.cci_js(singlePoint, 9);
    }, /Invalid period|Not enough valid data/);
});

test('CCI empty input', () => {
    // Test CCI fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.cci_js(empty, 14);
    }, /Input data slice is empty/);
});

test('CCI reinput', () => {
    // Test CCI applied twice (re-input) - mirrors check_cci_reinput
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.cci_js(close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply CCI to CCI output
    const secondResult = wasm.cci_js(firstResult, 14);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // After warmup period (28), no NaN values should exist
    if (secondResult.length > 28) {
        for (let i = 28; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Expected no NaN after index 28, found NaN at index ${i}`);
        }
    }
});

test('CCI NaN handling', () => {
    // Test CCI handles NaN values correctly - mirrors check_cci_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.cci_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First period-1 values should be NaN
    assertAllNaN(result.slice(0, 13), "Expected NaN in warmup period");
});

test('CCI all NaN input', () => {
    // Test CCI with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.cci_js(allNaN, 14);
    }, /All values are NaN/);
});

test('CCI batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Single parameter set: period=14
    const batchResult = wasm.cci_batch_js(
        close,
        14, 14, 0      // period range
    );
    
    // Should match single calculation
    const singleResult = wasm.cci_js(close, 14);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('CCI batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 10, 12, 14, 16, 18, 20
    const batchResult = wasm.cci_batch_js(
        close,
        10, 20, 2      // period range
    );
    
    // Should have 6 rows * 100 cols = 600 values
    assert.strictEqual(batchResult.length, 6 * 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 12, 14, 16, 18, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.cci_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('CCI batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.cci_batch_metadata_js(
        10, 20, 2      // period: 10, 12, 14, 16, 18, 20
    );
    
    // Should have 6 values (one per period)
    assert.strictEqual(metadata.length, 6);
    
    // Check values
    const expected = [10, 12, 14, 16, 18, 20];
    for (let i = 0; i < expected.length; i++) {
        assert.strictEqual(metadata[i], expected[i]);
    }
});

test('CCI batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.cci_batch_js(
        close,
        10, 14, 2      // 3 periods
    );
    
    const metadata = wasm.cci_batch_metadata_js(
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
        
        // First period-1 values should be NaN
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        // After warmup should have values
        for (let i = period - 1; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('CCI batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep
    const singleBatch = wasm.cci_batch_js(
        close,
        5, 5, 1
    );
    
    assert.strictEqual(singleBatch.length, 10);
    
    // Step larger than range
    const largeBatch = wasm.cci_batch_js(
        close,
        5, 7, 10 // Step larger than range
    );
    
    // Should only have period=5
    assert.strictEqual(largeBatch.length, 10);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.cci_batch_js(
            new Float64Array([]),
            14, 14, 0
        );
    }, /Input data slice is empty/);
});

// New API tests
test('CCI batch - new ergonomic API with single parameter', () => {
    // Test the new API with a single parameter combination
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Create hlc3 source
    const hlc3 = new Float64Array(high.length);
    for (let i = 0; i < high.length; i++) {
        hlc3[i] = (high[i] + low[i] + close[i]) / 3;
    }
    
    const result = wasm.cci_batch(hlc3, {
        period_range: [14, 14, 0]
    });
    
    // Verify structure
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');
    
    // Verify dimensions
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, hlc3.length);
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.values.length, hlc3.length);
    
    // Verify parameters
    const combo = result.combos[0];
    assert.strictEqual(combo.period, 14);
    
    // Compare with old API
    const oldResult = wasm.cci_js(hlc3, 14);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
    
    // Check last 5 values match expected
    const expected = EXPECTED_OUTPUTS.cci;
    const last5 = result.values.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-6,
        "CCI new API last 5 values mismatch"
    );
});

test('CCI batch - new API with multiple parameters', () => {
    // Test with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.cci_batch(close, {
        period_range: [10, 14, 2]      // 10, 12, 14
    });
    
    // Should have 3 combinations
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.values.length, 150);
    
    // Verify each combo
    const expectedCombos = [
        { period: 10 },
        { period: 12 },
        { period: 14 }
    ];
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedCombos[i].period);
    }
    
    // Extract and verify a specific row
    const secondRow = result.values.slice(result.cols, 2 * result.cols);
    assert.strictEqual(secondRow.length, 50);
    
    // Compare with old API for first combination
    const oldResult = wasm.cci_js(close, 10);
    const firstRow = result.values.slice(0, result.cols);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('CCI batch - new API matches old API results', () => {
    // Comprehensive comparison test
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const params = {
        period_range: [12, 18, 3]
    };
    
    // Old API
    const oldValues = wasm.cci_batch_js(
        close,
        params.period_range[0], params.period_range[1], params.period_range[2]
    );
    
    // New API
    const newResult = wasm.cci_batch(close, params);
    
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

test('CCI batch - new API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    // Invalid config structure
    assert.throws(() => {
        wasm.cci_batch(close, {
            period_range: [14, 14] // Missing step
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.cci_batch(close, {
            // Missing period_range
        });
    }, /Invalid config/);
    
    // Invalid data type
    assert.throws(() => {
        wasm.cci_batch(close, {
            period_range: "invalid"
        });
    }, /Invalid config/);
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('CCI WASM tests completed');
});