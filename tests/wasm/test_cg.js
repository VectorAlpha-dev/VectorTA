/**
 * WASM binding tests for CG (Center of Gravity) indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 * 
 * Warmup Period: CG requires period + 1 valid data points.
 * Output starts at index: first_valid + period
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


test('CG batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.cg_batch(close, {
        period_range: [10, 14, 2]  // 3 periods
    });
    
    // Should have 3 combinations
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.values.length, 3 * 50);
    
    // Verify structure
    for (let combo = 0; combo < result.rows; combo++) {
        const period = result.combos[combo].period;
        
        const rowStart = combo * 50;
        const rowData = result.values.slice(rowStart, rowStart + 50);
        
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
    const singleResult = wasm.cg_batch(close, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleResult.rows, 1);
    assert.strictEqual(singleResult.values.length, 12);
    
    // Step larger than range
    const largeResult = wasm.cg_batch(close, {
        period_range: [5, 7, 10] // Step larger than range
    });
    
    // Should only have period=5
    assert.strictEqual(largeResult.rows, 1);
    assert.strictEqual(largeResult.values.length, 12);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.cg_batch(new Float64Array([]), {
            period_range: [10, 10, 0]
        });
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

test('CG NaN injection at specific positions', () => {
    // Test CG handles NaN values injected at specific positions
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Inject NaN values at specific positions
    for (let i = 20; i < 25; i++) {
        close[i] = NaN;
    }
    
    const result = wasm.cg_js(close, 10);
    assert.strictEqual(result.length, close.length);
    
    // CG continues computing despite NaN in the window
    // It returns 0.0 when denom is too small (all NaN window)
    // First 10 values should be NaN (warmup period)
    assertAllNaN(result.slice(0, 10));
    
    // Check that computation continues after NaN injection
    // Some values during NaN window will be 0.0 due to division protection
    let hasNonNaNAfterInjection = false;
    for (let i = 30; i < result.length; i++) {
        if (!isNaN(result[i]) && result[i] !== 0.0) {
            hasNonNaNAfterInjection = true;
            break;
        }
    }
    assert(hasNonNaNAfterInjection, "Expected non-zero values after NaN injection");
});

test('CG batch accuracy verification', () => {
    // Test CG batch matches expected accuracy for default params
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.cg;
    
    // Run batch with default params only
    const result = wasm.cg_batch(close, {
        period_range: [10, 10, 0]
    });
    
    // Extract last 5 values from first row
    const last5 = result.values.slice(result.cols - 5, result.cols);
    
    // Verify the output matches expected
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-4,
        "CG batch accuracy mismatch"
    );
});

test('CG batch comprehensive parameter sweep', () => {
    // Test comprehensive batch parameter sweep like ALMA
    const close = new Float64Array(testData.close);
    
    // Test sweep of multiple periods
    const result = wasm.cg_batch(close, {
        period_range: [5, 20, 5]  // 5, 10, 15, 20
    });
    
    // Should have 4 combinations
    assert.strictEqual(result.rows, 4);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, 4);
    
    // Verify periods
    const expectedPeriods = [5, 10, 15, 20];
    for (let i = 0; i < expectedPeriods.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedPeriods[i]);
    }
    
    // Verify each row has correct warmup
    for (let i = 0; i < result.rows; i++) {
        const period = result.combos[i].period;
        const rowStart = i * result.cols;
        const rowData = result.values.slice(rowStart, rowStart + result.cols);
        
        // First period values should be NaN
        assertAllNaN(rowData.slice(0, period));
        
        // After warmup should have values
        if (close.length > period) {
            assert(!isNaN(rowData[period]));
        }
    }
});

test('CG numerical stability with extreme values', () => {
    // Test with very large values
    const largeData = new Float64Array([1e15, 1e15, 1e15, 1e15, 1e15, 1e15]);
    const resultLarge = wasm.cg_js(largeData, 2);
    assert.strictEqual(resultLarge.length, largeData.length);
    // CG with constant values should produce consistent output
    assert(!isNaN(resultLarge[2]));  // After warmup
    
    // Test with very small values
    const smallData = new Float64Array([1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15]);
    const resultSmall = wasm.cg_js(smallData, 2);
    assert.strictEqual(resultSmall.length, smallData.length);
    assert(!isNaN(resultSmall[2]));  // After warmup
});

test('CG full dataset test', () => {
    // Test CG with full dataset instead of slices
    const close = new Float64Array(testData.close);  // Use full dataset
    
    const result = wasm.cg_js(close, 10);
    assert.strictEqual(result.length, close.length);
    
    // Verify warmup period (first 10 values should be NaN)
    assertAllNaN(result.slice(0, 10));
    
    // After warmup, all values should be valid
    for (let i = 10; i < result.length; i++) {
        assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
    }
});

test('CG batch with invalid parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    // Test with period exceeding data length in batch
    // This should throw an error because periods exceed data length
    assert.throws(() => {
        wasm.cg_batch(close, {
            period_range: [15, 20, 5]  // All periods exceed data length
        });
    }, /Not enough valid data|Invalid period/);
});

test('CG batch memory validation', () => {
    // Test for potential memory issues with batch processing
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Run multiple batch operations to check for memory leaks or corruption
    for (let iteration = 0; iteration < 3; iteration++) {
        const result = wasm.cg_batch(close, {
            period_range: [5, 15, 5]
        });
        
        // Verify consistency across iterations
        assert.strictEqual(result.rows, 3);
        assert.strictEqual(result.cols, 100);
        
        // Check for poison patterns (common uninitialized memory values)
        for (let i = 0; i < result.values.length; i++) {
            const val = result.values[i];
            if (!isNaN(val)) {
                // Check value is reasonable (not a poison pattern)
                assert(Math.abs(val) < 1e100, `Unreasonable value at index ${i}: ${val}`);
            }
        }
    }
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('CG WASM tests completed');
});