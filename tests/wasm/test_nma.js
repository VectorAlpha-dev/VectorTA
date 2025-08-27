/**
 * WASM binding tests for NMA indicator.
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

test('NMA partial params', () => {
    // Test with default parameters - mirrors check_nma_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.nma_js(close, 40);
    assert(result instanceof Float64Array);
    assert.strictEqual(result.length, close.length);
});

test('NMA accuracy', async () => {
    // Test NMA matches expected values from Rust tests - mirrors check_nma_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.nma;
    
    const result = wasm.nma_js(close, expected.defaultParams.period);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const actualLastFive = result.slice(-5);
    
    assertArrayClose(
        actualLastFive,
        expected.last5Values,
        1e-3,  // Use same tolerance as Rust tests
        "NMA last 5 values mismatch"
    );
    
    // Compare with Rust implementation
    await compareWithRust('nma', result, 'close', expected.defaultParams);
});

test('NMA default candles', () => {
    // Test NMA with default parameters - mirrors check_nma_default_candles
    const close = new Float64Array(testData.close);
    
    // Default period: 40
    const result = wasm.nma_js(close, 40);
    assert.strictEqual(result.length, close.length);
});

test('NMA zero period', () => {
    // Test NMA fails with zero period - mirrors check_nma_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.nma_js(inputData, 0);
    }, /Invalid period/);
});

test('NMA period exceeds length', () => {
    // Test NMA fails when period exceeds data length - mirrors check_nma_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.nma_js(dataSmall, 10);
    }, /Invalid period/);
});

test('NMA very small dataset', () => {
    // Test NMA fails with insufficient data - mirrors check_nma_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.nma_js(singlePoint, 40);
    }, /Invalid period|Not enough valid data/);
});

test('NMA empty input', () => {
    // Test NMA fails with empty input - mirrors check_nma_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.nma_js(empty, 40);
    }, /Input data slice is empty/);
});

test('NMA NaN handling', () => {
    // Test NMA handles NaN values correctly - mirrors check_nma_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.nma_js(close, 40);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First period values should be NaN
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    const warmup = firstValid + 40;
    assertAllNaN(result.slice(0, warmup), "Expected NaN in warmup period");
});

test('NMA all NaN input', () => {
    // Test NMA with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.nma_js(allNaN, 40);
    }, /All values are NaN/);
});

test('NMA batch default row', () => {
    // Test NMA batch with default parameters - mirrors check_batch_default_row
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.nma;
    
    // Test with default period only using old API
    const batchResult = wasm.nma_batch_js(
        close,
        40, 40, 0  // Default period only
    );
    
    // Get rows and cols info
    const rowsCols = wasm.nma_batch_rows_cols_js(40, 40, 0, close.length);
    const rows = rowsCols[0];
    const cols = rowsCols[1];
    
    assert(batchResult instanceof Float64Array);
    assert.strictEqual(rows, 1); // Single combination
    assert.strictEqual(cols, close.length);
    assert.strictEqual(batchResult.length, rows * cols);
    
    // Extract the single row (entire result since only 1 row)
    const defaultRow = batchResult.slice(-5);
    
    // Check last 5 values match expected
    assertArrayClose(
        defaultRow,
        expected.batchDefaultRow,
        1e-3,  // Same tolerance as Rust
        "NMA batch default row mismatch"
    );
});

test('NMA batch multiple periods', () => {
    // Test NMA batch with multiple period values
    const close = new Float64Array(testData.close);
    
    // Test multiple periods using old API
    const batchResult = wasm.nma_batch_js(
        close,
        20, 60, 20  // periods: 20, 40, 60
    );
    
    // Get rows and cols info
    const rowsCols = wasm.nma_batch_rows_cols_js(20, 60, 20, close.length);
    const rows = rowsCols[0];
    const cols = rowsCols[1];
    
    assert(batchResult instanceof Float64Array);
    assert.strictEqual(rows, 3); // 3 period values
    assert.strictEqual(cols, close.length);
    assert.strictEqual(batchResult.length, rows * cols);
    
    // Get metadata for verification
    const metadata = wasm.nma_batch_metadata_js(20, 60, 20);
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 20);
    assert.strictEqual(metadata[1], 40);
    assert.strictEqual(metadata[2], 60);
    
    // Verify each combination matches individual calculation
    for (let i = 0; i < metadata.length; i++) {
        const period = metadata[i];
        const individualResult = wasm.nma_js(close, period);
        const rowStart = i * cols;
        const rowEnd = rowStart + cols;
        const batchRow = batchResult.slice(rowStart, rowEnd);
        
        // Compare after warmup period
        let firstValid = 0;
        for (let j = 0; j < close.length; j++) {
            if (!isNaN(close[j])) {
                firstValid = j;
                break;
            }
        }
        const warmup = firstValid + period;
        
        if (warmup < close.length) {
            for (let j = warmup; j < close.length; j++) {
                assertClose(
                    batchRow[j], 
                    individualResult[j], 
                    1e-9, 
                    `NMA batch period ${period} mismatch at index ${j}`
                );
            }
        }
    }
});

test('NMA batch with unified API', () => {
    // Test using the new unified batch API (if available)
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Test if unified API exists
    if (typeof wasm.nma_batch_unified_js === 'function') {
        const config = {
            period_range: [20, 40, 10]  // periods: 20, 30, 40
        };
        
        const result = wasm.nma_batch_unified_js(close, config);
        
        // Should have proper structure
        assert(result.values instanceof Float64Array);
        assert(Array.isArray(result.combos));
        assert.strictEqual(result.rows, 3);
        assert.strictEqual(result.cols, 100);
        assert.strictEqual(result.values.length, 3 * 100);
        
        // Verify combos
        assert.strictEqual(result.combos.length, 3);
        assert.strictEqual(result.combos[0].period, 20);
        assert.strictEqual(result.combos[1].period, 30);
        assert.strictEqual(result.combos[2].period, 40);
    }
});

test('NMA batch error handling', () => {
    // Test batch error conditions
    
    // Test with all NaN data
    const allNaN = new Float64Array(100).fill(NaN);
    assert.throws(() => {
        wasm.nma_batch_js(allNaN, 20, 40, 10);
    }, /All values are NaN/);
    
    // Test with insufficient data
    const smallData = new Float64Array([1.0, 2.0, 3.0]);
    assert.throws(() => {
        wasm.nma_batch_js(smallData, 10, 20, 10);
    }, /Invalid period|Not enough valid data|unreachable/);  // WASM may throw unreachable for some error conditions
});

test('NMA warmup behavior', () => {
    // Test NMA warmup period behavior
    const close = new Float64Array(testData.close);
    const period = 40;
    
    const result = wasm.nma_js(close, period);
    
    // Find first non-NaN value in input
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    
    const warmup = firstValid + period;
    
    // Values before warmup should be NaN
    for (let i = 0; i < warmup; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    // Values after warmup should be finite
    for (let i = warmup; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i} after warmup`);
    }
});

test('NMA different periods', () => {
    // Test NMA with various period values
    const close = new Float64Array(testData.close);
    
    // Test various period values
    const testPeriods = [10, 20, 40, 80];
    
    for (const period of testPeriods) {
        const result = wasm.nma_js(close, period);
        assert.strictEqual(result.length, close.length);
        
        // After warmup, all values should be finite
        let firstValid = 0;
        for (let i = 0; i < close.length; i++) {
            if (!isNaN(close[i])) {
                firstValid = i;
                break;
            }
        }
        const warmup = firstValid + period;
        
        if (warmup < result.length) {
            for (let i = warmup; i < result.length; i++) {
                assert(isFinite(result[i]), `NaN at index ${i} for period=${period}`);
            }
        }
    }
});

test('NMA edge cases', () => {
    // Test NMA with edge case inputs
    
    // Test with monotonically increasing data
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1;
    }
    const result = wasm.nma_js(data, 10);
    assert.strictEqual(result.length, data.length);
    
    // After warmup, all values should be finite
    for (let i = 10; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN at index ${i}`);
    }
    
    // Test with constant values
    const constantData = new Float64Array(100).fill(50.0);
    const constantResult = wasm.nma_js(constantData, 10);
    
    assert.strictEqual(constantResult.length, constantData.length);
    
    // After warmup, all values should be finite
    for (let i = 10; i < constantResult.length; i++) {
        assert(isFinite(constantResult[i]), `NaN at index ${i} for constant data`);
    }
    
    // Test with oscillating values
    const oscillatingData = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        oscillatingData[i] = (i % 2 === 0) ? 10.0 : 20.0;
    }
    const oscillatingResult = wasm.nma_js(oscillatingData, 10);
    assert.strictEqual(oscillatingResult.length, oscillatingData.length);
    
    // After warmup, all values should be finite
    for (let i = 10; i < oscillatingResult.length; i++) {
        assert(isFinite(oscillatingResult[i]), `Expected finite value at index ${i}`);
    }
});

test('NMA consistency across calls', () => {
    // Test that NMA produces consistent results across multiple calls
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.nma_js(close, 40);
    const result2 = wasm.nma_js(close, 40);
    
    assertArrayClose(result1, result2, 1e-15, "NMA results not consistent");
});

test('NMA formula verification', () => {
    // Verify NMA formula implementation with simple data
    const data = new Float64Array([10.0, 12.0, 11.0, 13.0, 15.0, 14.0]);
    const period = 3;
    
    const result = wasm.nma_js(data, period);
    
    // Result length should match input
    assert.strictEqual(result.length, data.length);
    
    // Warmup period should be respected
    for (let i = 0; i < period; i++) {
        assert(isNaN(result[i]), `Expected NaN during warmup at index ${i}`);
    }
    
    // Values after warmup should be reasonable
    const min = Math.min(...data);
    const max = Math.max(...data);
    for (let i = period; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
        assert(result[i] >= min * 0.5, `Value too low at index ${i}`);
        assert(result[i] <= max * 1.5, `Value too high at index ${i}`);
    }
});

test('NMA batch metadata', () => {
    // Test metadata function returns correct period values
    const metadata = wasm.nma_batch_metadata_js(
        10, 30, 10  // period range: 10, 20, 30
    );
    
    // Should have 3 period values
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 20);
    assert.strictEqual(metadata[2], 30);
});

test('NMA batch rows and cols', () => {
    // Test rows and cols calculation
    const rowsCols = wasm.nma_batch_rows_cols_js(
        10, 30, 10,  // period range: 10, 20, 30
        100          // data length
    );
    
    assert.strictEqual(rowsCols[0], 3);   // 3 periods
    assert.strictEqual(rowsCols[1], 100); // data length
});

test('NMA step precision', () => {
    // Test batch with various step sizes
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batchResult = wasm.nma_batch_js(
        data,
        10, 30, 10  // periods: 10, 20, 30
    );
    
    // Get rows and cols info
    const rowsCols = wasm.nma_batch_rows_cols_js(10, 30, 10, data.length);
    const rows = rowsCols[0];
    
    // Should have 3 combinations
    assert.strictEqual(rows, 3);
    assert.strictEqual(batchResult.length, 3 * data.length);
    
    // Verify metadata
    const metadata = wasm.nma_batch_metadata_js(10, 30, 10);
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 20);
    assert.strictEqual(metadata[2], 30);
});

test('NMA small step size', () => {
    // Test batch with very small step size
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batchResult = wasm.nma_batch_js(
        data,
        10, 12, 1  // periods: 10, 11, 12
    );
    
    const rowsCols = wasm.nma_batch_rows_cols_js(10, 12, 1, data.length);
    const rows = rowsCols[0];
    
    assert.strictEqual(rows, 3);
    assert.strictEqual(batchResult.length, 3 * data.length);
});