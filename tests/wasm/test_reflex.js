/**
 * WASM binding tests for Reflex indicator.
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

test('Reflex partial params', () => {
    // Test with default parameters - mirrors check_reflex_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.reflex_js(close, 20);
    assert(result instanceof Float64Array);
    assert.strictEqual(result.length, close.length);
});

test('Reflex accuracy', async () => {
    // Test Reflex matches expected values from Rust tests - mirrors check_reflex_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.reflex || {};
    
    const result = wasm.reflex_js(close, 20);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected values from Rust test (check_reflex_accuracy and check_batch_default_row)
    const expectedLastFive = [
        0.8085220962465361,
        0.445264715886137,
        0.13861699036615063,
        -0.03598639652007061,
        -0.224906760543743
    ];
    
    const actualLastFive = result.slice(-5);
    
    assertArrayClose(
        actualLastFive,
        expectedLastFive,
        1e-7,
        "Reflex last 5 values mismatch"
    );
    
    // Compare with Rust implementation
    await compareWithRust('reflex', result, 'close', { period: 20 });
});

test('Reflex default candles', () => {
    // Test Reflex with default parameters - mirrors check_reflex_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.reflex_js(close, 20);
    assert.strictEqual(result.length, close.length);
});

test('Reflex zero period', () => {
    // Test Reflex fails with zero period - mirrors check_reflex_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.reflex_js(inputData, 0);
    }, /period must be >=2|invalid period/i);
});

test('Reflex period less than two', () => {
    // Test Reflex fails when period < 2 - mirrors check_reflex_period_less_than_two
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.reflex_js(inputData, 1);
    }, /period must be >=2|invalid period/i);
});

test('Reflex period exceeds length', () => {
    // Test Reflex fails when period exceeds data length - mirrors check_reflex_very_small_data_set
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.reflex_js(dataSmall, 10);
    }, /invalid period|not enough valid data|not enough data/i);
});

test('Reflex very small dataset', () => {
    // Test Reflex fails with insufficient data - mirrors check_reflex_very_small_data_set
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.reflex_js(singlePoint, 5);
    }, /invalid period|not enough valid data|not enough data/i);
});

test('Reflex empty input', () => {
    // Test Reflex fails with empty input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.reflex_js(dataEmpty, 20);
    }, /empty/i);
});

test('Reflex NaN handling', () => {
    // Test Reflex handles NaN values correctly - mirrors check_reflex_nan_handling
    const close = new Float64Array(testData.close);
    const period = 14;
    
    const result = wasm.reflex_js(close, period);
    
    assert.strictEqual(result.length, close.length);
    
    // First period values should be 0.0 (Reflex specific warmup behavior)
    for (let i = 0; i < period; i++) {
        assert.strictEqual(result[i], 0.0, `Expected zero at index ${i} during warmup`);
    }
    
    // After warmup period, values should be finite (if input is valid)
    if (result.length > period) {
        for (let i = period; i < result.length; i++) {
            if (!isNaN(close[i])) {
                assert(isFinite(result[i]), `Found unexpected non-finite value at index ${i}`);
            }
        }
    }
});

test('Reflex batch', () => {
    // Test Reflex batch processing - mirrors check_batch_default_row
    const close = new Float64Array(testData.close);
    
    // Test with default period only
    const batch_result = wasm.reflex_batch_js(
        close, 
        20, 20, 0    // period range: just 20
    );
    
    // Get rows and cols info
    const rows_cols = wasm.reflex_batch_rows_cols_js(20, 20, 0, close.length);
    const rows = rows_cols[0];
    const cols = rows_cols[1];
    
    assert(batch_result instanceof Float64Array);
    assert.strictEqual(rows, 1); // Only 1 period value
    assert.strictEqual(cols, close.length);
    assert.strictEqual(batch_result.length, rows * cols);
    
    // Verify it matches individual calculation
    const individual_result = wasm.reflex_js(close, 20);
    assertArrayClose(batch_result, individual_result, 1e-9, "Batch vs single results");
    
    // Check last 5 values match expected from Rust
    const expectedLastFive = [
        0.8085220962465361,
        0.445264715886137,
        0.13861699036615063,
        -0.03598639652007061,
        -0.224906760543743
    ];
    
    const actualLastFive = batch_result.slice(-5);
    assertArrayClose(
        actualLastFive,
        expectedLastFive,
        1e-7,
        "Reflex batch last 5 values mismatch"
    );
});

test('Reflex batch multiple periods', () => {
    // Test Reflex batch computation with multiple periods
    const close = new Float64Array(testData.close);
    
    // Test parameter ranges
    const batch_result = wasm.reflex_batch_js(
        close, 
        10, 30, 5    // period range: 10, 15, 20, 25, 30
    );
    
    // Get rows and cols info
    const rows_cols = wasm.reflex_batch_rows_cols_js(10, 30, 5, close.length);
    const rows = rows_cols[0];
    const cols = rows_cols[1];
    
    assert(batch_result instanceof Float64Array);
    assert.strictEqual(rows, 5); // 5 period values
    assert.strictEqual(cols, close.length);
    assert.strictEqual(batch_result.length, rows * cols);
    
    // Verify metadata
    const metadata = wasm.reflex_batch_metadata_js(10, 30, 5);
    assert.strictEqual(metadata.length, 5);
    assert.deepStrictEqual(Array.from(metadata), [10, 15, 20, 25, 30]);
    
    // Verify first combination matches individual calculation
    const individual_result = wasm.reflex_js(close, 10);
    const batch_first = batch_result.slice(0, close.length);
    
    // Compare after warmup period
    const warmup = 10;
    for (let i = warmup; i < close.length; i++) {
        assertClose(batch_first[i], individual_result[i], 1e-9, `Batch mismatch at ${i}`);
    }
});

test('Reflex all NaN input', () => {
    // Test Reflex with all NaN values
    const allNaN = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        allNaN[i] = NaN;
    }
    
    assert.throws(() => {
        wasm.reflex_js(allNaN, 20);
    }, /All values.*NaN/);
});

test('Reflex batch error conditions', () => {
    // Test batch error handling with all NaN data
    const allNaN = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        allNaN[i] = NaN;
    }
    
    assert.throws(() => {
        wasm.reflex_batch_js(allNaN, 10, 20, 5);
    }, /All values.*NaN/);
    
    // Test with insufficient data
    const smallData = new Float64Array([1.0, 2.0, 3.0]);
    assert.throws(() => {
        wasm.reflex_batch_js(smallData, 10, 20, 5);
    }, /not enough valid data|invalid period|not enough/i);
});

test('Reflex edge cases', () => {
    // Test with monotonically increasing data
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1;
    }
    
    const result = wasm.reflex_js(data, 20);
    assert.strictEqual(result.length, data.length);
    
    // First 20 values should be zeros (warmup)
    for (let i = 0; i < 20; i++) {
        assert.strictEqual(result[i], 0.0, `Expected zero at index ${i} during warmup`);
    }
    
    // After warmup, all values should be finite
    for (let i = 20; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN at index ${i}`);
    }
    
    // Test with constant values
    const constantData = new Float64Array(100).fill(50.0);
    const constantResult = wasm.reflex_js(constantData, 20);
    
    assert.strictEqual(constantResult.length, constantData.length);
    
    // First period values should be zeros
    for (let i = 0; i < 20; i++) {
        assert.strictEqual(constantResult[i], 0.0, `Expected zero at index ${i} during warmup`);
    }
    // With constant input, Reflex produces NaN after warmup (division by zero variance)
    // This is expected behavior
    
    // Test with oscillating values
    const oscillatingData = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        oscillatingData[i] = (i % 2 === 0) ? 10.0 : 20.0;
    }
    
    const oscillatingResult = wasm.reflex_js(oscillatingData, 20);
    assert.strictEqual(oscillatingResult.length, oscillatingData.length);
    
    // First 20 values should be zeros
    for (let i = 0; i < 20; i++) {
        assert.strictEqual(oscillatingResult[i], 0.0, `Expected zero at index ${i} during warmup`);
    }
    
    // After warmup, all values should be finite
    for (let i = 20; i < oscillatingResult.length; i++) {
        assert(isFinite(oscillatingResult[i]), `Expected finite value at index ${i}`);
    }
});

test('Reflex warmup behavior', () => {
    // Test Reflex warmup period behavior
    const close = new Float64Array(testData.close);
    const period = 20;
    
    const result = wasm.reflex_js(close, period);
    
    // Values during warmup should be zeros (Reflex specific behavior)
    for (let i = 0; i < period; i++) {
        assert.strictEqual(result[i], 0.0, `Expected zero at index ${i} during warmup`);
    }
    
    // Find first non-NaN value in input
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    
    const warmup = firstValid + period;
    
    // Values after warmup should be finite
    if (warmup < result.length) {
        for (let i = warmup; i < result.length; i++) {
            assert(isFinite(result[i]), `Expected finite value at index ${i} after warmup`);
        }
    }
});

test('Reflex consistency', () => {
    // Test that Reflex produces consistent results across multiple calls
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.reflex_js(close, 20);
    const result2 = wasm.reflex_js(close, 20);
    
    assertArrayClose(result1, result2, 1e-15, "Reflex results not consistent");
});

test('Reflex batch metadata', () => {
    // Test metadata function returns correct period values
    const metadata = wasm.reflex_batch_metadata_js(
        10, 30, 5    // period range: 10, 15, 20, 25, 30
    );
    
    // Should have 5 period values
    assert.strictEqual(metadata.length, 5);
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 15);
    assert.strictEqual(metadata[2], 20);
    assert.strictEqual(metadata[3], 25);
    assert.strictEqual(metadata[4], 30);
});

test('Reflex step precision', () => {
    // Test batch with various step sizes
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.reflex_batch_js(
        data,
        10, 20, 2     // periods: 10, 12, 14, 16, 18, 20
    );
    
    // Get rows and cols info
    const rows_cols = wasm.reflex_batch_rows_cols_js(10, 20, 2, data.length);
    const rows = rows_cols[0];
    
    // Should have 6 combinations
    assert.strictEqual(rows, 6);
    assert.strictEqual(batch_result.length, 6 * data.length);
    
    // Verify metadata
    const metadata = wasm.reflex_batch_metadata_js(10, 20, 2);
    assert.strictEqual(metadata.length, 6);
    assert.deepStrictEqual(Array.from(metadata), [10, 12, 14, 16, 18, 20]);
});

test('Reflex formula verification', () => {
    // Test that Reflex detects patterns correctly
    const data = new Float64Array([10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
    const period = 5;
    
    const result = wasm.reflex_js(data, period);
    
    // Result length should match input
    assert.strictEqual(result.length, data.length);
    
    // Warmup period should have zeros
    for (let i = 0; i < period; i++) {
        assert.strictEqual(result[i], 0.0, `Expected zero during warmup at index ${i}`);
    }
    
    // After warmup, values should be finite
    for (let i = period; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
    
    // Reflex should detect the pattern - not all values should be the same
    const uniqueValues = new Set();
    for (let i = period; i < result.length; i++) {
        uniqueValues.add(result[i]);
    }
    assert(uniqueValues.size > 1, "Expected varying values for oscillating input");
});
