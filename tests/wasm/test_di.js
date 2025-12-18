/**
 * WASM binding tests for DI indicator.
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

test('DI accuracy - mirrors check_di_accuracy', async () => {
    const expected = EXPECTED_OUTPUTS.di;
    
    // Calculate DI with default parameters
    const result = wasm.di(
        new Float64Array(testData.high),
        new Float64Array(testData.low),
        new Float64Array(testData.close),
        expected.defaultParams.period
    );
    
    // New flattened format: {values: [...], rows: 2, cols: len}
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, testData.close.length);
    assert.strictEqual(result.values.length, 2 * testData.close.length);
    
    // Extract plus and minus from flattened format
    const plus = result.values.slice(0, result.cols);
    const minus = result.values.slice(result.cols, 2 * result.cols);
    
    // Check last 5 values match expected
    const plusTail = plus.slice(-5);
    const minusTail = minus.slice(-5);
    
    assertArrayClose(
        plusTail,
        expected.plusLast5Values,
        1e-6,
        "DI+ last 5 values mismatch"
    );
    assertArrayClose(
        minusTail,
        expected.minusLast5Values,
        1e-6,
        "DI- last 5 values mismatch"
    );
    
    // Compare with Rust implementation
    await compareWithRust('di', {plus, minus}, 'hlc', expected.defaultParams);
});

test('DI error handling - mirrors check_di_with_zero_period', () => {
    // Test with zero period
    assert.throws(() => {
        wasm.di(
            new Float64Array([10.0, 11.0, 12.0]),
            new Float64Array([9.0, 8.0, 7.0]),
            new Float64Array([9.5, 10.0, 11.0]),
            0
        );
    }, /Invalid period/);
    
    // Test with period exceeding data length - mirrors check_di_with_period_exceeding_data_length
    assert.throws(() => {
        wasm.di(
            new Float64Array([10.0, 11.0, 12.0]),
            new Float64Array([9.0, 8.0, 7.0]),
            new Float64Array([9.5, 10.0, 11.0]),
            10
        );
    }, /Invalid period/);
    
    // Test with empty data
    assert.throws(() => {
        wasm.di(
            new Float64Array([]),
            new Float64Array([]),
            new Float64Array([]),
            14
        );
    }, /Empty input data|Empty data/i);
});

test('DI NaN handling - mirrors check_di_accuracy_nan_check', () => {
    const result = wasm.di(
        new Float64Array(testData.high),
        new Float64Array(testData.low),
        new Float64Array(testData.close),
        14
    );
    
    // Extract plus and minus from flattened format
    const plus = result.values.slice(0, result.cols);
    const minus = result.values.slice(result.cols, 2 * result.cols);
    
    // Check warmup period has NaN
    for (let i = 0; i < 13; i++) {
        assert(isNaN(plus[i]), `Expected NaN at plus[${i}]`);
        assert(isNaN(minus[i]), `Expected NaN at minus[${i}]`);
    }
    
    // After warmup (beyond index 40), no NaN values should exist
    if (plus.length > 40) {
        for (let i = 40; i < plus.length; i++) {
            assert(!isNaN(plus[i]), `Unexpected NaN at plus[${i}]`);
            assert(!isNaN(minus[i]), `Unexpected NaN at minus[${i}]`);
        }
    }
});

test('DI batch processing - mirrors check_batch_period_range', () => {
    // Test batch with single period
    const config = {
        period_range: [14, 14, 1]
    };
    
    const result = wasm.di_batch(
        new Float64Array(testData.high),
        new Float64Array(testData.low),
        new Float64Array(testData.close),
        config
    );
    
    assert('values' in result);
    assert('periods' in result);
    assert('rows' in result);
    assert('cols' in result);
    
    // Check shape - new format has rows = 2 * combos (plus and minus interleaved)
    assert.strictEqual(result.rows, 2);  // 1 combo * 2 (plus and minus)
    assert.strictEqual(result.cols, testData.close.length);
    assert.strictEqual(result.values.length, result.rows * result.cols);
    assert.strictEqual(result.periods.length, 1);
    assert.strictEqual(result.periods[0], 14);
    
    // Test batch with multiple periods
    const configMulti = {
        period_range: [10, 20, 5]  // periods: 10, 15, 20
    };
    
    const resultMulti = wasm.di_batch(
        new Float64Array(testData.high),
        new Float64Array(testData.low),
        new Float64Array(testData.close),
        configMulti
    );
    
    assert.strictEqual(resultMulti.rows, 6);  // 3 combos * 2 (plus and minus)
    assert.strictEqual(resultMulti.cols, testData.close.length);
    assert.strictEqual(resultMulti.periods.length, 3);
    assert.deepStrictEqual(resultMulti.periods, [10, 15, 20]);
});

test('DI very small dataset - mirrors check_di_very_small_data_set', () => {
    // Test with single data point
    assert.throws(() => {
        wasm.di(
            new Float64Array([42.0]),
            new Float64Array([41.0]),
            new Float64Array([41.5]),
            14
        );
    }, /Invalid period|Not enough valid data/);
});

test('DI all NaN input', () => {
    // Test with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.di(allNaN, allNaN, allNaN, 14);
    }, /All values are NaN/);
});

test('DI re-input - mirrors check_di_with_slice_data_reinput', () => {
    const expected = EXPECTED_OUTPUTS.di;
    
    // First pass
    const firstResult = wasm.di(
        new Float64Array(testData.high),
        new Float64Array(testData.low),
        new Float64Array(testData.close),
        14
    );
    
    // Extract plus and minus from first pass
    const firstPlus = firstResult.values.slice(0, firstResult.cols);
    const firstMinus = firstResult.values.slice(firstResult.cols, 2 * firstResult.cols);
    
    // Second pass - apply DI to DI output (using plus as high, minus as low, close stays same)
    const secondResult = wasm.di(
        new Float64Array(firstPlus),
        new Float64Array(firstMinus),
        new Float64Array(testData.close),
        14
    );
    
    assert.strictEqual(secondResult.cols, firstResult.cols);
    assert.strictEqual(secondResult.rows, 2);
});

test('DI default parameters', () => {
    // Test with default period (14)
    const result = wasm.di(
        new Float64Array(testData.high),
        new Float64Array(testData.low),
        new Float64Array(testData.close),
        14  // Default period
    );
    
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, testData.close.length);
});

test.after(() => {
    console.log('DI WASM tests completed');
});
