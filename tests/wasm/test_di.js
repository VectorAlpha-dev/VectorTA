/**
 * WASM binding tests for DI indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
const test = require('node:test');
const assert = require('node:assert');
const path = require('path');
const { 
    loadTestData, 
    assertArrayClose, 
    assertClose,
    isNaN,
    assertAllNaN,
    assertNoNaN,
    EXPECTED_OUTPUTS 
} = require('./test_utils');

let wasm;
let testData;

test.before(async () => {
    // Load WASM module
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        wasm = await import(wasmPath);
        await wasm.default();
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('DI accuracy', () => {
    const expected = EXPECTED_OUTPUTS.di;
    
    // Calculate DI with default parameters
    const result = wasm.di_js(
        testData.high,
        testData.low,
        testData.close,
        expected.defaultParams.period
    );
    
    // Check output length
    assert.strictEqual(result.plus.length, testData.close.length);
    assert.strictEqual(result.minus.length, testData.close.length);
    
    // Check last 5 values match expected
    const plusTail = result.plus.slice(-5);
    const minusTail = result.minus.slice(-5);
    
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
});

test('DI error handling', () => {
    // Test with zero period
    assert.throws(() => {
        wasm.di_js(
            new Float64Array([10.0, 11.0, 12.0]),
            new Float64Array([9.0, 8.0, 7.0]),
            new Float64Array([9.5, 10.0, 11.0]),
            0
        );
    }, /Invalid period/);
    
    // Test with period exceeding data length
    assert.throws(() => {
        wasm.di_js(
            new Float64Array([10.0, 11.0, 12.0]),
            new Float64Array([9.0, 8.0, 7.0]),
            new Float64Array([9.5, 10.0, 11.0]),
            10
        );
    }, /Invalid period/);
    
    // Test with empty data
    assert.throws(() => {
        wasm.di_js(
            new Float64Array([]),
            new Float64Array([]),
            new Float64Array([]),
            14
        );
    }, /Empty data/);
});

test('DI NaN handling', () => {
    const result = wasm.di_js(
        testData.high,
        testData.low,
        testData.close,
        14
    );
    
    // Check warmup period has NaN
    for (let i = 0; i < 13; i++) {
        assert(isNaN(result.plus[i]), `Expected NaN at plus[${i}]`);
        assert(isNaN(result.minus[i]), `Expected NaN at minus[${i}]`);
    }
    
    // After warmup (beyond index 40), no NaN values should exist
    if (result.plus.length > 40) {
        for (let i = 40; i < result.plus.length; i++) {
            assert(!isNaN(result.plus[i]), `Unexpected NaN at plus[${i}]`);
            assert(!isNaN(result.minus[i]), `Unexpected NaN at minus[${i}]`);
        }
    }
});

test('DI batch processing', () => {
    // Test batch with single period
    const config = {
        period_range: [14, 14, 1]
    };
    
    const result = wasm.di_batch(
        testData.high,
        testData.low,
        testData.close,
        config
    );
    
    assert('plus' in result);
    assert('minus' in result);
    assert('periods' in result);
    assert('rows' in result);
    assert('cols' in result);
    
    // Check shape
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, testData.close.length);
    assert.strictEqual(result.plus.length, result.rows * result.cols);
    assert.strictEqual(result.minus.length, result.rows * result.cols);
    assert.strictEqual(result.periods.length, 1);
    assert.strictEqual(result.periods[0], 14);
    
    // Test batch with multiple periods
    const configMulti = {
        period_range: [10, 20, 5]  // periods: 10, 15, 20
    };
    
    const resultMulti = wasm.di_batch(
        testData.high,
        testData.low,
        testData.close,
        configMulti
    );
    
    assert.strictEqual(resultMulti.rows, 3);
    assert.strictEqual(resultMulti.cols, testData.close.length);
    assert.strictEqual(resultMulti.periods.length, 3);
    assert.deepStrictEqual(resultMulti.periods, [10, 15, 20]);
});

test.after(() => {
    console.log('DI WASM tests completed');
});
