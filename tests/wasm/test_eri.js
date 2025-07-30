/**
 * WASM binding tests for ERI indicator.
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

test('ERI accuracy', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    
    // Test with default parameters
    const result = wasm.eri_js(high, low, close, 13, "ema");
    
    // Result should be flattened [bull..., bear...]
    assert.strictEqual(result.length, close.length * 2);
    
    // Extract bull and bear arrays
    const bull = result.slice(0, close.length);
    const bear = result.slice(close.length);
    
    // Expected values from Rust tests
    const expectedBullLastFive = [
        -103.35343557205488,
        6.839912366813223,
        -42.851503685589705,
        -9.444146016219747,
        11.476446271808527,
    ];
    const expectedBearLastFive = [
        -433.3534355720549,
        -314.1600876331868,
        -414.8515036855897,
        -336.44414601621975,
        -925.5235537281915,
    ];
    
    // Check last 5 values
    assertArrayClose(
        bull.slice(-5),
        expectedBullLastFive,
        0.02,
        'ERI bull last 5 values mismatch'
    );
    
    assertArrayClose(
        bear.slice(-5),
        expectedBearLastFive,
        0.02,
        'ERI bear last 5 values mismatch'
    );
});

test('ERI error handling', () => {
    // Test with empty data
    assert.throws(
        () => wasm.eri_js([], [], [], 13, "ema"),
        /Invalid period/,
        'Should fail with empty data'
    );
    
    // Test with mismatched lengths
    assert.throws(
        () => wasm.eri_js([1, 2, 3], [1, 2], [1, 2, 3], 13, "ema"),
        /must have the same length/,
        'Should fail with mismatched lengths'
    );
    
    // Test with zero period
    assert.throws(
        () => wasm.eri_js([1, 2, 3], [1, 2, 3], [1, 2, 3], 0, "ema"),
        /Invalid period/,
        'Should fail with zero period'
    );
    
    // Test with period exceeding data length
    assert.throws(
        () => wasm.eri_js([1, 2, 3], [1, 2, 3], [1, 2, 3], 10, "ema"),
        /Invalid period/,
        'Should fail with period exceeding data length'
    );
});

test('ERI fast API', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    const len = close.length;
    
    // Allocate output buffers
    const bullPtr = wasm.eri_alloc(len);
    const bearPtr = wasm.eri_alloc(len);
    
    try {
        // Compute ERI using fast API
        wasm.eri_into(
            high,
            low,
            close,
            bullPtr,
            bearPtr,
            len,
            13,
            "ema"
        );
        
        // Create arrays from pointers
        const bull = new Float64Array(wasm.memory.buffer, bullPtr, len);
        const bear = new Float64Array(wasm.memory.buffer, bearPtr, len);
        
        // Compare with safe API results
        const safeResult = wasm.eri_js(high, low, close, 13, "ema");
        const safeBull = safeResult.slice(0, len);
        const safeBear = safeResult.slice(len);
        
        // Results should match
        assertArrayClose(Array.from(bull), safeBull, 1e-9, 'Bull values should match between safe and fast API');
        assertArrayClose(Array.from(bear), safeBear, 1e-9, 'Bear values should match between safe and fast API');
    } finally {
        // Clean up
        wasm.eri_free(bullPtr, len);
        wasm.eri_free(bearPtr, len);
    }
});

test('ERI batch processing', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    
    const config = {
        period_range: [10, 20, 5], // 10, 15, 20
        ma_type: "ema"
    };
    
    const result = wasm.eri_batch(high, low, close, config);
    
    assert.strictEqual(result.rows, 3, 'Should have 3 parameter combinations');
    assert.strictEqual(result.cols, close.length, 'Columns should match data length');
    assert.deepStrictEqual(result.periods, [10, 15, 20], 'Periods should match expected');
    
    // Verify output lengths
    assert.strictEqual(result.bull_values.length, 3 * close.length, 'Bull values length mismatch');
    assert.strictEqual(result.bear_values.length, 3 * close.length, 'Bear values length mismatch');
    
    // Test single period batch
    const singleConfig = {
        period_range: [13, 13, 0],
        ma_type: "ema"
    };
    
    const singleResult = wasm.eri_batch(high, low, close, singleConfig);
    assert.strictEqual(singleResult.rows, 1, 'Single period should have 1 row');
    
    // Compare with regular ERI for period 13
    const regularResult = wasm.eri_js(high, low, close, 13, "ema");
    const regularBull = regularResult.slice(0, close.length);
    const regularBear = regularResult.slice(close.length);
    
    assertArrayClose(
        singleResult.bull_values,
        regularBull,
        1e-9,
        'Batch bull should match regular ERI'
    );
    
    assertArrayClose(
        singleResult.bear_values,
        regularBear,
        1e-9,
        'Batch bear should match regular ERI'
    );
});

test.after(() => {
    console.log('ERI WASM tests completed');
});
