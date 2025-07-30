/**
 * WASM binding tests for KDJ indicator.
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

test('KDJ accuracy', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    
    // Test with default parameters
    const result = wasm.kdj_js(high, low, close, 9, 3, "sma", 3, "sma");
    
    assert.ok(result, 'KDJ should return a result');
    assert.equal(result.rows, 3, 'Should have 3 rows (K, D, J)');
    assert.equal(result.cols, close.length, 'Should have same number of columns as input');
    assert.equal(result.values.length, close.length * 3, 'Values array should be flattened K, D, J');
    
    // Extract K, D, J from flattened array
    const k = result.values.slice(0, close.length);
    const d = result.values.slice(close.length, close.length * 2);
    const j = result.values.slice(close.length * 2);
    
    // Expected values from Rust tests (last 5 values)
    const expectedK = [
        58.04341315415984,
        61.56034740940419,
        58.056304282719545,
        56.10961365678364,
        51.43992326447119,
    ];
    const expectedD = [
        49.57659409278555,
        56.81719223571944,
        59.22002161542779,
        58.57542178296905,
        55.20194706799139,
    ];
    const expectedJ = [
        74.97705127690843,
        71.04665775677368,
        55.72886961730306,
        51.17799740441281,
        43.91587565743079,
    ];
    
    // Check last 5 values
    assertArrayClose(k.slice(-5), expectedK, 'KDJ K last 5 values mismatch');
    assertArrayClose(d.slice(-5), expectedD, 'KDJ D last 5 values mismatch');
    assertArrayClose(j.slice(-5), expectedJ, 'KDJ J last 5 values mismatch');
});

test('KDJ error handling', () => {
    // Test empty input
    assert.throws(() => {
        wasm.kdj_js([], [], [], 9, 3, "sma", 3, "sma");
    }, 'Should throw on empty input');
    
    // Test period exceeds length
    const shortData = [1, 2, 3];
    assert.throws(() => {
        wasm.kdj_js(shortData, shortData, shortData, 10, 3, "sma", 3, "sma");
    }, 'Should throw when period exceeds data length');
    
    // Test zero period
    const data = [1, 2, 3, 4, 5];
    assert.throws(() => {
        wasm.kdj_js(data, data, data, 0, 3, "sma", 3, "sma");
    }, 'Should throw on zero period');
    
    // Test all NaN values
    const nanData = [NaN, NaN, NaN, NaN, NaN];
    assert.throws(() => {
        wasm.kdj_js(nanData, nanData, nanData, 3, 2, "sma", 2, "sma");
    }, 'Should throw on all NaN values');
});

test('KDJ fast API with aliasing', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    const len = close.length;
    
    // Allocate output memory
    const kPtr = wasm.kdj_alloc(len);
    const dPtr = wasm.kdj_alloc(len);
    const jPtr = wasm.kdj_alloc(len);
    
    try {
        // Create input memory
        const highMem = new Float64Array(wasm.memory.buffer, wasm.kdj_alloc(len), len);
        const lowMem = new Float64Array(wasm.memory.buffer, wasm.kdj_alloc(len), len);
        const closeMem = new Float64Array(wasm.memory.buffer, wasm.kdj_alloc(len), len);
        
        highMem.set(high);
        lowMem.set(low);
        closeMem.set(close);
        
        // Test normal operation (no aliasing)
        wasm.kdj_into(
            highMem.byteOffset, lowMem.byteOffset, closeMem.byteOffset,
            kPtr, dPtr, jPtr,
            len, 9, 3, "sma", 3, "sma"
        );
        
        const kResult = new Float64Array(wasm.memory.buffer, kPtr, len);
        const dResult = new Float64Array(wasm.memory.buffer, dPtr, len);
        const jResult = new Float64Array(wasm.memory.buffer, jPtr, len);
        
        // Verify some values exist
        assert.ok(!isNaN(kResult[len - 1]), 'K should have valid values');
        assert.ok(!isNaN(dResult[len - 1]), 'D should have valid values');
        assert.ok(!isNaN(jResult[len - 1]), 'J should have valid values');
        
        // Test with aliasing (output same as input) - should still work
        wasm.kdj_into(
            highMem.byteOffset, lowMem.byteOffset, closeMem.byteOffset,
            highMem.byteOffset, lowMem.byteOffset, closeMem.byteOffset,  // aliased output
            len, 9, 3, "sma", 3, "sma"
        );
        
        // Clean up input memory
        wasm.kdj_free(highMem.byteOffset, len);
        wasm.kdj_free(lowMem.byteOffset, len);
        wasm.kdj_free(closeMem.byteOffset, len);
        
    } finally {
        // Clean up output memory
        wasm.kdj_free(kPtr, len);
        wasm.kdj_free(dPtr, len);
        wasm.kdj_free(jPtr, len);
    }
});

test('KDJ batch processing', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    
    const config = {
        fast_k_period_range: [5, 15, 5],      // 5, 10, 15
        slow_k_period_range: [3, 3, 0],       // just 3
        slow_k_ma_type: "sma",
        slow_d_period_range: [3, 3, 0],       // just 3
        slow_d_ma_type: "sma"
    };
    
    const result = wasm.kdj_batch(high, low, close, config);
    
    assert.ok(result, 'Batch should return a result');
    assert.ok(result.combos, 'Should have parameter combinations');
    assert.equal(result.combos.length, 3, 'Should have 3 combinations (fast_k = 5, 10, 15)');
    assert.equal(result.rows, 3, 'Should have 3 rows');
    assert.equal(result.cols, close.length, 'Should have same columns as input');
    assert.equal(result.values.length, 3 * close.length * 3, 'Values should contain all K, D, J for all combos');
    
    // Verify parameter combinations
    assert.equal(result.combos[0].fast_k_period, 5);
    assert.equal(result.combos[1].fast_k_period, 10);
    assert.equal(result.combos[2].fast_k_period, 15);
    
    // All should have slow_k_period = 3
    result.combos.forEach(combo => {
        assert.equal(combo.slow_k_period, 3);
        assert.equal(combo.slow_d_period, 3);
    });
});

test.after(() => {
    console.log('KDJ WASM tests completed');
});
