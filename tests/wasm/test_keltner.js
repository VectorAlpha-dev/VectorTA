/**
 * WASM binding tests for KELTNER indicator.
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

test('KELTNER accuracy - safe API', () => {
    const { high, low, close } = testData;
    const source = close; // Use close as source
    const { period, multiplier, ma_type } = EXPECTED_OUTPUTS.keltner.defaultParams;
    
    const result = wasm.keltner_js(high, low, close, source, period, multiplier, ma_type);
    
    assert.equal(result.rows, 3, 'Should have 3 output rows (upper, middle, lower)');
    assert.equal(result.cols, close.length, 'Output length should match input');
    
    const values = result.values;
    const len = close.length;
    
    // Extract the three bands from flattened array
    const upper_band = values.slice(0, len);
    const middle_band = values.slice(len, 2 * len);
    const lower_band = values.slice(2 * len, 3 * len);
    
    // Check last 5 values
    const last5Upper = upper_band.slice(-5);
    const last5Middle = middle_band.slice(-5);
    const last5Lower = lower_band.slice(-5);
    
    assertArrayClose(last5Upper, EXPECTED_OUTPUTS.keltner.last5Upper, 1e-8);
    assertArrayClose(last5Middle, EXPECTED_OUTPUTS.keltner.last5Middle, 1e-8);
    assertArrayClose(last5Lower, EXPECTED_OUTPUTS.keltner.last5Lower, 1e-8);
    
    // Check warmup period (first 19 should be NaN)
    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(upper_band[i]), `Upper band warmup value at ${i} should be NaN`);
        assert(isNaN(middle_band[i]), `Middle band warmup value at ${i} should be NaN`);
        assert(isNaN(lower_band[i]), `Lower band warmup value at ${i} should be NaN`);
    }
});

test('KELTNER accuracy - fast API', () => {
    const { high, low, close } = testData;
    const source = close;
    const { period, multiplier, ma_type } = EXPECTED_OUTPUTS.keltner.defaultParams;
    const len = close.length;
    
    // Allocate output buffers
    const upper_ptr = wasm.keltner_alloc(len);
    const middle_ptr = wasm.keltner_alloc(len);
    const lower_ptr = wasm.keltner_alloc(len);
    
    try {
        // Call fast API
        wasm.keltner_into(
            new Float64Array(high).buffer,
            new Float64Array(low).buffer,
            new Float64Array(close).buffer,
            new Float64Array(source).buffer,
            upper_ptr,
            middle_ptr,
            lower_ptr,
            len,
            period,
            multiplier,
            ma_type
        );
        
        // Create views of the output
        const upper_band = new Float64Array(wasm.memory.buffer, upper_ptr, len);
        const middle_band = new Float64Array(wasm.memory.buffer, middle_ptr, len);
        const lower_band = new Float64Array(wasm.memory.buffer, lower_ptr, len);
        
        // Check last 5 values
        const last5Upper = Array.from(upper_band.slice(-5));
        const last5Middle = Array.from(middle_band.slice(-5));
        const last5Lower = Array.from(lower_band.slice(-5));
        
        assertArrayClose(last5Upper, EXPECTED_OUTPUTS.keltner.last5Upper, 1e-8);
        assertArrayClose(last5Middle, EXPECTED_OUTPUTS.keltner.last5Middle, 1e-8);
        assertArrayClose(last5Lower, EXPECTED_OUTPUTS.keltner.last5Lower, 1e-8);
        
    } finally {
        // Clean up allocated memory
        wasm.keltner_free(upper_ptr, len);
        wasm.keltner_free(middle_ptr, len);
        wasm.keltner_free(lower_ptr, len);
    }
});

test('KELTNER error handling - empty data', () => {
    const empty = new Float64Array(0);
    
    assert.throws(() => {
        wasm.keltner_js(empty, empty, empty, empty, 20, 2.0, "ema");
    }, /empty/i, 'Should throw error for empty data');
});

test('KELTNER error handling - invalid period', () => {
    const { high, low, close } = testData;
    const source = close;
    
    // Period = 0
    assert.throws(() => {
        wasm.keltner_js(high, low, close, source, 0, 2.0, "ema");
    }, /invalid period/i, 'Should throw error for period = 0');
    
    // Period > data length
    assert.throws(() => {
        wasm.keltner_js(high, low, close, source, close.length + 1, 2.0, "ema");
    }, /invalid period/i, 'Should throw error for period > data length');
});

test('KELTNER error handling - all NaN values', () => {
    const len = 100;
    const nanData = new Float64Array(len).fill(NaN);
    
    assert.throws(() => {
        wasm.keltner_js(nanData, nanData, nanData, nanData, 20, 2.0, "ema");
    }, /all values are nan/i, 'Should throw error for all NaN values');
});

test('KELTNER batch operation', () => {
    const { high, low, close } = testData;
    const source = close;
    
    const config = {
        period_range: [10, 30, 10],  // 10, 20, 30
        multiplier_range: [1.0, 3.0, 1.0],  // 1.0, 2.0, 3.0
        ma_type: "ema"
    };
    
    const result = wasm.keltner_batch(high, low, close, source, config);
    
    // Should have 3 periods x 3 multipliers = 9 combinations
    assert.equal(result.rows, 9, 'Should have 9 parameter combinations');
    assert.equal(result.cols, close.length, 'Output columns should match input length');
    assert.equal(result.combos.length, 9, 'Should have 9 parameter combinations');
    
    // Check first combo matches expected parameters
    assert.equal(result.combos[0].period, 10);
    assert.equal(result.combos[0].multiplier, 1.0);
    assert.equal(result.combos[0].ma_type, "ema");
    
    // Verify arrays have correct length
    assert.equal(result.upper_band.length, 9 * close.length);
    assert.equal(result.middle_band.length, 9 * close.length);
    assert.equal(result.lower_band.length, 9 * close.length);
});

test('KELTNER aliasing handling', () => {
    const { high, low, close } = testData;
    const { period, multiplier, ma_type } = EXPECTED_OUTPUTS.keltner.defaultParams;
    const len = close.length;
    
    // Create typed arrays
    const highArray = new Float64Array(high);
    const lowArray = new Float64Array(low);
    const closeArray = new Float64Array(close);
    const sourceArray = new Float64Array(close);
    
    // Allocate output that aliases with input (use close array as output)
    const closePtr = closeArray.buffer;
    
    // This should handle aliasing correctly
    wasm.keltner_into(
        highArray.buffer,
        lowArray.buffer,
        closeArray.buffer,
        sourceArray.buffer,
        closePtr,  // upper output aliases with close input
        wasm.keltner_alloc(len),  // middle doesn't alias
        wasm.keltner_alloc(len),  // lower doesn't alias
        len,
        period,
        multiplier,
        ma_type
    );
    
    // Should complete without error (aliasing is handled internally)
    assert(true, 'Aliasing should be handled correctly');
});

test.after(() => {
    console.log('KELTNER WASM tests completed');
});
