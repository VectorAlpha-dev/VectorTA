/**
 * WASM binding tests for Decycler Oscillator (DEC_OSC) indicator.
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

test('DEC_OSC partial params', () => {
    // Test with default parameters - mirrors check_dec_osc_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.dec_osc_js(close, 125, 1.0);
    assert.strictEqual(result.length, close.length);
});

test('DEC_OSC accuracy', () => {
    // Test DEC_OSC matches expected values from Rust tests - mirrors check_dec_osc_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.dec_osc_js(close, 125, 1.0);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected values from Rust tests
    const expectedLast5 = [
        -1.5036367540303395,
        -1.4037875172207006,
        -1.3174199471429475,
        -1.2245874070642693,
        -1.1638422627265639,
    ];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-7,
        "DEC_OSC last 5 values mismatch"
    );
});

test('DEC_OSC default candles', () => {
    // Test DEC_OSC with default parameters - mirrors check_dec_osc_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.dec_osc_js(close, 125, 1.0);
    assert.strictEqual(result.length, close.length);
});

test('DEC_OSC zero period', () => {
    // Test DEC_OSC fails with zero period - mirrors check_dec_osc_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.dec_osc_js(inputData, 0, 1.0);
    }, /Invalid period/);
});

test('DEC_OSC period exceeds length', () => {
    // Test DEC_OSC fails when period exceeds data length - mirrors check_dec_osc_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.dec_osc_js(dataSmall, 10, 1.0);
    }, /Invalid period/);
});

test('DEC_OSC very small dataset', () => {
    // Test DEC_OSC fails with insufficient data - mirrors check_dec_osc_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.dec_osc_js(singlePoint, 125, 1.0);
    }, /Invalid period|Not enough valid data/);
});

test('DEC_OSC empty input', () => {
    // Test DEC_OSC fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.dec_osc_js(empty, 125, 1.0);
    }, /Input data slice is empty/);
});

test('DEC_OSC invalid k', () => {
    // Test DEC_OSC fails with invalid k value
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // k = 0
    assert.throws(() => {
        wasm.dec_osc_js(data, 2, 0.0);
    }, /Invalid K/);
    
    // Negative k
    assert.throws(() => {
        wasm.dec_osc_js(data, 2, -1.0);
    }, /Invalid K/);
    
    // NaN k
    assert.throws(() => {
        wasm.dec_osc_js(data, 2, NaN);
    }, /Invalid K/);
});

test('DEC_OSC reinput', () => {
    // Test DEC_OSC using output as input - mirrors check_dec_osc_reinput
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.dec_osc_js(close, 50, 1.0);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass using first result as input
    const secondResult = wasm.dec_osc_js(firstResult, 50, 1.0);
    assert.strictEqual(secondResult.length, firstResult.length);
});

test('DEC_OSC NaN handling', () => {
    // Test DEC_OSC NaN handling
    const close = new Float64Array(testData.close);
    
    // Create data with some NaN values
    const dataWithNaN = new Float64Array(close);
    for (let i = 0; i < 5; i++) {
        dataWithNaN[i] = NaN;
    }
    
    const result = wasm.dec_osc_js(dataWithNaN, 10, 1.0);
    assert.strictEqual(result.length, dataWithNaN.length);
});

test('DEC_OSC batch calculation', async () => {
    // Test DEC_OSC batch calculation with parameter ranges
    const close = new Float64Array(testData.close);
    
    // Test batch calculation with parameter ranges
    const config = {
        hp_period_range: [100, 150, 25],
        k_range: [0.5, 1.5, 0.5]
    };
    
    const result = await wasm.dec_osc_batch(close, config);
    
    // Check result structure
    assert(result.values !== undefined, 'Result should have values');
    assert(result.combos !== undefined, 'Result should have combos');
    assert(result.rows !== undefined, 'Result should have rows');
    assert(result.cols !== undefined, 'Result should have cols');
    
    // Check dimensions
    const expectedCombinations = 3 * 3; // 3 periods * 3 k values
    assert.strictEqual(result.rows, expectedCombinations);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, expectedCombinations * close.length);
    assert.strictEqual(result.combos.length, expectedCombinations);
    
    // Verify parameter combinations
    const expectedPeriods = [100, 100, 100, 125, 125, 125, 150, 150, 150];
    const expectedKs = [0.5, 1.0, 1.5, 0.5, 1.0, 1.5, 0.5, 1.0, 1.5];
    
    for (let i = 0; i < expectedCombinations; i++) {
        assert.strictEqual(result.combos[i].hp_period, expectedPeriods[i]);
        assertClose(result.combos[i].k, expectedKs[i], 1e-10, `k value mismatch at index ${i}`);
    }
});

test('DEC_OSC edge cases', () => {
    // Test DEC_OSC with edge case inputs
    
    // Test with all same values
    const sameValues = new Float64Array(100).fill(50.0);
    const result1 = wasm.dec_osc_js(sameValues, 10, 1.0);
    assert.strictEqual(result1.length, sameValues.length);
    
    // Test with monotonically increasing values
    const increasing = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        increasing[i] = i;
    }
    const result2 = wasm.dec_osc_js(increasing, 10, 1.0);
    assert.strictEqual(result2.length, increasing.length);
    
    // Test with alternating values
    const alternating = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        alternating[i] = i % 2 === 0 ? 10.0 : 20.0;
    }
    const result3 = wasm.dec_osc_js(alternating, 10, 1.0);
    assert.strictEqual(result3.length, alternating.length);
});

test('DEC_OSC fast API (into)', () => {
    // Test the fast API with preallocated memory
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate memory
    const inPtr = wasm.dec_osc_alloc(len);
    const outPtr = wasm.dec_osc_alloc(len);
    
    try {
        // Copy input data to WASM memory
        const wasmMemory = new Float64Array(wasm.memory.buffer);
        wasmMemory.set(close, inPtr / 8);
        
        // Call fast API
        wasm.dec_osc_into(inPtr, outPtr, len, 125, 1.0);
        
        // Read result from WASM memory
        const result = new Float64Array(wasm.memory.buffer, outPtr, len);
        
        // Compare with safe API
        const safeResult = wasm.dec_osc_js(close, 125, 1.0);
        assertArrayClose(result, safeResult, 1e-15, "Fast API result mismatch");
        
    } finally {
        // Free allocated memory
        wasm.dec_osc_free(inPtr, len);
        wasm.dec_osc_free(outPtr, len);
    }
});

test('DEC_OSC fast API with aliasing', () => {
    // Test the fast API with in-place operation (aliasing)
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate memory
    const ptr = wasm.dec_osc_alloc(len);
    
    try {
        // Copy input data to WASM memory
        const wasmMemory = new Float64Array(wasm.memory.buffer);
        wasmMemory.set(close, ptr / 8);
        
        // Call fast API with same pointer for input and output (in-place)
        wasm.dec_osc_into(ptr, ptr, len, 125, 1.0);
        
        // Read result from WASM memory
        const result = new Float64Array(wasm.memory.buffer, ptr, len);
        
        // Compare with safe API
        const safeResult = wasm.dec_osc_js(close, 125, 1.0);
        assertArrayClose(result, safeResult, 1e-15, "Fast API aliasing result mismatch");
        
    } finally {
        // Free allocated memory
        wasm.dec_osc_free(ptr, len);
    }
});

test('DEC_OSC all NaN input', () => {
    // Test DEC_OSC with all NaN values
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.dec_osc_js(allNaN, 10, 1.0);
    }, /All values are NaN/);
});