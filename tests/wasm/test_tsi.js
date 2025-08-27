/**
 * WASM binding tests for TSI indicator.
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
    assertNoNaN
} from './test_utils.js';

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

test('TSI partial params', () => {
    // Test with default parameters - mirrors check_tsi_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.tsi_js(close, 25, 13);
    assert.strictEqual(result.length, close.length);
});

test('TSI accuracy', () => {
    // Test TSI matches expected values from Rust tests - mirrors check_tsi_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.tsi_js(close, 25, 13);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected last 5 values from Rust tests
    const expectedLastFive = [
        -17.757654061849838,
        -17.367527062626184,
        -17.305577681249513,
        -16.937565646991143,
        -17.61825617316731,
    ];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-7,
        "TSI last 5 values mismatch"
    );
});

test('TSI default candles', () => {
    // Test TSI with default parameters - mirrors check_tsi_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.tsi_js(close, 25, 13);
    assert.strictEqual(result.length, close.length);
});

test('TSI zero period', () => {
    // Test TSI fails with zero period - mirrors check_tsi_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.tsi_js(inputData, 0, 13);
    }, /Invalid period/);
    
    assert.throws(() => {
        wasm.tsi_js(inputData, 25, 0);
    }, /Invalid period/);
});

test('TSI period exceeds length', () => {
    // Test TSI fails when period exceeds data length - mirrors check_tsi_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.tsi_js(dataSmall, 25, 13);
    }, /Invalid period/);
});

test('TSI very small dataset', () => {
    // Test TSI fails with insufficient data - mirrors check_tsi_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.tsi_js(singlePoint, 25, 13);
    }, /Invalid period|Not enough valid data/);
});

test('TSI reinput', () => {
    // Test TSI applied twice (re-input) - mirrors check_tsi_reinput
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.tsi_js(close, 25, 13);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply TSI to TSI output
    const secondResult = wasm.tsi_js(firstResult, 25, 13);
    assert.strictEqual(secondResult.length, firstResult.length);
});

test('TSI NaN handling', () => {
    // Test TSI handles NaN values correctly - mirrors check_tsi_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.tsi_js(close, 25, 13);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        const afterWarmup = result.slice(240);
        assertNoNaN(afterWarmup, "Found unexpected NaN after warmup period");
    }
    
    // First values should be NaN during warmup
    // Warmup = first_valid + long + short
    const warmup = 25 + 13; // Assuming first valid at 0
    assertAllNaN(result.slice(0, warmup), `Expected NaN in warmup period (first ${warmup} values)`);
});

test('TSI all NaN input', () => {
    // Test TSI with all NaN values
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.tsi_js(allNaN, 25, 13);
    }, /All values are NaN/);
});

test('TSI batch processing', () => {
    // Test TSI batch processing - mirrors check_batch_default_row
    const close = new Float64Array(testData.close);
    
    const config = {
        long_period_range: [25, 25, 0],   // Default long period only
        short_period_range: [13, 13, 0]   // Default short period only
    };
    
    const result = wasm.tsi_batch(close, config);
    
    assert(result.values, 'Result should have values');
    assert(result.combos, 'Result should have combos');
    assert.strictEqual(result.rows, 1, 'Should have 1 combination');
    assert.strictEqual(result.cols, close.length, 'Columns should match input length');
    
    // Extract the single row
    const defaultRow = result.values.slice(0, close.length);
    const expectedLastFive = [
        -17.757654061849838,
        -17.367527062626184,
        -17.305577681249513,
        -16.937565646991143,
        -17.61825617316731,
    ];
    
    // Check last 5 values match
    assertArrayClose(
        defaultRow.slice(-5),
        expectedLastFive,
        1e-7,
        "TSI batch default row mismatch"
    );
});

test('TSI batch multiple params', () => {
    // Test TSI batch with multiple parameter combinations
    const close = new Float64Array(testData.close);
    
    const config = {
        long_period_range: [20, 30, 5],  // 20, 25, 30
        short_period_range: [10, 15, 5]  // 10, 15
    };
    
    const result = wasm.tsi_batch(close, config);
    
    assert(result.values, 'Result should have values');
    assert(result.combos, 'Result should have combos');
    assert.strictEqual(result.rows, 6, 'Should have 6 combinations (3*2)');
    assert.strictEqual(result.cols, close.length, 'Columns should match input length');
    
    // Check parameter combinations
    const expectedCombos = [
        { long_period: 20, short_period: 10 },
        { long_period: 20, short_period: 15 },
        { long_period: 25, short_period: 10 },
        { long_period: 25, short_period: 15 },
        { long_period: 30, short_period: 10 },
        { long_period: 30, short_period: 15 }
    ];
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].long_period, expectedCombos[i].long_period,
            `Combo ${i} long_period mismatch`);
        assert.strictEqual(result.combos[i].short_period, expectedCombos[i].short_period,
            `Combo ${i} short_period mismatch`);
    }
});

test('TSI mid-series NaN', () => {
    // Test TSI handles mid-series NaN values correctly
    const data = new Float64Array([
        100.0, 102.0, 101.0, 103.0, 104.0,
        NaN, NaN,  // Mid-series NaN gap
        105.0, 106.0, 107.0, 108.0, 109.0,
        110.0, 111.0, 112.0, 113.0, 114.0,
        115.0, 116.0, 117.0
    ]);
    
    // TSI should handle the gap and continue after
    const result = wasm.tsi_js(data, 5, 3);
    assert.strictEqual(result.length, data.length);
    
    // Should have NaN during gap
    assert(isNaN(result[5]), 'Should have NaN at gap position 5');
    assert(isNaN(result[6]), 'Should have NaN at gap position 6');
    
    // Should recover after gap (with new warmup)
    const validAfterGap = result.slice(10);
    const validCount = validAfterGap.filter(v => !isNaN(v)).length;
    assert(validCount > 0, "TSI should recover after mid-series NaN gap");
});

test('TSI constant data', () => {
    // Test TSI with constant data (zero momentum)
    const constant = new Float64Array(50).fill(100.0);
    const result = wasm.tsi_js(constant, 10, 5);
    
    // All values after warmup should be NaN (momentum is 0)
    const warmup = 10 + 5;
    const afterWarmup = result.slice(warmup);
    assertAllNaN(afterWarmup, "TSI should be NaN for constant prices");
});

test('TSI step data', () => {
    // Test TSI with step function data
    const step1 = new Float64Array(25).fill(100.0);
    const step2 = new Float64Array(25).fill(150.0);
    const data = new Float64Array([...step1, ...step2]);
    
    const result = wasm.tsi_js(data, 10, 5);
    assert.strictEqual(result.length, data.length);
    
    // Should detect the momentum change
    const lastValues = result.slice(-5);
    const validLast = lastValues.filter(v => !isNaN(v));
    
    if (validLast.length > 0) {
        // After initial jump momentum, should trend back toward 0
        for (const val of validLast) {
            assert(val >= -100 && val <= 100,
                `TSI value ${val} should be in [-100, 100] range`);
        }
    }
});

test('TSI into (in-place)', () => {
    // Test TSI in-place computation
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate memory for output
    const outPtr = wasm.tsi_alloc(len);
    const inPtr = wasm.tsi_alloc(len);
    
    try {
        // Copy data to WASM memory
        const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer);
        const inIndex = inPtr / 8;
        for (let i = 0; i < len; i++) {
            wasmMemory[inIndex + i] = close[i];
        }
        
        // Compute TSI in-place
        wasm.tsi_into(inPtr, outPtr, len, 25, 13);
        
        // Read results
        const outIndex = outPtr / 8;
        const result = new Float64Array(len);
        for (let i = 0; i < len; i++) {
            result[i] = wasmMemory[outIndex + i];
        }
        
        // Compare with regular function
        const expected = wasm.tsi_js(close, 25, 13);
        assertArrayClose(result, expected, 1e-10, "TSI into mismatch");
        
    } finally {
        // Clean up
        wasm.tsi_free(inPtr, len);
        wasm.tsi_free(outPtr, len);
    }
});

test('TSI batch into (in-place)', () => {
    // Test TSI batch in-place computation
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Single combination for simplicity
    const rows = 1;
    const totalSize = rows * len;
    
    // Allocate memory
    const outPtr = wasm.tsi_alloc(totalSize);
    const inPtr = wasm.tsi_alloc(len);
    
    try {
        // Copy data to WASM memory
        const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer);
        const inIndex = inPtr / 8;
        for (let i = 0; i < len; i++) {
            wasmMemory[inIndex + i] = close[i];
        }
        
        // Compute batch TSI in-place
        const resultRows = wasm.tsi_batch_into(
            inPtr, outPtr, len,
            25, 25, 0,  // long_period range
            13, 13, 0   // short_period range
        );
        
        assert.strictEqual(resultRows, rows, 'Should return correct number of rows');
        
        // Read results
        const outIndex = outPtr / 8;
        const result = new Float64Array(len);
        for (let i = 0; i < len; i++) {
            result[i] = wasmMemory[outIndex + i];
        }
        
        // Compare with regular function
        const expected = wasm.tsi_js(close, 25, 13);
        assertArrayClose(result, expected, 1e-10, "TSI batch into mismatch");
        
    } finally {
        // Clean up
        wasm.tsi_free(inPtr, len);
        wasm.tsi_free(outPtr, totalSize);
    }
});

test('TSI edge cases', () => {
    // Minimum valid data
    const minData = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    const result1 = wasm.tsi_js(minData, 3, 2);
    assert.strictEqual(result1.length, minData.length);
    
    // Long period equals short period
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = 100 + Math.random() * 20 - 10;
    }
    const result2 = wasm.tsi_js(data, 10, 10);
    assert.strictEqual(result2.length, data.length);
    
    // Very small periods
    const result3 = wasm.tsi_js(data, 2, 1);
    assert.strictEqual(result3.length, data.length);
});