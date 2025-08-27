/**
 * WASM binding tests for IFT RSI indicator.
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

test('IFT RSI partial params', () => {
    // Test with defaults (rsi_period=5, wma_period=9) - mirrors check_ift_rsi_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.ift_rsi_js(close, 5, 9);
    assert.strictEqual(result.length, close.length);
});

test('IFT RSI accuracy', () => {
    // Test IFT RSI matches expected values from Rust tests - mirrors check_ift_rsi_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.ift_rsi_js(close, 5, 9);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const expectedLastFive = [
        -0.35919800205778424,
        -0.3275464113984847,
        -0.39970276998138216,
        -0.36321812798797737,
        -0.5843346528346959,
    ];
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-8,
        "IFT RSI last 5 values mismatch"
    );
});

test('IFT RSI default candles', () => {
    // Test IFT RSI with default parameters - mirrors check_ift_rsi_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.ift_rsi_js(close, 5, 9);
    assert.strictEqual(result.length, close.length);
});

test('IFT RSI zero period', () => {
    // Test IFT RSI fails with zero period - mirrors check_ift_rsi_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.ift_rsi_js(inputData, 0, 9);
    }, /Invalid/);
});

test('IFT RSI period exceeds length', () => {
    // Test IFT RSI fails when period exceeds data length - mirrors check_ift_rsi_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.ift_rsi_js(dataSmall, 10, 9);
    }, /Invalid/);
});

test('IFT RSI very small dataset', () => {
    // Test IFT RSI fails with insufficient data - mirrors check_ift_rsi_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.ift_rsi_js(singlePoint, 5, 9);
    }, /Invalid/);
});

test('IFT RSI reinput', () => {
    // Test IFT RSI applied twice (re-input) - mirrors check_ift_rsi_reinput
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.ift_rsi_js(close, 5, 9);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply IFT RSI to IFT RSI output
    const secondResult = wasm.ift_rsi_js(firstResult, 5, 9);
    assert.strictEqual(secondResult.length, firstResult.length);
});

test('IFT RSI NaN handling', () => {
    // Test IFT RSI handles NaN values correctly - mirrors check_ift_rsi_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.ift_rsi_js(close, 5, 9);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        let nonNanCount = 0;
        for (let i = 240; i < result.length; i++) {
            if (!isNaN(result[i])) nonNanCount++;
        }
        assert.strictEqual(nonNanCount, result.length - 240, "Found unexpected NaN values after warmup");
    }
});

test('IFT RSI empty input', () => {
    // Test IFT RSI fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.ift_rsi_js(empty, 5, 9);
    }, /Input data slice is empty/);
});

test('IFT RSI all NaN input', () => {
    // Test IFT RSI with all NaN values
    const allNan = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.ift_rsi_js(allNan, 5, 9);
    }, /All values are NaN/);
});

test('IFT RSI fast API (in-place)', () => {
    // Test fast API with in-place operation (aliasing)
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate memory
    const ptr = wasm.ift_rsi_alloc(len);
    assert(ptr !== 0, "Failed to allocate memory");
    
    try {
        // Copy data to allocated memory
        const mem = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        mem.set(close);
        
        // Perform in-place operation (input and output are the same)
        wasm.ift_rsi_into(ptr, ptr, len, 5, 9);
        
        // Recreate view in case memory grew during operation
        const mem2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        
        // Check result
        assert.strictEqual(mem2.length, len);
        
        // Values should be bounded between -1 and 1 (IFT output range)
        let validCount = 0;
        for (let i = 0; i < len; i++) {
            if (!isNaN(mem2[i])) {
                assert(mem2[i] >= -1.0 && mem2[i] <= 1.0, 
                       `Value ${mem2[i]} at index ${i} out of IFT range [-1, 1]`);
                validCount++;
            }
        }
        assert(validCount > 0, "No valid values produced");
        
    } finally {
        // Always free allocated memory
        wasm.ift_rsi_free(ptr, len);
    }
});

test('IFT RSI fast API (separate buffers)', () => {
    // Test fast API with separate input/output buffers
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate separate buffers
    const inPtr = wasm.ift_rsi_alloc(len);
    const outPtr = wasm.ift_rsi_alloc(len);
    
    assert(inPtr !== 0, "Failed to allocate input memory");
    assert(outPtr !== 0, "Failed to allocate output memory");
    assert(inPtr !== outPtr, "Pointers should be different");
    
    try {
        // Copy input data
        const inMem = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inMem.set(close);
        
        // Perform operation
        wasm.ift_rsi_into(inPtr, outPtr, len, 5, 9);
        
        // Compare with safe API result
        const safeResult = wasm.ift_rsi_js(close, 5, 9);
        
        // IMPORTANT: Recreate the output view after calling safe API, as memory may have grown
        const finalOutMem = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        assertArrayClose(finalOutMem, safeResult, 1e-10, "Fast API result differs from safe API");
        
    } finally {
        // Free both buffers
        wasm.ift_rsi_free(inPtr, len);
        wasm.ift_rsi_free(outPtr, len);
    }
});

test('IFT RSI batch operation', () => {
    // Test batch processing with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    const config = {
        rsi_period_range: [5, 7, 1],    // 5, 6, 7
        wma_period_range: [9, 10, 1]    // 9, 10
    };
    
    const result = wasm.ift_rsi_batch(close, config);
    
    // Should have 3 * 2 = 6 combinations
    assert.strictEqual(result.rows, 6);
    assert.strictEqual(result.cols, 100);
    assert.strictEqual(result.values.length, 600);
    assert.strictEqual(result.combos.length, 6);
    
    // Verify first combination
    assert.strictEqual(result.combos[0].rsi_period, 5);
    assert.strictEqual(result.combos[0].wma_period, 9);
    
    // Check that values are bounded correctly
    for (let i = 0; i < result.values.length; i++) {
        if (!isNaN(result.values[i])) {
            assert(result.values[i] >= -1.0 && result.values[i] <= 1.0,
                   `Batch value ${result.values[i]} at index ${i} out of range`);
        }
    }
});

test('IFT RSI batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    const config = {
        rsi_period_range: [5, 5, 0],
        wma_period_range: [9, 9, 0]
    };
    
    const result = wasm.ift_rsi_batch(close, config);
    
    // Should have 1 combination
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    
    // Extract the single row and compare with regular calculation
    const batchRow = result.values;
    const singleResult = wasm.ift_rsi_js(close, 5, 9);
    
    assertArrayClose(batchRow, singleResult, 1e-10, 
                    "Batch result with single params differs from single calculation");
});

test('IFT RSI warmup period', () => {
    // Test IFT RSI warmup period calculation is correct
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.iftRsi;
    
    const result = wasm.ift_rsi_js(
        close,
        expected.defaultParams.rsiPeriod,
        expected.defaultParams.wmaPeriod
    );
    
    // Check warmup period
    // The warmup should be first + rsi_period + wma_period - 1
    // For data starting at index 0, warmup = 0 + 5 + 9 - 1 = 13
    const warmup = expected.warmupPeriod;
    
    // All values before warmup should be NaN
    for (let i = 0; i < Math.min(warmup, result.length); i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup, got ${result[i]}`);
    }
    
    // First non-NaN should be at warmup index
    if (warmup < result.length) {
        assert(!isNaN(result[warmup]), `Expected valid value at index ${warmup}, got NaN`);
    }
});

test('IFT RSI boundary values', () => {
    // Test IFT RSI with boundary parameter values
    // Minimum valid data for rsi_period=2, wma_period=2
    const minData = new Float64Array([100.0, 101.0, 102.0, 103.0, 104.0]);
    
    // Test minimum periods
    const result = wasm.ift_rsi_js(minData, 2, 2);
    assert.strictEqual(result.length, minData.length);
    
    // Test with larger periods but still valid
    const largeData = new Float64Array(200);
    for (let i = 0; i < 200; i++) {
        largeData[i] = 100 + Math.random() * 10;
    }
    const largeResult = wasm.ift_rsi_js(largeData, 50, 50);
    assert.strictEqual(largeResult.length, largeData.length);
    
    // Check warmup for large periods
    const warmupLarge = 0 + 50 + 50 - 1;  // 99
    for (let i = 0; i < Math.min(warmupLarge, largeResult.length); i++) {
        assert(isNaN(largeResult[i]), `Expected NaN during warmup at ${i}`);
    }
});

test('IFT RSI output bounds validation', () => {
    // Test IFT RSI output is bounded to [-1, 1] range
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.iftRsi;
    
    // Test various parameter combinations
    for (const params of expected.parameterCombinations) {
        const result = wasm.ift_rsi_js(
            close,
            params.rsiPeriod,
            params.wmaPeriod
        );
        
        // Check all non-NaN values are in [-1, 1]
        for (let i = 0; i < result.length; i++) {
            if (!isNaN(result[i])) {
                assert(result[i] >= -1.0, 
                       `Value ${result[i]} at index ${i} < -1 with params ${JSON.stringify(params)}`);
                assert(result[i] <= 1.0,
                       `Value ${result[i]} at index ${i} > 1 with params ${JSON.stringify(params)}`);
            }
        }
    }
});

test('IFT RSI batch_into pointer API', () => {
    // Test the batch_into pointer-based API
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;
    
    // Test parameters
    const rsiStart = 5, rsiEnd = 7, rsiStep = 1;  // 5, 6, 7
    const wmaStart = 9, wmaEnd = 10, wmaStep = 1; // 9, 10
    
    // Calculate expected rows
    const expectedRows = 3 * 2;  // 6 combinations
    
    // Allocate memory for input and output
    const inPtr = wasm.ift_rsi_alloc(len);
    const outPtr = wasm.ift_rsi_alloc(expectedRows * len);
    
    assert(inPtr !== 0, "Failed to allocate input memory");
    assert(outPtr !== 0, "Failed to allocate output memory");
    
    try {
        // Copy input data
        const inMem = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inMem.set(close);
        
        // Call batch_into
        const rows = wasm.ift_rsi_batch_into(
            inPtr, outPtr, len,
            rsiStart, rsiEnd, rsiStep,
            wmaStart, wmaEnd, wmaStep
        );
        
        assert.strictEqual(rows, expectedRows, "Unexpected number of rows");
        
        // Verify output
        const outMem = new Float64Array(wasm.__wasm.memory.buffer, outPtr, rows * len);
        
        // Check first combination matches single calculation
        const firstRow = new Float64Array(outMem.buffer, outMem.byteOffset, len);
        const singleResult = wasm.ift_rsi_js(close, rsiStart, wmaStart);
        
        assertArrayClose(firstRow, singleResult, 1e-10,
                        "batch_into first row differs from single calculation");
        
    } finally {
        // Always free allocated memory
        wasm.ift_rsi_free(inPtr, len);
        wasm.ift_rsi_free(outPtr, expectedRows * len);
    }
});

test('IFT RSI batch warmup periods', () => {
    // Test batch operations have correct warmup periods
    const close = new Float64Array(testData.close.slice(0, 200));
    
    const config = {
        rsi_period_range: [3, 5, 1],    // 3, 4, 5
        wma_period_range: [7, 9, 1]     // 7, 8, 9
    };
    
    const result = wasm.ift_rsi_batch(close, config);
    
    // Check each combination has correct warmup
    for (let idx = 0; idx < result.combos.length; idx++) {
        const rsiP = result.combos[idx].rsi_period;
        const wmaP = result.combos[idx].wma_period;
        const warmup = 0 + rsiP + wmaP - 1;  // first=0 for clean data
        
        // Extract the row
        const rowStart = idx * result.cols;
        const row = result.values.slice(rowStart, rowStart + result.cols);
        
        // Check NaN pattern matches warmup
        for (let i = 0; i < Math.min(warmup, row.length); i++) {
            assert(isNaN(row[i]), 
                   `Expected NaN at ${i} for combo ${idx} (rsi=${rsiP}, wma=${wmaP})`);
        }
        
        // First valid should be at warmup index
        if (warmup < row.length) {
            assert(!isNaN(row[warmup]), 
                   `Expected valid at ${warmup} for combo ${idx}`);
        }
    }
});