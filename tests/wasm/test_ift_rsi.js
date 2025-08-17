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
        -0.27763026899967286,
        -0.367418234207824,
        -0.1650156844504996,
        -0.26631220621545837,
        0.28324385010826775,
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