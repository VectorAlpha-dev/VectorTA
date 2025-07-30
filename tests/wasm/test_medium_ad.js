/**
 * WASM binding tests for Medium AD indicator.
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

test('Medium AD partial params', () => {
    // Test with default parameters - mirrors check_medium_ad_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.medium_ad_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('Medium AD accuracy', async () => {
    // Test Medium AD matches expected values from Rust tests - mirrors check_medium_ad_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.medium_ad_js(close, 5);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const expected_last_five = [220.0, 78.5, 126.5, 48.0, 28.5];
    const last5 = result.slice(-5);
    
    for (let i = 0; i < 5; i++) {
        const diff = Math.abs(last5[i] - expected_last_five[i]);
        assert(diff < 1e-1, 
            `Medium AD mismatch at idx ${i}: got ${last5[i]}, expected ${expected_last_five[i]}`);
    }
});

test('Medium AD default candles', () => {
    // Test Medium AD with default parameters - mirrors check_medium_ad_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.medium_ad_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('Medium AD zero period', () => {
    // Test Medium AD fails with zero period - mirrors check_medium_ad_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.medium_ad_js(inputData, 0);
    }, /Invalid period/);
});

test('Medium AD period exceeds length', () => {
    // Test Medium AD fails when period exceeds data length - mirrors check_medium_ad_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.medium_ad_js(dataSmall, 10);
    }, /Invalid period/);
});

test('Medium AD very small dataset', () => {
    // Test Medium AD fails with insufficient data - mirrors check_medium_ad_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.medium_ad_js(singlePoint, 5);
    }, /Invalid period/);
});

test('Medium AD re-input', async () => {
    // Test Medium AD applied twice (re-input) - mirrors check_medium_ad_reinput
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.medium_ad_js(close, 5);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply Medium AD to Medium AD output
    const secondResult = wasm.medium_ad_js(firstResult, 5);
    assert.strictEqual(secondResult.length, firstResult.length);
});

test('Medium AD NaN handling', () => {
    // Test Medium AD handles NaN values correctly - mirrors check_medium_ad_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.medium_ad_js(close, 5);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First period-1 values should be NaN
    for (let i = 0; i < 4; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}, got ${result[i]}`);
    }
});

test('Medium AD fast/unsafe API', () => {
    // Test the fast API with pre-allocated memory
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate memory
    const outPtr = wasm.medium_ad_alloc(len);
    const inPtr = wasm.medium_ad_alloc(len);
    
    try {
        // Copy data to WASM memory
        const wasmMemory = new Float64Array(wasm.memory.buffer);
        const inOffset = inPtr / 8; // Convert byte offset to f64 offset
        wasmMemory.set(close, inOffset);
        
        // Compute Medium AD
        wasm.medium_ad_into(inPtr, outPtr, len, 5);
        
        // Read result
        const outOffset = outPtr / 8;
        const result = new Float64Array(wasm.memory.buffer, outPtr, len);
        
        // Verify result matches safe API
        const safeResult = wasm.medium_ad_js(close, 5);
        assertArrayClose(result, safeResult, 1e-10, "Fast API result mismatch");
        
    } finally {
        // Free memory
        wasm.medium_ad_free(inPtr, len);
        wasm.medium_ad_free(outPtr, len);
    }
});

test('Medium AD fast API with aliasing', () => {
    // Test the fast API handles in-place operations correctly
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate memory
    const ptr = wasm.medium_ad_alloc(len);
    
    try {
        // Copy data to WASM memory
        const wasmMemory = new Float64Array(wasm.memory.buffer);
        const offset = ptr / 8;
        wasmMemory.set(close, offset);
        
        // Compute Medium AD in-place
        wasm.medium_ad_into(ptr, ptr, len, 5); // Same pointer for input and output
        
        // Read result
        const result = new Float64Array(wasm.memory.buffer, ptr, len);
        
        // Verify result matches safe API
        const safeResult = wasm.medium_ad_js(close, 5);
        assertArrayClose(result, safeResult, 1e-10, "In-place operation result mismatch");
        
    } finally {
        // Free memory
        wasm.medium_ad_free(ptr, len);
    }
});

test('Medium AD batch API', () => {
    // Test batch processing
    const close = new Float64Array(testData.close);
    
    const config = {
        period_range: [3, 7, 1] // periods 3, 4, 5, 6, 7
    };
    
    const result = wasm.medium_ad_batch(close, config);
    
    assert(result.values, "Batch result should have values");
    assert(result.combos, "Batch result should have combos");
    assert(result.rows, "Batch result should have rows");
    assert(result.cols, "Batch result should have cols");
    
    assert.strictEqual(result.rows, 5, "Should have 5 parameter combinations");
    assert.strictEqual(result.cols, close.length, "Columns should match data length");
    assert.strictEqual(result.values.length, result.rows * result.cols, "Values array size mismatch");
    assert.strictEqual(result.combos.length, result.rows, "Combos array size mismatch");
    
    // Verify each combo has the expected period
    for (let i = 0; i < result.combos.length; i++) {
        assert.strictEqual(result.combos[i].period, i + 3, `Combo ${i} has wrong period`);
    }
    
    // Verify first row matches single calculation with period=3
    const singleResult = wasm.medium_ad_js(close, 3);
    const firstRow = result.values.slice(0, result.cols);
    assertArrayClose(firstRow, singleResult, 1e-10, "Batch first row mismatch");
});

test('Medium AD memory management', () => {
    // Test memory allocation and deallocation
    const len = 1000;
    
    // Test allocation
    const ptr = wasm.medium_ad_alloc(len);
    assert(ptr !== 0, "Should allocate non-null pointer");
    
    // Test that we can write to the allocated memory
    const wasmMemory = new Float64Array(wasm.memory.buffer);
    const offset = ptr / 8;
    for (let i = 0; i < len; i++) {
        wasmMemory[offset + i] = i * 1.5;
    }
    
    // Test that we can read back the values
    for (let i = 0; i < len; i++) {
        assert.strictEqual(wasmMemory[offset + i], i * 1.5, `Memory corruption at index ${i}`);
    }
    
    // Test deallocation
    wasm.medium_ad_free(ptr, len);
    
    // Test null pointer handling in free
    wasm.medium_ad_free(0, len); // Should not crash
});

test('Medium AD error handling', () => {
    // Test various error conditions
    
    // Test null pointer errors
    assert.throws(() => {
        wasm.medium_ad_into(0, 100, 10, 5);
    }, /Null pointer/);
    
    assert.throws(() => {
        wasm.medium_ad_into(100, 0, 10, 5);
    }, /Null pointer/);
    
    // Test invalid parameters
    const ptr = wasm.medium_ad_alloc(10);
    try {
        assert.throws(() => {
            wasm.medium_ad_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        assert.throws(() => {
            wasm.medium_ad_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.medium_ad_free(ptr, 10);
    }
});