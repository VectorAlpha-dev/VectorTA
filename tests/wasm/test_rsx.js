/**
 * WASM binding tests for RSX indicator.
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

test('RSX partial params', () => {
    // Test with default parameters - mirrors check_rsx_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.rsx_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('RSX accuracy', async () => {
    // Test RSX matches expected values from Rust tests - mirrors check_rsx_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.rsx;
    
    const result = wasm.rsx_js(
        close,
        expected.default_params.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last_5_values,
        0.1,  // RSX uses 1e-1 tolerance in Python tests
        "RSX last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('rsx', result, 'close', expected.default_params);
});

test('RSX default candles', () => {
    // Test RSX with default parameters - mirrors check_rsx_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.rsx_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('RSX zero period', () => {
    // Test RSX fails with zero period - mirrors check_rsx_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.rsx_js(inputData, 0);
    }, /Invalid period/);
});

test('RSX period exceeds length', () => {
    // Test RSX fails when period exceeds data length - mirrors check_rsx_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.rsx_js(dataSmall, 10);
    }, /Invalid period/);
});

test('RSX all NaN', () => {
    // Test RSX fails on all NaN input
    const allNan = new Float64Array(10).fill(NaN);
    
    assert.throws(() => {
        wasm.rsx_js(allNan, 3);
    }, /All values are NaN/);
});

test('RSX NaN handling', () => {
    // Test RSX handles NaN values correctly - mirrors check_rsx_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.rsx_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    // Check no unexpected NaN values after warmup period
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
        }
    }
});

test('RSX empty input', () => {
    // Test RSX handles empty input - mirrors check_rsx_empty_data
    const emptyData = new Float64Array([]);
    
    assert.throws(() => {
        wasm.rsx_js(emptyData, 14);
    }, /All values are NaN|Invalid period|empty/i);
});

test('RSX fast API (rsx_into)', () => {
    // Test the fast API with pre-allocated memory
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate output buffer
    const outPtr = wasm.rsx_alloc(len);
    const inPtr = wasm.rsx_alloc(len);
    
    try {
        // Copy data to WASM memory
        const memory = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        memory.set(close);
        
        // Call fast API
        wasm.rsx_into(inPtr, outPtr, len, 14);
        
        // Read results
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const resultCopy = new Float64Array(result);
        
        // Compare with safe API
        const safeResult = wasm.rsx_js(close, 14);
        assertArrayClose(resultCopy, safeResult, 1e-10, "Fast API mismatch with safe API");
    } finally {
        // Clean up
        wasm.rsx_free(inPtr, len);
        wasm.rsx_free(outPtr, len);
    }
});

test('RSX fast API in-place', () => {
    // Test the fast API with in-place operation (aliasing)
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate single buffer for both input and output
    const ptr = wasm.rsx_alloc(len);
    
    try {
        // Copy data to WASM memory
        const memory = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        memory.set(close);
        
        // Call fast API with same pointer for input and output
        wasm.rsx_into(ptr, ptr, len, 14);
        
        // Read results
        const result = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        const resultCopy = new Float64Array(result);
        
        // Compare with safe API
        const safeResult = wasm.rsx_js(close, 14);
        assertArrayClose(resultCopy, safeResult, 1e-10, "In-place operation mismatch");
    } finally {
        // Clean up
        wasm.rsx_free(ptr, len);
    }
});

test('RSX memory allocation/deallocation', () => {
    // Test memory management functions
    const len = 1000;
    
    // Test allocation
    const ptr = wasm.rsx_alloc(len);
    assert(ptr !== 0, "Allocation returned null pointer");
    
    // Write some data
    const memory = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
    for (let i = 0; i < len; i++) {
        memory[i] = i * 1.5;
    }
    
    // Read it back
    assert.strictEqual(memory[0], 0);
    assert.strictEqual(memory[10], 15);
    
    // Free memory
    wasm.rsx_free(ptr, len);
    
    // Allocation after free should work
    const ptr2 = wasm.rsx_alloc(len);
    assert(ptr2 !== 0, "Re-allocation failed");
    wasm.rsx_free(ptr2, len);
});

test('RSX null pointer handling', () => {
    // Test fast API with null pointers
    assert.throws(() => {
        wasm.rsx_into(0, 0, 100, 14);
    }, /null pointer/);
});

test('RSX batch API', () => {
    // Test batch processing with multiple periods
    const close = new Float64Array(testData.close);
    
    const config = {
        period_range: [10, 20, 5]  // 10, 15, 20
    };
    
    const result = wasm.rsx_batch(close, config);
    
    assert(result.values, "Batch result should have values");
    assert(result.combos, "Batch result should have combos");
    assert.strictEqual(result.rows, 3, "Should have 3 parameter combinations");
    assert.strictEqual(result.cols, close.length, "Columns should match data length");
    assert.strictEqual(result.values.length, result.rows * result.cols, "Values array size mismatch");
    assert.strictEqual(result.combos.length, 3, "Should have 3 parameter combinations");
    
    // Check that combos match expected periods
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[1].period, 15);
    assert.strictEqual(result.combos[2].period, 20);
});

test('RSX batch fast API', () => {
    // Test fast batch API
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Calculate expected output size
    const periodStart = 10, periodEnd = 20, periodStep = 5;
    const expectedRows = Math.floor((periodEnd - periodStart) / periodStep) + 1; // 3 rows
    const outputSize = expectedRows * len;
    
    // Allocate buffers
    const inPtr = wasm.rsx_alloc(len);
    const outPtr = wasm.rsx_alloc(outputSize);
    
    try {
        // Copy data to WASM memory
        const memory = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        memory.set(close);
        
        // Call batch fast API
        const rows = wasm.rsx_batch_into(
            inPtr, outPtr, len,
            periodStart, periodEnd, periodStep
        );
        
        assert.strictEqual(rows, expectedRows, "Row count mismatch");
        
        // Read results
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, outputSize);
        
        // Verify first few values are NaN (warmup period)
        assert(isNaN(result[0]), "First value should be NaN");
        
        // Verify we got data for all combinations
        assert(result.length === outputSize, "Output size mismatch");
    } finally {
        // Clean up
        wasm.rsx_free(inPtr, len);
        wasm.rsx_free(outPtr, outputSize);
    }
});