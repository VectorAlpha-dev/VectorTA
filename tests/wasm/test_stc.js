/**
 * WASM binding tests for STC indicator.
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

test('STC default params', () => {
    // Test with default parameters
    const close = new Float64Array(testData.close);
    
    // Default params: fast=23, slow=50, k=10, d=3, fast_ma_type="ema", slow_ma_type="ema"
    const result = wasm.stc_js(close, 23, 50, 10, 3, "ema", "ema");
    assert.strictEqual(result.length, close.length);
    
    // Check that we have some valid values after warmup
    const warmup = 50; // max of default periods
    let hasValidValues = false;
    for (let i = warmup; i < result.length; i++) {
        if (!isNaN(result[i])) {
            hasValidValues = true;
            break;
        }
    }
    assert(hasValidValues, "Expected some valid values after warmup period");
});

test('STC with custom params', () => {
    // Test STC with custom parameters
    const close = new Float64Array(testData.close);
    
    const result = wasm.stc_js(close, 12, 26, 9, 3, "sma", "sma");
    assert.strictEqual(result.length, close.length);
});

test('STC accuracy', async () => {
    // Test STC accuracy with expected values if available
    const close = new Float64Array(testData.close);
    
    const result = wasm.stc_js(close, 23, 50, 10, 3, "ema", "ema");
    assert.strictEqual(result.length, close.length);
    
    // Check range - STC should be between 0 and 100
    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i])) {
            assert(result[i] >= -0.1, `STC value ${result[i]} should be >= 0`);
            assert(result[i] <= 100.1, `STC value ${result[i]} should be <= 100`);
        }
    }
    
    // Compare full output with Rust if expected values exist
    if (EXPECTED_OUTPUTS.stc) {
        const expected = EXPECTED_OUTPUTS.stc;
        const last5 = result.slice(-5);
        assertArrayClose(
            last5,
            expected.last5Values,
            1e-6,
            "STC last 5 values mismatch"
        );
        await compareWithRust('stc', result, 'close', expected.defaultParams);
    }
});

test('STC zero period', () => {
    // Test STC fails with zero period
    const inputData = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    assert.throws(() => {
        wasm.stc_js(inputData, 0, 50, 10, 3, "ema", "ema");
    }, /Not enough valid data|Invalid|Empty/);
    
    assert.throws(() => {
        wasm.stc_js(inputData, 23, 0, 10, 3, "ema", "ema");
    }, /Not enough valid data|Invalid|Empty/);
});

test('STC period exceeds length', () => {
    // Test STC fails when period exceeds data length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.stc_js(dataSmall, 10, 50, 10, 3, "ema", "ema");
    }, /Not enough valid data/);
});

test('STC empty data', () => {
    // Test STC fails with empty data
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.stc_js(empty, 23, 50, 10, 3, "ema", "ema");
    }, /Empty data/);
});

test('STC all NaN', () => {
    // Test STC handles all NaN input
    const allNaN = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        allNaN[i] = NaN;
    }
    
    assert.throws(() => {
        wasm.stc_js(allNaN, 23, 50, 10, 3, "ema", "ema");
    }, /All values are NaN/);
});

test('STC NaN handling', () => {
    // Test STC handles NaN values correctly
    const close = new Float64Array(testData.close);
    
    // Insert some NaN values
    for (let i = 10; i < 20; i++) {
        close[i] = NaN;
    }
    
    const result = wasm.stc_js(close, 23, 50, 10, 3, "ema", "ema");
    assert.strictEqual(result.length, close.length);
    
    // Should still produce some valid values after the NaN section
    let hasValidValues = false;
    for (let i = 100; i < result.length; i++) {
        if (!isNaN(result[i])) {
            hasValidValues = true;
            break;
        }
    }
    assert(hasValidValues, "Expected some valid values after NaN section");
});

test('STC fast API', () => {
    // Test the fast API with pre-allocated memory
    const close = new Float64Array(testData.close.slice(0, 500));
    const len = close.length;
    
    // Allocate input/output buffers in WASM memory
    const inPtr = wasm.stc_alloc(len);
    const outPtr = wasm.stc_alloc(len);
    assert(inPtr !== 0, 'Failed to allocate input buffer');
    assert(outPtr !== 0, 'Failed to allocate output buffer');
    
    try {
        // Copy input into WASM memory
        const memory = wasm.__wasm.memory.buffer;
        new Float64Array(memory, inPtr, len).set(close);
        
        // Call fast API
        wasm.stc_into(
            inPtr,
            outPtr,
            len,
            23, 50, 10, 3,
            "ema", "ema"
        );
        
        // Read results
        const memory2 = wasm.__wasm.memory.buffer;
        const result = new Float64Array(memory2, outPtr, len);
        
        assert.strictEqual(result.length, len);
        
        // Compare against the standard API
        const regular = wasm.stc_js(close, 23, 50, 10, 3, "ema", "ema");
        for (let i = 0; i < len; i++) {
            if (isNaN(regular[i]) && isNaN(result[i])) continue;
            assertClose(result[i], regular[i], 1e-10, `stc_into mismatch at ${i}`);
        }
    } finally {
        // Free allocated memory
        wasm.stc_free(inPtr, len);
        wasm.stc_free(outPtr, len);
    }
});

test('STC fast API in-place', () => {
    // Test the fast API with in-place operation (aliasing)
    const close = new Float64Array(testData.close.slice(0, 500));
    const len = close.length;
    
    const inOutPtr = wasm.stc_alloc(len);
    assert(inOutPtr !== 0, 'Failed to allocate in/out buffer');

    try {
        const regular = wasm.stc_js(close, 23, 50, 10, 3, "ema", "ema");

        const memory = wasm.__wasm.memory.buffer;
        new Float64Array(memory, inOutPtr, len).set(close);

        // Call fast API with same pointer for input and output
        wasm.stc_into(inOutPtr, inOutPtr, len, 23, 50, 10, 3, "ema", "ema");

        const memory2 = wasm.__wasm.memory.buffer;
        const result = new Float64Array(memory2, inOutPtr, len);

        for (let i = 0; i < len; i++) {
            if (isNaN(regular[i]) && isNaN(result[i])) continue;
            assertClose(result[i], regular[i], 1e-10, `stc_into aliasing mismatch at ${i}`);
        }
    } finally {
        wasm.stc_free(inOutPtr, len);
    }
});

test('STC batch processing', async () => {
    // Test STC batch processing
    const close = new Float64Array(testData.close);
    
    const config = {
        fast_period_range: [20, 30, 5],    // 20, 25, 30
        slow_period_range: [45, 55, 5],    // 45, 50, 55
        k_period_range: [10, 10, 1],       // 10 only
        d_period_range: [3, 3, 1]          // 3 only
    };
    
    const result = wasm.stc_batch(close, config);
    
    assert(result.values, "Expected values in result");
    assert(result.combos, "Expected combos in result");
    assert(result.rows, "Expected rows in result");
    assert(result.cols, "Expected cols in result");
    
    // Should have 3 * 3 * 1 * 1 = 9 combinations
    assert.strictEqual(result.rows, 9);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, 9 * close.length);
    assert.strictEqual(result.combos.length, 9);
    
    // Verify first combination matches single calculation
    const firstRow = result.values.slice(0, close.length);
    const singleResult = wasm.stc_js(close, 20, 45, 10, 3, "ema", "ema");
    
    assertArrayClose(
        firstRow,
        singleResult,
        1e-10,
        "Batch first row vs single calculation mismatch"
    );
});

test('STC batch single param', () => {
    // Test STC batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    const config = {
        fast_period_range: [23, 23, 0],
        slow_period_range: [50, 50, 0],
        k_period_range: [10, 10, 0],
        d_period_range: [3, 3, 0]
    };
    
    const result = wasm.stc_batch(close, config);
    
    // Should have exactly 1 combination
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    
    // Should match single calculation
    const singleResult = wasm.stc_js(close, 23, 50, 10, 3, "ema", "ema");
    assertArrayClose(
        result.values,
        singleResult,
        1e-10,
        "Batch single param vs single calculation mismatch"
    );
});
