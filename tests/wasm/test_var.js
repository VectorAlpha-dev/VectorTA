/**
 * WASM binding tests for VAR indicator.
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

test('VAR partial params', () => {
    // Test with default parameters - mirrors check_var_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.var_js(close, 14, 1.0);
    assert.strictEqual(result.length, close.length);
});

test('VAR accuracy', async () => {
    // Test VAR matches expected values from Rust tests - mirrors check_var_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.var;
    
    const result = wasm.var_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.nbdev
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        0.1,  // VAR values are large, so use larger tolerance
        "VAR last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('var', result, 'close', expected.defaultParams);
});

test('VAR default candles', () => {
    // Test VAR with default parameters - mirrors check_var_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.var_js(close, 14, 1.0);
    assert.strictEqual(result.length, close.length);
});

test('VAR zero period', () => {
    // Test VAR fails with zero period - mirrors check_var_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.var_js(inputData, 0, 1.0);
    }, /Invalid period/);
});

test('VAR period exceeds length', () => {
    // Test VAR fails when period exceeds data length - mirrors check_var_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.var_js(dataSmall, 10, 1.0);
    }, /Invalid period/);
});

test('VAR very small dataset', () => {
    // Test VAR fails with insufficient data - mirrors check_var_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.var_js(singlePoint, 14, 1.0);
    }, /Invalid period/);
});

test('VAR empty input', () => {
    // Test VAR fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.var_js(empty, 14, 1.0);
    }, /empty/i);
});

test('VAR all NaN', () => {
    // Test VAR fails with all NaN
    const nanData = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.var_js(nanData, 2, 1.0);
    }, /All values are NaN/);
});

test('VAR invalid nbdev', () => {
    // Test VAR fails with invalid nbdev
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    assert.throws(() => {
        wasm.var_js(data, 2, NaN);
    }, /nbdev is NaN/);
    
    assert.throws(() => {
        wasm.var_js(data, 2, Infinity);
    }, /nbdev is NaN/);
});

test('VAR NaN handling', () => {
    // Test VAR handles NaN values correctly - mirrors check_var_nan_handling
    const close = new Float64Array(testData.close);
    const period = 14;
    
    const result = wasm.var_js(close, period, 1.0);
    
    // First period-1 values should be NaN
    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    // After warmup period, no NaN values should exist
    if (result.length > 30) {
        for (let i = 30; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
        }
    }
});

test('VAR batch basic', () => {
    // Test batch processing
    const close = new Float64Array(testData.close);
    
    const config = {
        period_range: [10, 20, 5],  // 10, 15, 20
        nbdev_range: [1.0, 1.0, 0.0]  // Just 1.0
    };
    
    const result = wasm.var_batch(close, config);
    
    assert(result.values, 'Batch result should have values');
    assert(result.combos, 'Batch result should have combos');
    assert.strictEqual(result.rows, 3, 'Should have 3 parameter combinations');
    assert.strictEqual(result.cols, close.length, 'Should have same columns as input length');
    assert.strictEqual(result.values.length, 3 * close.length, 'Values should be flattened matrix');
    
    // Check parameter combinations
    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[1].period, 15);
    assert.strictEqual(result.combos[2].period, 20);
});

test('VAR fast API (in-place)', async () => {
    // Test fast API with in-place operation
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate memory
    const outPtr = wasm.var_alloc(len);
    
    try {
        // Create input data
        const inData = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        inData.set(close);
        
        // Apply VAR in-place (same pointer for input and output)
        wasm.var_into(outPtr, outPtr, len, 14, 1.0);
        
        // Read result
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        // Compare with safe API
        const expected = wasm.var_js(close, 14, 1.0);
        assertArrayClose(result, expected, 1e-10, 'Fast API should match safe API');
        
    } finally {
        // Free memory
        wasm.var_free(outPtr, len);
    }
});

test('VAR fast API (separate buffers)', async () => {
    // Test fast API with separate input/output buffers
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate memory
    const inPtr = wasm.var_alloc(len);
    const outPtr = wasm.var_alloc(len);
    
    try {
        // Create input data
        const inData = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inData.set(close);
        
        // Apply VAR with separate buffers
        wasm.var_into(inPtr, outPtr, len, 14, 1.0);
        
        // Read result
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        // Compare with safe API
        const expected = wasm.var_js(close, 14, 1.0);
        assertArrayClose(result, expected, 1e-10, 'Fast API should match safe API');
        
    } finally {
        // Free memory
        wasm.var_free(inPtr, len);
        wasm.var_free(outPtr, len);
    }
});
