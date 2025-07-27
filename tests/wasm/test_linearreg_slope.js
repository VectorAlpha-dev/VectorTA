/**
 * WASM binding tests for LINEARREG_SLOPE indicator.
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

test('LINEARREG_SLOPE partial params', () => {
    // Test with default parameters - mirrors check_linearreg_slope_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.linearreg_slope_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('LINEARREG_SLOPE accuracy', () => {
    // Test LINEARREG_SLOPE matches expected values from Rust tests
    const inputData = new Float64Array([100.0, 98.0, 95.0, 90.0, 85.0, 80.0, 78.0, 77.0, 79.0, 81.0]);
    
    const result = wasm.linearreg_slope_js(inputData, 5);
    
    assert.strictEqual(result.length, inputData.length);
    
    // Check warmup period
    for (let i = 0; i < 4; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}, got ${result[i]}`);
    }
    
    // Check that values after period-1 are not NaN
    for (let i = 4; i < result.length; i++) {
        assert(!isNaN(result[i]), `Expected valid slope value at index ${i}, got NaN`);
    }
});

test('LINEARREG_SLOPE zero period', () => {
    // Test LINEARREG_SLOPE fails with zero period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.linearreg_slope_js(inputData, 0);
    }, /Invalid period/);
});

test('LINEARREG_SLOPE period exceeds length', () => {
    // Test LINEARREG_SLOPE fails when period exceeds data length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.linearreg_slope_js(dataSmall, 10);
    }, /Invalid period/);
});

test('LINEARREG_SLOPE very small dataset', () => {
    // Test LINEARREG_SLOPE fails with insufficient data
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.linearreg_slope_js(singlePoint, 14);
    }, /Invalid period/);
});

test('LINEARREG_SLOPE empty input', () => {
    // Test LINEARREG_SLOPE fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.linearreg_slope_js(empty, 14);
    }, /Empty data provided/);
});

test('LINEARREG_SLOPE fast API', () => {
    // Test the fast/unsafe API
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const len = data.length;
    
    // Allocate output buffer
    const outPtr = wasm.linearreg_slope_alloc(len);
    
    try {
        // Compute linearreg_slope
        wasm.linearreg_slope_into(data, outPtr, len, 3);
        
        // Read results
        const result = new Float64Array(wasm.memory.buffer, outPtr, len);
        
        // Verify warmup period
        assert(isNaN(result[0]), 'Expected NaN at index 0');
        assert(isNaN(result[1]), 'Expected NaN at index 1');
        
        // Verify we have results after warmup
        assert(!isNaN(result[2]), 'Expected valid value at index 2');
    } finally {
        // Clean up
        wasm.linearreg_slope_free(outPtr, len);
    }
});

test('LINEARREG_SLOPE fast API in-place', () => {
    // Test the fast API with in-place operation (aliasing)
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const len = data.length;
    
    // Use the same buffer for input and output
    const ptr = wasm.linearreg_slope_alloc(len);
    
    try {
        // Copy data to WASM memory
        const wasmData = new Float64Array(wasm.memory.buffer, ptr, len);
        wasmData.set(data);
        
        // Compute in-place
        wasm.linearreg_slope_into(ptr, ptr, len, 3);
        
        // Results should be in the same buffer
        const result = new Float64Array(wasm.memory.buffer, ptr, len);
        
        // Verify warmup period
        assert(isNaN(result[0]), 'Expected NaN at index 0');
        assert(isNaN(result[1]), 'Expected NaN at index 1');
        
        // Verify we have results after warmup
        assert(!isNaN(result[2]), 'Expected valid value at index 2');
    } finally {
        // Clean up
        wasm.linearreg_slope_free(ptr, len);
    }
});

test('LINEARREG_SLOPE batch API', () => {
    // Test batch processing
    const close = new Float64Array(testData.close);
    
    const config = {
        period_range: [10, 20, 5]  // 3 values: 10, 15, 20
    };
    
    const result = wasm.linearreg_slope_batch(close, config);
    
    // Verify structure
    assert(result.values, 'Result should have values array');
    assert(result.combos, 'Result should have combos array');
    assert.strictEqual(result.rows, 3, 'Should have 3 rows');
    assert.strictEqual(result.cols, close.length, 'Columns should match input length');
    assert.strictEqual(result.values.length, 3 * close.length, 'Values array should be rows * cols');
    assert.strictEqual(result.combos.length, 3, 'Should have 3 parameter combinations');
    
    // Verify combos
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[1].period, 15);
    assert.strictEqual(result.combos[2].period, 20);
});

test.after(() => {
    console.log('LINEARREG_SLOPE WASM tests completed');
});
