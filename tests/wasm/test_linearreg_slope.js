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
    // Load WASM module from pkg (node target glue)
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --target nodejs -- --features wasm" first');
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
    // Test LINEARREG_SLOPE matches expected values from Rust tests - mirrors check_linearreg_slope_accuracy
    const inputData = new Float64Array([100.0, 98.0, 95.0, 90.0, 85.0, 80.0, 78.0, 77.0, 79.0, 81.0]);
    
    const result = wasm.linearreg_slope_js(inputData, 5);
    
    assert.strictEqual(result.length, inputData.length);
    
    // Check warmup period (first 4 values should be NaN)
    for (let i = 0; i < 4; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}, got ${result[i]}`);
    }
    
    // Check expected values after warmup (from Rust test output)
    const expectedValues = [-3.8, -4.6, -4.4, -3.3, -1.5, 0.3];
    
    for (let i = 0; i < expectedValues.length; i++) {
        const idx = i + 4;
        assertClose(result[idx], expectedValues[i], 1e-9, 
                   `LINEARREG_SLOPE value mismatch at index ${idx}`);
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
    
    // Allocate input and output buffers
    const inPtr = wasm.linearreg_slope_alloc(len);
    const outPtr = wasm.linearreg_slope_alloc(len);
    
    try {
        // Copy data to WASM memory
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(data);
        
        // Compute linearreg_slope
        wasm.linearreg_slope_into(inPtr, outPtr, len, 3);
        
        // Read results
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        // Verify warmup period
        assert(isNaN(result[0]), 'Expected NaN at index 0');
        assert(isNaN(result[1]), 'Expected NaN at index 1');
        
        // Verify we have results after warmup
        assert(!isNaN(result[2]), 'Expected valid value at index 2');
    } finally {
        // Clean up
        wasm.linearreg_slope_free(inPtr, len);
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
        const wasmData = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        wasmData.set(data);
        
        // Compute in-place
        wasm.linearreg_slope_into(ptr, ptr, len, 3);
        
        // Results should be in the same buffer
        const result = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        
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

test('LINEARREG_SLOPE all NaN input', () => {
    // Test LINEARREG_SLOPE with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.linearreg_slope_js(allNaN, 14);
    }, /All values are NaN/);
});

test('LINEARREG_SLOPE batch_into API', () => {
    // Test the new batch_into function for zero-copy batch processing
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const len = data.length;
    const periods = [5, 10]; // Two different periods
    const rows = periods.length;
    const totalSize = rows * len;
    
    // Allocate input and output buffers
    const inPtr = wasm.linearreg_slope_alloc(len);
    const outPtr = wasm.linearreg_slope_alloc(totalSize);
    
    try {
        // Copy data to WASM memory
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(data);
        
        // Test with period range 5 to 10 with step 5
        const resultRows = wasm.linearreg_slope_batch_into(
            inPtr,
            outPtr,
            len,
            5,  // period_start
            10, // period_end  
            5   // period_step
        );
        
        assert.strictEqual(resultRows, 2, 'Should return 2 rows');
        
        // Read results
        const results = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
        
        // Verify first row (period=5)
        for (let i = 0; i < 4; i++) {
            assert(isNaN(results[i]), `Expected NaN at row 0, index ${i}`);
        }
        assert(!isNaN(results[4]), 'Expected valid value at row 0, index 4');
        
        // Verify second row (period=10)
        const row2Start = len;
        for (let i = 0; i < 9; i++) {
            assert(isNaN(results[row2Start + i]), `Expected NaN at row 1, index ${i}`);
        }
        assert(!isNaN(results[row2Start + 9]), 'Expected valid value at row 1, index 9');
    } finally {
        // Clean up
        wasm.linearreg_slope_free(inPtr, len);
        wasm.linearreg_slope_free(outPtr, totalSize);
    }
});

test('LINEARREG_SLOPE linear data', () => {
    // Test with perfectly linear data: y = 2x + 10
    const linearData = new Float64Array(20);
    for (let i = 0; i < 20; i++) {
        linearData[i] = 2 * i + 10;
    }
    
    const result = wasm.linearreg_slope_js(linearData, 14);
    
    // After warmup, all slopes should be exactly 2.0
    for (let i = 13; i < result.length; i++) {
        assertClose(result[i], 2.0, 1e-9, 
                   `Expected slope=2.0 for linear data at index ${i}`);
    }
});

test('LINEARREG_SLOPE constant data', () => {
    // Test with constant data
    const constantData = new Float64Array(20);
    constantData.fill(100.0);
    
    const result = wasm.linearreg_slope_js(constantData, 10);
    
    // After warmup, all slopes should be exactly 0.0
    for (let i = 9; i < result.length; i++) {
        assertClose(result[i], 0.0, 1e-9, 
                   `Expected slope=0.0 for constant data at index ${i}`);
    }
});

test('LINEARREG_SLOPE batch with different warmup periods', () => {
    // Test batch processing with different periods to verify warmup handling
    const close = new Float64Array(30);
    for (let i = 0; i < 30; i++) {
        close[i] = Math.sin(i * 0.1) * 100 + 1000;
    }
    
    const config = {
        period_range: [5, 15, 5]  // periods: 5, 10, 15
    };
    
    const result = wasm.linearreg_slope_batch(close, config);
    
    // Verify structure
    assert.strictEqual(result.rows, 3, 'Should have 3 rows');
    assert.strictEqual(result.cols, close.length, 'Columns should match input length');
    assert.strictEqual(result.combos.length, 3, 'Should have 3 parameter combinations');
    
    // Verify warmup periods
    const periods = [5, 10, 15];
    for (let rowIdx = 0; rowIdx < periods.length; rowIdx++) {
        const period = periods[rowIdx];
        const rowStart = rowIdx * close.length;
        
        // Check NaN in warmup period
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(result.values[rowStart + i]), 
                  `Expected NaN at row ${rowIdx}, index ${i} for period ${period}`);
        }
        
        // Check non-NaN after warmup
        for (let i = period - 1; i < close.length; i++) {
            assert(!isNaN(result.values[rowStart + i]), 
                  `Unexpected NaN at row ${rowIdx}, index ${i} for period ${period}`);
        }
    }
});

test.after(() => {
    console.log('LINEARREG_SLOPE WASM tests completed');
});
