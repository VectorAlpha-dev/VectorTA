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
    
    try {
        const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
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
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.linearreg_slope_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('LINEARREG_SLOPE accuracy', () => {
    
    const inputData = new Float64Array([100.0, 98.0, 95.0, 90.0, 85.0, 80.0, 78.0, 77.0, 79.0, 81.0]);
    
    const result = wasm.linearreg_slope_js(inputData, 5);
    
    assert.strictEqual(result.length, inputData.length);
    
    
    for (let i = 0; i < 4; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}, got ${result[i]}`);
    }
    
    
    const expectedValues = [-3.8, -4.6, -4.4, -3.3, -1.5, 0.3];
    
    for (let i = 0; i < expectedValues.length; i++) {
        const idx = i + 4;
        assertClose(result[idx], expectedValues[i], 1e-9, 
                   `LINEARREG_SLOPE value mismatch at index ${idx}`);
    }
});

test('LINEARREG_SLOPE zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.linearreg_slope_js(inputData, 0);
    }, /Invalid period/);
});

test('LINEARREG_SLOPE period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.linearreg_slope_js(dataSmall, 10);
    }, /Invalid period/);
});

test('LINEARREG_SLOPE very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.linearreg_slope_js(singlePoint, 14);
    }, /Invalid period/);
});

test('LINEARREG_SLOPE empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.linearreg_slope_js(empty, 14);
    }, /Empty data provided/);
});

test('LINEARREG_SLOPE fast API', () => {
    
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const len = data.length;
    
    
    const inPtr = wasm.linearreg_slope_alloc(len);
    const outPtr = wasm.linearreg_slope_alloc(len);
    
    try {
        
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(data);
        
        
        wasm.linearreg_slope_into(inPtr, outPtr, len, 3);
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        
        assert(isNaN(result[0]), 'Expected NaN at index 0');
        assert(isNaN(result[1]), 'Expected NaN at index 1');
        
        
        assert(!isNaN(result[2]), 'Expected valid value at index 2');
    } finally {
        
        wasm.linearreg_slope_free(inPtr, len);
        wasm.linearreg_slope_free(outPtr, len);
    }
});

test('LINEARREG_SLOPE fast API in-place', () => {
    
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const len = data.length;
    
    
    const ptr = wasm.linearreg_slope_alloc(len);
    
    try {
        
        const wasmData = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        wasmData.set(data);
        
        
        wasm.linearreg_slope_into(ptr, ptr, len, 3);
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        
        
        assert(isNaN(result[0]), 'Expected NaN at index 0');
        assert(isNaN(result[1]), 'Expected NaN at index 1');
        
        
        assert(!isNaN(result[2]), 'Expected valid value at index 2');
    } finally {
        
        wasm.linearreg_slope_free(ptr, len);
    }
});

test('LINEARREG_SLOPE batch API', () => {
    
    const close = new Float64Array(testData.close);
    
    const config = {
        period_range: [10, 20, 5]  
    };
    
    const result = wasm.linearreg_slope_batch(close, config);
    
    
    assert(result.values, 'Result should have values array');
    assert(result.combos, 'Result should have combos array');
    assert.strictEqual(result.rows, 3, 'Should have 3 rows');
    assert.strictEqual(result.cols, close.length, 'Columns should match input length');
    assert.strictEqual(result.values.length, 3 * close.length, 'Values array should be rows * cols');
    assert.strictEqual(result.combos.length, 3, 'Should have 3 parameter combinations');
    
    
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[1].period, 15);
    assert.strictEqual(result.combos[2].period, 20);
});

test('LINEARREG_SLOPE all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.linearreg_slope_js(allNaN, 14);
    }, /All values are NaN/);
});

test('LINEARREG_SLOPE batch_into API', () => {
    
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const len = data.length;
    const periods = [5, 10]; 
    const rows = periods.length;
    const totalSize = rows * len;
    
    
    const inPtr = wasm.linearreg_slope_alloc(len);
    const outPtr = wasm.linearreg_slope_alloc(totalSize);
    
    try {
        
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(data);
        
        
        const resultRows = wasm.linearreg_slope_batch_into(
            inPtr,
            outPtr,
            len,
            5,  
            10, 
            5   
        );
        
        assert.strictEqual(resultRows, 2, 'Should return 2 rows');
        
        
        const results = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
        
        
        for (let i = 0; i < 4; i++) {
            assert(isNaN(results[i]), `Expected NaN at row 0, index ${i}`);
        }
        assert(!isNaN(results[4]), 'Expected valid value at row 0, index 4');
        
        
        const row2Start = len;
        for (let i = 0; i < 9; i++) {
            assert(isNaN(results[row2Start + i]), `Expected NaN at row 1, index ${i}`);
        }
        assert(!isNaN(results[row2Start + 9]), 'Expected valid value at row 1, index 9');
    } finally {
        
        wasm.linearreg_slope_free(inPtr, len);
        wasm.linearreg_slope_free(outPtr, totalSize);
    }
});

test('LINEARREG_SLOPE linear data', () => {
    
    const linearData = new Float64Array(20);
    for (let i = 0; i < 20; i++) {
        linearData[i] = 2 * i + 10;
    }
    
    const result = wasm.linearreg_slope_js(linearData, 14);
    
    
    for (let i = 13; i < result.length; i++) {
        assertClose(result[i], 2.0, 1e-9, 
                   `Expected slope=2.0 for linear data at index ${i}`);
    }
});

test('LINEARREG_SLOPE constant data', () => {
    
    const constantData = new Float64Array(20);
    constantData.fill(100.0);
    
    const result = wasm.linearreg_slope_js(constantData, 10);
    
    
    for (let i = 9; i < result.length; i++) {
        assertClose(result[i], 0.0, 1e-9, 
                   `Expected slope=0.0 for constant data at index ${i}`);
    }
});

test('LINEARREG_SLOPE batch with different warmup periods', () => {
    
    const close = new Float64Array(30);
    for (let i = 0; i < 30; i++) {
        close[i] = Math.sin(i * 0.1) * 100 + 1000;
    }
    
    const config = {
        period_range: [5, 15, 5]  
    };
    
    const result = wasm.linearreg_slope_batch(close, config);
    
    
    assert.strictEqual(result.rows, 3, 'Should have 3 rows');
    assert.strictEqual(result.cols, close.length, 'Columns should match input length');
    assert.strictEqual(result.combos.length, 3, 'Should have 3 parameter combinations');
    
    
    const periods = [5, 10, 15];
    for (let rowIdx = 0; rowIdx < periods.length; rowIdx++) {
        const period = periods[rowIdx];
        const rowStart = rowIdx * close.length;
        
        
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(result.values[rowStart + i]), 
                  `Expected NaN at row ${rowIdx}, index ${i} for period ${period}`);
        }
        
        
        for (let i = period - 1; i < close.length; i++) {
            assert(!isNaN(result.values[rowStart + i]), 
                  `Unexpected NaN at row ${rowIdx}, index ${i} for period ${period}`);
        }
    }
});

test.after(() => {
    console.log('LINEARREG_SLOPE WASM tests completed');
});
