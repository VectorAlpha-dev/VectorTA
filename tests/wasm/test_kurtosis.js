/**
 * WASM binding tests for Kurtosis indicator.
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
import { compareWithRust } from './rust-comparison.js';

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
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('Kurtosis partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.kurtosis_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('Kurtosis accuracy', async () => {
    
    const hl2 = new Float64Array(testData.hl2);
    
    const result = wasm.kurtosis_js(hl2, 5);
    
    assert.strictEqual(result.length, hl2.length);
    
    
    const expectedLast5 = [
        -0.5438903789933454,
        -1.6848139264816433,
        -1.6331336745945797,
        -0.6130805596586351,
        -0.027802601135927585,
    ];
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-10,
        "Kurtosis last 5 values mismatch"
    );
    
    
    await compareWithRust('kurtosis', result, 'hl2', { period: 5 });
});

test('Kurtosis default candles', () => {
    
    const hl2 = new Float64Array(testData.hl2);
    
    const result = wasm.kurtosis_js(hl2, 5);
    assert.strictEqual(result.length, hl2.length);
});

test('Kurtosis zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.kurtosis_js(inputData, 0);
    }, /Invalid period/);
});

test('Kurtosis period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.kurtosis_js(dataSmall, 10);
    }, /Invalid period/);
});

test('Kurtosis all nan', () => {
    
    const nanData = new Float64Array(20).fill(NaN);
    
    assert.throws(() => {
        wasm.kurtosis_js(nanData, 5);
    }, /All values are NaN/);
});

test('Kurtosis empty input', () => {
    
    const emptyData = new Float64Array([]);
    
    assert.throws(() => {
        wasm.kurtosis_js(emptyData, 5);
    }, /Input data slice is empty/);
});

test('Kurtosis nan prefix', () => {
    
    const nanPrefixData = new Float64Array(30);
    const period = 5;
    nanPrefixData.fill(NaN, 0, 10);
    for (let i = 10; i < 30; i++) {
        nanPrefixData[i] = 50.0 + i * 0.5;
    }
    
    const result = wasm.kurtosis_js(nanPrefixData, period);
    
    assert.strictEqual(result.length, nanPrefixData.length);
    
    
    const expectedNaNCount = 10 + (period - 1);
    for (let i = 0; i < expectedNaNCount; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    
    for (let i = expectedNaNCount; i < result.length; i++) {
        assert(!isNaN(result[i]), `Expected valid value at index ${i}`);
    }
});

test('Kurtosis batch operation', () => {
    
    const hl2 = new Float64Array(testData.hl2);
    
    
    const config = {
        period_range: [5, 5, 0]  
    };
    
    const result = wasm.kurtosis_batch(hl2, config);
    
    assert(result.values);
    assert(result.periods);
    assert.strictEqual(result.rows, 1); 
    assert.strictEqual(result.cols, hl2.length);
    
    
    const expectedLast5 = [
        -0.5438903789933454,
        -1.6848139264816433,
        -1.6331336745945797,
        -0.6130805596586351,
        -0.027802601135927585,
    ];
    
    const last5 = result.values.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-6,
        "Kurtosis batch last 5 values mismatch"
    );
});

test('Kurtosis batch multiple periods', () => {
    
    const close = new Float64Array(testData.close);
    
    const config = {
        period_range: [5, 20, 5]  
    };
    
    const result = wasm.kurtosis_batch(close, config);
    
    assert(result.values);
    assert(result.periods);
    assert.strictEqual(result.rows, 4); 
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, 4 * close.length);
    
    
    const expectedPeriods = [5, 10, 15, 20];
    assert.deepStrictEqual(result.periods, expectedPeriods);
});

test('Kurtosis fast API', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const inPtr = wasm.kurtosis_alloc(len);
    const outPtr = wasm.kurtosis_alloc(len);
    
    try {
        
        const inputArray = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inputArray.set(close);
        
        
        wasm.kurtosis_into(inPtr, outPtr, len, 5);
        
        
        const output = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        
        const expected = wasm.kurtosis_js(close, 5);
        assertArrayClose(output, expected, 1e-10, "Fast API mismatch");
        
    } finally {
        
        wasm.kurtosis_free(inPtr, len);
        wasm.kurtosis_free(outPtr, len);
    }
});

test('Kurtosis fast API aliasing', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const ptr = wasm.kurtosis_alloc(len);
    
    try {
        
        const dataArray = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        dataArray.set(close);
        
        
        wasm.kurtosis_into(ptr, ptr, len, 5);
        
        
        const resultArray = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        
        
        const expected = wasm.kurtosis_js(close, 5);
        assertArrayClose(resultArray, expected, 1e-10, "Fast API aliasing mismatch");
        
    } finally {
        
        wasm.kurtosis_free(ptr, len);
    }
});


test('Kurtosis very small dataset', () => {
    
    const smallData = new Float64Array([5.0, 10.0, 15.0, 20.0, 25.0]);
    const period = 5;
    
    const result = wasm.kurtosis_js(smallData, period);
    assert.strictEqual(result.length, smallData.length);
    
    
    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    
    assert(!isNaN(result[period - 1]), `Expected valid value at index ${period - 1}`);
});

test('Kurtosis batch metadata from result', () => {
    
    const close = new Float64Array(testData.close);
    
    const config = {
        period_range: [5, 10, 1]  
    };
    
    const result = wasm.kurtosis_batch(close, config);
    
    assert(result.periods);
    assert.strictEqual(result.periods.length, 6);
    
    
    const expectedPeriods = [5, 6, 7, 8, 9, 10];
    assert.deepStrictEqual(result.periods, expectedPeriods);
});

test('Kurtosis batch edge cases', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const config1 = {
        period_range: [5, 5, 0]
    };
    const result1 = wasm.kurtosis_batch(close, config1);
    assert.strictEqual(result1.rows, 1, "Single parameter should give 1 row");
    
    
    if (close.length > 50) {
        const config2 = {
            period_range: [5, 50, 45]
        };
        const result2 = wasm.kurtosis_batch(close.slice(0, 100), config2);
        assert.strictEqual(result2.rows, 2, "Large step should give 2 rows");
        assert.deepStrictEqual(result2.periods, [5, 50]);
    }
});

test('Kurtosis batch full parameter sweep', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50)); 
    
    
    const config = {
        period_range: [5, 9, 2]
    };
    
    const result = wasm.kurtosis_batch(close, config);
    
    assert(result.values);
    assert(result.periods);
    assert.strictEqual(result.rows, 3); 
    assert.strictEqual(result.cols, 50);
    
    
    const periods = [5, 7, 9];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 50;
        const rowEnd = rowStart + 50;
        const rowData = result.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.kurtosis_js(close, periods[i]);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Period ${periods[i]} mismatch`
        );
    }
});

