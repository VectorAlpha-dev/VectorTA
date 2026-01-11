/**
 * WASM binding tests for Midpoint indicator.
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
    
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
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

test('Midpoint partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.midpoint_js(close, 14);
    assert(result instanceof Float64Array);
    assert.strictEqual(result.length, close.length);
});

test('Midpoint accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.midpoint_js(close, 14);
    
    assert.strictEqual(result.length, close.length);
    
    
    const expectedLastFive = [59578.5, 59578.5, 59578.5, 58886.0, 58886.0];
    
    const actualLastFive = result.slice(-5);
    
    for (let i = 0; i < 5; i++) {
        assertClose(actualLastFive[i], expectedLastFive[i], 1e-1, 
            `Midpoint mismatch at index ${i}`);
    }
    
    
    await compareWithRust('midpoint', result, 'close', { period: 14 });
});

test('Midpoint invalid period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    
    assert.throws(() => {
        wasm.midpoint_js(inputData, 0);
    }, /Invalid period/);
});

test('Midpoint period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.midpoint_js(dataSmall, 10);
    }, /Invalid period/);
});

test('Midpoint very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.midpoint_js(singlePoint, 9);
    }, /Invalid period|Not enough valid data/);
});

test('Midpoint empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.midpoint_js(empty, 14);
    }, /All values are NaN/);
});

test('Midpoint reinput', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const firstResult = wasm.midpoint_js(close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    
    const secondResult = wasm.midpoint_js(firstResult, 14);
    assert.strictEqual(secondResult.length, firstResult.length);
});

test('Midpoint NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.midpoint_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    
    for (let i = 0; i < 13; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('Midpoint all NaN input', () => {
    
    const allNan = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.midpoint_js(allNan, 14);
    }, /All values are NaN/);
});

test('Midpoint simple case', () => {
    
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    const period = 3;
    
    const result = wasm.midpoint_js(data, period);
    
    
    assert(isNaN(result[0]));
    assert(isNaN(result[1]));
    
    
    assertClose(result[2], 2.0, 1e-10);
    
    
    assertClose(result[3], 3.0, 1e-10);
    
    
    assertClose(result[4], 4.0, 1e-10);
});

test('Midpoint fast API basic', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const inPtr = wasm.midpoint_alloc(len);
    
    const outPtr = wasm.midpoint_alloc(len);
    
    try {
        const memory = wasm.__wasm.memory;
        const inView = new Float64Array(memory.buffer, inPtr, len);
        inView.set(close);
        
        
        wasm.midpoint_into(inPtr, outPtr, len, 14);
        
        
        const result = new Float64Array(memory.buffer, outPtr, len);
        
        
        const expectedLastFive = [59578.5, 59578.5, 59578.5, 58886.0, 58886.0];
        const actualLastFive = result.slice(-5);
        
        for (let i = 0; i < 5; i++) {
            assertClose(actualLastFive[i], expectedLastFive[i], 1e-1);
        }
    } finally {
        
        wasm.midpoint_free(inPtr, len);
        wasm.midpoint_free(outPtr, len);
    }
});

test('Midpoint fast API in-place', () => {
    
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    const len = data.length;
    
    
    const ptr = wasm.midpoint_alloc(len);
    
    try {
        const memory = wasm.__wasm.memory;
        const memView = new Float64Array(memory.buffer, ptr, len);
        memView.set(data);
        
        
        wasm.midpoint_into(ptr, ptr, len, 3);
        
        
        const result = new Float64Array(memory.buffer, ptr, len);
        
        
        assert(isNaN(result[0]));
        assert(isNaN(result[1]));
        assertClose(result[2], 2.0, 1e-10);
        assertClose(result[3], 3.0, 1e-10);
        assertClose(result[4], 4.0, 1e-10);
    } finally {
        wasm.midpoint_free(ptr, len);
    }
});

test('Midpoint batch API', async () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    const config = {
        period_range: [10, 20, 2]  
    };
    
    const result = wasm.midpoint_batch(close, config);
    
    assert(result.values);
    assert(result.combos);
    assert.strictEqual(result.rows, 6); 
    assert.strictEqual(result.cols, 100);
    assert.strictEqual(result.values.length, 600); 
    
    
    const singleResult = wasm.midpoint_js(close, 10);
    const firstRow = result.values.slice(0, 100);
    
    for (let i = 0; i < 100; i++) {
        if (!isNaN(singleResult[i]) && !isNaN(firstRow[i])) {
            assertClose(firstRow[i], singleResult[i], 1e-10);
        }
    }
});

test('Midpoint batch single parameter', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const config = {
        period_range: [14, 14, 0]  
    };
    
    const result = wasm.midpoint_batch(close, config);
    
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, 100);
    
    
    const singleResult = wasm.midpoint_js(close, 14);
    const batchValues = result.values;
    
    for (let i = 0; i < 100; i++) {
        if (!isNaN(singleResult[i]) && !isNaN(batchValues[i])) {
            assertClose(batchValues[i], singleResult[i], 1e-10);
        }
    }
});

test('Midpoint batch zero-copy API', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;
    
    
    const periodStart = 10;
    const periodEnd = 18;
    const periodStep = 4;
    const expectedRows = 3; 
    
    
    const inPtr = wasm.midpoint_alloc(len);
    
    const outPtr = wasm.midpoint_alloc(len * expectedRows);
    
    try {
        
        const memory = wasm.__wasm.memory;
        const inView = new Float64Array(memory.buffer, inPtr, len);
        inView.set(close);
        
        
        const rows = wasm.midpoint_batch_into(
            inPtr, outPtr, len,
            periodStart, periodEnd, periodStep
        );
        
        assert.strictEqual(rows, expectedRows, 'Should return correct number of rows');
        
        
        const outView = new Float64Array(memory.buffer, outPtr, len * rows);
        
        
        const periods = [10, 14, 18];
        for (let row = 0; row < rows; row++) {
            const period = periods[row];
            const singleResult = wasm.midpoint_js(close, period);
            const rowStart = row * len;
            
            for (let col = 0; col < len; col++) {
                const batchVal = outView[rowStart + col];
                const singleVal = singleResult[col];
                
                if (!isNaN(singleVal) && !isNaN(batchVal)) {
                    assertClose(batchVal, singleVal, 1e-10,
                        `Row ${row} (period ${period}), col ${col} mismatch`);
                }
            }
        }
    } finally {
        
        wasm.midpoint_free(inPtr, len);
        wasm.midpoint_free(outPtr, len * expectedRows);
    }
});

test('Midpoint batch zero-copy with large dataset', () => {
    
    const size = 1000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    
    const periodStart = 10;
    const periodEnd = 30;
    const periodStep = 10;
    const expectedRows = 3; 
    
    
    const inPtr = wasm.midpoint_alloc(size);
    const outPtr = wasm.midpoint_alloc(size * expectedRows);
    
    try {
        
        const memory = wasm.__wasm.memory;
        const inView = new Float64Array(memory.buffer, inPtr, size);
        inView.set(data);
        
        
        const rows = wasm.midpoint_batch_into(
            inPtr, outPtr, size,
            periodStart, periodEnd, periodStep
        );
        
        assert.strictEqual(rows, expectedRows);
        
        
        const outView = new Float64Array(memory.buffer, outPtr, size * rows);
        
        
        const periods = [10, 20, 30];
        for (let row = 0; row < rows; row++) {
            const period = periods[row];
            const rowStart = row * size;
            
            
            for (let i = 0; i < period - 1; i++) {
                assert(isNaN(outView[rowStart + i]), 
                    `Row ${row} (period ${period}): Expected NaN at warmup index ${i}`);
            }
            
            
            for (let i = period - 1; i < Math.min(period + 10, size); i++) {
                assert(!isNaN(outView[rowStart + i]), 
                    `Row ${row} (period ${period}): Expected value at index ${i}, got NaN`);
            }
        }
    } finally {
        wasm.midpoint_free(inPtr, size);
        wasm.midpoint_free(outPtr, size * expectedRows);
    }
});
