/**
 * WASM binding tests for JSA indicator.
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

test('JSA partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.jsa_js(close, 30);
    assert.strictEqual(result.length, close.length);
});

test('JSA accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.jsa;
    
    const result = wasm.jsa_js(close, expected.defaultParams.period);
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-5,
        "JSA last 5 values mismatch"
    );
    
    
    
});

test('JSA default candles', async () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.jsa_js(close, 30);
    assert.strictEqual(result.length, close.length);
    
    
    await compareWithRust('jsa', result, 'close', { period: 30 });
});

test('JSA zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.jsa_js(inputData, 0);
    }, /Invalid period/);
});

test('JSA period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.jsa_js(dataSmall, 10);
    }, /Invalid period/);
});

test('JSA very small dataset', () => {
    
    const dataSingle = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.jsa_js(dataSingle, 5);
    }, /Invalid period|Not enough valid data/);
});

test('JSA empty input', () => {
    
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.jsa_js(dataEmpty, 30);
    }, /Input data slice is empty/);
});

test('JSA all NaN', () => {
    
    const data = new Float64Array([NaN, NaN, NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.jsa_js(data, 3);
    }, /All values are NaN/);
});

test('JSA NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.jsa_js(close, 30);
    
    assert.strictEqual(result.length, close.length);
    
    
    for (let i = 0; i < 30; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    
    for (let i = 30; i < result.length; i++) {
        assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
    }
});

test('JSA batch', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const config = {
        period_range: [10, 40, 10]  
    };
    
    const batch_result = wasm.jsa_batch(close, config);
    
    
    const values = new Float64Array(batch_result.values);
    const periods = new Float64Array(batch_result.periods);
    const rows = batch_result.rows;
    const cols = batch_result.cols;
    
    
    assert.strictEqual(rows, 4);  
    assert.strictEqual(cols, close.length);
    assert.strictEqual(periods.length, 4);
    assert.deepStrictEqual(Array.from(periods), [10, 20, 30, 40]);
    
    
    assert.strictEqual(values.length, rows * cols);
    
    
    for (let i = 0; i < rows; i++) {
        const period = periods[i];
        const individual_result = wasm.jsa_js(close, period);
        
        
        const row_start = i * cols;
        const row = values.slice(row_start, row_start + cols);
        
        assertArrayClose(row, individual_result, 1e-9, `Period ${period}`);
    }
});

test('JSA different periods', () => {
    
    const close = new Float64Array(testData.close);
    
    
    for (const period of [5, 10, 20, 50]) {
        const result = wasm.jsa_js(close, period);
        assert.strictEqual(result.length, close.length);
        
        
        for (let i = 0; i < period; i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} for period=${period}`);
        }
        
        
        for (let i = period; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for period=${period}`);
        }
    }
});

test('JSA batch performance', () => {
    
    const close = new Float64Array(testData.close.slice(0, 1000)); 
    
    
    const config = {
        period_range: [10, 50, 10]  
    };
    
    const startBatch = performance.now();
    const batchResult = wasm.jsa_batch(close, config);
    const batchTime = performance.now() - startBatch;
    
    const startSingle = performance.now();
    const singleResults = [];
    for (let period = 10; period <= 50; period += 10) {
        singleResults.push(...wasm.jsa_js(close, period));
    }
    const singleTime = performance.now() - startSingle;
    
    
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    
    const batchValues = new Float64Array(batchResult.values);
    
    
    assertArrayClose(batchValues, singleResults, 1e-9, 'Batch vs single results');
});

test('JSA edge cases', () => {
    
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    
    const result1 = wasm.jsa_js(data, 1);
    assert.strictEqual(result1.length, data.length);
    assert(isNaN(result1[0])); 
    
    for (let i = 1; i < data.length; i++) {
        const expected = (data[i] + data[i-1]) * 0.5;
        assertClose(result1[i], expected, 1e-9, `Value at index ${i}`);
    }
    
    
    const result2 = wasm.jsa_js(data, data.length);
    assert.strictEqual(result2.length, data.length);
});

test('JSA single value', () => {
    
    const data = new Float64Array([42.0]);
    
    
    const result = wasm.jsa_js(data, 1);
    assert.strictEqual(result.length, 1);
    assert(isNaN(result[0])); 
});

test('JSA two values', () => {
    
    const data = new Float64Array([1.0, 2.0]);
    
    
    const result = wasm.jsa_js(data, 1);
    assert.strictEqual(result.length, 2);
    assert(isNaN(result[0])); 
    assertClose(result[1], (data[1] + data[0]) * 0.5, 1e-9); 
    
    
    const result2 = wasm.jsa_js(data, 2);
    assert.strictEqual(result2.length, 2);
    
    assert(isNaN(result2[0]));
    assert(isNaN(result2[1]));
});

test('JSA batch metadata', () => {
    
    
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const config = {
        period_range: [15, 45, 15]  
    };
    
    
    const result = wasm.jsa_batch(data, config);
    
    
    const periods = new Float64Array(result.periods);
    assert.strictEqual(periods.length, 3);
    assert.strictEqual(periods[0], 15);
    assert.strictEqual(periods[1], 30);
    assert.strictEqual(periods[2], 45);
    
    
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, data.length);
});

test('JSA warmup period calculation', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const testCases = [
        { period: 5, expectedWarmup: 5 },
        { period: 10, expectedWarmup: 10 },
        { period: 20, expectedWarmup: 20 },
        { period: 30, expectedWarmup: 30 },
    ];
    
    for (const { period, expectedWarmup } of testCases) {
        const result = wasm.jsa_js(close, period);
        
        
        for (let i = 0; i < expectedWarmup && i < result.length; i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} for period=${period}`);
        }
        
        
        if (expectedWarmup < result.length) {
            assert(!isNaN(result[expectedWarmup]), 
                `Expected valid value at index ${expectedWarmup} for period=${period}`);
        }
    }
});

test('JSA consistency across calls', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.jsa_js(close, 30);
    const result2 = wasm.jsa_js(close, 30);
    
    assertArrayClose(result1, result2, 1e-15, "JSA results not consistent");
});

test('JSA parameter step precision', () => {
    
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    const config = {
        period_range: [2, 4, 1]  
    };
    
    const batch_result = wasm.jsa_batch(data, config);
    const values = new Float64Array(batch_result.values);
    const periods = new Float64Array(batch_result.periods);
    
    
    assert.strictEqual(batch_result.rows, 3);
    assert.strictEqual(batch_result.cols, data.length);
    assert.strictEqual(values.length, 3 * data.length);
    
    
    assert.deepStrictEqual(Array.from(periods), [2, 3, 4]);
});

test('JSA streaming', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    const period = 30;
    
    
    const batchResult = wasm.jsa_js(close, period);
    
    
    const streamResults = new Float64Array(close.length);
    streamResults.fill(NaN);
    
    
    for (let i = period; i < close.length; i++) {
        
        streamResults[i] = (close[i] + close[i - period]) * 0.5;
    }
    
    
    for (let i = period; i < close.length; i++) {
        assertClose(streamResults[i], batchResult[i], 1e-9, 
                   `Streaming mismatch at index ${i}`);
    }
});

test('JSA large period', () => {
    
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.sin(i * 0.1) * 100 + 1000; 
    }
    
    const result = wasm.jsa_js(data, 99);
    assert.strictEqual(result.length, data.length);
    
    
    for (let i = 0; i < 99; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    
    const expected = (data[99] + data[0]) * 0.5;
    assertClose(result[99], expected, 1e-9, "Last value mismatch");
});



test('JSA zero-copy API', () => {
    
    const data = new Float64Array(testData.close.slice(0, 100));
    const period = 30;
    
    
    const ptr = wasm.jsa_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    
    memView.set(data);
    
    
    try {
        wasm.jsa_into(ptr, ptr, data.length, period);
        
        
        const regularResult = wasm.jsa_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; 
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        
        wasm.jsa_free(ptr, data.length);
    }
});

test('JSA zero-copy with separate buffers', () => {
    
    const data = new Float64Array(testData.close.slice(0, 50));
    const period = 10;
    
    
    const inPtr = wasm.jsa_alloc(data.length);
    const outPtr = wasm.jsa_alloc(data.length);
    
    assert(inPtr !== 0, 'Failed to allocate input buffer');
    assert(outPtr !== 0, 'Failed to allocate output buffer');
    assert(inPtr !== outPtr, 'Buffers should be different');
    
    try {
        
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
        inView.set(data);
        
        
        wasm.jsa_into(inPtr, outPtr, data.length, period);
        
        
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, data.length);
        
        
        const regularResult = wasm.jsa_js(data, period);
        assertArrayClose(outView, regularResult, 1e-10, 'Zero-copy separate buffers');
    } finally {
        wasm.jsa_free(inPtr, data.length);
        wasm.jsa_free(outPtr, data.length);
    }
});

test('JSA batch fast API', () => {
    
    const data = new Float64Array(testData.close.slice(0, 100));
    
    const inPtr = wasm.jsa_alloc(data.length);
    const rows = 3; 
    const outPtr = wasm.jsa_alloc(data.length * rows);
    
    try {
        
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
        inView.set(data);
        
        
        const resultRows = wasm.jsa_batch_into(inPtr, outPtr, data.length, 10, 30, 10);
        assert.strictEqual(resultRows, rows, 'Batch should return correct row count');
        
        
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, data.length * rows);
        
        
        for (let i = 0; i < rows; i++) {
            const period = 10 + i * 10;
            const expected = wasm.jsa_js(data, period);
            const rowStart = i * data.length;
            const row = outView.slice(rowStart, rowStart + data.length);
            assertArrayClose(row, expected, 1e-10, `Batch row ${i} (period ${period})`);
        }
    } finally {
        wasm.jsa_free(inPtr, data.length);
        wasm.jsa_free(outPtr, data.length * rows);
    }
});

test('JSA zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.jsa_into(0, 0, 10, 5);
    }, /null pointer|Null pointer/i);
    
    
    const ptr = wasm.jsa_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.jsa_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        
        assert.throws(() => {
            wasm.jsa_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.jsa_free(ptr, 10);
    }
});

test('JSA memory management', () => {
    
    const sizes = [10, 100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.jsa_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 2.5;
        }
        
        
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 2.5, `Memory corruption at index ${i}`);
        }
        
        
        wasm.jsa_free(ptr, size);
    }
});