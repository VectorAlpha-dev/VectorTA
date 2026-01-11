/**
 * WASM binding tests for FWMA indicator.
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
        
        
        console.log('WASM exports:', Object.keys(wasm).filter(k => !k.startsWith('__')));
        
        
        if (wasm.__wasm && wasm.__wasm.memory) {
            console.log('Memory found at wasm.__wasm.memory');
        } else if (wasm.memory) {
            console.log('Memory found at wasm.memory');
        } else {
            console.log('Memory not directly accessible - fast API tests will be skipped');
        }
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('FWMA partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.fwma_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('FWMA accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.fwma_js(close, 5);
    
    assert.strictEqual(result.length, close.length);
    
    
    const period = 5;
    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} (warmup period)`);
    }
    
    assert(!isNaN(result[period - 1]), `Expected valid value at index ${period - 1}`);
    
    
    const expectedLastFive = [
        59273.583333333336,
        59252.5,
        59167.083333333336,
        59151.0,
        58940.333333333336
    ];
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-8,
        "FWMA last 5 values mismatch"
    );
    
    
    await compareWithRust('fwma', result, 'close', { period: 5 });
});

test('FWMA default candles', async () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.fwma_js(close, 5);
    assert.strictEqual(result.length, close.length);
    
    
    await compareWithRust('fwma', result, 'close', { period: 5 });
});

test('FWMA zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.fwma_js(inputData, 0);
    }, /Invalid period/);
});

test('FWMA empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.fwma_js(empty, 5);
    }, /Input data slice is empty/);
});

test('FWMA all NaN input', () => {
    
    const allNaN = new Float64Array(10);
    for (let i = 0; i < allNaN.length; i++) {
        allNaN[i] = NaN;
    }
    
    assert.throws(() => {
        wasm.fwma_js(allNaN, 3);
    }, /All values are NaN/);
});

test('FWMA period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.fwma_js(dataSmall, 5);
    }, /Invalid period/);
});

test('FWMA very small dataset', () => {
    
    const dataSingle = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.fwma_js(dataSingle, 5);
    }, /Invalid period|Not enough valid data/);
});

test('FWMA reinput', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const firstResult = wasm.fwma_js(close, 5);
    
    
    const secondResult = wasm.fwma_js(firstResult, 3);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    
    for (let i = 240; i < secondResult.length; i++) {
        assert(!isNaN(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('FWMA NaN handling', () => {
    
    const data = new Float64Array([NaN, NaN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    const period = 3;
    
    const result = wasm.fwma_js(data, period);
    
    assert.strictEqual(result.length, data.length);
    
    
    for (let i = 0; i < 2 + period - 1; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    
    for (let i = 2 + period - 1; i < result.length; i++) {
        assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
    }
});

test('FWMA batch', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const period_start = 3;
    const period_end = 9;
    const period_step = 2; 
    
    const batch_result = wasm.fwma_batch_js(close, period_start, period_end, period_step);
    const metadata = wasm.fwma_batch_metadata_js(period_start, period_end, period_step);
    
    const expected_periods = [3, 5, 7, 9];
    assert.deepStrictEqual(Array.from(metadata), expected_periods, 'Metadata periods mismatch');
    
    
    assert.strictEqual(batch_result.length, expected_periods.length * close.length, 'Batch result length mismatch');
    
    
    for (let i = 0; i < expected_periods.length; i++) {
        const period = expected_periods[i];
        const individual_result = wasm.fwma_js(close, period);
        
        
        const row_start = i * close.length;
        const row = batch_result.slice(row_start, row_start + close.length);
        
        assertArrayClose(row, individual_result, 1e-9, `Period ${period}`);
        
        
        for (let j = 0; j < period - 1; j++) {
            assert(isNaN(row[j]), `Row ${i} (period ${period}): Expected NaN at index ${j} (warmup period)`);
        }
        assert(!isNaN(row[period - 1]), `Row ${i} (period ${period}): Expected valid value at index ${period - 1}`);
    }
    
    
    const single_batch = wasm.fwma_batch_js(close, 5, 5, 0);
    const single_metadata = wasm.fwma_batch_metadata_js(5, 5, 0);
    
    assert.deepStrictEqual(Array.from(single_metadata), [5], 'Single period metadata mismatch');
    assert.strictEqual(single_batch.length, close.length, 'Single batch result length mismatch');
    
    
    const individual_5 = wasm.fwma_js(close, 5);
    assertArrayClose(single_batch, individual_5, 1e-9, 'Single period batch vs individual');
});

test('FWMA Fibonacci weights calculation', () => {
    
    
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    const period = 5;
    
    const result = wasm.fwma_js(data, period);
    
    
    const expected = (1*1 + 2*1 + 3*2 + 4*3 + 5*5) / 12;
    assertClose(result[4], expected, 1e-9, 'FWMA calculation');
});

test('FWMA batch performance', () => {
    
    const close = new Float64Array(testData.close.slice(0, 1000)); 
    const periods = Array.from({length: 20}, (_, i) => i + 3); 
    
    const startBatch = performance.now();
    const batchResult = wasm.fwma_batch_js(close, 3, 22, 1);
    const batchTime = performance.now() - startBatch;
    
    const startSingle = performance.now();
    const singleResults = [];
    for (const period of periods) {
        singleResults.push(...wasm.fwma_js(close, period));
    }
    const singleTime = performance.now() - startSingle;
    
    
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});




function hasWasmMemory() {
    if (!wasm) return false;
    return (wasm.__wasm && wasm.__wasm.memory) || wasm.memory;
}


function getMemoryBuffer() {
    if (wasm.__wasm && wasm.__wasm.memory) {
        return getMemoryBuffer();
    } else if (wasm.memory) {
        return wasm.memory.buffer;
    }
    throw new Error('WASM memory not accessible');
}

test('FWMA memory allocation and deallocation', { skip: !hasWasmMemory() }, () => {
    const len = 100;
    
    
    const ptr = wasm.fwma_alloc(len);
    assert(ptr !== 0, 'Allocated pointer should not be null');
    
    
    const data = new Float64Array(wasm.__getMemoryBuffer(), ptr, len);
    for (let i = 0; i < len; i++) {
        data[i] = i + 1;
    }
    
    
    assert.strictEqual(data[0], 1);
    assert.strictEqual(data[len - 1], len);
    
    
    wasm.fwma_free(ptr, len);
    
});

test('FWMA fast API (fwma_into) without aliasing', { skip: !hasWasmMemory() }, () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;
    const period = 5;
    
    
    const inPtr = wasm.fwma_alloc(len);
    const outPtr = wasm.fwma_alloc(len);
    
    try {
        
        const inBuf = new Float64Array(getMemoryBuffer(), inPtr, len);
        inBuf.set(close);
        
        
        wasm.fwma_into(inPtr, outPtr, len, period);
        
        
        const results = new Float64Array(getMemoryBuffer(), outPtr, len);
        const resultsCopy = new Float64Array(results); 
        
        
        const safeResults = wasm.fwma_js(close, period);
        assertArrayClose(resultsCopy, safeResults, 1e-9, 'Fast vs safe API');
    } finally {
        wasm.fwma_free(inPtr, len);
        wasm.fwma_free(outPtr, len);
    }
});

test('FWMA fast API (fwma_into) with aliasing', { skip: !hasWasmMemory() }, () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;
    const period = 5;
    
    
    const ptr = wasm.fwma_alloc(len);
    
    try {
        
        const buf = new Float64Array(getMemoryBuffer(), ptr, len);
        buf.set(close);
        
        
        wasm.fwma_into(ptr, ptr, len, period);
        
        
        const results = new Float64Array(getMemoryBuffer(), ptr, len);
        const resultsCopy = new Float64Array(results); 
        
        
        const safeResults = wasm.fwma_js(close, period);
        assertArrayClose(resultsCopy, safeResults, 1e-9, 'In-place vs safe API');
    } finally {
        wasm.fwma_free(ptr, len);
    }
});

test('FWMA unified batch API', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    
    
    const config = {
        period_range: [3, 9, 2] 
    };
    
    const result = wasm.fwma_batch(close, config);
    
    
    assert(result.values, 'Result should have values array');
    assert(result.combos, 'Result should have combos array');
    assert.strictEqual(result.rows, 4, 'Should have 4 rows');
    assert.strictEqual(result.cols, close.length, 'Should have cols equal to input length');
    
    
    assert.strictEqual(result.combos.length, 4);
    assert.deepStrictEqual(
        result.combos.map(c => c.period),
        [3, 5, 7, 9],
        'Combo periods should match'
    );
    
    
    for (let i = 0; i < result.combos.length; i++) {
        const period = result.combos[i].period;
        const rowStart = i * close.length;
        const row = result.values.slice(rowStart, rowStart + close.length);
        
        const individual = wasm.fwma_js(close, period);
        assertArrayClose(row, individual, 1e-9, `Batch row ${i} (period ${period})`);
    }
});

test('FWMA batch_into fast API', { skip: !hasWasmMemory() }, () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;
    const periodStart = 3;
    const periodEnd = 9;
    const periodStep = 2; 
    const numPeriods = Math.floor((periodEnd - periodStart) / periodStep) + 1;
    
    
    const inPtr = wasm.fwma_alloc(len);
    const outPtr = wasm.fwma_alloc(len * numPeriods);
    
    try {
        
        const inBuf = new Float64Array(getMemoryBuffer(), inPtr, len);
        inBuf.set(close);
        
        
        const rows = wasm.fwma_batch_into(inPtr, outPtr, len, periodStart, periodEnd, periodStep);
        assert.strictEqual(rows, numPeriods, 'Should return correct number of rows');
        
        
        const results = new Float64Array(getMemoryBuffer(), outPtr, len * rows);
        
        
        const safeBatch = wasm.fwma_batch_js(close, periodStart, periodEnd, periodStep);
        assertArrayClose(results, safeBatch, 1e-9, 'Batch fast vs safe API');
    } finally {
        wasm.fwma_free(inPtr, len);
        wasm.fwma_free(outPtr, len * numPeriods);
    }
});

test('FWMA fast API null pointer handling', () => {
    
    assert.throws(() => {
        wasm.fwma_into(0, 0, 100, 5); 
    }, 'Should throw on null pointers');
});

test('FWMA fast API performance comparison', { skip: !hasWasmMemory() }, () => {
    const close = new Float64Array(testData.close);
    const len = close.length;
    const period = 20;
    const iterations = 10;
    
    
    wasm.fwma_js(close, period);
    
    
    const safeStart = performance.now();
    for (let i = 0; i < iterations; i++) {
        wasm.fwma_js(close, period);
    }
    const safeTime = performance.now() - safeStart;
    
    
    const inPtr = wasm.fwma_alloc(len);
    const outPtr = wasm.fwma_alloc(len);
    
    try {
        const inBuf = new Float64Array(getMemoryBuffer(), inPtr, len);
        inBuf.set(close);
        
        const fastStart = performance.now();
        for (let i = 0; i < iterations; i++) {
            wasm.fwma_into(inPtr, outPtr, len, period);
        }
        const fastTime = performance.now() - fastStart;
        
        console.log(`Safe API: ${safeTime.toFixed(2)}ms, Fast API: ${fastTime.toFixed(2)}ms`);
        console.log(`Fast API is ${(safeTime / fastTime).toFixed(2)}x faster`);
        
        
        assert(fastTime <= safeTime * 1.1, 'Fast API should not be significantly slower');
    } finally {
        wasm.fwma_free(inPtr, len);
        wasm.fwma_free(outPtr, len);
    }
});