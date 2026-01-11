/**
 * WASM binding tests for CORREL_HL indicator.
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
let usingFallback = false;
let testData;

test.before(async () => {
    
    const esmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const esmImportPath = process.platform === 'win32'
        ? 'file:///' + esmPath.replace(/\\/g, '/')
        : esmPath;
    try {
        
        wasm = await import(esmImportPath);
    } catch (error) {
        
        
        const { createRequire } = await import('node:module');
        const require = createRequire(import.meta.url);
        wasm = require(path.join(__dirname, '../../pkg/vector_ta.js'));
        usingFallback = true;
    }

    testData = loadTestData();
});

test('CORREL_HL partial params', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.correl_hl_js(high, low, 9);
    assert.strictEqual(result.length, high.length);
});

test('CORREL_HL accuracy', async () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const expected = EXPECTED_OUTPUTS.correl_hl;
    
    const result = wasm.correl_hl_js(
        high,
        low,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, high.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-7,  
        "CORREL_HL last 5 values mismatch"
    );
    
    
    const resultDefault = wasm.correl_hl_js(high, low, 9);
    
    await compareWithRust('correl_hl', resultDefault, 'high,low', { period: 9 }, 1e-7);
});

test('CORREL_HL from candles', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.correl_hl_js(high, low, 9);
    assert.strictEqual(result.length, high.length);
});

test('CORREL_HL zero period', () => {
    
    const high = new Float64Array([1.0, 2.0, 3.0]);
    const low = new Float64Array([1.0, 2.0, 3.0]);
    
    assert.throws(() => {
        wasm.correl_hl_js(high, low, 0);
    }, /Invalid period/);
});

test('CORREL_HL period exceeds length', () => {
    
    const high = new Float64Array([1.0, 2.0, 3.0]);
    const low = new Float64Array([1.0, 2.0, 3.0]);
    
    assert.throws(() => {
        wasm.correl_hl_js(high, low, 10);
    }, /Invalid period/);
});

test('CORREL_HL data length mismatch', () => {
    
    const high = new Float64Array([1.0, 2.0, 3.0]);
    const low = new Float64Array([1.0, 2.0]);
    
    assert.throws(() => {
        wasm.correl_hl_js(high, low, 2);
    }, /Data length mismatch/);
});

test('CORREL_HL all NaN', () => {
    
    const high = new Float64Array([NaN, NaN, NaN]);
    const low = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.correl_hl_js(high, low, 2);
    }, /All values are NaN/);
});

test('CORREL_HL empty input', () => {
    
    const high = new Float64Array([]);
    const low = new Float64Array([]);
    
    assert.throws(() => {
        wasm.correl_hl_js(high, low, 9);
    }, /Empty data/);
});

test('CORREL_HL reinput', () => {
    
    const high = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    const low = new Float64Array([0.5, 1.0, 1.5, 2.0, 2.5]);
    
    const firstResult = wasm.correl_hl_js(high, low, 2);
    const secondResult = wasm.correl_hl_js(firstResult, low, 2);
    assert.strictEqual(secondResult.length, low.length);
});

test('CORREL_HL very small dataset', () => {
    
    const singleHigh = new Float64Array([42.0]);
    const singleLow = new Float64Array([21.0]);
    
    const result = wasm.correl_hl_js(singleHigh, singleLow, 1);
    assert.strictEqual(result.length, 1);
    
    assert(isNaN(result[0]) || Math.abs(result[0]) < Number.EPSILON,
           `Expected NaN or 0 for period=1, got ${result[0]}`);
});

test('CORREL_HL all NaN input', () => {
    
    const allNanHigh = new Float64Array(100);
    const allNanLow = new Float64Array(100);
    allNanHigh.fill(NaN);
    allNanLow.fill(NaN);
    
    assert.throws(() => {
        wasm.correl_hl_js(allNanHigh, allNanLow, 9);
    }, /All values are NaN/);
});

test('CORREL_HL NaN handling', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.correl_hl_js(high, low, 9);
    assert.strictEqual(result.length, high.length);
    
    
    assertAllNaN(result.slice(0, 8));
    
    
    assertNoNaN(result.slice(8));
});

test('CORREL_HL fast API (in-place)', () => {
    
    const high = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    const low = new Float64Array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]);
    const period = 3;
    const len = high.length;
    
    
    const expected = wasm.correl_hl_js(high, low, period);
    
    
    const highPtr = wasm.correl_hl_alloc(len);
    const lowPtr = wasm.correl_hl_alloc(len);
    const outPtr = wasm.correl_hl_alloc(len);
    
    try {
        
        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        highMem.set(high);
        lowMem.set(low);
        
        
        wasm.correl_hl_into(highPtr, lowPtr, outPtr, len, period);
        
        
        const outMem = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const result = new Float64Array(outMem);
        
        
        assertArrayClose(result, expected, 1e-10, "Fast API mismatch");
        
        
        wasm.correl_hl_into(highPtr, lowPtr, highPtr, len, period);
        
        
        const highMem2 = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        
        
        assertArrayClose(highMem2, expected, 1e-10, "Fast API aliasing mismatch");
    } finally {
        wasm.correl_hl_free(highPtr, len);
        wasm.correl_hl_free(lowPtr, len);
        wasm.correl_hl_free(outPtr, len);
    }
});

test('CORREL_HL batch single period', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    
    const config = {
        period_range: [9, 9, 1]
    };
    
    const result = wasm.correl_hl_batch(high, low, config);
    
    
    assert(result.values, 'Missing values in batch result');
    assert(result.periods, 'Missing periods in batch result');
    assert(result.rows, 'Missing rows in batch result');
    assert(result.cols, 'Missing cols in batch result');
    
    
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, 100);
    assert.strictEqual(result.values.length, 100);
    assert.strictEqual(result.periods.length, 1);
    assert.strictEqual(result.periods[0], 9);
    
    
    const singleResult = wasm.correl_hl_js(high, low, 9);
    assertArrayClose(result.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('CORREL_HL batch multiple periods', () => {
    
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    
    const config = {
        period_range: [5, 15, 5]  
    };
    
    const result = wasm.correl_hl_batch(high, low, config);
    
    
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.values.length, 150);  
    assert.strictEqual(result.periods.length, 3);
    assert.deepStrictEqual(Array.from(result.periods), [5, 10, 15]);
    
    
    for (let i = 0; i < 3; i++) {
        const period = result.periods[i];
        const rowStart = i * 50;
        const rowEnd = rowStart + 50;
        const row = result.values.slice(rowStart, rowEnd);
        
        
        assertAllNaN(row.slice(0, period - 1));
        
        assertNoNaN(row.slice(period - 1));
    }
});

test('CORREL_HL batch fast API', () => {
    if (usingFallback) {
        
        
        return;
    }
    
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const len = high.length;
    
    
    const periodStart = 5;
    const periodEnd = 15;
    const periodStep = 5;
    const expectedRows = 3; 
    
    
    const highPtr = wasm.correl_hl_alloc(len);
    const lowPtr = wasm.correl_hl_alloc(len);
    const outPtr = wasm.correl_hl_alloc(len * expectedRows);
    
    try {
        
        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        highMem.set(high);
        lowMem.set(low);
        
        const rows = wasm.correl_hl_batch_into(
            highPtr, lowPtr, outPtr, len,
            periodStart, periodEnd, periodStep
        );
        
        assert.strictEqual(rows, expectedRows);
        
        
        const outMem = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len * rows);
        const result = new Float64Array(outMem);
        
        
        const config = {
            period_range: [periodStart, periodEnd, periodStep]
        };
        const safeBatchResult = wasm.correl_hl_batch(high, low, config);
        
        assertArrayClose(
            result,
            safeBatchResult.values,
            1e-10,
            "Fast batch API mismatch"
        );
    } finally {
        wasm.correl_hl_free(highPtr, len);
        wasm.correl_hl_free(lowPtr, len);
        wasm.correl_hl_free(outPtr, len * expectedRows);
    }
});
