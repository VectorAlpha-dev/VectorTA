/**
 * WASM binding tests for CMO indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';
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
    
    
    
    const pkgPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const esmImportPath = process.platform === 'win32'
        ? 'file:///' + pkgPath.replace(/\\/g, '/')
        : pkgPath;
    try {
        wasm = await import(esmImportPath);
        
        if (wasm && wasm.default) wasm = wasm.default;
        
        
        if (!wasm || !wasm.__wasm) throw new Error('no __wasm in ESM module');
    } catch (err) {
        const require = createRequire(import.meta.url);
        wasm = require(pkgPath);
    }

    testData = loadTestData();
});

test('CMO partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.cmo_js(close);
    assert.strictEqual(result.length, close.length);
    
    
    const result2 = wasm.cmo_js(close, 10);
    assert.strictEqual(result2.length, close.length);
});

test('CMO accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    
    
    const expectedLastFive = [
        -13.152504931406101,
        -14.649876201213106,
        -16.760170709240303,
        -14.274505732779227,
        -21.984038127126716,
    ];
    
    const result = wasm.cmo_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-6,
        "CMO last 5 values mismatch"
    );
    
    
    await compareWithRust('cmo', result, 'close', { period: 14 });
});

test('CMO default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.cmo_js(close);
    assert.strictEqual(result.length, close.length);
});

test('CMO zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cmo_js(inputData, 0);
    }, /Invalid period/);
});

test('CMO period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cmo_js(dataSmall, 10);
    }, /Invalid period/);
});

test('CMO very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.cmo_js(singlePoint, 14);
    }, /Invalid period/);
});

test('CMO empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.cmo_js(empty, 14);
    }, /Empty data provided/);
});

test('CMO reinput', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const firstResult = wasm.cmo_js(close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    
    const secondResult = wasm.cmo_js(firstResult, 14);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    
    if (secondResult.length > 28) {
        for (let i = 28; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Expected no NaN after index 28, found NaN at ${i}`);
        }
    }
});

test('CMO NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.cmo_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    
    assertAllNaN(result.slice(0, 14), "Expected NaN in warmup period");
});

test('CMO all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.cmo_js(allNaN, 14);
    }, /All values are NaN/);
});


test('CMO fast API - basic operation', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const inPtr = wasm.cmo_alloc(len);
    const outPtr = wasm.cmo_alloc(len);
    
    
    const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
    inView.set(close);
    
    
    wasm.cmo_into(inPtr, outPtr, len, 14);
    
    
    const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
    
    
    const resultArray = Array.from(result);
    
    
    const safeResult = wasm.cmo_js(close, 14);
    
    assertArrayClose(resultArray, safeResult, 1e-10, "Fast API vs Safe API mismatch");
    
    
    wasm.cmo_free(inPtr, len);
    wasm.cmo_free(outPtr, len);
});

test('CMO fast API - in-place operation (aliasing)', () => {
    
    const data = new Float64Array(testData.close.slice(0, 100)); 
    const len = data.length;
    
    
    const bufPtr = wasm.cmo_alloc(len);
    const buffer = new Float64Array(wasm.__wasm.memory.buffer, bufPtr, len);
    buffer.set(data);
    
    
    const expected = wasm.cmo_js(data, 14);
    
    
    wasm.cmo_into(bufPtr, bufPtr, len, 14);
    
    
    assertArrayClose(buffer, expected, 1e-10, "In-place operation mismatch");
    
    
    wasm.cmo_free(bufPtr, len);
});

test('CMO fast API - null pointer error', () => {
    
    const len = 100;
    
    
    assert.throws(() => {
        wasm.cmo_into(0, wasm.cmo_alloc(len), len, 14);
    }, /Null pointer/);
    
    
    const inPtr = wasm.cmo_alloc(len);
    assert.throws(() => {
        wasm.cmo_into(inPtr, 0, len, 14);
    }, /Null pointer/);
    wasm.cmo_free(inPtr, len);
});


test('CMO batch single parameter set', () => {
    
    const close = new Float64Array(testData.close);
    
    const batchResult = wasm.cmo_batch(close, {
        period_range: [14, 14, 0]
    });
    
    
    const singleResult = wasm.cmo_js(close, 14);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('CMO batch multiple periods', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const batchResult = wasm.cmo_batch(close, {
        period_range: [10, 20, 2]
    });
    
    
    assert.strictEqual(batchResult.values.length, 6 * 100);
    assert.strictEqual(batchResult.rows, 6);
    assert.strictEqual(batchResult.cols, 100);
    
    
    const periods = [10, 12, 14, 16, 18, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.cmo_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('CMO batch metadata', () => {
    
    const close = new Float64Array(50);
    close.fill(100);
    
    const result = wasm.cmo_batch(close, {
        period_range: [10, 20, 5] 
    });
    
    
    assert.strictEqual(result.combos.length, 3);
    
    
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[1].period, 15);
    assert.strictEqual(result.combos[2].period, 20);
});

test('CMO batch warmup verification', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.cmo_batch(close, {
        period_range: [10, 20, 10] 
    });
    
    assert.strictEqual(batchResult.rows, 2);
    assert.strictEqual(batchResult.cols, 50);
    
    
    for (let row = 0; row < 2; row++) {
        const period = batchResult.combos[row].period;
        const rowStart = row * 50;
        const rowData = batchResult.values.slice(rowStart, rowStart + 50);
        
        
        for (let i = 0; i < period; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        
        for (let i = period; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('CMO batch edge cases', () => {
    
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    
    
    const singleBatch = wasm.cmo_batch(close, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 15);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    
    const largeBatch = wasm.cmo_batch(close, {
        period_range: [5, 7, 10] 
    });
    
    
    assert.strictEqual(largeBatch.values.length, 15);
    assert.strictEqual(largeBatch.combos.length, 1);
    assert.strictEqual(largeBatch.combos[0].period, 5);
});


test('CMO fast batch API - basic operation', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;
    
    
    const periodStart = 10;
    const periodEnd = 20;
    const periodStep = 5;
    const expectedRows = 3;
    
    
    const inPtr = wasm.cmo_alloc(len);
    const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
    inView.set(close);
    
    
    const outPtr = wasm.cmo_alloc(len * expectedRows);
    
    
    const rows = wasm.cmo_batch_into(inPtr, outPtr, len, periodStart, periodEnd, periodStep);
    assert.strictEqual(rows, expectedRows);
    
    
    const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len * rows);
    
    
    const safeBatch = wasm.cmo_batch(close, {
        period_range: [periodStart, periodEnd, periodStep]
    });
    
    assertArrayClose(result, safeBatch.values, 1e-10, "Fast batch vs safe batch mismatch");
    
    
    wasm.cmo_free(inPtr, len);
    wasm.cmo_free(outPtr, len * rows);
});

test('CMO fast batch API - null pointer error', () => {
    
    const len = 100;
    
    assert.throws(() => {
        wasm.cmo_batch_into(0, wasm.cmo_alloc(len * 3), len, 10, 20, 5);
    }, /null pointer/);
});

test.after(() => {
    console.log('CMO WASM tests completed');
});
