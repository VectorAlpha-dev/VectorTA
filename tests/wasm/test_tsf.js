
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

test('TSF partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.tsf_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('TSF accuracy', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.tsf_js(close, 14);


    const expectedLastFive = [
        58846.945054945056,
        58818.83516483516,
        58854.57142857143,
        59083.846153846156,
        58962.25274725275,
    ];

    const lastFive = result.slice(-5);
    assertArrayClose(lastFive, expectedLastFive, 0.1, 'TSF last 5 values mismatch');
});

test('TSF zero period', () => {

    const data = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.tsf_js(data, 0);
    }, /Period must be at least 2/, 'TSF should fail with zero period');
});

test('TSF period one', () => {

    const data = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);

    assert.throws(() => {
        wasm.tsf_js(data, 1);
    }, /Period must be at least 2/, 'TSF should fail with period=1');
});

test('TSF period exceeds length', () => {

    const data = new Float64Array([1.0, 2.0, 3.0]);

    assert.throws(() => {
        wasm.tsf_js(data, 5);
    }, /Invalid period/, 'TSF should fail when period exceeds length');
});

test('TSF NaN handling', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.tsf_js(close, 14);
    assert.strictEqual(result.length, close.length);



    assert(isNaN(result[0]), 'First value should be NaN');


    if (result.length > 20) {
        for (let i = 20; i < Math.min(result.length, 240); i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('TSF empty input', () => {

    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.tsf_js(empty, 14);
    }, /Input data slice is empty/, 'TSF should fail with empty input');
});

test('TSF fast API', async (t) => {
    await t.test('basic computation', () => {
        const close = new Float64Array(testData.close);
        const len = close.length;


        const outPtr = wasm.tsf_alloc(len);
        const inPtr = wasm.tsf_alloc(len);

        try {

            const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
            wasmMemory.set(close);


            wasm.tsf_into(inPtr, outPtr, len, 14);


            const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
            const resultCopy = new Float64Array(result);


            const safeResult = wasm.tsf_js(close, 14);
            assertArrayClose(resultCopy, safeResult, 1e-10, 'Fast API should match safe API');

        } finally {
            wasm.tsf_free(inPtr, len);
            wasm.tsf_free(outPtr, len);
        }
    });

    await t.test('in-place computation (aliasing)', () => {
        const close = new Float64Array(testData.close);
        const len = close.length;


        const ptr = wasm.tsf_alloc(len);

        try {

            const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
            wasmMemory.set(close);


            wasm.tsf_into(ptr, ptr, len, 14);


            const result = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
            const resultCopy = new Float64Array(result);


            const safeResult = wasm.tsf_js(close, 14);
            assertArrayClose(resultCopy, safeResult, 1e-10, 'In-place computation should match safe API');

        } finally {
            wasm.tsf_free(ptr, len);
        }
    });
});

test('TSF batch operation', () => {

    const close = new Float64Array(testData.close.slice(0, 1000));

    const config = {
        period_range: [10, 20, 2],
    };

    const result = wasm.tsf_batch(close, config);

    assert(result.values, 'Result should have values');
    assert(result.combos, 'Result should have combos');
    assert(result.rows, 'Result should have rows');
    assert(result.cols, 'Result should have cols');


    assert.strictEqual(result.rows, 6);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, result.rows * result.cols);


    const expectedPeriods = [10, 12, 14, 16, 18, 20];
    const actualPeriods = result.combos.map(c => c.period);
    assert.deepStrictEqual(actualPeriods, expectedPeriods);
});

test('TSF batch fast API', () => {
    const close = new Float64Array(testData.close.slice(0, 500));
    const len = close.length;


    const periodStart = 10;
    const periodEnd = 20;
    const periodStep = 5;
    const expectedRows = 3;


    const inPtr = wasm.tsf_alloc(len);
    const outPtr = wasm.tsf_alloc(len * expectedRows);

    try {

        const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        wasmMemory.set(close);


        const rows = wasm.tsf_batch_into(
            inPtr, outPtr, len,
            periodStart, periodEnd, periodStep
        );

        assert.strictEqual(rows, expectedRows, 'Should return correct number of rows');


        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len * rows);


        for (let i = 0; i < rows; i++) {
            const period = periodStart + i * periodStep;
            const rowResult = result.slice(i * len, (i + 1) * len);
            const singleResult = wasm.tsf_js(close, period);

            assertArrayClose(
                new Float64Array(rowResult),
                singleResult,
                1e-10,
                `Batch row ${i} (period ${period}) should match single computation`
            );
        }

    } finally {
        wasm.tsf_free(inPtr, len);
        wasm.tsf_free(outPtr, len * expectedRows);
    }
});

test('TSF memory management', () => {

    const len = 1000;


    const ptr = wasm.tsf_alloc(len);
    assert(ptr > 0, 'Should return valid pointer');


    const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
    const data = new Float64Array(len);
    for (let i = 0; i < len; i++) {
        data[i] = Math.random();
    }
    wasmMemory.set(data);


    const readBack = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
    assertArrayClose(new Float64Array(readBack), data, 1e-15, 'Should read back same data');


    wasm.tsf_free(ptr, len);


    assert.doesNotThrow(() => {
        wasm.tsf_free(0, len);
    });
});