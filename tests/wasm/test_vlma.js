
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

test('VLMA partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.vlma_js(close, 5, 50, "sma", 0);
    assert.strictEqual(result.length, close.length);
});

test('VLMA accuracy', async () => {

    const close = new Float64Array(testData.close);

    const result = wasm.vlma_js(close, 5, 50, "sma", 0);

    assert.strictEqual(result.length, close.length);


    const expectedLast5 = [
        59376.252799490234,
        59343.71066624187,
        59292.92555520155,
        59269.93796266796,
        59167.4483022233,
    ];

    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-1,
        "VLMA last 5 values mismatch"
    );
});

test('VLMA zero or inverted periods', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0, 40.0]);


    assert.throws(() => {
        wasm.vlma_js(inputData, 10, 5, "sma", 0);
    }, /min_period.*is greater than max_period/);


    assert.throws(() => {
        wasm.vlma_js(inputData, 5, 0, "sma", 0);
    }, /(Invalid period|greater than max_period)/);
});

test('VLMA not enough data', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.vlma_js(inputData, 5, 10, "sma", 0);
    }, /Invalid period|Not enough valid data/);
});

test('VLMA all NaN', () => {

    const inputData = new Float64Array([NaN, NaN, NaN]);

    assert.throws(() => {
        wasm.vlma_js(inputData, 2, 3, "sma", 0);
    }, /All values are NaN/);
});

test('VLMA empty input', () => {

    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.vlma_js(empty, 5, 50, "sma", 0);
    }, /Empty data/);
});

test('VLMA slice reinput', () => {

    const close = new Float64Array(testData.close);


    const firstResult = wasm.vlma_js(close, 5, 20, "ema", 1);


    const secondResult = wasm.vlma_js(firstResult, 5, 20, "ema", 1);

    assert.strictEqual(secondResult.length, firstResult.length);
});

test('VLMA batch single parameter set', () => {

    const close = new Float64Array(testData.close);


    const batchResult = wasm.vlma_batch(close, {
        min_period_range: [5, 5, 0],
        max_period_range: [50, 50, 0],
        devtype_range: [0, 0, 0],
        matype: "sma"
    });


    const singleResult = wasm.vlma_js(close, 5, 50, "sma", 0);

    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('VLMA batch multiple periods', () => {

    const close = new Float64Array(testData.close.slice(0, 100));


    const batchResult = wasm.vlma_batch(close, {
        min_period_range: [5, 15, 5],
        max_period_range: [50, 50, 0],
        devtype_range: [0, 0, 0],
        matype: "sma"
    });


    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);


    const minPeriods = [5, 10, 15];
    for (let i = 0; i < minPeriods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);

        const singleResult = wasm.vlma_js(close, minPeriods[i], 50, "sma", 0);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Min period ${minPeriods[i]} mismatch`
        );
    }
});

test('VLMA batch metadata from result', () => {

    const close = new Float64Array(60);
    close.fill(100);

    const result = wasm.vlma_batch(close, {
        min_period_range: [5, 10, 5],
        max_period_range: [30, 40, 10],
        devtype_range: [0, 2, 1],
        matype: "sma"
    });


    assert.strictEqual(result.combos.length, 12);


    assert.strictEqual(result.combos[0].min_period, 5);
    assert.strictEqual(result.combos[0].max_period, 30);
    assert.strictEqual(result.combos[0].devtype, 0);
    assert.strictEqual(result.combos[0].matype, "sma");


    assert.strictEqual(result.combos[11].min_period, 10);
    assert.strictEqual(result.combos[11].max_period, 40);
    assert.strictEqual(result.combos[11].devtype, 2);
    assert.strictEqual(result.combos[11].matype, "sma");
});



test('VLMA zero-copy API in-place operation', () => {

    const data = new Float64Array([
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
        110, 120, 130, 140, 150, 160, 170, 180, 190, 200
    ]);
    const minPeriod = 3;
    const maxPeriod = 10;
    const matype = "sma";
    const devtype = 0;


    const ptr = wasm.vlma_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');


    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );


    memView.set(data);


    try {
        wasm.vlma_into(ptr, ptr, data.length, minPeriod, maxPeriod, matype, devtype);


        const regularResult = wasm.vlma_js(data, minPeriod, maxPeriod, matype, devtype);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue;
            }
            assertClose(memView[i], regularResult[i], 1e-10, `Mismatch at index ${i}`);
        }
    } finally {
        wasm.vlma_free(ptr, data.length);
    }
});

test('VLMA zero-copy API separate buffers', () => {

    const size = 100;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }

    const inPtr = wasm.vlma_alloc(size);
    const outPtr = wasm.vlma_alloc(size);
    assert(inPtr !== 0, 'Failed to allocate input buffer');
    assert(outPtr !== 0, 'Failed to allocate output buffer');

    try {

        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, size);
        inView.set(data);


        wasm.vlma_into(inPtr, outPtr, size, 5, 20, "ema", 1);


        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, size);


        const regularResult = wasm.vlma_js(data, 5, 20, "ema", 1);
        assertArrayClose(outView, regularResult, 1e-10, "Zero-copy mismatch");

    } finally {
        wasm.vlma_free(inPtr, size);
        wasm.vlma_free(outPtr, size);
    }
});

test('VLMA zero-copy error handling', () => {

    assert.throws(() => {
        wasm.vlma_into(0, 0, 10, 5, 50, "sma", 0);
    }, /null pointer|Null pointer provided/i);


    const ptr = wasm.vlma_alloc(10);
    try {

        assert.throws(() => {
            wasm.vlma_into(ptr, ptr, 10, 20, 5, "sma", 0);
        }, /min_period.*is greater than max_period/);


        assert.throws(() => {
            wasm.vlma_into(ptr, ptr, 10, 5, 0, "sma", 0);
        }, /(Invalid period|greater than max_period)/);
    } finally {
        wasm.vlma_free(ptr, 10);
    }
});

test('VLMA zero-copy memory management', () => {

    const sizes = [100, 1000, 10000, 100000];

    for (const size of sizes) {
        const ptr = wasm.vlma_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);


        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }


        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Pattern mismatch at ${i}`);
        }

        wasm.vlma_free(ptr, size);
    }
});

test('VLMA batch zero-copy API', () => {

    const size = 50;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.1) * 100 + 100;
    }


    const minPeriods = 3;
    const maxPeriods = 1;
    const devtypes = 2;
    const totalCombos = minPeriods * maxPeriods * devtypes;
    const totalSize = totalCombos * size;

    const inPtr = wasm.vlma_alloc(size);
    const outPtr = wasm.vlma_alloc(totalSize);

    try {

        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, size);
        inView.set(data);


        const numCombos = wasm.vlma_batch_into(
            inPtr, outPtr, size,
            5, 15, 5,
            50, 50, 0,
            0, 1, 1,
            "sma"
        );

        assert.strictEqual(numCombos, totalCombos, 'Unexpected number of combinations');


        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);


        const firstRowResult = new Float64Array(size);
        for (let i = 0; i < size; i++) {
            firstRowResult[i] = outView[i];
        }

        const expectedFirst = wasm.vlma_js(data, 5, 50, "sma", 0);
        assertArrayClose(firstRowResult, expectedFirst, 1e-10, "First batch row mismatch");

    } finally {
        wasm.vlma_free(inPtr, size);
        wasm.vlma_free(outPtr, totalSize);
    }
});