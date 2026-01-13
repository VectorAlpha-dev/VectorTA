
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

test('SMMA partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.smma(close, 7);
    assert.strictEqual(result.length, close.length);
});

test('SMMA accuracy', async () => {

    const close = new Float64Array(testData.close);

    const result = wasm.smma(close, 7);

    assert.strictEqual(result.length, close.length);


    const expectedLast5 = [59434.4, 59398.2, 59346.9, 59319.4, 59224.5];


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-1,
        "SMMA last 5 values mismatch"
    );


    await compareWithRust('smma', result, 'close', { period: 7 });
});

test('SMMA default candles', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.smma(close, 7);
    assert.strictEqual(result.length, close.length);
});

test('SMMA zero period', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.smma(inputData, 0);
    }, /Invalid period/);
});

test('SMMA empty input', () => {

    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.smma(empty, 7);
    }, /Input data slice is empty/);
});

test('SMMA period exceeds length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.smma(dataSmall, 10);
    }, /Invalid period/);
});

test('SMMA very small dataset', () => {

    const singlePoint = new Float64Array([42.0]);

    assert.throws(() => {
        wasm.smma(singlePoint, 9);
    }, /Invalid period|Not enough valid data/);
});

test('SMMA reinput', () => {

    const close = new Float64Array(testData.close);


    const firstResult = wasm.smma(close, 7);
    assert.strictEqual(firstResult.length, close.length);


    const expectedFirstPass = [59434.4, 59398.2, 59346.9, 59319.4, 59224.5];
    assertArrayClose(
        firstResult.slice(-5),
        expectedFirstPass,
        1e-1,
        "First pass SMMA values mismatch"
    );


    const secondResult = wasm.smma(firstResult, 5);
    assert.strictEqual(secondResult.length, firstResult.length);



    const calcStdDev = (arr) => {
        const validValues = arr.filter(v => !isNaN(v));
        const mean = validValues.reduce((a, b) => a + b, 0) / validValues.length;
        const variance = validValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / validValues.length;
        return Math.sqrt(variance);
    };

    assert(calcStdDev(secondResult) < calcStdDev(firstResult),
           "Second pass should produce smoother results");
});

test('SMMA NaN handling', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.smma(close, 7);
    assert.strictEqual(result.length, close.length);


    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }



    assertAllNaN(result.slice(0, 6), "Expected NaN in warmup period (indices 0-5)");

    assert(!isNaN(result[6]), "Expected valid value at index 6 (period-1)");
});

test('SMMA all NaN input', () => {

    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);

    assert.throws(() => {
        wasm.smma(allNaN, 7);
    }, /All values are NaN/);
});

test('SMMA batch single period', () => {

    const close = new Float64Array(testData.close);

    const batchValues = wasm.smma_batch_legacy(close, 7, 7, 0);
    const metadata = wasm.smma_batch_metadata(7, 7, 0);
    const dims = wasm.smma_batch_rows_cols(7, 7, 0, close.length);

    assert(batchValues instanceof Float64Array, "Values should be Float64Array");
    assert(metadata instanceof Uint32Array, "Metadata should be Uint32Array");
    assert.strictEqual(dims[0], 1);
    assert.strictEqual(dims[1], close.length);

    const expected = [59434.4, 59398.2, 59346.9, 59319.4, 59224.5];


    const last5 = batchValues.slice(-5);
    assertArrayClose(
        last5,
        expected,
        1e-1,
        "SMMA batch default row mismatch"
    );
});

test('SMMA batch multiple periods', () => {

    const close = new Float64Array(testData.close.slice(0, 100));

    const batchValues = wasm.smma_batch_legacy(close, 5, 10, 1);
    const metadata = wasm.smma_batch_metadata(5, 10, 1);
    const dims = wasm.smma_batch_rows_cols(5, 10, 1, 100);

    assert.strictEqual(dims[0], 6);
    assert.strictEqual(dims[1], 100);
    assert.strictEqual(batchValues.length, 6 * 100);
    assert.strictEqual(metadata.length, 6);


    const expectedPeriods = [5, 6, 7, 8, 9, 10];
    for (let i = 0; i < expectedPeriods.length; i++) {
        assert.strictEqual(metadata[i], expectedPeriods[i]);
    }


    const firstRow = batchValues.slice(0, 100);
    const singleResult = wasm.smma(close, 5);
    assertArrayClose(firstRow, singleResult, 1e-10, "First row mismatch");
});

test('SMMA batch large range', () => {

    const close = new Float64Array(testData.close.slice(0, 200));

    const batchValues = wasm.smma_batch_legacy(close, 7, 100, 1);
    const metadata = wasm.smma_batch_metadata(7, 100, 1);
    const dims = wasm.smma_batch_rows_cols(7, 100, 1, 200);

    const expectedPeriodCount = 94;
    assert.strictEqual(dims[0], expectedPeriodCount);
    assert.strictEqual(dims[1], 200);
    assert.strictEqual(metadata.length, expectedPeriodCount);
});

test('SMMA edge case period one', () => {

    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    const result = wasm.smma(data, 1);


    assertArrayClose(result, data, 1e-10, "SMMA with period=1 should equal input");
});

test('SMMA constant values', () => {

    const constantValue = 42.0;
    const data = new Float64Array(50);
    data.fill(constantValue);

    const result = wasm.smma(data, 10);


    for (let i = 10; i < result.length; i++) {
        assertClose(result[i], constantValue, 1e-10,
                   `SMMA of constant should be ${constantValue} at index ${i}`);
    }
});

test('SMMA formula verification', () => {

    const data = new Float64Array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]);
    const period = 3;

    const result = wasm.smma(data, period);


    assertAllNaN(result.slice(0, period - 1), "First period-1 values should be NaN");


    const expectedFirst = (10 + 12 + 14) / 3;
    assertClose(result[period - 1], expectedFirst, 1e-10, "First SMMA value incorrect");


    const expectedSecond = (expectedFirst * (period - 1) + data[period]) / period;

    assertClose(result[period], expectedSecond, 1e-10, "Second SMMA value incorrect");
});


test('SMMA warmup period', () => {

    const close = new Float64Array(testData.close.slice(0, 50));
    const period = 10;

    const result = wasm.smma(close, period);



    assertAllNaN(result.slice(0, period - 1), `Expected NaN in first ${period - 1} values`);


    assert(!isNaN(result[period - 1]), `Expected valid value at index ${period - 1}`);
});

test('SMMA batch step parameter', () => {

    const close = new Float64Array(testData.close.slice(0, 50));


    const batchValues = wasm.smma_batch_legacy(close, 5, 9, 2);
    const metadata = wasm.smma_batch_metadata(5, 9, 2);
    const dims = wasm.smma_batch_rows_cols(5, 9, 2, 50);

    assert.strictEqual(dims[0], 3);
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 5);
    assert.strictEqual(metadata[1], 7);
    assert.strictEqual(metadata[2], 9);
});

test('SMMA batch zero step', () => {

    const close = new Float64Array(testData.close.slice(0, 50));

    const batchValues = wasm.smma_batch_legacy(close, 7, 7, 0);
    const metadata = wasm.smma_batch_metadata(7, 7, 0);
    const dims = wasm.smma_batch_rows_cols(7, 7, 0, 50);

    assert.strictEqual(dims[0], 1);
    assert.strictEqual(metadata.length, 1);
    assert.strictEqual(metadata[0], 7);
});



test('SMMA fast API basic', () => {

    const close = new Float64Array(testData.close.slice(0, 100));
    const period = 7;


    const safeResult = wasm.smma(close, period);


    const inPtr = wasm.smma_alloc(close.length);
    const outPtr = wasm.smma_alloc(close.length);

    try {

        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        memory.set(close, inPtr / 8);


        wasm.smma_into(inPtr, outPtr, close.length, period);


        const fastResult = new Float64Array(wasm.__wasm.memory.buffer, outPtr, close.length);


        assertArrayClose(fastResult, safeResult, 1e-10, "Fast API should match safe API");
    } finally {
        wasm.smma_free(inPtr, close.length);
        wasm.smma_free(outPtr, close.length);
    }
});

test('SMMA fast API aliasing', () => {

    const close = new Float64Array(testData.close.slice(0, 50));
    const period = 7;


    const expected = wasm.smma(close, period);


    const ptr = wasm.smma_alloc(close.length);

    try {

        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        memory.set(close, ptr / 8);


        wasm.smma_into(ptr, ptr, close.length, period);


        const result = new Float64Array(wasm.__wasm.memory.buffer, ptr, close.length);


        assertArrayClose(result, expected, 1e-10, "In-place computation should work correctly");
    } finally {
        wasm.smma_free(ptr, close.length);
    }
});

test('SMMA fast API null pointer', () => {

    assert.throws(() => {
        wasm.smma_into(0, 0, 100, 7);
    }, /null pointer/);
});

test('SMMA batch new API', () => {

    const close = new Float64Array(testData.close.slice(0, 50));

    const config = {
        period_range: [5, 10, 1]
    };

    const result = wasm.smma_batch(close, config);

    assert.strictEqual(result.rows, 6);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.values.length, 6 * 50);
    assert.strictEqual(result.combos.length, 6);


    for (let i = 0; i < 6; i++) {
        assert.strictEqual(result.combos[i].period, 5 + i);
    }
});

test('SMMA batch fast API', () => {

    const close = new Float64Array(testData.close.slice(0, 50));

    const inPtr = wasm.smma_alloc(close.length);
    const outPtr = wasm.smma_alloc(6 * close.length);

    try {

        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        memory.set(close, inPtr / 8);


        const rows = wasm.smma_batch_into(inPtr, outPtr, close.length, 5, 10, 1);

        assert.strictEqual(rows, 6);


        const batchResult = new Float64Array(wasm.__wasm.memory.buffer, outPtr, rows * close.length);


        const firstRow = batchResult.slice(0, close.length);
        const singleResult = wasm.smma(close, 5);
        assertArrayClose(firstRow, singleResult, 1e-10, "First batch row should match single calculation");
    } finally {
        wasm.smma_free(inPtr, close.length);
        wasm.smma_free(outPtr, 6 * close.length);
    }
});

test('SMMA streaming', () => {

    const close = new Float64Array(testData.close.slice(0, 100));
    const period = 7;


    const batchResult = wasm.smma(close, period);


    const streamValues = new Float64Array(close.length);
    streamValues.fill(NaN);


    for (let i = period; i <= close.length; i++) {
        const partialData = close.slice(0, i);
        const partialResult = wasm.smma(partialData, period);
        streamValues[i - 1] = partialResult[i - 1];
    }


    for (let i = period - 1; i < close.length; i++) {
        if (!isNaN(batchResult[i]) && !isNaN(streamValues[i])) {
            assertClose(batchResult[i], streamValues[i], 1e-10,
                       `Streaming mismatch at index ${i}`);
        }
    }
});

test('SMMA batch validation', () => {

    const close = new Float64Array(testData.close.slice(0, 50));


    assert.throws(() => {
        wasm.smma_batch(close, {});
    }, /Invalid config/);


    const desc = wasm.smma_batch(close, {
        period_range: [10, 5, 1]
    });
    assert.strictEqual(desc.rows, 6);
    assert.strictEqual(desc.combos.length, 6);


    assert.throws(() => {
        wasm.smma_batch(close, {
            period_range: [0, 5, 1]
        });
    }, /Invalid period|unreachable/);
});

test('SMMA batch matches individual', () => {

    const close = new Float64Array(testData.close.slice(0, 100));


    const periods = [5, 7, 10, 14];
    const batchResult = wasm.smma_batch(close, {
        period_range: [5, 14, 1]
    });


    for (const period of periods) {
        const rowIdx = period - 5;
        const rowStart = rowIdx * close.length;
        const rowEnd = rowStart + close.length;
        const batchRow = batchResult.values.slice(rowStart, rowEnd);


        const individualResult = wasm.smma(close, period);


        assertArrayClose(
            batchRow,
            individualResult,
            1e-10,
            `Batch row for period ${period} doesn't match individual calculation`
        );
    }
});

test('SMMA leading NaNs', () => {

    const data = new Float64Array([NaN, NaN, NaN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    const period = 3;

    const result = wasm.smma(data, period);




    assertAllNaN(result.slice(0, 5), "Expected NaN through index 4");
    assert(!isNaN(result[5]), "Expected valid value at index 5");


    const expectedFirst = (1.0 + 2.0 + 3.0) / 3;
    assertClose(result[5], expectedFirst, 1e-10, "First valid value incorrect");
});

test('SMMA batch fast API error handling', () => {

    const close = new Float64Array([1, 2, 3, 4, 5]);


    assert.throws(() => {
        wasm.smma_batch_into(0, 0, close.length, 2, 3, 1);
    }, /null pointer/);


    const inPtr = wasm.smma_alloc(close.length);
    const outPtr = wasm.smma_alloc(close.length);

    try {
        assert.throws(() => {
            wasm.smma_batch_into(inPtr, outPtr, close.length, 10, 5, 1);
        });
    } finally {
        wasm.smma_free(inPtr, close.length);
        wasm.smma_free(outPtr, close.length);
    }
});
