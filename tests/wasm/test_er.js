
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

test('ER accuracy', async () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.er;

    const result = wasm.er_js(
        close,
        expected.defaultParams.period
    );

    assert.strictEqual(result.length, close.length);


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "ER last 5 values mismatch"
    );


    const valuesAt100 = result.slice(100, 105);
    assertArrayClose(
        valuesAt100,
        expected.valuesAt100_104,
        1e-8,
        "ER values at indices 100-104 mismatch"
    );


});

test('ER partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.er_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('ER default candles', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.er_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('ER zero period', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.er_js(inputData, 0);
    }, /Invalid period/);
});

test('ER period exceeds length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.er_js(dataSmall, 10);
    }, /Invalid period/);
});

test('ER very small dataset', () => {

    const singlePoint = new Float64Array([42.0]);

    assert.throws(() => {
        wasm.er_js(singlePoint, 5);
    }, /Invalid period|Not enough valid data/);
});

test('ER empty input', () => {

    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.er_js(empty, 5);
    }, /Input data slice is empty|All values are NaN/);
});

test('ER all NaN input', () => {

    const allNan = new Float64Array(100);
    allNan.fill(NaN);

    assert.throws(() => {
        wasm.er_js(allNan, 5);
    }, /All input data values are NaN/);
});

test('ER NaN handling', () => {

    const close = new Float64Array(testData.close);
    const period = 5;

    const result = wasm.er_js(close, period);
    assert.strictEqual(result.length, close.length);


    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }


    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    const warmupEnd = firstValid + period - 1;


    assertAllNaN(result.slice(0, warmupEnd), "Expected NaN in warmup period");
});

test('ER reinput', () => {

    const close = new Float64Array(testData.close);


    const firstResult = wasm.er_js(close, 5);
    assert.strictEqual(firstResult.length, close.length);


    const secondResult = wasm.er_js(firstResult, 5);
    assert.strictEqual(secondResult.length, firstResult.length);


    const validValues = secondResult.filter(v => !isNaN(v));
    validValues.forEach(v => {
        assert(v >= 0.0 && v <= 1.0, `ER value ${v} outside valid range [0.0, 1.0]`);
    });
});

test('ER consistency', () => {

    const expected = EXPECTED_OUTPUTS.er;


    const trendingData = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const result = wasm.er_js(trendingData, 5);


    const validValues = result.slice(4);
    assertArrayClose(
        validValues,
        expected.trendingDataValues,
        1e-10,
        "ER trending data mismatch"
    );


    const choppyData = new Float64Array([1, 5, 2, 6, 3, 7, 4, 8, 5, 9]);
    const result2 = wasm.er_js(choppyData, 5);


    const validValues2 = result2.slice(4);
    assertArrayClose(
        validValues2,
        expected.choppyDataValues,
        1e-10,
        "ER choppy data mismatch"
    );
});

test('ER fast API - basic calculation', async () => {

    const close = new Float64Array(testData.close);
    const len = close.length;


    const inPtr = wasm.er_alloc(len);
    const outPtr = wasm.er_alloc(len);

    try {

        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        const inOffset = inPtr / 8;
        const outOffset = outPtr / 8;


        memory.set(close, inOffset);


        wasm.er_into(inPtr, outPtr, len, 5);


        const result = new Float64Array(memory.buffer, outPtr, len);
        const resultCopy = new Float64Array(result);


        const safeResult = wasm.er_js(close, 5);
        assertArrayClose(resultCopy, safeResult, 1e-10, "Fast API result differs from safe API");

    } finally {

        wasm.er_free(inPtr, len);
        wasm.er_free(outPtr, len);
    }
});

test('ER fast API - in-place operation (aliasing)', async () => {

    const close = new Float64Array(testData.close);
    const len = close.length;


    const ptr = wasm.er_alloc(len);

    try {

        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        const offset = ptr / 8;


        memory.set(close, offset);


        wasm.er_into(ptr, ptr, len, 5);


        const result = new Float64Array(memory.buffer, ptr, len);
        const resultCopy = new Float64Array(result);


        const safeResult = wasm.er_js(close, 5);
        assertArrayClose(resultCopy, safeResult, 1e-10, "In-place operation result differs from safe API");

    } finally {

        wasm.er_free(ptr, len);
    }
});

test('ER batch - single period', () => {

    const close = new Float64Array(testData.close.slice(0, 100));
    const expected = EXPECTED_OUTPUTS.er;

    const config = {
        period_range: [expected.defaultParams.period, expected.defaultParams.period, 0]
    };

    const result = wasm.er_batch(close, config);

    assert.strictEqual(typeof result, 'object');
    assert(Array.isArray(result.values));
    assert(Array.isArray(result.combos));
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, close.length);


    assert.strictEqual(result.combos[0].period, expected.defaultParams.period);


    const singleResult = wasm.er_js(close, expected.defaultParams.period);
    assertArrayClose(
        new Float64Array(result.values),
        singleResult,
        1e-10,
        "Batch single period differs from single calculation"
    );
});

test('ER batch - multiple periods', () => {

    const close = new Float64Array(testData.close.slice(0, 100));

    const config = {
        period_range: [5, 15, 5]
    };

    const result = wasm.er_batch(close, config);

    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, 3 * close.length);
    assert.strictEqual(result.combos.length, 3);


    const periods = [5, 10, 15];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * close.length;
        const rowEnd = rowStart + close.length;
        const rowData = new Float64Array(result.values.slice(rowStart, rowEnd));

        const singleResult = wasm.er_js(close, periods[i]);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Batch row ${i} (period=${periods[i]}) differs from single calculation`
        );
    }
});

test('ER batch metadata', () => {

    const close = new Float64Array(20);
    close.fill(100);

    const result = wasm.er_batch(close, {
        period_range: [5, 7, 2]
    });


    assert.strictEqual(result.combos.length, 2);


    assert.strictEqual(result.combos[0].period, 5);


    assert.strictEqual(result.combos[1].period, 7);
});

test('ER SIMD128 consistency', () => {


    const testCases = [
        { size: 10, period: 5 },
        { size: 100, period: 10 },
        { size: 1000, period: 20 },
        { size: 10000, period: 50 }
    ];

    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }

        const result = wasm.er_js(data, testCase.period);


        assert.strictEqual(result.length, data.length);


        for (let i = 0; i < testCase.period - 1; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }


        for (let i = testCase.period - 1; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            assert(result[i] >= 0 && result[i] <= 1, `ER value ${result[i]} outside [0,1] range`);
        }
    }
});

test('ER zero-copy memory management', () => {

    const sizes = [100, 1000, 10000];

    for (const size of sizes) {
        const ptr = wasm.er_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);


        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }


        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }


        wasm.er_free(ptr, size);
    }
});

test('ER batch - fast API', () => {

    const close = new Float64Array(testData.close.slice(0, 100));
    const len = close.length;


    const periods = [5, 10, 15];
    const outputLen = periods.length * len;


    const inPtr = wasm.er_alloc(len);
    const outPtr = wasm.er_alloc(outputLen);

    try {
        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        const inOffset = inPtr / 8;
        const outOffset = outPtr / 8;


        memory.set(close, inOffset);


        const rows = wasm.er_batch_into(inPtr, outPtr, len, 5, 15, 5);

        assert.strictEqual(rows, 3);


        const result = new Float64Array(memory.buffer, outPtr, outputLen);
        const resultCopy = new Float64Array(result);


        const config = { period_range: [5, 15, 5] };
        const safeResult = wasm.er_batch(close, config);

        assertArrayClose(
            resultCopy,
            new Float64Array(safeResult.values),
            1e-10,
            "Fast batch API differs from safe batch API"
        );

    } finally {
        wasm.er_free(inPtr, len);
        wasm.er_free(outPtr, outputLen);
    }
});