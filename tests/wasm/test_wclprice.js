
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

test('WCLPRICE slices', () => {

    const high = new Float64Array([59230.0, 59220.0, 59077.0, 59160.0, 58717.0]);
    const low = new Float64Array([59222.0, 59211.0, 59077.0, 59143.0, 58708.0]);
    const close = new Float64Array([59225.0, 59210.0, 59080.0, 59150.0, 58710.0]);

    const result = wasm.wclprice_js(high, low, close);
    const expected = [59225.5, 59212.75, 59078.5, 59150.75, 58711.25];

    assertArrayClose(result, expected, 1e-2, "WCLPRICE values mismatch");
});

test('WCLPRICE candles', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);

    const result = wasm.wclprice_js(high, low, close);
    assert.strictEqual(result.length, close.length);


    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i])) {
            assert(low[i] <= result[i] && result[i] <= high[i],
                `WCLPRICE value ${result[i]} at index ${i} is outside range [${low[i]}, ${high[i]}]`);
        }
    }
});

test('WCLPRICE empty data', () => {

    const high = new Float64Array([]);
    const low = new Float64Array([]);
    const close = new Float64Array([]);

    assert.throws(() => {
        wasm.wclprice_js(high, low, close);
    }, /Empty data/);
});

test('WCLPRICE all NaN', () => {

    const high = new Float64Array([NaN, NaN]);
    const low = new Float64Array([NaN, NaN]);
    const close = new Float64Array([NaN, NaN]);

    assert.throws(() => {
        wasm.wclprice_js(high, low, close);
    }, /all values are NaN/);
});

test('WCLPRICE partial NaN', () => {

    const high = new Float64Array([NaN, 59000.0]);
    const low = new Float64Array([NaN, 58950.0]);
    const close = new Float64Array([NaN, 58975.0]);

    const result = wasm.wclprice_js(high, low, close);


    assert(isNaN(result[0]));

    const expected = (59000.0 + 58950.0 + 2.0 * 58975.0) / 4.0;
    assertClose(result[1], expected, 1e-8, "WCLPRICE calculation incorrect");
});

test('WCLPRICE formula', () => {

    const high = new Float64Array([100.0]);
    const low = new Float64Array([90.0]);
    const close = new Float64Array([95.0]);

    const result = wasm.wclprice_js(high, low, close);
    const expected = (100.0 + 90.0 + 2.0 * 95.0) / 4.0;

    assertClose(result[0], expected, 1e-10, "WCLPRICE formula incorrect");
});

test('WCLPRICE mismatched lengths', () => {

    const high = new Float64Array([100.0, 101.0, 102.0]);
    const low = new Float64Array([90.0, 91.0]);
    const close = new Float64Array([95.0, 96.0, 97.0]);


    const result = wasm.wclprice_js(high, low, close);
    assert.strictEqual(result.length, 2);
});

test('WCLPRICE fast API (no aliasing)', () => {

    const high = new Float64Array([100.0, 101.0, 102.0]);
    const low = new Float64Array([90.0, 91.0, 92.0]);
    const close = new Float64Array([95.0, 96.0, 97.0]);
    const output = new Float64Array(3);

    const highPtr = wasm.wclprice_alloc(3);
    const lowPtr = wasm.wclprice_alloc(3);
    const closePtr = wasm.wclprice_alloc(3);
    const outPtr = wasm.wclprice_alloc(3);

    const memory = new Float64Array(wasm.__wasm.memory.buffer);
    memory.set(high, highPtr / 8);
    memory.set(low, lowPtr / 8);
    memory.set(close, closePtr / 8);

    wasm.wclprice_into(highPtr, lowPtr, closePtr, outPtr, 3);

    const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, 3);


    for (let i = 0; i < 3; i++) {
        const expected = (high[i] + low[i] + 2.0 * close[i]) / 4.0;
        assertClose(result[i], expected, 1e-10, `Fast API value mismatch at index ${i}`);
    }


    wasm.wclprice_free(highPtr, 3);
    wasm.wclprice_free(lowPtr, 3);
    wasm.wclprice_free(closePtr, 3);
    wasm.wclprice_free(outPtr, 3);
});

test('WCLPRICE fast API (with aliasing)', () => {

    const data = new Float64Array([100.0, 101.0, 102.0]);
    const low = new Float64Array([90.0, 91.0, 92.0]);
    const close = new Float64Array([95.0, 96.0, 97.0]);

    const dataPtr = wasm.wclprice_alloc(3);
    const lowPtr = wasm.wclprice_alloc(3);
    const closePtr = wasm.wclprice_alloc(3);

    const memory = new Float64Array(wasm.__wasm.memory.buffer);
    memory.set(data, dataPtr / 8);
    memory.set(low, lowPtr / 8);
    memory.set(close, closePtr / 8);


    wasm.wclprice_into(dataPtr, lowPtr, closePtr, dataPtr, 3);

    const result = new Float64Array(wasm.__wasm.memory.buffer, dataPtr, 3);


    for (let i = 0; i < 3; i++) {
        const expected = (data[i] + low[i] + 2.0 * close[i]) / 4.0;
        assertClose(result[i], expected, 1e-10, `Fast API aliasing value mismatch at index ${i}`);
    }


    wasm.wclprice_free(dataPtr, 3);
    wasm.wclprice_free(lowPtr, 3);
    wasm.wclprice_free(closePtr, 3);
});

test('WCLPRICE batch', () => {

    const high = new Float64Array([100.0, 101.0, 102.0]);
    const low = new Float64Array([90.0, 91.0, 92.0]);
    const close = new Float64Array([95.0, 96.0, 97.0]);


    const config = {};

    const result = wasm.wclprice_batch(high, low, close, config);

    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, 3);
    assert.strictEqual(result.values.length, 3);


    const singleResult = wasm.wclprice_js(high, low, close);
    assertArrayClose(result.values, singleResult, 1e-10, "Batch values mismatch");
});

test('WCLPRICE with test data', async () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);

    const result = wasm.wclprice_js(high, low, close);


    await compareWithRust('wclprice', result, 'hlc', null);
});

test('WCLPRICE reference last five', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);

    const result = wasm.wclprice_js(high, low, close);
    const expectedLastFive = [59225.5, 59212.75, 59078.5, 59150.75, 58711.25];
    const start = result.length - 5;
    const actualLastFive = Array.from(result.slice(start));
    assertArrayClose(actualLastFive, expectedLastFive, 1e-8, 'Reference last five mismatch');
});
