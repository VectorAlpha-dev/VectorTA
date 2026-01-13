
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

test('ULTOSC basic calculation', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);

    const result = wasm.ultosc_js(high, low, close, 7, 14, 28);
    assert.strictEqual(result.length, close.length);


    const expectedLastFive = [
        41.25546890298435,
        40.83865967175865,
        48.910324164909625,
        45.43113094857947,
        42.163165136766295,
    ];


    for (let i = 0; i < 5; i++) {
        assertClose(result[result.length - 5 + i], expectedLastFive[i], 1e-8);
    }
});

test('ULTOSC custom parameters', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);


    const result = wasm.ultosc_js(high, low, close, 5, 10, 20);

    assert.strictEqual(result.length, close.length);
    assert.ok(!isNaN(result[result.length - 1]));


    for (let i = 0; i < 19; i++) {
        assert.ok(isNaN(result[i]));
    }
});

test('ULTOSC memory allocation/deallocation', () => {
    const size = 1000;
    const ptr = wasm.ultosc_alloc(size);
    assert.ok(ptr !== 0, 'Should allocate non-null pointer');


    wasm.ultosc_free(ptr, size);
});

test('ULTOSC fast API', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const len = close.length;


    const highPtr = wasm.ultosc_alloc(len);
    const lowPtr = wasm.ultosc_alloc(len);
    const closePtr = wasm.ultosc_alloc(len);
    const outPtr = wasm.ultosc_alloc(len);

    try {

        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);

        highView.set(high);
        lowView.set(low);
        closeView.set(close);


        wasm.ultosc_into(
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            len,
            7,
            14,
            28
        );


        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const output = new Float64Array(outView);


        const safeResult = wasm.ultosc_js(high, low, close, 7, 14, 28);
        assertArrayClose(output, safeResult, 1e-10);
    } finally {

        wasm.ultosc_free(highPtr, len);
        wasm.ultosc_free(lowPtr, len);
        wasm.ultosc_free(closePtr, len);
        wasm.ultosc_free(outPtr, len);
    }
});

test('ULTOSC fast API with aliasing', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const len = close.length;


    const closeCopy = new Float64Array(close);


    const highPtr = wasm.ultosc_alloc(len);
    const lowPtr = wasm.ultosc_alloc(len);
    const closePtr = wasm.ultosc_alloc(len);

    try {

        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);

        highView.set(high);
        lowView.set(low);
        closeView.set(close);


        wasm.ultosc_into(
            highPtr,
            lowPtr,
            closePtr,
            closePtr,
            len,
            7,
            14,
            28
        );


        const modifiedClose = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const result = new Float64Array(modifiedClose);


        const expectedResult = wasm.ultosc_js(high, low, closeCopy, 7, 14, 28);
        assertArrayClose(result, expectedResult, 1e-10);
    } finally {

        wasm.ultosc_free(highPtr, len);
        wasm.ultosc_free(lowPtr, len);
        wasm.ultosc_free(closePtr, len);
    }
});

test('ULTOSC batch calculation', async () => {
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));

    const config = {
        timeperiod1_range: [5, 9, 2],
        timeperiod2_range: [12, 16, 2],
        timeperiod3_range: [26, 30, 2]
    };

    const result = await wasm.ultosc_batch(high, low, close, config);

    assert.ok(result.values);
    assert.ok(result.combos);
    assert.strictEqual(result.rows, 27);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, result.rows * result.cols);
    assert.strictEqual(result.combos.length, result.rows);


    let targetIdx = -1;
    for (let i = 0; i < result.combos.length; i++) {
        const combo = result.combos[i];
        if (combo.timeperiod1 === 7 && combo.timeperiod2 === 14 && combo.timeperiod3 === 28) {
            targetIdx = i;
            break;
        }
    }

    assert.ok(targetIdx >= 0, 'Should find (7, 14, 28) combination');


    const rowValues = new Float64Array(result.cols);
    for (let j = 0; j < result.cols; j++) {
        rowValues[j] = result.values[targetIdx * result.cols + j];
    }


    const singleResult = wasm.ultosc_js(high, low, close, 7, 14, 28);
    assertArrayClose(rowValues, singleResult, 1e-10);
});

test('ULTOSC error handling - empty data', () => {
    assert.throws(() => {
        wasm.ultosc_js(new Float64Array([]), new Float64Array([]), new Float64Array([]), 7, 14, 28);
    }, /EmptyData|empty/i);
});

test('ULTOSC error handling - mismatched lengths', () => {
    assert.throws(() => {
        wasm.ultosc_js(
            new Float64Array([1, 2]),
            new Float64Array([0.5, 1.5, 2.5]),
            new Float64Array([0.8, 1.8]),
            7, 14, 28
        );
    });
});

test('ULTOSC error handling - zero period', () => {
    const high = new Float64Array([10.0, 11.0, 12.0]);
    const low = new Float64Array([9.0, 10.0, 11.0]);
    const close = new Float64Array([9.5, 10.5, 11.5]);

    assert.throws(() => {
        wasm.ultosc_js(high, low, close, 0, 14, 28);
    }, /Invalid period/);
});

test('ULTOSC error handling - period exceeding data length', () => {
    const high = new Float64Array([10.0, 11.0, 12.0]);
    const low = new Float64Array([9.0, 10.0, 11.0]);
    const close = new Float64Array([9.5, 10.5, 11.5]);

    assert.throws(() => {
        wasm.ultosc_js(high, low, close, 7, 14, 50);
    }, /Period exceeds data length|Invalid periods/i);
});

test('ULTOSC NaN handling', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);


    high[0] = NaN;
    high[1] = NaN;
    low[0] = NaN;
    low[1] = NaN;
    close[0] = NaN;
    close[1] = NaN;

    const result = wasm.ultosc_js(high, low, close, 7, 14, 28);

    assert.strictEqual(result.length, close.length);

    for (let i = 0; i < 30; i++) {
        assert.ok(isNaN(result[i]));
    }
});

test('ULTOSC consistency', () => {
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));


    const result1 = wasm.ultosc_js(high, low, close, 7, 14, 28);
    const result2 = wasm.ultosc_js(high, low, close, 7, 14, 28);


    assertArrayClose(result1, result2, 0.0);
});

test('ULTOSC edge case - minimum data', () => {


    const size = 29;
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);


    for (let i = 0; i < size; i++) {
        high[i] = 10 + i * 0.1;
        low[i] = 9 + i * 0.1;
        close[i] = 9.5 + i * 0.1;
    }

    const result = wasm.ultosc_js(high, low, close, 7, 14, 28);
    assert.strictEqual(result.length, size);


    assert.ok(!isNaN(result[size - 1]));
});


test.skip('ULTOSC WASM vs Rust comparison', async () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);

    const wasmResult = wasm.ultosc_js(high, low, close, 7, 14, 28);
    const rustResult = await compareWithRust('ultosc', { high, low, close, timeperiod1: 7, timeperiod2: 14, timeperiod3: 28 });

    assertArrayClose(wasmResult, rustResult, 1e-10);
});