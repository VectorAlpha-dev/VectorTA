
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

test('VWAP partial params', () => {
    const timestamps = testData.timestamp;
    const volumes = testData.volume;

    const prices = testData.high.map((h, i) =>
        (h + testData.low[i] + testData.close[i]) / 3.0
    );


    const result = wasm.vwap_js(timestamps, volumes, prices);
    assert.strictEqual(result.length, prices.length, 'Output length should match input');


    const result2 = wasm.vwap_js(timestamps, volumes, prices, undefined, "scalar");
    assert.strictEqual(result2.length, prices.length, 'Output length should match input with scalar kernel');
});

test('VWAP accuracy', () => {
    const timestamps = testData.timestamp;
    const volumes = testData.volume;

    const prices = testData.high.map((h, i) =>
        (h + testData.low[i] + testData.close[i]) / 3.0
    );


    const expectedLastFive = [
        59353.05963230107,
        59330.15815713043,
        59289.94649532547,
        59274.6155462414,
        58730.0
    ];

    const result = wasm.vwap_js(timestamps, volumes, prices, "1D", undefined);

    assert.strictEqual(result.length, prices.length, 'Output length should match input');


    const last5 = result.slice(-5);
    assertArrayClose(last5, expectedLastFive, 1e-5, 'VWAP last 5 values mismatch');
});

test('VWAP anchor parsing error', () => {
    const timestamps = testData.timestamp;
    const volumes = testData.volume;
    const prices = testData.close;

    assert.throws(() => {
        wasm.vwap_js(timestamps, volumes, prices, "xyz");
    }, /Error parsing anchor/, 'Should throw error for invalid anchor');
});

test('VWAP kernel parameter', () => {
    const timestamps = testData.timestamp;
    const volumes = testData.volume;
    const prices = testData.close;


    const result1 = wasm.vwap_js(timestamps, volumes, prices, "1d", "scalar");
    assert.strictEqual(result1.length, prices.length, 'Scalar kernel should work');

    const result2 = wasm.vwap_js(timestamps, volumes, prices, "1d", "scalar_batch");
    assert.strictEqual(result2.length, prices.length, 'Scalar batch kernel should work');


    assert.throws(() => {
        wasm.vwap_js(timestamps, volumes, prices, "1d", "invalid_kernel");
    }, /Unknown kernel/, 'Should throw error for invalid kernel');
});

test('VWAP mismatch lengths', () => {
    const timestamps = [1000, 2000, 3000];
    const volumes = [100.0, 200.0];
    const prices = [10.0, 20.0, 30.0];

    assert.throws(() => {
        wasm.vwap_js(timestamps, volumes, prices);
    }, /Mismatch in length/, 'Should throw error for mismatched array lengths');
});

test('VWAP empty data', () => {
    const empty = [];

    assert.throws(() => {
        wasm.vwap_js(empty, empty, empty);
    }, /No data/, 'Should throw error for empty input');
});

test('VWAP batch processing', () => {
    const timestamps = testData.timestamp;
    const volumes = testData.volume;

    const prices = testData.high.map((h, i) =>
        (h + testData.low[i] + testData.close[i]) / 3.0
    );


    const result_old = wasm.vwap_batch(
        timestamps,
        volumes,
        prices,
        "1d", "3d", 1
    );


    assert(result_old.values, 'Result should have values array');
    assert(result_old.combos, 'Result should have combos array');
    assert.strictEqual(result_old.rows, 3, 'Should have 3 rows (1d, 2d, 3d)');
    assert.strictEqual(result_old.cols, prices.length, 'Cols should match input length');
    assert.deepStrictEqual(result_old.combos.map(c => c.anchor), ["1d", "2d", "3d"], 'Anchors should match expected');


    assert.strictEqual(result_old.values.length, result_old.rows * result_old.cols,
        'Values array should have rows × cols elements');


    const metadata = wasm.vwap_batch_metadata_js("1d", "3d", 1);
    assert.deepStrictEqual(metadata, ["1d", "2d", "3d"],
        'Metadata should contain expected anchors');


    const single_vwap = wasm.vwap_js(timestamps, volumes, prices, "1d");
    const first_row = result_old.values.slice(0, prices.length);
    assertArrayClose(first_row, single_vwap, 1e-9, 'Batch 1d row should match single calculation');
});

test('VWAP default params', () => {
    const timestamps = testData.timestamp;
    const volumes = testData.volume;
    const prices = testData.close;


    const result = wasm.vwap_js(timestamps, volumes, prices);
    assert.strictEqual(result.length, prices.length, 'Output length should match input');
});

test('VWAP nan handling', () => {
    const timestamps = testData.timestamp;
    const volumes = testData.volume;

    const prices = testData.high.map((h, i) =>
        (h + testData.low[i] + testData.close[i]) / 3.0
    );

    const result = wasm.vwap_js(timestamps, volumes, prices);
    assert.strictEqual(result.length, prices.length, 'Output length should match input');


    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i])) {
            assert(isFinite(result[i]), `Found non-finite value at index ${i}`);
        }
    }
});

test('VWAP fast API (vwap_into)', () => {
    const timestamps = testData.timestamp;
    const volumes = testData.volume;
    const prices = testData.close;
    const len = prices.length;


    const outPtr = wasm.vwap_alloc(len);

    try {

        const timestampsPtr = wasm.vwap_alloc(len);
        const volumesPtr = wasm.vwap_alloc(len);
        const pricesPtr = wasm.vwap_alloc(len);

        const timestampsView = new Float64Array(wasm.__wasm.memory.buffer, timestampsPtr, len);
        const volumesView = new Float64Array(wasm.__wasm.memory.buffer, volumesPtr, len);
        const pricesView = new Float64Array(wasm.__wasm.memory.buffer, pricesPtr, len);

        timestampsView.set(timestamps);
        volumesView.set(volumes);
        pricesView.set(prices);


        wasm.vwap_into(timestampsPtr, volumesPtr, pricesPtr, outPtr, len, "1d");


        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const resultCopy = Array.from(result);


        const expected = wasm.vwap_js(timestamps, volumes, prices, "1d");
        assertArrayClose(resultCopy, expected, 1e-9, 'Fast API should match safe API');


        wasm.vwap_free(timestampsPtr, len);
        wasm.vwap_free(volumesPtr, len);
        wasm.vwap_free(pricesPtr, len);
    } finally {

        wasm.vwap_free(outPtr, len);
    }
});

test('VWAP fast API aliasing', () => {
    const timestamps = testData.timestamp.slice(0, 100);
    const volumes = testData.volume.slice(0, 100);
    const prices = testData.close.slice(0, 100);
    const len = prices.length;


    const pricesPtr = wasm.vwap_alloc(len);
    const timestampsPtr = wasm.vwap_alloc(len);
    const volumesPtr = wasm.vwap_alloc(len);

    try {

        const timestampsView = new Float64Array(wasm.__wasm.memory.buffer, timestampsPtr, len);
        const volumesView = new Float64Array(wasm.__wasm.memory.buffer, volumesPtr, len);
        const pricesView = new Float64Array(wasm.__wasm.memory.buffer, pricesPtr, len);

        timestampsView.set(timestamps);
        volumesView.set(volumes);
        pricesView.set(prices);


        const originalPrices = Array.from(pricesView);


        wasm.vwap_into(timestampsPtr, volumesPtr, pricesPtr, pricesPtr, len, "1d");


        const result = Array.from(pricesView);


        const expected = wasm.vwap_js(timestamps, volumes, originalPrices, "1d");
        assertArrayClose(result, expected, 1e-9, 'Aliased fast API should match safe API');
    } finally {
        wasm.vwap_free(timestampsPtr, len);
        wasm.vwap_free(volumesPtr, len);
        wasm.vwap_free(pricesPtr, len);
    }
});

test('VWAP batch with serde config', () => {
    const timestamps = testData.timestamp;
    const volumes = testData.volume;
    const prices = testData.close;


    const result = wasm.vwap_batch(timestamps, volumes, prices, "1d", "3d", 1);


    assert(result.values, 'Result should have values array');
    assert(result.combos, 'Result should have combos array');
    assert.strictEqual(result.rows, 3, 'Should have 3 rows (1d, 2d, 3d)');
    assert.strictEqual(result.cols, prices.length, 'Cols should match input length');
    assert.deepStrictEqual(result.combos.map(c => c.anchor), ["1d", "2d", "3d"], 'Anchors should match expected');


    assert.strictEqual(result.values.length, result.rows * result.cols,
        'Values array should have rows × cols elements');
});

test('VWAP batch_into API', () => {
    const timestamps = testData.timestamp.slice(0, 100);
    const volumes = testData.volume.slice(0, 100);
    const prices = testData.close.slice(0, 100);
    const len = prices.length;


    const rows = 3;
    const totalSize = rows * len;


    const timestampsPtr = wasm.vwap_alloc(len);
    const volumesPtr = wasm.vwap_alloc(len);
    const pricesPtr = wasm.vwap_alloc(len);
    const outPtr = wasm.vwap_alloc(totalSize);

    try {

        const timestampsView = new Float64Array(wasm.__wasm.memory.buffer, timestampsPtr, len);
        const volumesView = new Float64Array(wasm.__wasm.memory.buffer, volumesPtr, len);
        const pricesView = new Float64Array(wasm.__wasm.memory.buffer, pricesPtr, len);

        timestampsView.set(timestamps);
        volumesView.set(volumes);
        pricesView.set(prices);


        const actualRows = wasm.vwap_batch_into(
            timestampsPtr, volumesPtr, pricesPtr, outPtr, len,
            "1d", "3d", 1
        );

        assert.strictEqual(actualRows, rows, 'Should return correct number of rows');


        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
        const resultCopy = Array.from(result);


        const expected1d = wasm.vwap_js(timestamps, volumes, prices, "1d");
        const firstRow = resultCopy.slice(0, len);
        assertArrayClose(firstRow, expected1d, 1e-9, 'First row should match 1d VWAP');
    } finally {
        wasm.vwap_free(timestampsPtr, len);
        wasm.vwap_free(volumesPtr, len);
        wasm.vwap_free(pricesPtr, len);
        wasm.vwap_free(outPtr, totalSize);
    }
});

test('VWAP all NaN input', () => {

    const timestamps = Array.from({length: 100}, (_, i) => i * 1000);
    const volumes = new Array(100).fill(100.0);
    const allNaN = new Array(100).fill(NaN);


    const result = wasm.vwap_js(timestamps, volumes, allNaN);
    assert.strictEqual(result.length, allNaN.length, 'Output length should match input');
    assertAllNaN(result, 'Expected all NaN output for all NaN prices');
});

test('VWAP zero volume', () => {

    const timestamps = testData.timestamp.slice(0, 100);
    const prices = testData.close.slice(0, 100);
    const volumes = [...testData.volume.slice(0, 100)];


    for (let i = 10; i < 20; i++) {
        volumes[i] = 0.0;
    }

    const result = wasm.vwap_js(timestamps, volumes, prices);
    assert.strictEqual(result.length, prices.length, 'Output length should match input');



    let nonNanCount = 0;
    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i])) {
            nonNanCount++;
        }
    }
    assert(nonNanCount > 0, 'Expected some valid VWAP values');
});

test('VWAP warmup period', () => {

    const timestamps = testData.timestamp.slice(0, 100);
    const volumes = testData.volume.slice(0, 100);
    const prices = testData.close.slice(0, 100);


    const result1m = wasm.vwap_js(timestamps, volumes, prices, "1m");
    assert.strictEqual(result1m.length, prices.length, 'Output length should match input');


    const result1d = wasm.vwap_js(timestamps, volumes, prices, "1d");
    assert.strictEqual(result1d.length, prices.length, 'Output length should match input');


    let nonNan1m = false;
    let nonNan1d = false;
    for (let i = 0; i < result1m.length; i++) {
        if (!isNaN(result1m[i])) nonNan1m = true;
        if (!isNaN(result1d[i])) nonNan1d = true;
    }
    assert(nonNan1m, 'Expected some valid values for 1m anchor');
    assert(nonNan1d, 'Expected some valid values for 1d anchor');
});

test('VWAP volume weighting', () => {


    const baseTs = 1609459200000;
    const timestamps = [baseTs, baseTs + 3600000, baseTs + 7200000];
    const prices = [100.0, 200.0, 300.0];
    const volumes = [1.0, 2.0, 3.0];

    const result = wasm.vwap_js(timestamps, volumes, prices, "1d");





    const expected = [100.0, 500.0/3.0, 1400.0/6.0];

    assertArrayClose(result, expected, 1e-9, 'VWAP volume weighting incorrect');
});

test('VWAP batch multi-anchor', () => {

    const timestamps = testData.timestamp.slice(0, 200);
    const volumes = testData.volume.slice(0, 200);
    const prices = testData.close.slice(0, 200);


    const result = wasm.vwap_batch(
        timestamps,
        volumes,
        prices,
        "1h", "4h", 1
    );

    assert(result.values, 'Result should have values array');
    assert(result.combos, 'Result should have combos array');


    assert.strictEqual(result.rows, 4, 'Should have 4 rows');
    assert.strictEqual(result.cols, prices.length, 'Cols should match input length');


    const anchors = result.combos.map(c => c.anchor);
    assert.deepStrictEqual(anchors, ["1h", "2h", "3h", "4h"], 'Anchors should match expected');


    const single1h = wasm.vwap_js(timestamps, volumes, prices, "1h");
    const firstRow = result.values.slice(0, prices.length);
    assertArrayClose(firstRow, single1h, 1e-9, "Batch 1h row doesn't match single calculation");


    const single4h = wasm.vwap_js(timestamps, volumes, prices, "4h");
    const lastRowStart = 3 * prices.length;
    const lastRow = result.values.slice(lastRowStart, lastRowStart + prices.length);
    assertArrayClose(lastRow, single4h, 1e-9, "Batch 4h row doesn't match single calculation");
});

test('VWAP batch static anchor', () => {

    const timestamps = testData.timestamp.slice(0, 100);
    const volumes = testData.volume.slice(0, 100);
    const prices = testData.close.slice(0, 100);


    const result = wasm.vwap_batch(
        timestamps,
        volumes,
        prices,
        "1d", "1d", 0
    );

    assert.strictEqual(result.rows, 1, 'Should have 1 row');
    assert.strictEqual(result.cols, prices.length, 'Cols should match input length');
    assert.strictEqual(result.combos[0].anchor, "1d", 'Anchor should be 1d');


    const single = wasm.vwap_js(timestamps, volumes, prices, "1d");
    assertArrayClose(result.values, single, 1e-9, "Static batch doesn't match single calculation");
});

test.after(() => {
    console.log('VWAP WASM tests completed');
});
