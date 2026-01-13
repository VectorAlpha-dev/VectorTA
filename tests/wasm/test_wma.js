
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

test('WMA partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.wma_js(close, 30);
    assert.strictEqual(result.length, close.length);
});

test('WMA accuracy', async () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.wma;

    const result = wasm.wma_js(
        close,
        expected.defaultParams.period
    );

    assert.strictEqual(result.length, close.length);


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-6,
        "WMA last 5 values mismatch"
    );


    await compareWithRust('wma', result, 'close', expected.defaultParams);
});

test('WMA default candles', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.wma_js(close, 30);
    assert.strictEqual(result.length, close.length);
});

test('WMA empty input', () => {

    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.wma_js(empty, 30);
    }, /Input data slice is empty/);
});

test('WMA zero period', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.wma_js(inputData, 0);
    }, /Invalid period/);
});

test('WMA period exceeds length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.wma_js(dataSmall, 10);
    }, /Invalid period/);
});

test('WMA very small dataset', () => {

    const singlePoint = new Float64Array([42.0]);

    assert.throws(() => {
        wasm.wma_js(singlePoint, 9);
    }, /Invalid period|Not enough valid data/);
});

test('WMA reinput', () => {

    const close = new Float64Array(testData.close);


    const firstResult = wasm.wma_js(close, 14);
    assert.strictEqual(firstResult.length, close.length);


    const secondResult = wasm.wma_js(firstResult, 5);
    assert.strictEqual(secondResult.length, firstResult.length);


    if (secondResult.length > 50) {
        for (let i = 50; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Unexpected NaN at index ${i}`);
        }
    }
});

test('WMA NaN handling', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.wma_js(close, 14);
    assert.strictEqual(result.length, close.length);


    if (result.length > 50) {
        for (let i = 50; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }




    assertAllNaN(result.slice(0, 13), "Expected NaN in warmup period (indices 0-12)");
    assert(!isNaN(result[13]), "Expected first valid value at index 13");
});

test('WMA all NaN input', () => {

    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);

    assert.throws(() => {
        wasm.wma_js(allNaN, 30);
    }, /All values are NaN/);
});

test('WMA batch single parameter set', () => {

    const close = new Float64Array(testData.close);


    const batchResult = wasm.wma_batch(close, {
        period_range: [30, 30, 0]
    });


    const singleResult = wasm.wma_js(close, 30);

    assert.strictEqual(batchResult.values.length, singleResult.length);
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, close.length);
    assert.strictEqual(batchResult.combos.length, 1);
    assert.strictEqual(batchResult.combos[0].period, 30);


    const p = 30;
    const firstValid = p - 1;

    assertClose(batchResult.values[firstValid], singleResult[firstValid], 1e-6, "First valid mismatch");

    const end = batchResult.values.length;
    assertArrayClose(batchResult.values.slice(end - 5, end), singleResult.slice(end - 5, end), 1e-6, "Last 5 mismatch");


});

test('WMA batch multiple periods', () => {

    const close = new Float64Array(testData.close.slice(0, 100));


    const batchResult = wasm.wma_batch(close, {
        period_range: [10, 30, 10]
    });


    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.combos.length, 3);


    const periods = [10, 20, 30];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);

        const singleResult = wasm.wma_js(close, periods[i]);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Period ${periods[i]} mismatch`
        );


        assert.strictEqual(batchResult.combos[i].period, periods[i]);
    }
});

test('WMA batch metadata', () => {

    const metadata = wasm.wma_batch_metadata_js(
        10, 30, 10
    );


    assert.strictEqual(metadata.length, 3);


    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 20);
    assert.strictEqual(metadata[2], 30);
});

test('WMA batch full parameter sweep', () => {

    const close = new Float64Array(testData.close.slice(0, 50));

    const batchResult = wasm.wma_batch_js(
        close,
        10, 14, 2
    );

    const metadata = wasm.wma_batch_metadata_js(
        10, 14, 2
    );


    const numCombos = metadata.length;
    assert.strictEqual(numCombos, 3);
    assert.strictEqual(batchResult.length, 3 * 50);


    for (let combo = 0; combo < numCombos; combo++) {
        const period = metadata[combo];

        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);


        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }


        for (let i = period - 1; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('WMA batch edge cases', () => {

    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);


    const singleBatch = wasm.wma_batch_js(
        close,
        5, 5, 1
    );

    assert.strictEqual(singleBatch.length, 10);


    const largeBatch = wasm.wma_batch_js(
        close,
        5, 7, 10
    );


    assert.strictEqual(largeBatch.length, 10);
});



test.after(() => {
    console.log('WMA WASM tests completed');
});
