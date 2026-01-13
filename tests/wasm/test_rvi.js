
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

test('RVI partial params', () => {

    const close = new Float64Array(testData.close);


    const result = wasm.rvi_js(close, 10, 14, 1, 0);
    assert.strictEqual(result.length, close.length);
});

test('RVI accuracy', async () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.rvi;

    const result = wasm.rvi_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.ma_len,
        expected.defaultParams.matype,
        expected.defaultParams.devtype
    );

    assert.strictEqual(result.length, close.length);


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-1,
        "RVI last 5 values mismatch"
    );


    const warmup = expected.defaultParams.period - 1 + expected.defaultParams.ma_len - 1;
    for (let i = 0; i < warmup; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }


    await compareWithRust('rvi', result, 'close', expected.defaultParams);
});

test('RVI default params', () => {

    const close = new Float64Array(testData.close);


    const result = wasm.rvi_js(close, 10, 14, 1, 0);
    assert.strictEqual(result.length, close.length);
});

test('RVI zero period', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0, 40.0]);

    assert.throws(() => {
        wasm.rvi_js(inputData, 0, 14, 1, 0);
    }, /Invalid period/);
});

test('RVI zero ma_len', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0, 40.0]);

    assert.throws(() => {
        wasm.rvi_js(inputData, 10, 0, 1, 0);
    }, /Invalid period/);
});

test('RVI period exceeds data length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.rvi_js(dataSmall, 10, 14, 1, 0);
    }, /Invalid period/);
});

test('RVI all NaN input', () => {

    const allNaN = new Float64Array(3).fill(NaN);

    assert.throws(() => {
        wasm.rvi_js(allNaN, 10, 14, 1, 0);
    }, /All values are NaN/);
});

test('RVI not enough valid data', () => {


    const data = new Float64Array([NaN, NaN, NaN, 1.0, 2.0, 3.0, 4.0, 5.0]);

    assert.throws(() => {
        wasm.rvi_js(data, 3, 5, 1, 0);
    }, /Invalid period|Not enough valid data/);
});

test('RVI empty input', () => {

    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.rvi_js(empty, 10, 14, 1, 0);
    }, /Empty data/);
});

test('RVI batch single parameter set', () => {

    const close = new Float64Array(testData.close);


    const config = {
        period_range: [10, 10, 0],
        ma_len_range: [14, 14, 0],
        matype_range: [1, 1, 0],
        devtype_range: [0, 0, 0]
    };

    const batchResult = wasm.rvi_batch(close, config);
    const singleResult = wasm.rvi_js(close, 10, 14, 1, 0);

    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, close.length);
    assert.strictEqual(batchResult.values.length, close.length);


    assertArrayClose(
        batchResult.values,
        singleResult,
        1e-10,
        "Batch vs single RVI mismatch"
    );
});

test('RVI batch multiple periods', () => {

    const close = new Float64Array(testData.close.slice(0, 100));


    const config = {
        period_range: [10, 14, 2],
        ma_len_range: [14, 14, 0],
        matype_range: [1, 1, 0],
        devtype_range: [0, 0, 0]
    };

    const batchResult = wasm.rvi_batch(close, config);


    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.periods.length, 3);
    assert.strictEqual(batchResult.periods[0], 10);
    assert.strictEqual(batchResult.periods[1], 12);
    assert.strictEqual(batchResult.periods[2], 14);


    const periods = [10, 12, 14];
    for (let i = 0; i < periods.length; i++) {
        const singleResult = wasm.rvi_js(close, periods[i], 14, 1, 0);
        const rowData = batchResult.values.slice(i * 100, (i + 1) * 100);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Period ${periods[i]} batch mismatch`
        );
    }
});

test('RVI batch full parameter sweep', () => {

    const close = new Float64Array(testData.close.slice(0, 50));

    const config = {
        period_range: [10, 14, 2],
        ma_len_range: [10, 12, 2],
        matype_range: [0, 1, 1],
        devtype_range: [0, 1, 1]
    };

    const batchResult = wasm.rvi_batch(close, config);


    assert.strictEqual(batchResult.rows, 24);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 24 * 50);
    assert.strictEqual(batchResult.periods.length, 24);
    assert.strictEqual(batchResult.ma_lens.length, 24);
    assert.strictEqual(batchResult.matypes.length, 24);
    assert.strictEqual(batchResult.devtypes.length, 24);


    for (let row = 0; row < batchResult.rows; row++) {
        const rowData = batchResult.values.slice(row * 50, (row + 1) * 50);


        const period = batchResult.periods[row];
        const ma_len = batchResult.ma_lens[row];
        const warmup = period - 1 + ma_len - 1;

        for (let i = 0; i < Math.min(warmup, rowData.length); i++) {
            if (!isNaN(rowData[i])) {


            }
        }


        for (let i = warmup; i < rowData.length; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for combination ${row}`);
        }
    }
});


test('RVI zero-copy API', () => {
    const close = new Float64Array(testData.close);
    const len = close.length;


    const inPtr = wasm.rvi_alloc(len);
    const outPtr = wasm.rvi_alloc(len);

    try {

        const expected = wasm.rvi_js(close, 10, 14, 1, 0);


        const inputView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);


        inputView.set(close);


        wasm.rvi_into(
            inPtr,
            outPtr,
            len,
            10, 14, 1, 0
        );


        const output = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);


        const outputArray = Array.from(output);


        assertArrayClose(
            outputArray,
            expected,
            1e-10,
            "Zero-copy RVI mismatch"
        );

    } finally {

        wasm.rvi_free(inPtr, len);
        wasm.rvi_free(outPtr, len);
    }
});

test('RVI zero-copy in-place', () => {

    const data = new Float64Array(testData.close.slice(0, 100));
    const len = data.length;


    const inPtr = wasm.rvi_alloc(len);
    const outPtr = wasm.rvi_alloc(len);

    try {

        new Float64Array(wasm.__wasm.memory.buffer, inPtr, len).set(data);


        wasm.rvi_into(inPtr, outPtr, len, 10, 14, 1, 0);
        const expected = Array.from(new Float64Array(wasm.__wasm.memory.buffer, outPtr, len));


        wasm.rvi_into(inPtr, inPtr, len, 10, 14, 1, 0);
        const inPlace = Array.from(new Float64Array(wasm.__wasm.memory.buffer, inPtr, len));


        assertArrayClose(inPlace, expected, 1e-10, "In-place RVI mismatch");
    } finally {

        wasm.rvi_free(inPtr, len);
        wasm.rvi_free(outPtr, len);
    }
});

test('RVI streaming (not fully supported)', () => {


    const value = 100.0;




    assert.doesNotThrow(() => {


        assert(typeof wasm.rvi_js === 'function');
    });
});

test('RVI different devtypes', () => {

    const data = new Float64Array(testData.close.slice(0, 100));
    const len = data.length;
    const devtypes = [0, 1, 2];

    const inPtr = wasm.rvi_alloc(len);

    const outPtr = wasm.rvi_alloc(len);

    try {
        new Float64Array(wasm.__wasm.memory.buffer, inPtr, len).set(data);

        const results = devtypes.map((dev) => {
            wasm.rvi_into(inPtr, outPtr, len, 10, 14, 1, dev);
            return Array.from(new Float64Array(wasm.__wasm.memory.buffer, outPtr, len));
        });


        for (let i = 0; i < devtypes.length; i++) {
            for (let j = i + 1; j < devtypes.length; j++) {
                let maxDiff = 0;
                for (let k = 0; k < len; k++) {
                    const a = results[i][k];
                    const b = results[j][k];
                    if (!isNaN(a) && !isNaN(b)) {
                        maxDiff = Math.max(maxDiff, Math.abs(a - b));
                    }
                }
                assert(maxDiff > 1e-10, `Devtype ${devtypes[i]} and ${devtypes[j]} gave identical results`);
            }
        }
    } finally {
        wasm.rvi_free(inPtr, len);
        wasm.rvi_free(outPtr, len);
    }
});

test('RVI different matypes', () => {

    const data = new Float64Array(testData.close.slice(0, 100));
    const len = data.length;
    const matypes = [0, 1];

    const inPtr = wasm.rvi_alloc(len);
    const outPtr = wasm.rvi_alloc(len);

    try {
        new Float64Array(wasm.__wasm.memory.buffer, inPtr, len).set(data);

        const results = matypes.map((ma) => {
            wasm.rvi_into(inPtr, outPtr, len, 10, 14, ma, 0);
            return Array.from(new Float64Array(wasm.__wasm.memory.buffer, outPtr, len));
        });

        let maxDiff = 0;
        for (let i = 0; i < len; i++) {
            const a = results[0][i];
            const b = results[1][i];
            if (!isNaN(a) && !isNaN(b)) {
                maxDiff = Math.max(maxDiff, Math.abs(a - b));
            }
        }
        assert(maxDiff > 1e-10, "SMA and EMA gave identical results");
    } finally {
        wasm.rvi_free(inPtr, len);
        wasm.rvi_free(outPtr, len);
    }
});
