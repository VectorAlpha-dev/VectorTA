
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

test('HWMA partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.hwma_js(close, 0.2, 0.1, 0.1);
    assert.strictEqual(result.length, close.length);
});

test('HWMA accuracy', async () => {

    const close = new Float64Array(testData.close);

    const result = wasm.hwma_js(close, 0.2, 0.1, 0.1);

    assert.strictEqual(result.length, close.length);


    const expectedLastFive = [
        57941.04005793378,
        58106.90324194954,
        58250.474156632234,
        58428.90005831887,
        58499.37021151028,
    ];


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-3,
        "HWMA last 5 values mismatch"
    );



});

test('HWMA default candles', async () => {

    const close = new Float64Array(testData.close);

    const result = wasm.hwma_js(close, 0.2, 0.1, 0.1);
    assert.strictEqual(result.length, close.length);


    await compareWithRust('hwma', result, 'close', { na: 0.2, nb: 0.1, nc: 0.1 });
});

test('HWMA invalid na', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);


    assert.throws(() => {
        wasm.hwma_js(inputData, 1.5, 0.1, 0.1);
    });


    assert.throws(() => {
        wasm.hwma_js(inputData, 0.0, 0.1, 0.1);
    });


    assert.throws(() => {
        wasm.hwma_js(inputData, NaN, 0.1, 0.1);
    });
});

test('HWMA invalid nb', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);


    assert.throws(() => {
        wasm.hwma_js(inputData, 0.2, 1.5, 0.1);
    });


    assert.throws(() => {
        wasm.hwma_js(inputData, 0.2, 0.0, 0.1);
    });


    assert.throws(() => {
        wasm.hwma_js(inputData, 0.2, NaN, 0.1);
    });
});

test('HWMA invalid nc', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);


    assert.throws(() => {
        wasm.hwma_js(inputData, 0.2, 0.1, 1.5);
    });


    assert.throws(() => {
        wasm.hwma_js(inputData, 0.2, 0.1, 0.0);
    });


    assert.throws(() => {
        wasm.hwma_js(inputData, 0.2, 0.1, NaN);
    });
});

test('HWMA empty input', () => {

    const dataEmpty = new Float64Array([]);

    assert.throws(() => {
        wasm.hwma_js(dataEmpty, 0.2, 0.1, 0.1);
    });
});

test('HWMA all NaN', () => {

    const data = new Float64Array([NaN, NaN, NaN, NaN, NaN]);

    assert.throws(() => {
        wasm.hwma_js(data, 0.2, 0.1, 0.1);
    });
});

test('HWMA reinput', () => {

    const close = new Float64Array(testData.close);


    const firstResult = wasm.hwma_js(close, 0.2, 0.1, 0.1);


    const secondResult = wasm.hwma_js(firstResult, 0.2, 0.1, 0.1);

    assert.strictEqual(secondResult.length, firstResult.length);


    let finiteCount = 0;
    for (let i = 0; i < secondResult.length; i++) {
        if (isFinite(secondResult[i])) {
            finiteCount++;
            assert(secondResult[i] > 0, `HWMA reinput produced negative value at index ${i}`);
        }
    }
    assert(finiteCount > 0, "No finite values in HWMA reinput result");
});

test('HWMA NaN handling', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.hwma_js(close, 0.2, 0.1, 0.1);

    assert.strictEqual(result.length, close.length);


    if (result.length > 3) {
        for (let i = 3; i < result.length; i++) {
            assert(isFinite(result[i]), `Unexpected non-finite value at index ${i}`);
        }
    }
});

test('HWMA batch', () => {

    const close = new Float64Array(testData.close);


    const na_start = 0.1, na_end = 0.3, na_step = 0.1;
    const nb_start = 0.05, nb_end = 0.15, nb_step = 0.05;
    const nc_start = 0.05, nc_end = 0.15, nc_step = 0.05;

    const batch_result = wasm.hwma_batch_js(
        close,
        na_start, na_end, na_step,
        nb_start, nb_end, nb_step,
        nc_start, nc_end, nc_step
    );
    const metadata = wasm.hwma_batch_metadata_js(
        na_start, na_end, na_step,
        nb_start, nb_end, nb_step,
        nc_start, nc_end, nc_step
    );


    assert.strictEqual(metadata.length, 27 * 3);


    assert.strictEqual(batch_result.length, 27 * close.length);


    const individual_result = wasm.hwma_js(close, 0.1, 0.05, 0.05);


    const row = batch_result.slice(0, close.length);

    assertArrayClose(row, individual_result, 1e-9, 'First combination');
});

test('HWMA different params', () => {

    const close = new Float64Array(testData.close);


    const paramSets = [
        [0.1, 0.05, 0.05],
        [0.2, 0.1, 0.1],
        [0.3, 0.15, 0.15],
        [0.5, 0.25, 0.25],
    ];

    for (const [na, nb, nc] of paramSets) {
        const result = wasm.hwma_js(close, na, nb, nc);
        assert.strictEqual(result.length, close.length);


        let finiteCount = 0;
        for (let i = 0; i < result.length; i++) {
            if (isFinite(result[i])) finiteCount++;
        }
        assert(finiteCount > close.length - 4,
            `Too many non-finite values for params (${na}, ${nb}, ${nc})`);
    }
});

test('HWMA batch performance', () => {

    const close = new Float64Array(testData.close.slice(0, 1000));


    const startBatch = performance.now();
    const batchResult = wasm.hwma_batch_js(
        close,
        0.1, 0.2, 0.1,
        0.05, 0.1, 0.05,
        0.05, 0.1, 0.05
    );
    const batchTime = performance.now() - startBatch;

    const startSingle = performance.now();
    const singleResults = [];
    for (let na = 0.1; na <= 0.2; na += 0.1) {
        for (let nb = 0.05; nb <= 0.1; nb += 0.05) {
            for (let nc = 0.05; nc <= 0.1; nc += 0.05) {
                singleResults.push(...wasm.hwma_js(close, na, nb, nc));
            }
        }
    }
    const singleTime = performance.now() - startSingle;


    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);


    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});

test('HWMA edge cases', () => {

    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);


    const result1 = wasm.hwma_js(data, 0.001, 0.001, 0.001);
    assert.strictEqual(result1.length, data.length);
    let finiteCount1 = 0;
    for (let val of result1) {
        if (isFinite(val)) finiteCount1++;
    }
    assert(finiteCount1 > 0, "No finite values with small parameters");


    const result2 = wasm.hwma_js(data, 0.999, 0.999, 0.999);
    assert.strictEqual(result2.length, data.length);
    let finiteCount2 = 0;
    for (let val of result2) {
        if (isFinite(val)) finiteCount2++;
    }
    assert(finiteCount2 > 0, "No finite values with large parameters");
});

test('HWMA single value', () => {

    const data = new Float64Array([42.0]);

    const result = wasm.hwma_js(data, 0.2, 0.1, 0.1);
    assert.strictEqual(result.length, 1);
    assert(Math.abs(result[0] - data[0]) < 1e-12);
});

test('HWMA two values', () => {

    const data = new Float64Array([1.0, 2.0]);

    const result = wasm.hwma_js(data, 0.2, 0.1, 0.1);
    assert.strictEqual(result.length, 2);

    assert(isFinite(result[0]));
    assert(isFinite(result[1]));
});

test('HWMA three values', () => {

    const data = new Float64Array([1.0, 2.0, 3.0]);

    const result = wasm.hwma_js(data, 0.2, 0.1, 0.1);
    assert.strictEqual(result.length, 3);

    assert(isFinite(result[0]));
    assert(isFinite(result[1]));
    assert(isFinite(result[2]));
});

test('HWMA batch metadata', () => {

    const metadata = wasm.hwma_batch_metadata_js(
        0.1, 0.2, 0.1,
        0.05, 0.1, 0.05,
        0.05, 0.1, 0.05
    );



    assert.strictEqual(metadata.length, 24);


    const expectedFirstCombinations = [
        0.1, 0.05, 0.05,
        0.1, 0.05, 0.1,
        0.1, 0.1, 0.05,
        0.1, 0.1, 0.1,
    ];

    for (let i = 0; i < expectedFirstCombinations.length; i++) {
        assertClose(
            metadata[i],
            expectedFirstCombinations[i],
            1e-9,
            `Metadata value at index ${i}`
        );
    }
});

test('HWMA warmup period', () => {

    const close = new Float64Array(testData.close.slice(0, 50));

    const result = wasm.hwma_js(close, 0.2, 0.1, 0.1);



    for (let i = 0; i < Math.min(result.length, 10); i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

test('HWMA consistency across calls', () => {

    const close = new Float64Array(testData.close.slice(0, 100));

    const result1 = wasm.hwma_js(close, 0.2, 0.1, 0.1);
    const result2 = wasm.hwma_js(close, 0.2, 0.1, 0.1);

    assertArrayClose(result1, result2, 1e-15, "HWMA results not consistent");
});

test('HWMA parameter step precision', () => {

    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

    const batch_result = wasm.hwma_batch_js(
        data,
        0.1, 0.11, 0.01,
        0.05, 0.06, 0.01,
        0.05, 0.06, 0.01
    );


    assert.strictEqual(batch_result.length, 8 * data.length);
});