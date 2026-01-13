import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import {
    loadTestData,
    EXPECTED_OUTPUTS
} from './test_utils.js';
import { compareWithRust } from './rust-comparison.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

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
});

test('SuperSmoother3Pole partial params', (t) => {
    const data = loadTestData();
    const closePrices = data.close;


    const result = wasm.supersmoother_3_pole_js(closePrices, 14);
    assert.strictEqual(result.length, closePrices.length);
});

test('SuperSmoother3Pole accuracy', async (t) => {
    const data = loadTestData();
    const closePrices = data.close;
    const expected = EXPECTED_OUTPUTS.supersmoother_3_pole;


    const result = wasm.supersmoother_3_pole_js(
        closePrices,
        expected.defaultParams.period
    );


    assert.strictEqual(result.length, closePrices.length);


    const last5 = result.slice(-5);
    const expectedLast5 = expected.last5Values;

    for (let i = 0; i < 5; i++) {
        const diff = Math.abs(last5[i] - expectedLast5[i]);
        assert.ok(
            diff < 1e-8,
            `Value mismatch at position ${i}: expected ${expectedLast5[i]}, got ${last5[i]}, diff ${diff}`
        );
    }




    const rustResult = await compareWithRust(
        'supersmoother_3_pole',
        result,
        'close',
        expected.defaultParams,
        1e-13
    );
    assert.ok(rustResult, 'Comparison with Rust succeeded');
});

test('SuperSmoother3Pole default candles', (t) => {
    const data = loadTestData();
    const closePrices = data.close;


    const result = wasm.supersmoother_3_pole_js(closePrices, 14);
    assert.strictEqual(result.length, closePrices.length);
});

test('SuperSmoother3Pole zero period', (t) => {
    const data = [10.0, 20.0, 30.0];

    assert.throws(
        () => wasm.supersmoother_3_pole_js(data, 0),
        /Invalid period/,
        'Should throw error for period = 0'
    );
});

test('SuperSmoother3Pole period exceeds length', (t) => {
    const data = [10.0, 20.0, 30.0];

    assert.throws(
        () => wasm.supersmoother_3_pole_js(data, 10),
        /Invalid period/,
        'Should throw error when period exceeds data length'
    );
});

test('SuperSmoother3Pole very small dataset', (t) => {
    const data = [42.0];

    assert.throws(
        () => wasm.supersmoother_3_pole_js(data, 9),
        /Invalid period|Not enough valid data/,
        'Should throw error for insufficient data'
    );
});

test('SuperSmoother3Pole empty input', (t) => {
    const data = [];

    assert.throws(
        () => wasm.supersmoother_3_pole_js(data, 14),
        /Input data slice is empty/,
        'Should throw error for empty input'
    );
});

test('SuperSmoother3Pole all NaN input', (t) => {
    const data = new Float64Array(100).fill(NaN);

    assert.throws(
        () => wasm.supersmoother_3_pole_js(data, 14),
        /All values are NaN/,
        'Should throw error for all NaN input'
    );
});

test('SuperSmoother3Pole batch processing', (t) => {
    const data = loadTestData();
    const closePrices = data.close;


    const batchResult = wasm.supersmoother_3_pole_batch_js(
        closePrices,
        10,
        20,
        5
    );


    const metadata = wasm.supersmoother_3_pole_batch_metadata_js(10, 20, 5);
    const numPeriods = metadata.length;


    assert.strictEqual(batchResult.length, numPeriods * closePrices.length);
    assert.strictEqual(numPeriods, 3);


    for (let p = 0; p < numPeriods; p++) {
        const period = metadata[p];
        const rowStart = p * closePrices.length;
        const row = batchResult.slice(rowStart, rowStart + closePrices.length);


        for (let i = 0; i < Math.min(3, row.length); i++) {
            assert.ok(!isNaN(row[i]), `Value at index ${i} for period ${period} should not be NaN`);
        }
    }
});

test('SuperSmoother3Pole NaN handling', (t) => {
    const data = loadTestData();
    const closePrices = data.close;

    const result = wasm.supersmoother_3_pole_js(closePrices, 14);
    assert.strictEqual(result.length, closePrices.length);


    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert.ok(!isNaN(result[i]), `Found unexpected NaN at index ${i} after warmup period`);
        }
    }



    for (let i = 0; i < Math.min(3, result.length); i++) {
        assert.ok(!isNaN(result[i]), `Value at index ${i} should not be NaN`);
    }
});

test('SuperSmoother3Pole with leading NaNs', (t) => {

    const data = new Float64Array(20);
    for (let i = 0; i < 5; i++) {
        data[i] = NaN;
    }
    for (let i = 5; i < 20; i++) {
        data[i] = i - 4;
    }

    const period = 3;
    const result = wasm.supersmoother_3_pole_js(data, period);







    for (let i = 0; i < 5; i++) {

        assert.strictEqual(result[i], 0, `WASM converts NaN to 0 at index ${i}`);
    }


    assert.strictEqual(result[5], 1, 'First valid value passed through');
    assert.strictEqual(result[6], 2, 'Second valid value passed through');
    assert.strictEqual(result[7], 3, 'Third valid value passed through');


    assert.ok(!isNaN(result[8]) && result[8] !== 0, `Filter starts calculating at index 8`);
});


test('SuperSmoother3Pole edge cases', (t) => {
    const data = loadTestData();
    const closePrices = data.close;


    const result1 = wasm.supersmoother_3_pole_js(closePrices, 1);
    assert.strictEqual(result1.length, closePrices.length);


    const smallData = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    const result2 = wasm.supersmoother_3_pole_js(smallData, 2);
    assert.strictEqual(result2.length, smallData.length);


    assert.strictEqual(result2[0], 1.0);
    assert.strictEqual(result2[1], 2.0);
    assert.strictEqual(result2[2], 3.0);
});

test('SuperSmoother3Pole consistency check', (t) => {
    const data = loadTestData();
    const closePrices = data.close;
    const period = 14;


    const result1 = wasm.supersmoother_3_pole_js(closePrices, period);
    const result2 = wasm.supersmoother_3_pole_js(closePrices, period);


    for (let i = 0; i < result1.length; i++) {
        if (isNaN(result1[i]) && isNaN(result2[i])) continue;
        assert.strictEqual(result1[i], result2[i], `Inconsistent result at index ${i}`);
    }
});


test('SuperSmoother3Pole batch metadata', (t) => {

    const metadata = wasm.supersmoother_3_pole_batch_metadata_js(5, 15, 5);


    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 5);
    assert.strictEqual(metadata[1], 10);
    assert.strictEqual(metadata[2], 15);


    const singleMeta = wasm.supersmoother_3_pole_batch_metadata_js(7, 7, 0);
    assert.strictEqual(singleMeta.length, 1);
    assert.strictEqual(singleMeta[0], 7);
});




test('SuperSmoother3Pole fast API basic test', (t) => {
    const data = loadTestData();
    const closePrices = data.close;
    const period = 14;
    const len = closePrices.length;


    const inPtr = wasm.supersmoother_3_pole_alloc(len);
    const outPtr = wasm.supersmoother_3_pole_alloc(len);

    try {

        const memory = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        memory.set(closePrices);


        wasm.supersmoother_3_pole_into(inPtr, outPtr, len, period);


        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const resultCopy = Array.from(result);


        const safeResult = wasm.supersmoother_3_pole_js(closePrices, period);

        assert.strictEqual(resultCopy.length, safeResult.length);
        for (let i = 0; i < resultCopy.length; i++) {
            if (isNaN(resultCopy[i]) && isNaN(safeResult[i])) continue;
            assert.ok(
                Math.abs(resultCopy[i] - safeResult[i]) < 1e-10,
                `Mismatch at index ${i}: fast=${resultCopy[i]}, safe=${safeResult[i]}`
            );
        }
    } finally {

        wasm.supersmoother_3_pole_free(inPtr, len);
        wasm.supersmoother_3_pole_free(outPtr, len);
    }
});

test('SuperSmoother3Pole fast API aliasing test', (t) => {
    const data = loadTestData();
    const closePrices = data.close.slice(0, 100);
    const period = 14;
    const len = closePrices.length;


    const ptr = wasm.supersmoother_3_pole_alloc(len);

    try {

        const memory = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        memory.set(closePrices);


        wasm.supersmoother_3_pole_into(ptr, ptr, len, period);


        const result = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        const resultCopy = Array.from(result);


        const safeResult = wasm.supersmoother_3_pole_js(closePrices, period);

        assert.strictEqual(resultCopy.length, safeResult.length);
        for (let i = 0; i < resultCopy.length; i++) {
            if (isNaN(resultCopy[i]) && isNaN(safeResult[i])) continue;
            assert.ok(
                Math.abs(resultCopy[i] - safeResult[i]) < 1e-10,
                `In-place mismatch at index ${i}: fast=${resultCopy[i]}, safe=${safeResult[i]}`
            );
        }
    } finally {

        wasm.supersmoother_3_pole_free(ptr, len);
    }
});

test('SuperSmoother3Pole fast batch API test', (t) => {
    const data = loadTestData();
    const closePrices = data.close.slice(0, 1000);
    const len = closePrices.length;


    const periodStart = 10;
    const periodEnd = 20;
    const periodStep = 5;
    const numPeriods = Math.floor((periodEnd - periodStart) / periodStep) + 1;
    const totalSize = numPeriods * len;


    const inPtr = wasm.supersmoother_3_pole_alloc(len);
    const outPtr = wasm.supersmoother_3_pole_alloc(totalSize);

    try {

        const memory = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        memory.set(closePrices);


        const rows = wasm.supersmoother_3_pole_batch_into(
            inPtr,
            outPtr,
            len,
            periodStart,
            periodEnd,
            periodStep
        );

        assert.strictEqual(rows, numPeriods, 'Batch should return correct number of rows');


        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);


        for (let p = 0; p < numPeriods; p++) {
            const rowStart = p * len;
            const row = result.slice(rowStart, rowStart + len);


            for (let i = 0; i < Math.min(3, row.length); i++) {
                assert.ok(!isNaN(row[i]), `Row ${p}, value at index ${i} should not be NaN`);
            }
        }
    } finally {

        wasm.supersmoother_3_pole_free(inPtr, len);
        wasm.supersmoother_3_pole_free(outPtr, totalSize);
    }
});

test('SuperSmoother3Pole unified batch API test', (t) => {
    const data = loadTestData();
    const closePrices = data.close.slice(0, 100);


    const config = {
        period_range: [10, 20, 5]
    };

    const result = wasm.supersmoother_3_pole_batch(closePrices, config);


    assert.ok(result.values, 'Result should have values');
    assert.ok(result.rows === 3, 'Should have 3 rows (periods: 10, 15, 20)');
    assert.ok(result.cols === closePrices.length, 'Columns should match input length');
    assert.ok(result.periods, 'Result should have periods array');
    assert.deepStrictEqual(result.periods, [10, 15, 20], 'Periods should match config');
});
