
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

test('HighPass partial params', () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.highpass;

    const result = wasm.highpass_js(close, expected.defaultParams.period);
    assert.strictEqual(result.length, close.length);
});

test('HighPass accuracy', async () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.highpass;

    const result = wasm.highpass_js(close, expected.defaultParams.period);

    assert.strictEqual(result.length, close.length);


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-6,
        "HighPass last 5 values mismatch"
    );


    await compareWithRust('highpass', result, 'close', expected.defaultParams);
});

test('HighPass default candles', async () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.highpass;

    const result = wasm.highpass_js(close, expected.defaultParams.period);
    assert.strictEqual(result.length, close.length);


    await compareWithRust('highpass', result, 'close', expected.defaultParams);
});

test('HighPass zero period', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.highpass_js(inputData, 0);
    });
});

test('HighPass period exceeds length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.highpass_js(dataSmall, 48);
    });
});

test('HighPass very small dataset', () => {

    const dataSmall = new Float64Array([42.0, 43.0]);

    assert.throws(() => {
        wasm.highpass_js(dataSmall, 2);
    });
});

test('HighPass empty input', () => {

    const dataEmpty = new Float64Array([]);

    assert.throws(() => {
        wasm.highpass_js(dataEmpty, 48);
    });
});

test('HighPass invalid alpha', () => {

    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);


    assert.throws(() => {
        wasm.highpass_js(data, 4);
    });
});

test('HighPass all NaN', () => {

    const data = new Float64Array([NaN, NaN, NaN, NaN, NaN]);

    assert.throws(() => {
        wasm.highpass_js(data, 3);
    });
});

test('HighPass reinput', () => {

    const close = new Float64Array(testData.close);


    const firstResult = wasm.highpass_js(close, 36);


    const secondResult = wasm.highpass_js(firstResult, 24);

    assert.strictEqual(secondResult.length, firstResult.length);


    for (let i = 240; i < secondResult.length; i++) {
        assert(!isNaN(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('HighPass NaN handling', () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.highpass;
    const period = expected.defaultParams.period;

    const result = wasm.highpass_js(close, period);

    assert.strictEqual(result.length, close.length);



    assert(!isNaN(result[0]), "HighPass should produce value at index 0");


    for (let i = 0; i < result.length; i++) {
        assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
    }
});

test('HighPass warmup period', () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.highpass;

    const result = wasm.highpass_js(close, expected.defaultParams.period);


    assert(!isNaN(result[0]), "HighPass should produce value at index 0 (no warmup)");


    assert.strictEqual(expected.hasWarmup, false, "HighPass should have no warmup period");
    assert.strictEqual(expected.warmupLength, 0, "HighPass warmup length should be 0");


    assertNoNaN(result, "HighPass should have no NaN values");
});

test('HighPass leading NaN input', () => {

    const close = new Float64Array(testData.close.slice(0, 100));


    for (let i = 0; i < 5; i++) {
        close[i] = NaN;
    }


    const result = wasm.highpass_js(close, 48);
    assert.strictEqual(result.length, close.length);

    
    
    for (let i = 0; i < 5; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} for leading-NaN prefix`);
    }
    for (let i = 5; i < result.length; i++) {
        assert(!isNaN(result[i]), `Unexpected NaN at index ${i} after first valid sample`);
    }
});

test('HighPass batch', () => {

    const close = new Float64Array(testData.close.slice(0, 500));


    const batchResult = wasm.highpass_batch(close, {
        period_range: [30, 60, 10]
    });


    assert(batchResult.values, "Batch result should have values");
    assert(batchResult.combos, "Batch result should have combos");
    assert.strictEqual(batchResult.rows, 4, "Should have 4 rows");
    assert.strictEqual(batchResult.cols, close.length, "Should have cols equal to data length");


    for (let row = 0; row < 4; row++) {
        const rowStart = row * close.length;
        assert(!isNaN(batchResult.values[rowStart]), `Row ${row} should have value at index 0`);
    }


    const periods = [30, 40, 50, 60];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * close.length;
        const rowEnd = rowStart + close.length;
        const rowData = batchResult.values.slice(rowStart, rowEnd);

        const individualResult = wasm.highpass_js(close, periods[i]);
        assertArrayClose(rowData, individualResult, 1e-9, `Period ${periods[i]} mismatch`);
    }
});

test('HighPass edge cases', () => {

    const period = 10;


    const dataExact = new Float64Array(testData.close.slice(0, period));
    const resultExact = wasm.highpass_js(dataExact, period);
    assert.strictEqual(resultExact.length, dataExact.length);
    assert(!isNaN(resultExact[0]), "Should have value at index 0");


    const dataPlusOne = new Float64Array(testData.close.slice(0, period + 1));
    const resultPlusOne = wasm.highpass_js(dataPlusOne, period);
    assert.strictEqual(resultPlusOne.length, dataPlusOne.length);
    assert(!isNaN(resultPlusOne[0]), "Should have value at index 0");


    const constantData = new Float64Array(100);
    constantData.fill(50.0);
    const resultConstant = wasm.highpass_js(constantData, 20);



    const stabilizedStart = 3 * 20;
    for (let i = stabilizedStart; i < resultConstant.length; i++) {
        assert(Math.abs(resultConstant[i]) < 1e-3,
            `DC component not removed at index ${i}: ${resultConstant[i]}`);
    }
});

test('HighPass different periods', () => {

    const close = new Float64Array(testData.close);


    for (const period of [10, 20, 30, 48, 60]) {
        const result = wasm.highpass_js(close, period);
        assert.strictEqual(result.length, close.length);


        let firstValid = null;
        for (let i = 0; i < result.length; i++) {
            if (!isNaN(result[i])) {
                firstValid = i;
                break;
            }
        }


        assert(firstValid !== null, `No valid data found for period=${period}`);


        for (let i = firstValid; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for period=${period}`);
        }
    }
});

test('HighPass batch multiple parameters', () => {

    const close = new Float64Array(testData.close.slice(0, 200));


    const periods = [10, 20, 30, 40, 50];
    const batchResult = wasm.highpass_batch_js(close, 10, 50, 10);
    const metadata = wasm.highpass_batch_metadata_js(10, 50, 10);

    assert.strictEqual(metadata.length, 5);
    assert.strictEqual(batchResult.length, 5 * close.length);


    for (let row = 0; row < 5; row++) {
        const rowStart = row * close.length;
        assert(!isNaN(batchResult[rowStart]), `Row ${row} should have value at index 0`);


        const individualResult = wasm.highpass_js(close, periods[row]);
        const rowData = batchResult.slice(rowStart, rowStart + close.length);
        assertArrayClose(rowData, individualResult, 1e-9,
            `Batch row ${row} (period=${periods[row]}) mismatch`);
    }
});

test('HighPass batch performance', () => {

    const close = new Float64Array(testData.close.slice(0, 1000));


    const startBatch = performance.now();
    const batchResult = wasm.highpass_batch_js(close, 20, 60, 10);
    const batchTime = performance.now() - startBatch;

    const startSingle = performance.now();
    const singleResults = [];
    for (let period = 20; period <= 60; period += 10) {
        singleResults.push(...wasm.highpass_js(close, period));
    }
    const singleTime = performance.now() - startSingle;


    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);


    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});
