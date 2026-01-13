
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

test('TSI partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.tsi_js(close, 25, 13);
    assert.strictEqual(result.length, close.length);
});

test('TSI accuracy', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.tsi_js(close, 25, 13);

    assert.strictEqual(result.length, close.length);


    const expectedLastFive = [
        -17.757654061849838,
        -17.367527062626184,
        -17.305577681249513,
        -16.937565646991143,
        -17.61825617316731,
    ];


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-7,
        "TSI last 5 values mismatch"
    );
});

test('TSI default candles', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.tsi_js(close, 25, 13);
    assert.strictEqual(result.length, close.length);
});

test('TSI zero period', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.tsi_js(inputData, 0, 13);
    }, /Invalid period/);

    assert.throws(() => {
        wasm.tsi_js(inputData, 25, 0);
    }, /Invalid period/);
});

test('TSI period exceeds length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.tsi_js(dataSmall, 25, 13);
    }, /Invalid period/);
});

test('TSI very small dataset', () => {

    const singlePoint = new Float64Array([42.0]);

    assert.throws(() => {
        wasm.tsi_js(singlePoint, 25, 13);
    }, /Invalid period|Not enough valid data/);
});

test('TSI reinput', () => {

    const close = new Float64Array(testData.close);


    const firstResult = wasm.tsi_js(close, 25, 13);
    assert.strictEqual(firstResult.length, close.length);


    const secondResult = wasm.tsi_js(firstResult, 25, 13);
    assert.strictEqual(secondResult.length, firstResult.length);
});

test('TSI NaN handling', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.tsi_js(close, 25, 13);
    assert.strictEqual(result.length, close.length);


    if (result.length > 240) {
        const afterWarmup = result.slice(240);
        assertNoNaN(afterWarmup, "Found unexpected NaN after warmup period");
    }



    const warmup = 25 + 13;
    assertAllNaN(result.slice(0, warmup), `Expected NaN in warmup period (first ${warmup} values)`);
});

test('TSI all NaN input', () => {

    const allNaN = new Float64Array(100).fill(NaN);

    assert.throws(() => {
        wasm.tsi_js(allNaN, 25, 13);
    }, /All values are NaN/);
});

test('TSI batch processing', () => {

    const close = new Float64Array(testData.close);

    const config = {
        long_period_range: [25, 25, 0],
        short_period_range: [13, 13, 0]
    };

    const result = wasm.tsi_batch(close, config);

    assert(result.values, 'Result should have values');
    assert(result.combos, 'Result should have combos');
    assert.strictEqual(result.rows, 1, 'Should have 1 combination');
    assert.strictEqual(result.cols, close.length, 'Columns should match input length');


    const defaultRow = result.values.slice(0, close.length);
    const expectedLastFive = [
        -17.757654061849838,
        -17.367527062626184,
        -17.305577681249513,
        -16.937565646991143,
        -17.61825617316731,
    ];


    assertArrayClose(
        defaultRow.slice(-5),
        expectedLastFive,
        1e-7,
        "TSI batch default row mismatch"
    );
});

test('TSI batch multiple params', () => {

    const close = new Float64Array(testData.close);

    const config = {
        long_period_range: [20, 30, 5],
        short_period_range: [10, 15, 5]
    };

    const result = wasm.tsi_batch(close, config);

    assert(result.values, 'Result should have values');
    assert(result.combos, 'Result should have combos');
    assert.strictEqual(result.rows, 6, 'Should have 6 combinations (3*2)');
    assert.strictEqual(result.cols, close.length, 'Columns should match input length');


    const expectedCombos = [
        { long_period: 20, short_period: 10 },
        { long_period: 20, short_period: 15 },
        { long_period: 25, short_period: 10 },
        { long_period: 25, short_period: 15 },
        { long_period: 30, short_period: 10 },
        { long_period: 30, short_period: 15 }
    ];

    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].long_period, expectedCombos[i].long_period,
            `Combo ${i} long_period mismatch`);
        assert.strictEqual(result.combos[i].short_period, expectedCombos[i].short_period,
            `Combo ${i} short_period mismatch`);
    }
});

test('TSI mid-series NaN', () => {

    const data = new Float64Array([
        100.0, 102.0, 101.0, 103.0, 104.0,
        NaN, NaN,
        105.0, 106.0, 107.0, 108.0, 109.0,
        110.0, 111.0, 112.0, 113.0, 114.0,
        115.0, 116.0, 117.0
    ]);


    const result = wasm.tsi_js(data, 5, 3);
    assert.strictEqual(result.length, data.length);


    assert(isNaN(result[5]), 'Should have NaN at gap position 5');
    assert(isNaN(result[6]), 'Should have NaN at gap position 6');


    const validAfterGap = result.slice(10);
    const validCount = validAfterGap.filter(v => !isNaN(v)).length;
    assert(validCount > 0, "TSI should recover after mid-series NaN gap");
});

test('TSI constant data', () => {

    const constant = new Float64Array(50).fill(100.0);
    const result = wasm.tsi_js(constant, 10, 5);


    const warmup = 10 + 5;
    const afterWarmup = result.slice(warmup);
    assertAllNaN(afterWarmup, "TSI should be NaN for constant prices");
});

test('TSI step data', () => {

    const step1 = new Float64Array(25).fill(100.0);
    const step2 = new Float64Array(25).fill(150.0);
    const data = new Float64Array([...step1, ...step2]);

    const result = wasm.tsi_js(data, 10, 5);
    assert.strictEqual(result.length, data.length);


    const lastValues = result.slice(-5);
    const validLast = lastValues.filter(v => !isNaN(v));

    if (validLast.length > 0) {

        for (const val of validLast) {
            assert(val >= -100 && val <= 100,
                `TSI value ${val} should be in [-100, 100] range`);
        }
    }
});

test('TSI into (in-place)', () => {

    const close = new Float64Array(testData.close);
    const len = close.length;


    const outPtr = wasm.tsi_alloc(len);
    const inPtr = wasm.tsi_alloc(len);

    try {

        const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer);
        const inIndex = inPtr / 8;
        for (let i = 0; i < len; i++) {
            wasmMemory[inIndex + i] = close[i];
        }


        wasm.tsi_into(inPtr, outPtr, len, 25, 13);


        const outIndex = outPtr / 8;
        const result = new Float64Array(len);
        for (let i = 0; i < len; i++) {
            result[i] = wasmMemory[outIndex + i];
        }


        const expected = wasm.tsi_js(close, 25, 13);
        assertArrayClose(result, expected, 1e-10, "TSI into mismatch");

    } finally {

        wasm.tsi_free(inPtr, len);
        wasm.tsi_free(outPtr, len);
    }
});

test('TSI batch into (in-place)', () => {

    const close = new Float64Array(testData.close);
    const len = close.length;


    const rows = 1;
    const totalSize = rows * len;


    const outPtr = wasm.tsi_alloc(totalSize);
    const inPtr = wasm.tsi_alloc(len);

    try {

        const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer);
        const inIndex = inPtr / 8;
        for (let i = 0; i < len; i++) {
            wasmMemory[inIndex + i] = close[i];
        }


        const resultRows = wasm.tsi_batch_into(
            inPtr, outPtr, len,
            25, 25, 0,
            13, 13, 0
        );

        assert.strictEqual(resultRows, rows, 'Should return correct number of rows');


        const outIndex = outPtr / 8;
        const result = new Float64Array(len);
        for (let i = 0; i < len; i++) {
            result[i] = wasmMemory[outIndex + i];
        }


        const expected = wasm.tsi_js(close, 25, 13);
        assertArrayClose(result, expected, 1e-10, "TSI batch into mismatch");

    } finally {

        wasm.tsi_free(inPtr, len);
        wasm.tsi_free(outPtr, totalSize);
    }
});

test('TSI edge cases', () => {

    const minData = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    const result1 = wasm.tsi_js(minData, 3, 2);
    assert.strictEqual(result1.length, minData.length);


    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = 100 + Math.random() * 20 - 10;
    }
    const result2 = wasm.tsi_js(data, 10, 10);
    assert.strictEqual(result2.length, data.length);


    const result3 = wasm.tsi_js(data, 2, 1);
    assert.strictEqual(result3.length, data.length);
});