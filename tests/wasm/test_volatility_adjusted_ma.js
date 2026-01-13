
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


import * as wasm from '../../pkg/vector_ta.js';


const EXPECTED_LAST_5 = [
    61437.31013970,
    61409.77885185,
    61381.24752811,
    61352.71733871,
    61321.57890702,
];

const DEFAULT_PARAMS = {
    basePeriod: 113,
    volPeriod: 51,
    smoothing: true,
    smoothType: 3,
    smoothPeriod: 5
};

let testData;

test.before(async () => {
    testData = loadTestData();
});

test('VAMA accuracy', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.vama_js(
        close,
        DEFAULT_PARAMS.basePeriod,
        DEFAULT_PARAMS.volPeriod,
        DEFAULT_PARAMS.smoothing,
        DEFAULT_PARAMS.smoothType,
        DEFAULT_PARAMS.smoothPeriod
    );

    assert.strictEqual(result.length, close.length, 'Output length should match input');


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        EXPECTED_LAST_5,
        1e-6,
        "VAMA reference value mismatch"
    );
});

test('VAMA partial params', () => {

    const close = new Float64Array(testData.close);


    const result = wasm.vama_js(close, 113, 51, false, 1, 5);
    assert.strictEqual(result.length, close.length);
});

test('VAMA warmup NaN', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.vama_js(
        close,
        DEFAULT_PARAMS.basePeriod,
        DEFAULT_PARAMS.volPeriod,
        DEFAULT_PARAMS.smoothing,
        DEFAULT_PARAMS.smoothType,
        DEFAULT_PARAMS.smoothPeriod
    );


    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }


    const warmup = firstValid + Math.max(DEFAULT_PARAMS.basePeriod, DEFAULT_PARAMS.volPeriod) - 1;


    if (warmup > 0) {
        const warmupValues = result.slice(0, Math.min(warmup, result.length));
        assertAllNaN(warmupValues, `Expected NaN in warmup period [0:${warmup})`);
    }


    const nonNaNCount = result.filter(v => !isNaN(v)).length;
    assert(nonNaNCount > 0, 'Should have some non-NaN values after warmup');
});

test('VAMA edge cases', () => {

    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);

    const result = wasm.vama_js(data, 2, 2, false, 1, 5);

    assert.strictEqual(result.length, data.length, 'Output length should match input');




    const warmup = Math.max(2, 2) - 1;
    if (warmup > 0) {
        const warmupValues = result.slice(0, warmup);
        assertAllNaN(warmupValues, `Expected NaN in warmup [0:${warmup})`);
    }


    if (data.length > warmup) {
        const afterWarmup = result.slice(warmup);
        const hasValues = afterWarmup.some(v => !isNaN(v));
        assert(hasValues, 'Should have values after warmup');
    }
});

test('VAMA smoothing variations', () => {

    const close = new Float64Array(testData.close.slice(0, 100));

    const results = {};


    const smoothTypes = [
        { type: 1, name: 'SMA' },
        { type: 2, name: 'EMA' },
        { type: 3, name: 'WMA' }
    ];

    for (const { type, name } of smoothTypes) {
        const result = wasm.vama_js(close, 10, 5, true, type, 3);
        results[name] = result;
        assert.strictEqual(result.length, close.length, `${name} output length mismatch`);
    }


    const noSmooth = wasm.vama_js(close, 10, 5, false, 1, 5);


    for (const [name, smoothResult] of Object.entries(results)) {
        let foundDifference = false;
        for (let i = 0; i < smoothResult.length; i++) {
            if (!isNaN(smoothResult[i]) && !isNaN(noSmooth[i])) {
                if (Math.abs(smoothResult[i] - noSmooth[i]) > 1e-10) {
                    foundDifference = true;
                    break;
                }
            }
        }
        assert(foundDifference, `${name} smoothing should produce different values`);
    }
});

test('VAMA invalid periods', () => {

    const data = new Float64Array(10).fill(1.0);


    assert.throws(() => {
        wasm.vama_js(data, 0, 5, false, 1, 5);
    }, /Invalid period/, 'Should throw on zero base period');


    assert.throws(() => {
        wasm.vama_js(data, 5, 0, false, 1, 5);
    }, /Invalid period/, 'Should throw on zero vol period');


    assert.throws(() => {
        wasm.vama_js(data, 20, 5, false, 1, 5);
    }, /Invalid period/, 'Should throw when period exceeds data length');
});

test('VAMA invalid smooth type', () => {

    const data = new Float64Array(100).fill(1.0);


    assert.throws(() => {
        wasm.vama_js(data, 10, 5, true, 0, 5);
    }, /Invalid smooth/, 'Should throw on invalid smooth type 0');


    assert.throws(() => {
        wasm.vama_js(data, 10, 5, true, 4, 5);
    }, /Invalid smooth/, 'Should throw on invalid smooth type 4');
});

test('VAMA empty input', () => {

    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.vama_js(empty, 113, 51, false, 1, 5);
    }, /Empty|empty/, 'Should throw on empty array');
});

test('VAMA all NaN input', () => {

    const allNaN = new Float64Array(100).fill(NaN);

    assert.throws(() => {
        wasm.vama_js(allNaN, 113, 51, false, 1, 5);
    }, /All values are NaN|all NaN/, 'Should throw on all NaN values');
});

test('VAMA batch single params', () => {

    const close = new Float64Array(testData.close.slice(0, 200));

    const result = wasm.vama_batch(close, {
        base_period_range: [113, 113, 0],
        vol_period_range: [51, 51, 0]
    });

    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert.strictEqual(result.rows, 1, 'Should have 1 row');
    assert.strictEqual(result.cols, close.length, 'Cols should match data length');
    assert.strictEqual(result.combos.length, 1, 'Should have 1 combination');


    const batchRow = result.values.slice(0, close.length);
    const singleResult = wasm.vama_js(close, 113, 51, false, 3, 5);


    for (let i = 0; i < batchRow.length; i++) {
        if (!isNaN(batchRow[i]) && !isNaN(singleResult[i])) {
            assertClose(batchRow[i], singleResult[i], 1e-10,
                       `Batch vs single mismatch at ${i}`);
        }
    }
});

test('VAMA batch sweep', () => {

    const close = new Float64Array(testData.close.slice(0, 150));

    const result = wasm.vama_batch(close, {
        base_period_range: [100, 104, 2],
        vol_period_range: [40, 44, 2]
    });


    assert.strictEqual(result.rows, 9, 'Should have 9 rows');
    assert.strictEqual(result.cols, close.length, 'Cols should match data length');
    assert.strictEqual(result.combos.length, 9, 'Should have 9 combinations');


    const expectedBases = [100, 100, 100, 102, 102, 102, 104, 104, 104];
    const expectedVols = [40, 42, 44, 40, 42, 44, 40, 42, 44];

    for (let i = 0; i < result.combos.length; i++) {
        assert.strictEqual(result.combos[i].base_period, expectedBases[i],
                          `Base period mismatch at combo ${i}`);
        assert.strictEqual(result.combos[i].vol_period, expectedVols[i],
                          `Vol period mismatch at combo ${i}`);
    }
});

test('VAMA zero-copy functionality', () => {

    const close = new Float64Array(testData.close.slice(0, 200));
    const len = close.length;


    const inPtr = wasm.vama_alloc(len);
    const outPtr = wasm.vama_alloc(len);


    const memory = wasm.__wasm.memory;
    const inView = new Float64Array(memory.buffer, inPtr, len);
    const outView = new Float64Array(memory.buffer, outPtr, len);


    inView.set(close);


    wasm.vama_into(
        inPtr, outPtr, len,
        DEFAULT_PARAMS.basePeriod,
        DEFAULT_PARAMS.volPeriod,
        DEFAULT_PARAMS.smoothing,
        DEFAULT_PARAMS.smoothType,
        DEFAULT_PARAMS.smoothPeriod
    );


    const result = Array.from(outView);
    assert.strictEqual(result.length, close.length);


    const firstValidIdx = result.findIndex(v => !isNaN(v));
    assert(firstValidIdx > 0, 'Should have warmup NaN values');


    const regularResult = wasm.vama_js(
        close,
        DEFAULT_PARAMS.basePeriod,
        DEFAULT_PARAMS.volPeriod,
        DEFAULT_PARAMS.smoothing,
        DEFAULT_PARAMS.smoothType,
        DEFAULT_PARAMS.smoothPeriod
    );

    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i]) && !isNaN(regularResult[i])) {
            assertClose(result[i], regularResult[i], 1e-10,
                       `Zero-copy mismatch at index ${i}`);
        }
    }


    wasm.vama_free(inPtr, len);
    wasm.vama_free(outPtr, len);
});

test('VAMA with real market data', () => {

    const candles = loadTestData();
    const close = new Float64Array(candles.close);

    const result = wasm.vama_js(
        close,
        DEFAULT_PARAMS.basePeriod,
        DEFAULT_PARAMS.volPeriod,
        DEFAULT_PARAMS.smoothing,
        DEFAULT_PARAMS.smoothType,
        DEFAULT_PARAMS.smoothPeriod
    );

    assert.strictEqual(result.length, close.length);


    const validValues = result.filter(v => !isNaN(v));
    assert(validValues.length > 0, 'Should have valid values');


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        EXPECTED_LAST_5,
        1e-6,
        "Real data should produce expected reference values"
    );
});

test('VAMA performance characteristics', () => {

    const sizes = [100, 200, 400, 800];
    const times = [];

    for (const size of sizes) {

        const data = new Float64Array(testData.close.slice(0, size));

        const start = performance.now();
        wasm.vama_js(data, 20, 10, false, 1, 5);
        const end = performance.now();

        times.push(end - start);
    }


    for (let i = 1; i < times.length; i++) {
        const sizeRatio = sizes[i] / sizes[i-1];
        const timeRatio = times[i] / times[i-1];

        assert(timeRatio < sizeRatio * 3,
               `Performance should scale linearly (${timeRatio} vs ${sizeRatio})`);
    }
});