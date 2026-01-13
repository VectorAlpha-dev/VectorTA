
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

test('ZLEMA partial params', () => {
    const data = testData.close;


    const result = wasm.zlema_js(data, 14);
    assert.strictEqual(result.length, data.length, 'Output length should match input');
});

test('ZLEMA accuracy', async () => {
    const data = testData.close;
    const expected = EXPECTED_OUTPUTS.zlema;

    const result = wasm.zlema_js(data, expected.defaultParams.period);

    assert.strictEqual(result.length, data.length, 'Output length should match input');


    const last5 = result.slice(-5);
    assertArrayClose(last5, expected.last5Values, 1e-1, 'ZLEMA last 5 values mismatch');


    await compareWithRust('zlema', result, 'close', expected.defaultParams);
});

test('ZLEMA zero period', () => {
    const data = [10.0, 20.0, 30.0];

    assert.throws(() => {
        wasm.zlema_js(data, 0);
    }, /Invalid period/, 'Should throw error for zero period');
});

test('ZLEMA period exceeds length', () => {
    const data = [10.0, 20.0, 30.0];

    assert.throws(() => {
        wasm.zlema_js(data, 10);
    }, /Invalid period/, 'Should throw error when period exceeds data length');
});

test('ZLEMA very small dataset', () => {
    const data = [42.0];

    assert.throws(() => {
        wasm.zlema_js(data, 14);
    }, /Invalid period/, 'Should throw error for insufficient data');
});

test('ZLEMA empty input', () => {
    const data = [];

    assert.throws(() => {
        wasm.zlema_js(data, 14);
    }, /Input data slice is empty/, 'Should throw error for empty input');
});

test('ZLEMA default candles', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.zlema_js(close, 14);
    assert.strictEqual(result.length, close.length, 'Output length should match input');
});

test('ZLEMA reinput', () => {
    const data = testData.close;


    const firstResult = wasm.zlema_js(data, 21);


    const secondResult = wasm.zlema_js(firstResult, 14);

    assert.strictEqual(secondResult.length, firstResult.length, 'Output length should match input');


    let firstValidIdx = 0;
    while (firstValidIdx < secondResult.length && isNaN(secondResult[firstValidIdx])) {
        firstValidIdx++;
    }


    for (let idx = firstValidIdx; idx < secondResult.length; idx++) {
        assert(isFinite(secondResult[idx]), `NaN found at index ${idx} after first valid at ${firstValidIdx}`);
    }
});

test('ZLEMA nan handling', () => {
    const data = testData.close;
    const period = 14;

    const result = wasm.zlema_js(data, period);
    assert.strictEqual(result.length, data.length, 'Output length should match input');


    assertAllNaN(result.slice(0, period - 1), `Expected NaN in warmup period [0:${period - 1}]`);


    if (result.length > period) {
        for (let i = period; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i} after warmup period`);
        }
    }
});

test('ZLEMA batch processing', () => {
    const data = testData.close;

    const result = wasm.zlema_batch_js(
        data,
        14,
        40,
        1
    );


    assert.strictEqual(result.length, 27 * data.length,
        'Batch result should have 27 rows × data length values');


    const metadata = wasm.zlema_batch_metadata_js(14, 40, 1);
    assert.strictEqual(metadata.length, 27, 'Metadata should have 27 periods');
    assert.strictEqual(metadata[0], 14, 'First period should be 14');
    assert.strictEqual(metadata[26], 40, 'Last period should be 40');


    const single_zlema = wasm.zlema_js(data, 14);
    const first_row = result.slice(0, data.length);
    assertArrayClose(first_row, single_zlema, 1e-9, 'Batch period 14 row should match single calculation');
});

test('ZLEMA all nan input', () => {
    const data = [NaN, NaN, NaN];

    assert.throws(() => {
        wasm.zlema_js(data, 2);
    }, /All values are NaN/, 'Should throw error for all NaN input');
});

test('ZLEMA batch metadata', () => {

    const metadata = wasm.zlema_batch_metadata_js(10, 20, 2);


    assert.strictEqual(metadata.length, 6, 'Metadata should have 6 periods');
    assert.deepStrictEqual(Array.from(metadata), [10, 12, 14, 16, 18, 20],
        'Metadata should contain correct periods');
});

test('ZLEMA batch single period', () => {
    const data = testData.close;


    const result = wasm.zlema_batch_js(
        data,
        14,
        14,
        0
    );


    assert.strictEqual(result.length, data.length,
        'Single period batch should have 1 row × data length values');


    const single_zlema = wasm.zlema_js(data, 14);
    assertArrayClose(result, single_zlema, 1e-9, 'Single period batch should match single calculation');
});

test('ZLEMA fast API basic', () => {
    const data = testData.close;
    const period = 14;


    const inPtr = wasm.zlema_alloc(data.length);
    const outPtr = wasm.zlema_alloc(data.length);

    try {

        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        memory.set(data, inPtr / 8);


        wasm.zlema_into(inPtr, outPtr, data.length, period);


        const result = Array.from(memory.slice(outPtr / 8, outPtr / 8 + data.length));


        const safeResult = wasm.zlema_js(data, period);
        assertArrayClose(result, safeResult, 1e-9, 'Fast API should match safe API');
    } finally {

        wasm.zlema_free(inPtr, data.length);
        wasm.zlema_free(outPtr, data.length);
    }
});

test('ZLEMA fast API in-place (aliasing)', () => {
    const data = testData.close;
    const period = 14;


    const ptr = wasm.zlema_alloc(data.length);

    try {

        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        memory.set(data, ptr / 8);


        wasm.zlema_into(ptr, ptr, data.length, period);


        const result = Array.from(memory.slice(ptr / 8, ptr / 8 + data.length));


        const safeResult = wasm.zlema_js(data, period);
        assertArrayClose(result, safeResult, 1e-9, 'In-place computation should match safe API');
    } finally {

        wasm.zlema_free(ptr, data.length);
    }
});

test('ZLEMA unified batch API', () => {
    const data = testData.close;


    const config = {
        period_range: [14, 20, 2]
    };

    const result = wasm.zlema_batch(data, config);

    assert(result.values, 'Result should have values array');
    assert(result.combos, 'Result should have combos array');
    assert.strictEqual(result.rows, 4, 'Should have 4 rows (periods)');
    assert.strictEqual(result.cols, data.length, 'Columns should match data length');
    assert.strictEqual(result.values.length, 4 * data.length, 'Values should be flattened 2D array');


    const firstRowValues = result.values.slice(0, data.length);
    const singleResult = wasm.zlema_js(data, 14);
    assertArrayClose(firstRowValues, singleResult, 1e-9, 'First row should match period=14 calculation');
});

test('ZLEMA batch fast API', () => {
    const data = testData.close;
    const periodStart = 10;
    const periodEnd = 20;
    const periodStep = 5;


    const inPtr = wasm.zlema_alloc(data.length);
    const outPtr = wasm.zlema_alloc(3 * data.length);

    try {

        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        memory.set(data, inPtr / 8);


        const rows = wasm.zlema_batch_into(inPtr, outPtr, data.length, periodStart, periodEnd, periodStep);
        assert.strictEqual(rows, 3, 'Should return 3 rows');


        const result = Array.from(memory.slice(outPtr / 8, outPtr / 8 + 3 * data.length));


        const firstRow = result.slice(0, data.length);
        const expected10 = wasm.zlema_js(data, 10);
        assertArrayClose(firstRow, expected10, 1e-9, 'First batch row should match period=10');
    } finally {

        wasm.zlema_free(inPtr, data.length);
        wasm.zlema_free(outPtr, 3 * data.length);
    }
});

test('ZLEMA batch edge cases', () => {

    const data = testData.close.slice(0, 100);


    const singleBatch = wasm.zlema_batch(data, {
        period_range: [14, 14, 0]
    });

    assert.strictEqual(singleBatch.values.length, 100);
    assert.strictEqual(singleBatch.combos.length, 1);
    assert.strictEqual(singleBatch.combos[0].period, 14);


    const largeBatch = wasm.zlema_batch(data, {
        period_range: [10, 15, 10]
    });


    assert.strictEqual(largeBatch.values.length, 100);
    assert.strictEqual(largeBatch.combos.length, 1);
    assert.strictEqual(largeBatch.combos[0].period, 10);


    const multiPeriod = wasm.zlema_batch(data, {
        period_range: [10, 20, 5]
    });

    assert.strictEqual(multiPeriod.rows, 3);
    assert.strictEqual(multiPeriod.cols, 100);
    assert.strictEqual(multiPeriod.combos.length, 3);


    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = multiPeriod.values.slice(rowStart, rowEnd);

        const singleResult = wasm.zlema_js(data, periods[i]);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Period ${periods[i]} mismatch`
        );
    }
});

test('ZLEMA batch metadata validation', () => {

    const data = testData.close.slice(0, 50);

    const result = wasm.zlema_batch(data, {
        period_range: [10, 20, 5]
    });


    assert.strictEqual(result.combos.length, 3);


    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[1].period, 15);
    assert.strictEqual(result.combos[2].period, 20);






    for (let combo = 0; combo < result.combos.length; combo++) {
        const period = result.combos[combo].period;
        const rowStart = combo * 50;
        const rowData = result.values.slice(rowStart, rowStart + 50);


        for (let i = period; i < Math.min(50, rowData.length); i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('ZLEMA SIMD128 consistency', () => {


    const testCases = [
        { size: 10, period: 5 },
        { size: 100, period: 14 },
        { size: 1000, period: 20 },
        { size: 10000, period: 50 }
    ];

    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }

        const result = wasm.zlema_js(data, testCase.period);


        assert.strictEqual(result.length, data.length);



        const warmupEnd = Math.min(testCase.period, result.length);
        for (let i = 0; i < warmupEnd - 1; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}, period=${testCase.period}`);
        }


        if (result.length >= testCase.period) {
            let sumAfterWarmup = 0;
            let countAfterWarmup = 0;

            for (let i = testCase.period; i < result.length; i++) {
                assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
                sumAfterWarmup += result[i];
                countAfterWarmup++;
            }


            if (countAfterWarmup > 0) {
                const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
                assert(Math.abs(avgAfterWarmup) < 10, `Average value ${avgAfterWarmup} seems unreasonable`);
            }
        }
    }
});

test('ZLEMA zero-copy error handling', () => {

    assert.throws(() => {
        wasm.zlema_into(0, 0, 10, 14);
    }, /null pointer|invalid memory/i);


    const ptr = wasm.zlema_alloc(10);
    try {

        assert.throws(() => {
            wasm.zlema_into(ptr, ptr, 10, 0);
        }, /Invalid period/);


        assert.throws(() => {
            wasm.zlema_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.zlema_free(ptr, 10);
    }
});

test('ZLEMA zero-copy memory management', () => {

    const sizes = [100, 1000, 10000, 100000];

    for (const size of sizes) {
        const ptr = wasm.zlema_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);


        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }


        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }


        wasm.zlema_free(ptr, size);
    }
});

test('ZLEMA zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }

    const ptr = wasm.zlema_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');

    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);

        wasm.zlema_into(ptr, ptr, size, 14);


        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);



        for (let i = 0; i < 13; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }


        for (let i = 13; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i} after warmup`);
        }
    } finally {
        wasm.zlema_free(ptr, size);
    }
});

test.after(() => {
    console.log('ZLEMA WASM tests completed');
});