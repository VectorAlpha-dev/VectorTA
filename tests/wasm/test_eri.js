
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

test('ERI accuracy', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;


    const result = wasm.eri_js(high, low, close, 13, "ema");


    assert.strictEqual(result.length, close.length * 2);


    const bull = result.slice(0, close.length);
    const bear = result.slice(close.length);


    const expectedBullLastFive = [
        -103.35343557205488,
        6.839912366813223,
        -42.851503685589705,
        -9.444146016219747,
        11.476446271808527,
    ];
    const expectedBearLastFive = [
        -433.3534355720549,
        -314.1600876331868,
        -414.8515036855897,
        -336.44414601621975,
        -925.5235537281915,
    ];



    assertArrayClose(
        bull.slice(-5),
        expectedBullLastFive,
        0.01,
        'ERI bull last 5 values mismatch'
    );


    assertArrayClose(
        bear.slice(-5),
        expectedBearLastFive,
        0.01,
        'ERI bear last 5 values mismatch'
    );
});

test('ERI error handling', () => {

    assert.throws(
        () => wasm.eri_js([], [], [], 13, "ema"),
        /Empty input data|Empty data/i,
        'Should fail with empty data'
    );


    assert.throws(
        () => wasm.eri_js([1, 2, 3], [1, 2], [1, 2, 3], 13, "ema"),
        /must have the same length/,
        'Should fail with mismatched lengths'
    );


    assert.throws(
        () => wasm.eri_js([1, 2, 3], [1, 2, 3], [1, 2, 3], 0, "ema"),
        /Invalid period/,
        'Should fail with zero period'
    );


    assert.throws(
        () => wasm.eri_js([1, 2, 3], [1, 2, 3], [1, 2, 3], 10, "ema"),
        /Invalid period/,
        'Should fail with period exceeding data length'
    );


    const allNaN = new Float64Array(10).fill(NaN);
    assert.throws(
        () => wasm.eri_js(allNaN, allNaN, allNaN, 5, "ema"),
        /All.*NaN/,
        'Should fail with all NaN values'
    );


    assert.throws(
        () => wasm.eri_js([42.0], [40.0], [41.0], 9, "ema"),
        /Invalid period|Not enough/,
        'Should fail with single data point and large period'
    );


    const withInf = [1, Infinity, 3, 4, 5];

    const result = wasm.eri_js(withInf, withInf, withInf, 2, "ema");
    assert(result.length === withInf.length * 2, 'Should handle infinite values');
});

test('ERI warmup period verification', () => {
    const high = testData.high.slice(0, 50);
    const low = testData.low.slice(0, 50);
    const close = testData.close.slice(0, 50);


    const period = 13;
    const result = wasm.eri_js(high, low, close, period, "ema");
    const bull = result.slice(0, close.length);
    const bear = result.slice(close.length);


    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(bull[i]), `Expected NaN in bull warmup at index ${i}`);
        assert(isNaN(bear[i]), `Expected NaN in bear warmup at index ${i}`);
    }


    assert(!isNaN(bull[period - 1]), `Expected value at index ${period - 1}`);
    assert(!isNaN(bear[period - 1]), `Expected value at index ${period - 1}`);




    const highWithNaN = new Float64Array(high);
    const lowWithNaN = new Float64Array(low);
    const closeWithNaN = new Float64Array(close);


    highWithNaN[0] = NaN;
    lowWithNaN[0] = NaN;
    closeWithNaN[0] = NaN;

    const result2 = wasm.eri_js(highWithNaN, lowWithNaN, closeWithNaN, 5, "ema");
    const bull2 = result2.slice(0, closeWithNaN.length);
    const bear2 = result2.slice(closeWithNaN.length);


    for (let i = 0; i < 5; i++) {
        assert(isNaN(bull2[i]), `Expected NaN at index ${i} with NaN at start`);
        assert(isNaN(bear2[i]), `Expected NaN at index ${i} with NaN at start`);
    }

    assert(!isNaN(bull2[5]), 'Expected value at index 5 after warmup');
    assert(!isNaN(bear2[5]), 'Expected value at index 5 after warmup');
});

test.skip('ERI fast API', () => {

    const high = testData.high;
    const low = testData.low;
    const close = testData.close;
    const len = close.length;


    const bullPtr = wasm.eri_alloc(len);
    const bearPtr = wasm.eri_alloc(len);

    try {

        wasm.eri_into(
            high,
            low,
            close,
            bullPtr,
            bearPtr,
            len,
            13,
            "ema"
        );


        const bull = new Float64Array(wasm.memory.buffer, bullPtr, len);
        const bear = new Float64Array(wasm.memory.buffer, bearPtr, len);


        const safeResult = wasm.eri_js(high, low, close, 13, "ema");
        const safeBull = safeResult.slice(0, len);
        const safeBear = safeResult.slice(len);


        assertArrayClose(Array.from(bull), safeBull, 1e-9, 'Bull values should match between safe and fast API');
        assertArrayClose(Array.from(bear), safeBear, 1e-9, 'Bear values should match between safe and fast API');
    } finally {

        wasm.eri_free(bullPtr, len);
        wasm.eri_free(bearPtr, len);
    }
});

test('ERI batch processing', () => {
    const high = testData.high;
    const low = testData.low;
    const close = testData.close;

    const config = {
        period_range: [10, 20, 5],
        ma_type: "ema"
    };

    const result = wasm.eri_batch(high, low, close, config);


    const numParams = 3;
    assert.strictEqual(result.rows, numParams * 2, 'Should have 6 rows (3 params * 2 for bull/bear)');
    assert.strictEqual(result.cols, close.length, 'Columns should match data length');
    assert.deepStrictEqual(result.periods, [10, 15, 20], 'Periods should match expected');


    assert.strictEqual(result.values.length, result.rows * result.cols, 'Values length should be rows * cols');



    const bullRows = numParams;
    const bearRows = numParams;
    const bullValues = result.values.slice(0, bullRows * result.cols);
    const bearValues = result.values.slice(bullRows * result.cols);


    const singleConfig = {
        period_range: [13, 13, 0],
        ma_type: "ema"
    };

    const singleResult = wasm.eri_batch(high, low, close, singleConfig);
    assert.strictEqual(singleResult.rows, 2, 'Single period should have 2 rows (bull and bear)');


    const regularResult = wasm.eri_js(high, low, close, 13, "ema");
    const regularBull = regularResult.slice(0, close.length);
    const regularBear = regularResult.slice(close.length);


    const singleBull = singleResult.values.slice(0, close.length);
    const singleBear = singleResult.values.slice(close.length, 2 * close.length);

    assertArrayClose(
        singleBull,
        regularBull,
        1e-9,
        'Batch bull should match regular ERI'
    );

    assertArrayClose(
        singleBear,
        regularBear,
        1e-9,
        'Batch bear should match regular ERI'
    );
});

test('ERI batch with different MA types', () => {
    const high = testData.high.slice(0, 100);
    const low = testData.low.slice(0, 100);
    const close = testData.close.slice(0, 100);

    const maTypes = ["ema", "sma", "wma", "hma"];

    for (const maType of maTypes) {
        const config = {
            period_range: [13, 13, 0],
            ma_type: maType
        };

        const batchResult = wasm.eri_batch(high, low, close, config);





        const singleResult = wasm.eri_js(high, low, close, 13, maType);
        const singleBull = singleResult.slice(0, close.length);
        const singleBear = singleResult.slice(close.length);


        const batchBull = batchResult.values.slice(0, close.length);
        const batchBear = batchResult.values.slice(close.length, 2 * close.length);

        assertArrayClose(
            batchBull,
            singleBull,
            1e-9,
            `Bull mismatch for MA type ${maType}`
        );

        assertArrayClose(
            batchBear,
            singleBear,
            1e-9,
            `Bear mismatch for MA type ${maType}`
        );
    }
});

test('ERI batch matches individual calculations', () => {
    const high = testData.high.slice(0, 100);
    const low = testData.low.slice(0, 100);
    const close = testData.close.slice(0, 100);

    const periods = [10, 15, 20];
    const maType = "sma";


    const batchResult = wasm.eri_batch(high, low, close, {
        period_range: [10, 20, 5],
        ma_type: maType
    });


    for (let i = 0; i < periods.length; i++) {
        const period = periods[i];


        const bullBatch = batchResult.values.slice(i * close.length, (i + 1) * close.length);
        const bearBatch = batchResult.values.slice((periods.length + i) * close.length, (periods.length + i + 1) * close.length);


        const individualResult = wasm.eri_js(high, low, close, period, maType);
        const bullSingle = individualResult.slice(0, close.length);
        const bearSingle = individualResult.slice(close.length);


        assertArrayClose(
            bullBatch,
            bullSingle,
            1e-9,
            `Bull mismatch for period ${period}`
        );
        assertArrayClose(
            bearBatch,
            bearSingle,
            1e-9,
            `Bear mismatch for period ${period}`
        );
    }
});

test('ERI batch warmup consistency', () => {
    const high = new Float64Array(50);
    const low = new Float64Array(50);
    const close = new Float64Array(50);


    for (let i = 0; i < 50; i++) {
        high[i] = 100 + i + Math.random();
        low[i] = 100 + i - Math.random();
        close[i] = 100 + i;
    }


    high[0] = NaN;
    high[1] = NaN;

    const result = wasm.eri_batch(high, low, close, {
        period_range: [5, 15, 5],
        ma_type: "ema"
    });


    const expectedWarmups = [2 + 5 - 1, 2 + 10 - 1, 2 + 15 - 1];
    const numParams = 3;

    for (let row = 0; row < numParams; row++) {
        const warmup = expectedWarmups[row];

        const bullRow = result.values.slice(row * 50, (row + 1) * 50);
        const bearRow = result.values.slice((numParams + row) * 50, (numParams + row + 1) * 50);


        for (let i = 0; i < Math.min(warmup, 50); i++) {
            assert(isNaN(bullRow[i]), `Row ${row}: Expected NaN at index ${i}`);
            assert(isNaN(bearRow[i]), `Row ${row}: Expected NaN at index ${i}`);
        }


        if (warmup < 50) {
            assert(!isNaN(bullRow[warmup]), `Row ${row}: Expected value at index ${warmup}`);
            assert(!isNaN(bearRow[warmup]), `Row ${row}: Expected value at index ${warmup}`);
        }
    }
});

test('ERI batch edge cases', () => {
    const high = testData.high.slice(0, 50);
    const low = testData.low.slice(0, 50);
    const close = testData.close.slice(0, 50);


    const largeStepConfig = {
        period_range: [13, 15, 10],
        ma_type: "ema"
    };

    const largeStepResult = wasm.eri_batch(high, low, close, largeStepConfig);
    assert.strictEqual(largeStepResult.rows, 2, 'Should have 2 rows for period=13 (bull and bear)');
    assert.strictEqual(largeStepResult.periods[0], 13, 'Period should be 13');


    assert.throws(
        () => wasm.eri_batch([], [], [], { period_range: [13, 13, 0], ma_type: "ema" }),
        /All input values are NaN/,
        'Should fail with empty data'
    );


    assert.throws(
        () => wasm.eri_batch(high, low, close, { period_range: [13], ma_type: "ema" }),
        /Invalid config/,
        'Should fail with invalid config'
    );
});

test.skip('ERI zero-copy API', () => {

    const high = new Float64Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    const low = new Float64Array([8, 18, 28, 38, 48, 58, 68, 78, 88, 98]);
    const close = new Float64Array([9, 19, 29, 39, 49, 59, 69, 79, 89, 99]);
    const len = close.length;
    const period = 5;
    const maType = "ema";


    const bullPtr = wasm.eri_alloc(len);
    const bearPtr = wasm.eri_alloc(len);

    assert(bullPtr !== 0, 'Failed to allocate bull buffer');
    assert(bearPtr !== 0, 'Failed to allocate bear buffer');

    try {

        wasm.eri_into(high, low, close, bullPtr, bearPtr, len, period, maType);


        const bullView = new Float64Array(wasm.memory.buffer, bullPtr, len);
        const bearView = new Float64Array(wasm.memory.buffer, bearPtr, len);


        const safeResult = wasm.eri_js(high, low, close, period, maType);
        const safeBull = safeResult.slice(0, len);
        const safeBear = safeResult.slice(len);


        for (let i = 0; i < len; i++) {
            if (isNaN(safeBull[i]) && isNaN(bullView[i])) {
                continue;
            }
            assertClose(bullView[i], safeBull[i], 1e-10, `Bull mismatch at index ${i}`);
            assertClose(bearView[i], safeBear[i], 1e-10, `Bear mismatch at index ${i}`);
        }


        for (let i = 0; i < period; i++) {
            assert(isNaN(bullView[i]), `Expected NaN in bull at warmup index ${i}`);
            assert(isNaN(bearView[i]), `Expected NaN in bear at warmup index ${i}`);
        }


        assert(!isNaN(bullView[period]), `Expected value in bull after warmup`);
        assert(!isNaN(bearView[period]), `Expected value in bear after warmup`);
    } finally {

        wasm.eri_free(bullPtr, len);
        wasm.eri_free(bearPtr, len);
    }
});

test.skip('ERI memory management', () => {


    const sizes = [100, 1000, 10000];

    for (const size of sizes) {
        const bullPtr = wasm.eri_alloc(size);
        const bearPtr = wasm.eri_alloc(size);

        assert(bullPtr !== 0, `Failed to allocate bull buffer of size ${size}`);
        assert(bearPtr !== 0, `Failed to allocate bear buffer of size ${size}`);
        assert(bullPtr !== bearPtr, 'Bull and bear buffers should be different');


        const bullView = new Float64Array(wasm.memory.buffer, bullPtr, size);
        const bearView = new Float64Array(wasm.memory.buffer, bearPtr, size);

        for (let i = 0; i < Math.min(10, size); i++) {
            bullView[i] = i * 2.5;
            bearView[i] = i * 3.5;
        }


        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(bullView[i], i * 2.5, `Bull memory corruption at index ${i}`);
            assert.strictEqual(bearView[i], i * 3.5, `Bear memory corruption at index ${i}`);
        }


        wasm.eri_free(bullPtr, size);
        wasm.eri_free(bearPtr, size);
    }
});

test('ERI SIMD128 consistency', () => {


    const testCases = [
        { size: 10, period: 5 },
        { size: 100, period: 13 },
        { size: 1000, period: 20 }
    ];

    for (const testCase of testCases) {
        const high = new Float64Array(testCase.size);
        const low = new Float64Array(testCase.size);
        const close = new Float64Array(testCase.size);


        for (let i = 0; i < testCase.size; i++) {
            const base = 100 + Math.sin(i * 0.1) * 10;
            high[i] = base + 2;
            low[i] = base - 2;
            close[i] = base;
        }

        const result = wasm.eri_js(high, low, close, testCase.period, "ema");
        const bull = result.slice(0, testCase.size);
        const bear = result.slice(testCase.size);


        assert.strictEqual(bull.length, testCase.size);
        assert.strictEqual(bear.length, testCase.size);


        for (let i = 0; i < testCase.period - 1; i++) {
            assert(isNaN(bull[i]), `Expected NaN in bull at warmup index ${i} for size=${testCase.size}`);
            assert(isNaN(bear[i]), `Expected NaN in bear at warmup index ${i} for size=${testCase.size}`);
        }


        for (let i = testCase.period - 1; i < Math.min(testCase.period + 10, testCase.size); i++) {
            assert(!isNaN(bull[i]), `Unexpected NaN in bull at index ${i} for size=${testCase.size}`);
            assert(!isNaN(bear[i]), `Unexpected NaN in bear at index ${i} for size=${testCase.size}`);
        }
    }
});

test.skip('ERI flat API', () => {

    const high = testData.high.slice(0, 100);
    const low = testData.low.slice(0, 100);
    const close = testData.close.slice(0, 100);


    const flatResult = wasm.eri_js_flat(high, low, close, 13, "ema");


    assert(flatResult.values, 'Should have values array');
    assert.strictEqual(flatResult.rows, 2, 'Should have 2 rows (bull and bear)');
    assert.strictEqual(flatResult.cols, close.length, 'Columns should match data length');
    assert.strictEqual(flatResult.values.length, 2 * close.length, 'Values should be flattened');


    const regularResult = wasm.eri_js(high, low, close, 13, "ema");


    assertArrayClose(
        flatResult.values,
        regularResult,
        1e-10,
        'Flat API should match regular API'
    );


    const flatBull = flatResult.values.slice(0, flatResult.cols);
    const flatBear = flatResult.values.slice(flatResult.cols);
    const regularBull = regularResult.slice(0, close.length);
    const regularBear = regularResult.slice(close.length);


    assertArrayClose(flatBull, regularBull, 1e-10, 'Bull values should match');
    assertArrayClose(flatBear, regularBear, 1e-10, 'Bear values should match');


    const flatResult2 = wasm.eri_js_flat(high, low, close, 20, "sma");
    assert.strictEqual(flatResult2.values.length, 2 * close.length, 'Should work with different params');
});

test('ERI large dataset performance', () => {
    const size = 50000;
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);


    for (let i = 0; i < size; i++) {
        const base = 100 + Math.sin(i * 0.01) * 20 + Math.random() * 5;
        high[i] = base + Math.random() * 5;
        low[i] = base - Math.random() * 5;
        close[i] = base;
    }

    const startTime = Date.now();
    const result = wasm.eri_js(high, low, close, 13, "ema");
    const endTime = Date.now();

    console.log(`ERI large dataset (${size} points) completed in ${endTime - startTime}ms`);


    assert.strictEqual(result.length, size * 2, 'Result should have 2x size elements');

    const bull = result.slice(0, size);
    const bear = result.slice(size);


    for (let i = 0; i < 12; i++) {
        assert(isNaN(bull[i]), `Expected NaN in bull warmup at ${i}`);
        assert(isNaN(bear[i]), `Expected NaN in bear warmup at ${i}`);
    }


    for (let i = 12; i < 100; i++) {
        assert(!isNaN(bull[i]), `Unexpected NaN in bull at ${i}`);
        assert(!isNaN(bear[i]), `Unexpected NaN in bear at ${i}`);
    }
});

test.after(() => {
    console.log('ERI WASM tests completed');
});
