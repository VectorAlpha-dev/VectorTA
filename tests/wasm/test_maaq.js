
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

test('MAAQ partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.maaq_js(close, 11, 2, 30);
    assert.strictEqual(result.length, close.length);
});

test('MAAQ accuracy', async () => {

    const close = new Float64Array(testData.close);

    const result = wasm.maaq_js(close, 11, 2, 30);

    assert.strictEqual(result.length, close.length);


    const expectedLastFive = [
        59747.657115949725,
        59740.803138018055,
        59724.24153333905,
        59720.60576365108,
        59673.9954445178,
    ];


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        0.01,
        "MAAQ last 5 values mismatch"
    );


    await compareWithRust('maaq', result, 'close', {
        period: 11,
        fast_period: 2,
        slow_period: 30
    });
});

test('MAAQ default candles', async () => {

    const close = new Float64Array(testData.close);

    const result = wasm.maaq_js(close, 11, 2, 30);
    assert.strictEqual(result.length, close.length);


    await compareWithRust('maaq', result, 'close', {
        period: 11,
        fast_period: 2,
        slow_period: 30
    });
});

test('MAAQ zero period', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);


    assert.throws(() => {
        wasm.maaq_js(inputData, 0, 2, 30);
    });


    assert.throws(() => {
        wasm.maaq_js(inputData, 11, 0, 30);
    });


    assert.throws(() => {
        wasm.maaq_js(inputData, 11, 2, 0);
    });
});

test('MAAQ period exceeds length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.maaq_js(dataSmall, 10, 2, 30);
    });
});

test('MAAQ very small dataset', () => {

    const dataSingle = new Float64Array([42.0]);

    assert.throws(() => {
        wasm.maaq_js(dataSingle, 11, 2, 30);
    });
});

test('MAAQ empty input', () => {

    const dataEmpty = new Float64Array([]);

    assert.throws(() => {
        wasm.maaq_js(dataEmpty, 11, 2, 30);
    });
});

test('MAAQ all NaN', () => {

    const data = new Float64Array([NaN, NaN, NaN, NaN, NaN]);

    assert.throws(() => {
        wasm.maaq_js(data, 3, 2, 5);
    });
});

test('MAAQ reinput', () => {

    const close = new Float64Array(testData.close);


    const firstResult = wasm.maaq_js(close, 11, 2, 30);


    const secondResult = wasm.maaq_js(firstResult, 20, 3, 25);

    assert.strictEqual(secondResult.length, firstResult.length);



    for (let i = 40; i < secondResult.length; i++) {
        assert(isFinite(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('MAAQ NaN handling', () => {

    const close = new Float64Array(testData.close);
    const period = 11;

    const result = wasm.maaq_js(close, period, 2, 30);

    assert.strictEqual(result.length, close.length);


    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(result[i]), `Expected NaN at warmup index ${i}, got ${result[i]}`);
    }



    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('MAAQ batch', () => {

    const close = new Float64Array(testData.close);


    const batch_result = wasm.maaq_batch_js(
        close,
        {
            period_range: [11, 41, 10],
            fast_period_range: [2, 2, 0],
            slow_period_range: [30, 30, 0]
        }
    );
    const metadata = wasm.maaq_batch_metadata_js(
        11, 41, 10,
        2, 2, 0,
        30, 30, 0
    );



    assert.strictEqual(metadata.length, 12);

    assert.strictEqual(metadata[0], 11);
    assert.strictEqual(metadata[1], 2);
    assert.strictEqual(metadata[2], 30);


    assert.strictEqual(batch_result.length, 4 * close.length);


    let row_idx = 0;
    for (let p = 11; p <= 41; p += 10) {
        const individual_result = wasm.maaq_js(close, p, 2, 30);


        const row_start = row_idx * close.length;
        const row = batch_result.slice(row_start, row_start + close.length);

        assertArrayClose(row, individual_result, 1e-9, `Period ${p}`);
        row_idx++;
    }
});

test('MAAQ different periods', () => {

    const close = new Float64Array(testData.close);


    const testCases = [
        [5, 2, 10],
        [10, 3, 20],
        [20, 5, 40],
        [50, 10, 100],
    ];

    for (const [period, fast_p, slow_p] of testCases) {
        const result = wasm.maaq_js(close, period, fast_p, slow_p);
        assert.strictEqual(result.length, close.length);


        let validCount = 0;
        for (let i = period; i < result.length; i++) {
            if (!isNaN(result[i])) validCount++;
        }
        assert(validCount > close.length - period - 5,
            `Too many NaN values for params=(${period}, ${fast_p}, ${slow_p})`);
    }
});

test('MAAQ batch performance', () => {

    const close = new Float64Array(testData.close.slice(0, 1000));


    const startBatch = performance.now();
    const batchResult = wasm.maaq_batch_js(
        close,
        {
            period_range: [10, 30, 10],
            fast_period_range: [2, 2, 0],
            slow_period_range: [25, 35, 5]
        }
    );
    const batchTime = performance.now() - startBatch;

    const startSingle = performance.now();
    const singleResults = [];
    for (let p = 10; p <= 30; p += 10) {
        for (let s = 25; s <= 35; s += 5) {
            singleResults.push(...wasm.maaq_js(close, p, 2, s));
        }
    }
    const singleTime = performance.now() - startSingle;


    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);


    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});

test('MAAQ edge cases', () => {



    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1;
    }
    const result = wasm.maaq_js(data, 10, 2, 20);
    assert.strictEqual(result.length, data.length);


    for (let i = 10; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN at index ${i}`);
    }


    const constantData = new Float64Array(100).fill(5.0);
    const constantResult = wasm.maaq_js(constantData, 10, 2, 20);
    assert.strictEqual(constantResult.length, constantData.length);


    for (let i = 20; i < constantResult.length; i++) {
        assertClose(constantResult[i], 5.0, 1e-9, `Constant prediction failed at index ${i}`);
    }
});

test('MAAQ batch metadata', () => {

    const metadata = wasm.maaq_batch_metadata_js(
        11, 31, 10,
        2, 4, 2,
        25, 35, 10
    );


    assert.strictEqual(metadata.length, 36);


    assert.strictEqual(metadata[0], 11);
    assert.strictEqual(metadata[1], 2);
    assert.strictEqual(metadata[2], 25);


    assert.strictEqual(metadata[3], 11);
    assert.strictEqual(metadata[4], 2);
    assert.strictEqual(metadata[5], 35);
});

test('MAAQ warmup period calculation', () => {


    const close = new Float64Array(testData.close.slice(0, 50));

    const testCases = [
        { period: 5, fast_p: 2, slow_p: 10 },
        { period: 10, fast_p: 3, slow_p: 20 },
        { period: 20, fast_p: 5, slow_p: 30 },
        { period: 30, fast_p: 10, slow_p: 40 },
    ];

    for (const { period, fast_p, slow_p } of testCases) {
        const result = wasm.maaq_js(close, period, fast_p, slow_p);


        for (let i = 0; i < period - 1 && i < result.length; i++) {
            assert(isNaN(result[i]),
                `Expected NaN at warmup index ${i} for period=${period}, got ${result[i]}`);
        }


        if (period - 1 < result.length) {
            assert(!isNaN(result[period - 1]),
                `Expected valid value at index ${period - 1} for period=${period}`);
        }
    }
});

test('MAAQ consistency across calls', () => {

    const close = new Float64Array(testData.close.slice(0, 100));

    const result1 = wasm.maaq_js(close, 11, 2, 30);
    const result2 = wasm.maaq_js(close, 11, 2, 30);

    assertArrayClose(result1, result2, 1e-15, "MAAQ results not consistent");
});

test('MAAQ parameter step precision', () => {

    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }

    const batch_result = wasm.maaq_batch_js(
        data,
        {
            period_range: [5, 7, 1],
            fast_period_range: [2, 3, 1],
            slow_period_range: [10, 10, 0]
        }
    );


    assert.strictEqual(batch_result.length, 6 * data.length);


    const metadata = wasm.maaq_batch_metadata_js(5, 7, 1, 2, 3, 1, 10, 10, 0);
    assert.strictEqual(metadata.length, 18);
});

test('MAAQ streaming simulation', () => {

    const close = new Float64Array(testData.close.slice(0, 100));
    const period = 11;
    const fast_period = 2;
    const slow_period = 30;


    const batchResult = wasm.maaq_js(close, period, fast_period, slow_period);


    assert.strictEqual(batchResult.length, close.length);


    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(batchResult[i]), `Expected NaN at warmup index ${i}`);
    }


    let hasDifferentValues = false;
    for (let i = period - 1; i < close.length; i++) {
        assert(!isNaN(batchResult[i]), `Unexpected NaN at index ${i}`);
        if (Math.abs(batchResult[i] - close[i]) > 1e-9) {
            hasDifferentValues = true;
        }
    }
    assert(hasDifferentValues, "MAAQ should produce smoothed values after warmup");
});

test('MAAQ large period', () => {

    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.sin(i * 0.1) * 100 + 1000;
    }

    const period = 50;
    const result = wasm.maaq_js(data, period, 5, 60);
    assert.strictEqual(result.length, data.length);


    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(result[i]), `Expected NaN at warmup index ${i}`);
    }


    for (let i = period - 1; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});




test('MAAQ batch with invalid ranges', () => {

    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }


    const desc = wasm.maaq_batch_js(data, {
        period_range: [20, 10, 5],
        fast_period_range: [2, 2, 0],
        slow_period_range: [30, 30, 0]
    });
    assert.strictEqual(desc.length, 3 * data.length, 'Expected 3 period combos');


    assert.throws(() => {
        wasm.maaq_batch_js(data, {
            period_range: [100, 200, 50],
            fast_period_range: [2, 2, 0],
            slow_period_range: [30, 30, 0]
        });
    });
});

test('MAAQ accuracy with expected values', () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.maaq;

    const result = wasm.maaq_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.fast_period,
        expected.defaultParams.slow_period
    );


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        0.01,
        "MAAQ last 5 values mismatch with expected"
    );


    const warmupEnd = expected.defaultParams.period - 1;
    for (let i = 0; i < warmupEnd; i++) {
        assert(isNaN(result[i]), `Expected NaN at warmup index ${i}`);
    }
});

test('MAAQ single value with period 1', () => {

    const data = new Float64Array([42.0]);


    assert.throws(() => {
        wasm.maaq_js(data, 1, 1, 1);
    });
});
