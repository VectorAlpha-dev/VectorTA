
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

test('DevStop partial params', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);

    const result = wasm.devstop(high, low, 20, 0.0, 0, 'long', 'sma');
    assert.strictEqual(result.length, high.length);


    const resultCustom = wasm.devstop(high, low, 20, 1.0, 2, 'short', 'ema');
    assert.strictEqual(resultCustom.length, high.length);
});

test('DevStop accuracy', async () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const expected = EXPECTED_OUTPUTS.devstop;

    const result = wasm.devstop(
        high, low,
        expected.defaultParams.period,
        expected.defaultParams.mult,
        expected.defaultParams.devtype,
        expected.defaultParams.direction,
        expected.defaultParams.maType
    );

    assert.strictEqual(result.length, high.length);


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "DevStop last 5 values mismatch"
    );


    const params = {
        high: 'high',
        low: 'low',
        ...expected.defaultParams
    };
    await compareWithRust('devstop', result, null, params);
});

test('DevStop default candles', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);

    const result = wasm.devstop(high, low, 20, 0.0, 0, 'long', 'sma');
    assert.strictEqual(result.length, high.length);
});

test('DevStop zero period', () => {

    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);

    assert.throws(() => {
        wasm.devstop(high, low, 0, 1.0, 0, 'long', 'sma');
    }, /Invalid period/);
});

test('DevStop period exceeds length', () => {

    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);

    assert.throws(() => {
        wasm.devstop(high, low, 10, 1.0, 0, 'long', 'sma');
    }, /Invalid period/);
});

test('DevStop very small dataset', () => {

    const high = new Float64Array([100.0]);
    const low = new Float64Array([90.0]);

    assert.throws(() => {
        wasm.devstop(high, low, 20, 2.0, 0, 'long', 'sma');
    }, /Invalid period|Not enough valid data/);
});

test('DevStop NaN handling', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);

    const result = wasm.devstop(high, low, 20, 0.0, 0, 'long', 'sma');
    assert.strictEqual(result.length, high.length);


    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }



    const expectedWarmup = 39;
    if (result.length > expectedWarmup) {

        let hasNaN = false;
        for (let i = 0; i < expectedWarmup; i++) {
            if (isNaN(result[i])) {
                hasNaN = true;
                break;
            }
        }
        assert(hasNaN, "Expected NaN in warmup period");
    }
});

test('DevStop all NaN input', () => {

    const allNaNHigh = new Float64Array(100);
    const allNaNLow = new Float64Array(100);
    allNaNHigh.fill(NaN);
    allNaNLow.fill(NaN);

    assert.throws(() => {
        wasm.devstop(allNaNHigh, allNaNLow, 20, 0.0, 0, 'long', 'sma');
    }, /All values are NaN/);
});

test('DevStop mismatched lengths', () => {

    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]);

    assert.throws(() => {
        wasm.devstop(high, low, 2, 0.0, 0, 'long', 'sma');
    }, /length mismatch/);
});

test('DevStop batch single parameter set', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);


    const batchResult = wasm.devstop_batch(high, low, {
        period_range: [20, 20, 0],
        mult_range: [0.0, 0.0, 0],
        devtype_range: [0, 0, 0]
    });


    const singleResult = wasm.devstop(high, low, 20, 0.0, 0, 'long', 'sma');

    assert.strictEqual(batchResult.values.length, singleResult.length);


    assertArrayClose(batchResult.values, singleResult, 1e-9, "Batch vs single mismatch");
});

test('DevStop batch multiple periods', () => {

    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));


    const batchResult = wasm.devstop_batch(high, low, {
        period_range: [10, 20, 5],
        mult_range: [0.0, 0.0, 0],
        devtype_range: [0, 0, 0]
    });


    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);


    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);

        const singleResult = wasm.devstop(high, low, periods[i], 0.0, 0, 'long', 'sma');

        assertArrayClose(
            Array.from(rowData),
            Array.from(singleResult),
            1e-9,
            `Batch row ${i} (period=${periods[i]}) mismatch`
        );
    }
});

test('DevStop direction types', () => {

    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));


    const resultLong = wasm.devstop(high, low, 20, 1.0, 0, 'long', 'sma');
    assert.strictEqual(resultLong.length, high.length);


    const resultShort = wasm.devstop(high, low, 20, 1.0, 0, 'short', 'sma');
    assert.strictEqual(resultShort.length, high.length);


    let isDifferent = false;
    for (let i = 0; i < resultLong.length; i++) {
        if (!isNaN(resultLong[i]) && !isNaN(resultShort[i])) {
            if (Math.abs(resultLong[i] - resultShort[i]) > 1e-10) {
                isDifferent = true;
                break;
            }
        }
    }
    assert(isDifferent, "Long and short directions should produce different results");
});

test('DevStop MA types', () => {

    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));

    const maTypes = ['sma', 'ema', 'wma', 'hma', 'dema'];
    const results = {};

    for (const maType of maTypes) {
        results[maType] = wasm.devstop(high, low, 20, 1.0, 0, 'long', maType);
        assert.strictEqual(results[maType].length, high.length, `MA type ${maType} length mismatch`);
    }


    for (let i = 0; i < maTypes.length - 1; i++) {
        const ma1 = maTypes[i];
        for (let j = i + 1; j < maTypes.length; j++) {
            const ma2 = maTypes[j];
            let isDifferent = false;
            for (let k = 0; k < results[ma1].length; k++) {
                if (!isNaN(results[ma1][k]) && !isNaN(results[ma2][k])) {
                    if (Math.abs(results[ma1][k] - results[ma2][k]) > 1e-10) {
                        isDifferent = true;
                        break;
                    }
                }
            }
            assert(isDifferent, `${ma1} and ${ma2} should produce different results`);
        }
    }
});

test('DevStop devtype variations', () => {

    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));


    const results = {};
    for (const devtype of [0, 1, 2]) {
        results[devtype] = wasm.devstop(high, low, 20, 1.0, devtype, 'long', 'sma');
        assert.strictEqual(results[devtype].length, high.length, `Devtype ${devtype} length mismatch`);
    }


    for (const [dt1, dt2] of [[0, 1], [0, 2], [1, 2]]) {
        let isDifferent = false;
        for (let i = 0; i < results[dt1].length; i++) {
            if (!isNaN(results[dt1][i]) && !isNaN(results[dt2][i])) {
                if (Math.abs(results[dt1][i] - results[dt2][i]) > 1e-10) {
                    isDifferent = true;
                    break;
                }
            }
        }
        assert(isDifferent, `Devtype ${dt1} and ${dt2} should produce different results`);
    }
});

test('DevStop batch parameter sweep', () => {


    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));

    const batchResult = wasm.devstop_batch(high, low, {
        period_range: [10, 30, 10],
        mult_range: [0.0, 1.0, 0.5],
        devtype_range: [0, 2, 1]
    });


    const expectedCombos = 3 * 3 * 3;
    assert.strictEqual(batchResult.rows, expectedCombos);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.values.length, expectedCombos * 100);
});
