
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

test('SQWMA partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.sqwma_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('SQWMA accuracy', async () => {

    const close = new Float64Array(testData.close);


    const expectedLastFive = [
        59229.72287968442,
        59211.30867850099,
        59172.516765286,
        59167.73471400394,
        59067.97928994083,
    ];

    const result = wasm.sqwma_js(close, 14);

    assert.strictEqual(result.length, close.length);


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-8,
        "SQWMA last 5 values mismatch"
    );


    await compareWithRust('sqwma', result, 'close', { period: 14 });
});

test('SQWMA zero period', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.sqwma_js(inputData, 0);
    }, /Invalid period/);
});

test('SQWMA period exceeds length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.sqwma_js(dataSmall, 10);
    }, /Invalid period/);
});

test('SQWMA very small dataset', () => {

    const singlePoint = new Float64Array([42.0]);

    assert.throws(() => {
        wasm.sqwma_js(singlePoint, 9);
    }, /Invalid period/);
});

test('SQWMA empty input', () => {

    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.sqwma_js(empty, 14);
    }, /Input data slice is empty/);
});

test('SQWMA NaN handling', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.sqwma_js(close, 14);
    assert.strictEqual(result.length, close.length);


    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }



    for (let i = 0; i < 15; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} in warmup period`);
    }
});

test('SQWMA batch single period', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.sqwma_batch_js(close, 14, 14, 0);


    assert.strictEqual(result.length, close.length);


    const expectedLastFive = [
        59229.72287968442,
        59211.30867850099,
        59172.516765286,
        59167.73471400394,
        59067.97928994083,
    ];


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-8,
        "SQWMA batch default row mismatch"
    );
});

test('SQWMA batch metadata', () => {

    const periods = wasm.sqwma_batch_metadata_js(10, 20, 2);


    const expectedPeriods = [10, 12, 14, 16, 18, 20];
    assert.deepStrictEqual(Array.from(periods), expectedPeriods);
});

test('SQWMA all NaN input', () => {

    const allNan = new Float64Array(100).fill(NaN);

    assert.throws(() => {
        wasm.sqwma_js(allNan, 14);
    }, /All values are NaN/);
});

test('SQWMA period less than 2', () => {

    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);

    assert.throws(() => {
        wasm.sqwma_js(data, 1);
    }, /Invalid period/);
});

test('SQWMA batch multiple periods', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.sqwma_batch_js(close, 10, 20, 2);


    assert.strictEqual(result.length, 6 * close.length);


    const periods = wasm.sqwma_batch_metadata_js(10, 20, 2);
    assert.strictEqual(periods.length, 6);


    for (let i = 0; i < 6; i++) {
        const period = periods[i];
        const startIdx = i * close.length;
        const rowData = result.slice(startIdx, startIdx + close.length);


        const warmupEnd = period + 1;
        for (let j = 0; j < warmupEnd; j++) {
            assert(isNaN(rowData[j]), `Expected NaN in warmup for period ${period} at index ${j}`);
        }


        if (rowData.length > warmupEnd + 10) {
            let hasValidValues = false;
            for (let j = warmupEnd; j < warmupEnd + 10; j++) {
                if (!isNaN(rowData[j])) {
                    hasValidValues = true;
                    break;
                }
            }
            assert(hasValidValues, `Expected valid values after warmup for period ${period}`);
        }
    }
});

test('SQWMA edge case period 2', () => {

    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);

    const result = wasm.sqwma_js(data, 2);
    assert.strictEqual(result.length, data.length);


    assertAllNaN(result.slice(0, 3));


    assertNoNaN(result.slice(3));
});

test('SQWMA batch with step 0', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.sqwma_batch_js(close, 15, 15, 0);
    assert.strictEqual(result.length, close.length);


    const periods = wasm.sqwma_batch_metadata_js(15, 15, 0);
    assert.deepStrictEqual(Array.from(periods), [15]);
});

test('SQWMA consistency check', () => {

    const close = new Float64Array(testData.close);
    const period = 14;

    const single = wasm.sqwma_js(close, period);
    const batch = wasm.sqwma_batch_js(close, period, period, 0);

    assert.strictEqual(single.length, batch.length);



    for (let i = 0; i < single.length; i++) {
        if (isNaN(single[i]) && isNaN(batch[i])) {
            continue;
        }




        assertClose(single[i], batch[i], 2e-9, 2e-9,
            `SQWMA single vs batch mismatch at index ${i}`);
    }
});

test('SQWMA with leading NaNs', () => {


    const data = new Float64Array(15);
    for (let i = 0; i < 5; i++) {
        data[i] = NaN;
    }
    for (let i = 5; i < 15; i++) {
        data[i] = i - 4;
    }

    const result = wasm.sqwma_js(data, 3);
    assert.strictEqual(result.length, data.length);


    for (let i = 0; i < 9; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} in warmup period including leading NaNs`);
    }

    for (let i = 9; i < result.length; i++) {
        assert(!isNaN(result[i]), `Expected valid value at index ${i} after warmup`);
    }
});

test('SQWMA unified batch API', () => {

    const close = new Float64Array(testData.close.slice(0, 100));


    const batchResult = wasm.sqwma_batch(close, {
        period_range: [10, 20, 5]
    });


    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);


    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);

        const singleResult = wasm.sqwma_js(close, periods[i]);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Period ${periods[i]} mismatch in unified batch API`
        );
    }
});

test('SQWMA batch metadata from result', () => {

    const close = new Float64Array(30);
    close.fill(100);

    const result = wasm.sqwma_batch(close, {
        period_range: [10, 20, 10]
    });


    assert.strictEqual(result.combos.length, 2);


    assert.strictEqual(result.combos[0].period, 10);


    assert.strictEqual(result.combos[1].period, 20);
});

test('SQWMA improved warmup assertions', () => {

    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }


    const testCases = [
        { period: 5, warmupEnd: 6 },
        { period: 10, warmupEnd: 11 },
        { period: 15, warmupEnd: 16 },
    ];

    for (const { period, warmupEnd } of testCases) {
        const result = wasm.sqwma_js(data, period);


        for (let i = 0; i < warmupEnd; i++) {
            assert(isNaN(result[i]),
                `Period ${period}: Expected NaN at index ${i} (warmup ends at ${warmupEnd})`);
        }


        for (let i = warmupEnd; i < Math.min(warmupEnd + 5, result.length); i++) {
            assert(!isNaN(result[i]),
                `Period ${period}: Expected valid value at index ${i} (warmup ended at ${warmupEnd})`);
        }
    }
});

test('SQWMA batch with NaN injection', () => {

    const data = new Float64Array(30);
    for (let i = 0; i < 30; i++) {
        data[i] = i + 1;
    }

    data[5] = NaN;
    data[6] = NaN;

    const result = wasm.sqwma_batch_js(data, 5, 10, 5);


    const periods = wasm.sqwma_batch_metadata_js(5, 10, 5);
    assert.strictEqual(periods.length, 2);




    for (let p = 0; p < 2; p++) {
        const period = periods[p];
        const rowStart = p * data.length;
        const rowData = result.slice(rowStart, rowStart + data.length);




        const theoreticalEarliest = 7 + (period - 2);


        for (let i = 0; i < theoreticalEarliest; i++) {
            assert(
                isNaN(rowData[i]),
                `Period ${period}: Expected NaN at index ${i} (pre-window-clean)`
            );
        }


        let firstFinite = -1;
        for (let i = theoreticalEarliest; i < rowData.length; i++) {
            if (!isNaN(rowData[i])) { firstFinite = i; break; }
        }
        if (firstFinite !== -1) {

            for (let i = firstFinite; i < Math.min(firstFinite + 3, rowData.length); i++) {
                assert(!isNaN(rowData[i]),
                    `Period ${period}: Expected valid value at index ${i} (first finite at ${firstFinite})`);
            }
        }
    }
});
