
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

test('LinReg partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.linreg_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('LinReg accuracy', async () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.linreg;

    const result = wasm.linreg_js(close, expected.defaultParams.period);

    assert.strictEqual(result.length, close.length);


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        0.1,
        "LinReg last 5 values mismatch"
    );


    await compareWithRust('linreg', result, 'close', expected.defaultParams);
});

test('LinReg default candles', async () => {

    const close = new Float64Array(testData.close);

    const result = wasm.linreg_js(close, 14);
    assert.strictEqual(result.length, close.length);


    await compareWithRust('linreg', result, 'close', { period: 14 });
});

test('LinReg zero period', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.linreg_js(inputData, 0);
    }, /Invalid period/);
});

test('LinReg period exceeds length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.linreg_js(dataSmall, 10);
    }, /Invalid period|Not enough valid data/);
});

test('LinReg very small dataset', () => {

    const dataSingle = new Float64Array([42.0]);

    assert.throws(() => {
        wasm.linreg_js(dataSingle, 14);
    }, /Invalid period|Not enough valid data/);
});

test('LinReg empty input', () => {

    const dataEmpty = new Float64Array([]);

    assert.throws(() => {
        wasm.linreg_js(dataEmpty, 14);
    }, /no data provided|empty|all values are nan/i);
});

test('LinReg all NaN', () => {

    const data = new Float64Array([NaN, NaN, NaN, NaN, NaN]);

    assert.throws(() => {
        wasm.linreg_js(data, 3);
    }, /All values are NaN/);
});

test('LinReg reinput', () => {

    const close = new Float64Array(testData.close);


    const firstResult = wasm.linreg_js(close, 14);


    const secondResult = wasm.linreg_js(firstResult, 10);

    assert.strictEqual(secondResult.length, firstResult.length);





    for (let i = 24; i < secondResult.length; i++) {
        assert(isFinite(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('LinReg NaN handling', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.linreg_js(close, 14);

    assert.strictEqual(result.length, close.length);



    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('LinReg batch', () => {

    const close = new Float64Array(testData.close);


    const batchResult = wasm.linreg_batch(close, {
        period_range: [10, 40, 10]
    });


    assert(batchResult.values, 'Should have values array');
    assert(batchResult.combos, 'Should have combos array');
    assert(typeof batchResult.rows === 'number', 'Should have rows count');
    assert(typeof batchResult.cols === 'number', 'Should have cols count');


    assert.strictEqual(batchResult.rows, 4);
    assert.strictEqual(batchResult.cols, close.length);
    assert.strictEqual(batchResult.combos.length, 4);
    assert.strictEqual(batchResult.values.length, 4 * close.length);


    const expectedPeriods = [10, 20, 30, 40];
    for (let i = 0; i < 4; i++) {
        assert.strictEqual(batchResult.combos[i].period, expectedPeriods[i]);
    }


    for (let i = 0; i < 4; i++) {
        const period = expectedPeriods[i];
        const individual_result = wasm.linreg_js(close, period);


        const row_start = i * close.length;
        const row = batchResult.values.slice(row_start, row_start + close.length);

        assertArrayClose(row, individual_result, 1e-9, `Period ${period}`);
    }
});

test('LinReg different periods', () => {

    const close = new Float64Array(testData.close);


    for (const period of [5, 10, 20, 50]) {
        const result = wasm.linreg_js(close, period);
        assert.strictEqual(result.length, close.length);



        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} for period=${period}`);
        }


        for (let i = period - 1; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for period=${period}`);
        }
    }
});

test('LinReg batch performance', () => {

    const close = new Float64Array(testData.close.slice(0, 1000));


    const startBatch = performance.now();
    const batchResult = wasm.linreg_batch(close, {
        period_range: [10, 50, 10]
    });
    const batchTime = performance.now() - startBatch;

    const startSingle = performance.now();
    const singleResults = [];
    for (let period = 10; period <= 50; period += 10) {
        singleResults.push(...wasm.linreg_js(close, period));
    }
    const singleTime = performance.now() - startSingle;


    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);


    assertArrayClose(batchResult.values, singleResults, 1e-9, 'Batch vs single results');
});

test('LinReg edge cases', () => {



    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    const result = wasm.linreg_js(data, 3);
    assert.strictEqual(result.length, data.length);



    assertClose(result[result.length - 1], 10.0, 1e-9, "Perfect linear regression failed");


    const constantData = new Float64Array(20).fill(5.0);
    const constantResult = wasm.linreg_js(constantData, 5);
    assert.strictEqual(constantResult.length, constantData.length);


    for (let i = 5; i < constantResult.length; i++) {
        assertClose(constantResult[i], 5.0, 1e-9, `Constant prediction failed at index ${i}`);
    }
});

test('LinReg single value', () => {

    const data = new Float64Array([42.0]);


    const result = wasm.linreg_js(data, 1);
    assert.strictEqual(result.length, 1);
    assert(isNaN(result[0]));
});

test('LinReg two values', () => {

    const data = new Float64Array([1.0, 2.0]);



    const result = wasm.linreg_js(data, 1);
    assert.strictEqual(result.length, 2);
    assert(isNaN(result[0]));
    assert(isNaN(result[1]));


    const result2 = wasm.linreg_js(data, 2);
    assert.strictEqual(result2.length, 2);
    assert(isNaN(result2[0]));
    assertClose(result2[1], 2.0, 1e-9, "Two-value prediction failed");
});

test('LinReg batch metadata', () => {

    const close = new Float64Array(50);
    close.fill(100);

    const result = wasm.linreg_batch(close, {
        period_range: [15, 45, 15]
    });


    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.combos[0].period, 15);
    assert.strictEqual(result.combos[1].period, 30);
    assert.strictEqual(result.combos[2].period, 45);
});

test('LinReg warmup period calculation', () => {

    const close = new Float64Array(testData.close.slice(0, 50));

    const testCases = [
        { period: 5, expectedWarmup: 5 },
        { period: 10, expectedWarmup: 10 },
        { period: 20, expectedWarmup: 20 },
        { period: 30, expectedWarmup: 30 },
    ];

    for (const { period, expectedWarmup } of testCases) {
        const result = wasm.linreg_js(close, period);



        for (let i = 0; i < period - 1 && i < result.length; i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} for period=${period}`);
        }


        if (period - 1 < result.length) {
            assert(!isNaN(result[period - 1]),
                `Expected valid value at index ${period - 1} for period=${period}`);
        }
    }
});

test('LinReg consistency across calls', () => {

    const close = new Float64Array(testData.close.slice(0, 100));

    const result1 = wasm.linreg_js(close, 14);
    const result2 = wasm.linreg_js(close, 14);

    assertArrayClose(result1, result2, 1e-15, "LinReg results not consistent");
});

test('LinReg parameter step precision', () => {

    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

    const batchResult = wasm.linreg_batch(data, {
        period_range: [2, 4, 1]
    });


    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.combos.length, 3);
    assert.strictEqual(batchResult.values.length, 3 * data.length);


    assert.strictEqual(batchResult.combos[0].period, 2);
    assert.strictEqual(batchResult.combos[1].period, 3);
    assert.strictEqual(batchResult.combos[2].period, 4);
});

test('LinReg slope calculation', () => {


    const data = new Float64Array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    const result = wasm.linreg_js(data, 4);



    assertClose(result[result.length - 1], 16.0, 1e-9, "Slope calculation failed");
});

test('LinReg streaming simulation', () => {

    const close = new Float64Array(testData.close.slice(0, 100));
    const period = 14;


    const batchResult = wasm.linreg_js(close, period);



    assert.strictEqual(batchResult.length, close.length);



    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(batchResult[i]), `Expected NaN at index ${i}`);
    }


    for (let i = period - 1; i < close.length; i++) {
        assert(isFinite(batchResult[i]), `Expected finite value at index ${i}`);
    }
});

test('LinReg large period', () => {

    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.sin(i * 0.1) * 100 + 1000;
    }

    const result = wasm.linreg_js(data, 99);
    assert.strictEqual(result.length, data.length);


    for (let i = 0; i < 98; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }


    assert(isFinite(result[98]), "Expected finite value at index 98");
    assert(isFinite(result[99]), "Expected finite value at last index");
});



test('LinReg zero-copy basic', () => {

    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 3;


    const ptr = wasm.linreg_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');


    const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, data.length);


    memView.set(data);


    try {
        wasm.linreg_into(ptr, ptr, data.length, period);


        const regularResult = wasm.linreg_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue;
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {

        wasm.linreg_free(ptr, data.length);
    }
});

test('LinReg zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }

    const ptr = wasm.linreg_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');

    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);

        wasm.linreg_into(ptr, ptr, size, 14);


        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);


        for (let i = 0; i < 13; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }


        for (let i = 13; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.linreg_free(ptr, size);
    }
});

test('LinReg zero-copy error handling', () => {

    assert.throws(() => {
        wasm.linreg_into(0, 0, 10, 5);
    }, /null pointer/);


    const ptr = wasm.linreg_alloc(10);
    try {
        assert.throws(() => {
            wasm.linreg_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
    } finally {
        wasm.linreg_free(ptr, 10);
    }
});

test('LinReg zero-copy batch API', () => {
    const size = 100;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.1) * 10;
    }


    const periods = 3;
    const totalSize = periods * size;
    const inPtr = wasm.linreg_alloc(size);
    const outPtr = wasm.linreg_alloc(totalSize);

    try {
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, size);
        inView.set(data);

        const rows = wasm.linreg_batch_into(inPtr, outPtr, size, 10, 30, 10);
        assert.strictEqual(rows, 3, 'Expected 3 rows');

        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);


        const periodValues = [10, 20, 30];
        for (let i = 0; i < periods; i++) {
            const period = periodValues[i];
            const individual = wasm.linreg_js(data, period);
            const rowStart = i * size;
            const row = Array.from(outView.slice(rowStart, rowStart + size));

            assertArrayClose(row, individual, 1e-9, `Batch row ${i} mismatch`);
        }
    } finally {
        wasm.linreg_free(inPtr, size);
        wasm.linreg_free(outPtr, totalSize);
    }
});



test('LinReg SIMD consistency', () => {


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

        const result = wasm.linreg_js(data, testCase.period);


        assert.strictEqual(result.length, data.length);


        for (let i = 0; i < testCase.period - 1; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }


        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.period - 1; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }


        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(Math.abs(avgAfterWarmup) < 1000, `Average value ${avgAfterWarmup} seems unreasonable`);
    }
});



test('LinReg batch metadata validation', () => {

    const close = new Float64Array(50);
    close.fill(100);

    const result = wasm.linreg_batch(close, {
        period_range: [10, 30, 10]
    });


    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);


    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[1].period, 20);
    assert.strictEqual(result.combos[2].period, 30);
});

test('LinReg batch warmup consistency', () => {

    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.random() * 100;
    }

    const result = wasm.linreg_batch(data, {
        period_range: [5, 15, 5]
    });

    const periods = [5, 10, 15];
    for (let row = 0; row < periods.length; row++) {
        const period = periods[row];
        const rowStart = row * 100;
        const rowData = result.values.slice(rowStart, rowStart + 100);


        const expectedWarmup = period - 1;
        for (let i = 0; i < expectedWarmup; i++) {
            assert(isNaN(rowData[i]), `Row ${row}: Expected NaN at index ${i}`);
        }


        for (let i = expectedWarmup; i < rowData.length; i++) {
            assert(!isNaN(rowData[i]), `Row ${row}: Unexpected NaN at index ${i}`);
        }
    }
});
