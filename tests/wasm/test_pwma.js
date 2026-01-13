
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

test('PWMA partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.pwma_js(close, 5);
    assert(result instanceof Float64Array);
    assert.strictEqual(result.length, close.length);
});

test('PWMA accuracy', async () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.pwma;

    const result = wasm.pwma_js(close, expected.defaultParams.period);

    assert.strictEqual(result.length, close.length);


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-3,
        "PWMA last 5 values mismatch"
    );


    await compareWithRust('pwma', result, 'close', expected.defaultParams);
});

test('PWMA zero period', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.pwma_js(inputData, 0);
    });
});

test('PWMA period exceeds length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.pwma_js(dataSmall, 10);
    });
});

test('PWMA very small dataset', () => {

    const singlePoint = new Float64Array([42.0]);

    assert.throws(() => {
        wasm.pwma_js(singlePoint, 5);
    });
});

test('PWMA empty input', () => {

    const dataEmpty = new Float64Array([]);

    assert.throws(() => {
        wasm.pwma_js(dataEmpty, 5);
    });
});

test('PWMA reinput', () => {

    const close = new Float64Array(testData.close);


    const firstResult = wasm.pwma_js(close, 5);


    const secondResult = wasm.pwma_js(firstResult, 3);

    assert.strictEqual(secondResult.length, firstResult.length);


    const warmup = 240 + (5 - 1) + (3 - 1);
    for (let i = warmup; i < secondResult.length; i++) {
        assert(isFinite(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('PWMA NaN handling', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.pwma_js(close, 5);

    assert.strictEqual(result.length, close.length);


    for (let i = 245; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN found at index ${i}`);
    }
});

test('PWMA batch', () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.pwma;


    const batch_result = wasm.pwma_batch_js(
        close,
        expected.batchRange.start,
        expected.batchRange.end,
        expected.batchRange.step
    );


    const rows_cols = wasm.pwma_batch_rows_cols_js(
        expected.batchRange.start,
        expected.batchRange.end,
        expected.batchRange.step,
        close.length
    );
    const rows = rows_cols[0];
    const cols = rows_cols[1];

    assert(batch_result instanceof Float64Array);
    assert.strictEqual(rows, expected.batchPeriods.length);
    assert.strictEqual(cols, close.length);
    assert.strictEqual(batch_result.length, rows * cols);


    const individual_result = wasm.pwma_js(close, expected.batchPeriods[0]);
    const batch_first = batch_result.slice(0, close.length);


    const warmup = 240 + expected.batchPeriods[0] - 1;
    for (let i = warmup; i < close.length; i++) {
        assertClose(batch_first[i], individual_result[i], 1e-9, `Batch mismatch at ${i}`);
    }
});

test('PWMA different periods', () => {

    const close = new Float64Array(testData.close);


    const testPeriods = [2, 5, 10, 20];

    for (const period of testPeriods) {
        const result = wasm.pwma_js(close, period);
        assert.strictEqual(result.length, close.length);


        const warmup = 240 + period - 1;
        if (warmup < result.length) {
            for (let i = warmup; i < result.length; i++) {
                assert(isFinite(result[i]), `NaN at index ${i} for period=${period}`);
            }
        }
    }
});

test('PWMA batch performance', () => {

    const close = new Float64Array(testData.close.slice(0, 1000));


    const startBatch = performance.now();
    const batchResult = wasm.pwma_batch_js(
        close,
        5, 30, 5
    );
    const batchTime = performance.now() - startBatch;


    const metadata = wasm.pwma_batch_metadata_js(5, 30, 5);

    const startSingle = performance.now();
    const singleResults = [];

    for (const period of metadata) {
        const result = wasm.pwma_js(close, period);
        singleResults.push(...result);
    }
    const singleTime = performance.now() - startSingle;


    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);


    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});

test('PWMA edge cases', () => {

    const expected = EXPECTED_OUTPUTS.pwma;
    const period = expected.defaultParams.period;


    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1;
    }
    const result = wasm.pwma_js(data, period);
    assert.strictEqual(result.length, data.length);


    for (let i = period - 1; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN at index ${i}`);
    }


    const constantVal = expected.constantValue;
    const constantData = new Float64Array(100).fill(constantVal);
    const constantResult = wasm.pwma_js(constantData, period);

    assert.strictEqual(constantResult.length, constantData.length);


    for (let i = period - 1; i < constantResult.length; i++) {
        assert(isFinite(constantResult[i]), `NaN at index ${i} for constant data`);
        assertClose(constantResult[i], constantVal, 1e-9, `Constant value mismatch at ${i}`);
    }
});

test('PWMA batch metadata', () => {

    const expected = EXPECTED_OUTPUTS.pwma;
    const metadata = wasm.pwma_batch_metadata_js(
        expected.batchRange.start,
        expected.batchRange.end,
        expected.batchRange.step
    );


    assert.strictEqual(metadata.length, expected.batchPeriods.length);
    for (let i = 0; i < metadata.length; i++) {
        assert.strictEqual(metadata[i], expected.batchPeriods[i]);
    }
});

test('PWMA consistency across calls', () => {

    const close = new Float64Array(testData.close.slice(0, 100));

    const result1 = wasm.pwma_js(close, 5);
    const result2 = wasm.pwma_js(close, 5);

    assertArrayClose(result1, result2, 1e-15, "PWMA results not consistent");
});

test('PWMA step precision', () => {

    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }

    const batch_result = wasm.pwma_batch_js(
        data,
        5, 15, 5
    );


    const rows_cols = wasm.pwma_batch_rows_cols_js(5, 15, 5, data.length);
    const rows = rows_cols[0];


    assert.strictEqual(rows, 3);
    assert.strictEqual(batch_result.length, 3 * data.length);


    const metadata = wasm.pwma_batch_metadata_js(5, 15, 5);
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 5);
    assert.strictEqual(metadata[1], 10);
    assert.strictEqual(metadata[2], 15);
});

test('PWMA warmup behavior', () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.pwma;
    const period = expected.defaultParams.period;

    const result = wasm.pwma_js(close, period);


    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    const warmup = firstValid + period - 1;


    for (let i = 0; i < warmup; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }


    for (let i = warmup; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i} after warmup`);
    }
});

test('PWMA oscillating data', () => {

    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = (i % 2 === 0) ? 10.0 : 20.0;
    }

    const result = wasm.pwma_js(data, 5);
    assert.strictEqual(result.length, data.length);


    for (let i = 4; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

test('PWMA small step size', () => {

    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }

    const batch_result = wasm.pwma_batch_js(
        data,
        5, 7, 1
    );

    const rows_cols = wasm.pwma_batch_rows_cols_js(5, 7, 1, data.length);
    const rows = rows_cols[0];

    assert.strictEqual(rows, 3);
    assert.strictEqual(batch_result.length, 3 * data.length);
});

test('PWMA formula verification', () => {

    const expected = EXPECTED_OUTPUTS.pwma.formulaTest;
    const data = new Float64Array(expected.data);

    const result = wasm.pwma_js(data, expected.period);


    assert.strictEqual(result.length, data.length);


    for (let i = 0; i < expected.expected.length; i++) {
        if (isNaN(expected.expected[i])) {
            assert(isNaN(result[i]), `Expected NaN at index ${i}`);
        } else {
            assertClose(result[i], expected.expected[i], 1e-9,
                       `PWMA formula mismatch at index ${i}`);
        }
    }
});

test('PWMA all NaN input', () => {

    const allNaN = new Float64Array(100).fill(NaN);

    assert.throws(() => {
        wasm.pwma_js(allNaN, 5);
    }, /All values are NaN/);
});

test('PWMA batch error conditions', () => {

    const data = new Float64Array([1, 2, 3, 4, 5]);


    assert.throws(() => {
        wasm.pwma_batch_js(data, 10, 20, 5);
    });


    const empty = new Float64Array([]);
    assert.throws(() => {
        wasm.pwma_batch_js(empty, 5, 10, 5);
    });
});

test('PWMA pascal weights verification', () => {

    const expected = EXPECTED_OUTPUTS.pwma;
    const constantVal = expected.constantValue;
    const data = new Float64Array(10).fill(constantVal);


    for (const period of [2, 3, 4, 5]) {
        const result = wasm.pwma_js(data, period);


        for (let i = period - 1; i < result.length; i++) {
            assertClose(result[i], constantVal, 1e-9,
                `PWMA constant test failed at index ${i} for period=${period}`);
        }
    }
});


test('PWMA zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;


    const ptr = wasm.pwma_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');


    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );


    memView.set(data);


    try {
        wasm.pwma_into(ptr, ptr, data.length, period);


        const regularResult = wasm.pwma_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue;
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {

        wasm.pwma_free(ptr, data.length);
    }
});

test('PWMA zero-copy with large dataset', () => {
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }

    const ptr = wasm.pwma_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');

    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);

        wasm.pwma_into(ptr, ptr, size, 5);


        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);


        for (let i = 0; i < 4; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }


        for (let i = 4; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.pwma_free(ptr, size);
    }
});

test('PWMA zero-copy error handling', () => {

    assert.throws(() => {
        wasm.pwma_into(0, 0, 10, 5);
    }, /null pointer|invalid memory/i);


    const ptr = wasm.pwma_alloc(10);
    try {

        assert.throws(() => {
            wasm.pwma_into(ptr, ptr, 10, 0);
        }, /Invalid period/);


        assert.throws(() => {
            wasm.pwma_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.pwma_free(ptr, 10);
    }
});

test('PWMA zero-copy memory management', () => {

    const sizes = [100, 1000, 5000];

    for (const size of sizes) {
        const ptr = wasm.pwma_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);


        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }


        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }


        wasm.pwma_free(ptr, size);
    }
});


test('PWMA SIMD128 consistency', () => {

    const testCases = [
        { size: 10, period: 3 },
        { size: 100, period: 5 },
        { size: 1000, period: 10 },
        { size: 5000, period: 20 }
    ];

    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }

        const result = wasm.pwma_js(data, testCase.period);


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
        assert(Math.abs(avgAfterWarmup) < 10, `Average value ${avgAfterWarmup} seems unreasonable`);
    }
});