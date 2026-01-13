
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

test('CWMA partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.cwma_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('CWMA accuracy', async () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.cwma;

    const result = wasm.cwma_js(close, expected.defaultParams.period);

    assert.strictEqual(result.length, close.length);


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1.0,
        "CWMA last 5 values mismatch"
    );


    await compareWithRust('cwma', result, 'close', expected.defaultParams);
});

test('CWMA default candles', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.cwma_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('CWMA zero period', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.cwma_js(inputData, 0);
    }, /Invalid period/);
});

test('CWMA period exceeds length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.cwma_js(dataSmall, 10);
    }, /Invalid period/);
});

test('CWMA very small dataset', () => {

    const singlePoint = new Float64Array([42.0]);

    assert.throws(() => {
        wasm.cwma_js(singlePoint, 9);
    }, /Invalid period|Not enough valid data/);
});

test('CWMA empty input', () => {

    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.cwma_js(empty, 9);
    }, /Input data slice is empty/);
});

test('CWMA reinput', () => {

    const close = new Float64Array(testData.close);


    const firstResult = wasm.cwma_js(close, 80);
    assert.strictEqual(firstResult.length, close.length);


    const secondResult = wasm.cwma_js(firstResult, 60);
    assert.strictEqual(secondResult.length, firstResult.length);


    if (secondResult.length > 240) {
        for (let i = 240; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('CWMA NaN handling', () => {

    const close = new Float64Array(testData.close);
    const period = 9;

    const result = wasm.cwma_js(close, period);
    assert.strictEqual(result.length, close.length);


    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }


    const warmupLength = period - 1;
    assertAllNaN(result.slice(0, warmupLength), `Expected NaN in warmup period (first ${warmupLength} values)`);
    assert(!isNaN(result[warmupLength]), `Expected valid value at index ${warmupLength}`);
});

test('CWMA all NaN input', () => {

    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);

    assert.throws(() => {
        wasm.cwma_js(allNaN, 9);
    }, /All values are NaN/);
});

test('CWMA batch single parameter set', () => {

    const close = new Float64Array(testData.close);


    const batchResult = wasm.cwma_batch_js(
        close,
        14, 14, 0
    );


    const singleResult = wasm.cwma_js(close, 14);

    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 10.0, "Batch vs single mismatch");
});

test('CWMA batch multiple periods', () => {

    const close = new Float64Array(testData.close.slice(0, 100));


    const batchResult = wasm.cwma_batch_js(
        close,
        10, 25, 5
    );


    assert.strictEqual(batchResult.length, 4 * 100);


    const periods = [10, 15, 20, 25];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);

        const singleResult = wasm.cwma_js(close, periods[i]);
        assertArrayClose(
            rowData,
            singleResult,
            2.0,
            `Period ${periods[i]} mismatch`
        );
    }
});

test('CWMA batch metadata', () => {

    const metadata = wasm.cwma_batch_metadata_js(
        10, 30, 5
    );


    assert.strictEqual(metadata.length, 5);


    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 15);
    assert.strictEqual(metadata[2], 20);
    assert.strictEqual(metadata[3], 25);
    assert.strictEqual(metadata[4], 30);
});

test('CWMA batch warmup validation', () => {

    const close = new Float64Array(testData.close.slice(0, 50));

    const batchResult = wasm.cwma_batch_js(
        close,
        5, 15, 5
    );

    const metadata = wasm.cwma_batch_metadata_js(5, 15, 5);
    const numCombos = metadata.length;
    assert.strictEqual(numCombos, 3);


    for (let combo = 0; combo < numCombos; combo++) {
        const period = metadata[combo];
        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);


        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }


        for (let i = period - 1; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('CWMA batch edge cases', () => {

    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);


    const singleBatch = wasm.cwma_batch_js(
        close,
        7, 7, 1
    );

    assert.strictEqual(singleBatch.length, 15);


    const stepZero = wasm.cwma_batch_js(
        close,
        5, 5, 0
    );
    assert.strictEqual(stepZero.length, 15);


    const largeBatch = wasm.cwma_batch_js(
        close,
        5, 7, 10
    );


    assert.strictEqual(largeBatch.length, 15);


    assert.throws(() => {
        wasm.cwma_batch_js(
            new Float64Array([]),
            9, 9, 0
        );
    }, /Input data slice is empty/);
});

test('CWMA batch performance test', () => {

    const close = new Float64Array(testData.close.slice(0, 200));


    const startBatch = Date.now();
    const batchResult = wasm.cwma_batch_js(
        close,
        10, 50, 2
    );
    const batchTime = Date.now() - startBatch;


    const startSingle = Date.now();
    const singleResults = [];
    for (let period = 10; period <= 50; period += 2) {
        singleResults.push(...wasm.cwma_js(close, period));
    }
    const singleTime = Date.now() - startSingle;


    assert.strictEqual(batchResult.length, singleResults.length);


    console.log(`  CWMA Batch time: ${batchTime}ms, Single calls time: ${singleTime}ms`);
});

test('CWMA period one invalid', () => {

    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);

    assert.throws(() => {
        wasm.cwma_js(data, 1);
    }, /Invalid period/, 'CWMA should fail with period=1');
});

test('CWMA leading NaN handling', () => {

    const close = new Float64Array(testData.close);


    for (let i = 0; i < 10; i++) {
        close[i] = NaN;
    }

    const result = wasm.cwma_js(close, 14);
    assert.strictEqual(result.length, close.length);


    const expectedNans = 10 + 13;
    for (let i = 0; i < expectedNans; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    assert(!isNaN(result[expectedNans]), `Expected valid value at index ${expectedNans}`);
});

test('CWMA warmup validation', () => {

    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1;
    }

    const testPeriods = [2, 3, 5, 10, 14, 20, 30];
    for (const period of testPeriods) {
        const result = wasm.cwma_js(data, period);


        let nanCount = 0;
        for (const val of result) {
            if (isNaN(val)) {
                nanCount++;
            } else {
                break;
            }
        }

        const expectedWarmup = period - 1;
        assert.strictEqual(
            nanCount,
            expectedWarmup,
            `Period ${period}: Expected ${expectedWarmup} NaN values, got ${nanCount}`
        );


        assert(!isNaN(result[expectedWarmup]),
            `Period ${period}: Expected valid value at index ${expectedWarmup}`);
        if (expectedWarmup > 0) {
            assert(isNaN(result[expectedWarmup - 1]),
                `Period ${period}: Expected NaN at index ${expectedWarmup - 1}`);
        }
    }
});

test('CWMA unified batch API', () => {

    const close = new Float64Array(testData.close.slice(0, 100));

    const config = {
        period_range: [10, 20, 5]
    };

    const result = wasm.cwma_batch(close, config);

    assert(result.values, 'Result should have values');
    assert(result.combos, 'Result should have combos');
    assert.strictEqual(result.rows, 3, 'Should have 3 rows');
    assert.strictEqual(result.cols, 100, 'Should have 100 columns');
    assert.strictEqual(result.values.length, 300, 'Should have 300 total values');
    assert.strictEqual(result.combos.length, 3, 'Should have 3 combinations');


    const periods = result.combos.map(c => c.period);
    assert.deepStrictEqual(periods, [10, 15, 20], 'Periods should match');
});

test('CWMA unified batch API validation', () => {

    const close = new Float64Array([1, 2, 3, 4, 5]);


    assert.throws(() => {
        wasm.cwma_batch(close, {});
    }, /Invalid config/, 'Should error on missing period_range');

    assert.throws(() => {
        wasm.cwma_batch(close, { period_range: [10] });
    }, /Invalid config/, 'Should error on invalid period_range');

    assert.throws(() => {
        wasm.cwma_batch(close, 'invalid');
    }, /Invalid config/, 'Should error on non-object config');
});





test('CWMA memory allocation and deallocation', () => {

    const len = 100;


    const ptr = wasm.cwma_alloc(len);
    assert(ptr !== 0, 'Memory allocation should return non-zero pointer');


    assert.doesNotThrow(() => {
        wasm.cwma_free(ptr, len);
    }, 'Memory deallocation should not throw');


});

test('CWMA fast API basic functionality', () => {

    const close = new Float64Array(testData.close.slice(0, 1000));
    const period = 14;


    const safeResult = wasm.cwma_js(close, period);


    const inPtr = wasm.cwma_alloc(close.length);
    const outPtr = wasm.cwma_alloc(close.length);

    try {

        const memView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, close.length);
        memView.set(close);


        wasm.cwma_into(inPtr, outPtr, close.length, period);


        const fastResult = new Float64Array(wasm.__wasm.memory.buffer, outPtr, close.length);


        assertArrayClose(
            Array.from(fastResult),
            safeResult,
            1e-10,
            "Fast API should produce same results as safe API"
        );
    } finally {

        wasm.cwma_free(inPtr, close.length);
        wasm.cwma_free(outPtr, close.length);
    }
});

test('CWMA fast API with aliasing (in-place)', () => {

    const close = new Float64Array(testData.close.slice(0, 500));
    const period = 20;


    const expected = wasm.cwma_js(close, period);


    const ptr = wasm.cwma_alloc(close.length);

    try {

        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, close.length);
        memView.set(close);


        wasm.cwma_into(ptr, ptr, close.length, period);


        const result = new Float64Array(wasm.__wasm.memory.buffer, ptr, close.length);


        assertArrayClose(
            Array.from(result),
            expected,
            1e-10,
            "In-place computation should produce correct results"
        );
    } finally {
        wasm.cwma_free(ptr, close.length);
    }
});

test('CWMA fast API error handling', () => {
    const len = 100;
    const inPtr = wasm.cwma_alloc(len);
    const outPtr = wasm.cwma_alloc(len);

    try {

        assert.throws(() => {
            wasm.cwma_into(0, outPtr, len, 14);
        }, /null pointer/, 'Should error on null input pointer');

        assert.throws(() => {
            wasm.cwma_into(inPtr, 0, len, 14);
        }, /null pointer/, 'Should error on null output pointer');


        assert.throws(() => {
            wasm.cwma_into(inPtr, outPtr, len, 0);
        }, /Invalid period/, 'Should error on zero period');

        assert.throws(() => {
            wasm.cwma_into(inPtr, outPtr, len, 200);
        }, /Invalid period/, 'Should error when period exceeds length');
    } finally {
        wasm.cwma_free(inPtr, len);
        wasm.cwma_free(outPtr, len);
    }
});

test('CWMA batch into performance', () => {

    const close = new Float64Array(testData.close.slice(0, 200));
    const periods = [10, 15, 20, 25, 30];


    const inPtr = wasm.cwma_alloc(close.length);
    const outPtr = wasm.cwma_alloc(close.length * periods.length);

    try {

        const memView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, close.length);
        memView.set(close);


        const startBatch = Date.now();
        const rows = wasm.cwma_batch_into(
            inPtr,
            outPtr,
            close.length,
            10,
            30,
            5
        );
        const batchTime = Date.now() - startBatch;

        assert.strictEqual(rows, periods.length, 'Should return correct number of rows');


        const startSingle = Date.now();
        for (let i = 0; i < periods.length; i++) {
            const singleOutPtr = wasm.cwma_alloc(close.length);
            try {
                wasm.cwma_into(inPtr, singleOutPtr, close.length, periods[i]);
            } finally {
                wasm.cwma_free(singleOutPtr, close.length);
            }
        }
        const singleTime = Date.now() - startSingle;

        console.log(`  CWMA Batch into time: ${batchTime}ms, Multiple into calls: ${singleTime}ms`);
    } finally {
        wasm.cwma_free(inPtr, close.length);
        wasm.cwma_free(outPtr, close.length * periods.length);
    }
});

test('CWMA batch into validation', () => {

    const close = new Float64Array(testData.close.slice(0, 50));
    const periods = [5, 10, 15];


    const inPtr = wasm.cwma_alloc(close.length);
    const outPtr = wasm.cwma_alloc(close.length * periods.length);

    try {

        const memView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, close.length);
        memView.set(close);


        const rows = wasm.cwma_batch_into(
            inPtr,
            outPtr,
            close.length,
            5,
            15,
            5
        );

        assert.strictEqual(rows, 3, 'Should return 3 rows');


        const batchResult = new Float64Array(wasm.__wasm.memory.buffer, outPtr, close.length * rows);
        const regularBatch = wasm.cwma_batch_js(close, 5, 15, 5);

        assertArrayClose(
            Array.from(batchResult),
            regularBatch,
            2.0,
            'batch_into should match batch_js'
        );
    } finally {
        wasm.cwma_free(inPtr, close.length);
        wasm.cwma_free(outPtr, close.length * periods.length);
    }
});

test.after(() => {
    console.log('CWMA WASM tests completed');
});