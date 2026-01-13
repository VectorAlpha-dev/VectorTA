
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

test('Chande partial params', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);

    const result = wasm.chande_js(high, low, close, 22, 3.0, 'long');
    assert.strictEqual(result.length, close.length);
});

test('Chande accuracy', async () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.chande;

    const result = wasm.chande_js(
        high, low, close,
        expected.defaultParams.period,
        expected.defaultParams.mult,
        expected.defaultParams.direction
    );

    assert.strictEqual(result.length, close.length);


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "Chande last 5 values mismatch"
    );


    const warmupPeriod = expected.warmupPeriod;
    for (let i = 0; i < warmupPeriod; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup period`);
    }

    assert(!isNaN(result[warmupPeriod]), `Expected valid value at index ${warmupPeriod} (after warmup)`);


    await compareWithRust('chande', result, 'candles', expected.defaultParams);
});

test('Chande zero period', () => {

    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    const close = new Float64Array([8.0, 18.0, 28.0]);

    assert.throws(() => {
        wasm.chande_js(high, low, close, 0, 3.0, 'long');
    }, /invalid period/i);
});

test('Chande period exceeds length', () => {

    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    const close = new Float64Array([8.0, 18.0, 28.0]);

    assert.throws(() => {
        wasm.chande_js(high, low, close, 10, 3.0, 'long');
    }, /invalid period/i);
});

test('Chande bad direction', () => {

    const high = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    const low = new Float64Array([5.0, 15.0, 25.0, 35.0, 45.0]);
    const close = new Float64Array([8.0, 18.0, 28.0, 38.0, 48.0]);

    assert.throws(() => {
        wasm.chande_js(high, low, close, 2, 3.0, 'bad');
    }, /invalid direction/i);
});

test('Chande empty input', () => {

    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.chande_js(empty, empty, empty, 22, 3.0, 'long');
    }, /input series are empty/i);
});

test('Chande mismatched lengths', () => {

    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]);
    const close = new Float64Array([8.0, 18.0, 28.0]);

    assert.throws(() => {
        wasm.chande_js(high, low, close, 2, 3.0, 'long');
    }, /length mismatch/);
});

test('Chande directions', () => {

    const high = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    const low = new Float64Array([5.0, 15.0, 25.0, 35.0, 45.0]);
    const close = new Float64Array([8.0, 18.0, 28.0, 38.0, 48.0]);


    const resultLong = wasm.chande_js(high, low, close, 3, 2.0, 'long');
    assert.strictEqual(resultLong.length, close.length);
    assert(isNaN(resultLong[0]));
    assert(isNaN(resultLong[1]));
    assert(!isNaN(resultLong[2]));


    const resultShort = wasm.chande_js(high, low, close, 3, 2.0, 'short');
    assert.strictEqual(resultShort.length, close.length);
    assert(isNaN(resultShort[0]));
    assert(isNaN(resultShort[1]));
    assert(!isNaN(resultShort[2]));


    assert.notStrictEqual(resultLong[2], resultShort[2]);
});

test('Chande batch single params', () => {

    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));


    const result = wasm.chande_batch_js(
        high, low, close,
        22, 22, 0,
        3.0, 3.0, 0.0,
        'long'
    );


    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, 100);
    assert.strictEqual(result.values.length, 100);


    const singleResult = wasm.chande_js(high, low, close, 22, 3.0, 'long');
    assertArrayClose(
        result.values,
        singleResult,
        1e-10,
        "Batch vs single mismatch"
    );


    assert.strictEqual(result.periods.length, 1);
    assert.strictEqual(result.periods[0], 22);
    assert.strictEqual(result.mults.length, 1);
    assert.strictEqual(result.mults[0], 3.0);
    assert.strictEqual(result.directions.length, 1);
    assert.strictEqual(result.directions[0], 'long');
});

test('Chande batch multiple params', () => {

    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));


    const result = wasm.chande_batch_js(
        high, low, close,
        10, 20, 10,
        2.0, 3.0, 0.5,
        'short'
    );


    assert.strictEqual(result.rows, 6);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.values.length, 6 * 50);


    assert.strictEqual(result.periods.length, 6);
    assert.strictEqual(result.mults.length, 6);
    assert.strictEqual(result.directions.length, 6);


    assert(result.directions.every(d => d === 'short'));


    const expectedParams = [
        [10, 2.0], [10, 2.5], [10, 3.0],
        [20, 2.0], [20, 2.5], [20, 3.0]
    ];

    for (let i = 0; i < expectedParams.length; i++) {
        const [period, mult] = expectedParams[i];
        const singleResult = wasm.chande_js(high, low, close, period, mult, 'short');
        const batchRow = result.values.slice(i * 50, (i + 1) * 50);

        assertArrayClose(
            batchRow,
            singleResult,
            1e-10,
            `Batch row ${i} (period=${period}, mult=${mult}) mismatch`
        );
    }
});

test('Chande zero-copy API', () => {

    const len = 100;
    const high = new Float64Array(testData.high.slice(0, len));
    const low = new Float64Array(testData.low.slice(0, len));
    const close = new Float64Array(testData.close.slice(0, len));


    const highPtr = wasm.chande_alloc(len);
    const lowPtr = wasm.chande_alloc(len);
    const closePtr = wasm.chande_alloc(len);
    const outPtr = wasm.chande_alloc(len);

    try {

        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeMem = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        highMem.set(high);
        lowMem.set(low);
        closeMem.set(close);


        wasm.chande_into(
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            len,
            22,
            3.0,
            'long'
        );


        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);


        const expected = wasm.chande_js(high, low, close, 22, 3.0, 'long');
        assertArrayClose(
            result,
            expected,
            1e-10,
            "Zero-copy API mismatch"
        );
    } finally {

        wasm.chande_free(highPtr, len);
        wasm.chande_free(lowPtr, len);
        wasm.chande_free(closePtr, len);
        wasm.chande_free(outPtr, len);
    }
});

test('Chande zero-copy API with aliasing', () => {

    const len = 100;
    const high = new Float64Array(testData.high.slice(0, len));
    const low = new Float64Array(testData.low.slice(0, len));
    const close = new Float64Array(testData.close.slice(0, len));


    const highPtr = wasm.chande_alloc(len);
    const lowPtr = wasm.chande_alloc(len);
    const closePtr = wasm.chande_alloc(len);

    try {

        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeMem = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        highMem.set(high);
        lowMem.set(low);
        closeMem.set(close);


        wasm.chande_into(
            highPtr,
            lowPtr,
            closePtr,
            closePtr,
            len,
            22,
            3.0,
            'long'
        );


        const result = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);


        const expected = wasm.chande_js(high, low, close, 22, 3.0, 'long');
        assertArrayClose(
            result,
            expected,
            1e-10,
            "Zero-copy API with aliasing mismatch"
        );
    } finally {

        wasm.chande_free(highPtr, len);
        wasm.chande_free(lowPtr, len);
        wasm.chande_free(closePtr, len);
    }
});


test('Chande batch metadata structure', () => {

    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    const expected = EXPECTED_OUTPUTS.chande;


    const result = wasm.chande_batch_js(
        high, low, close,
        expected.defaultParams.period, expected.defaultParams.period, 0,
        expected.defaultParams.mult, expected.defaultParams.mult, 0,
        expected.defaultParams.direction
    );


    assert(result.values, 'Should have values array');
    assert(result.periods, 'Should have periods array');
    assert(result.mults, 'Should have mults array');
    assert(result.directions, 'Should have directions array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');


    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.periods.length, 1);
    assert.strictEqual(result.values.length, 50);
});

test('Chande batch multiple parameter combinations', () => {

    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));

    const result = wasm.chande_batch_js(
        high, low, close,
        10, 20, 10,
        2.0, 3.0, 0.5,
        'short'
    );


    assert.strictEqual(result.rows, 6);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.periods.length, 6);
    assert.strictEqual(result.mults.length, 6);
    assert.strictEqual(result.directions.length, 6);
    assert.strictEqual(result.values.length, 300);


    const expectedPeriods = [10, 10, 10, 20, 20, 20];
    const expectedMults = [2.0, 2.5, 3.0, 2.0, 2.5, 3.0];

    for (let i = 0; i < 6; i++) {
        assert.strictEqual(result.periods[i], expectedPeriods[i], `Period mismatch at index ${i}`);
        assertClose(result.mults[i], expectedMults[i], 1e-10, `Mult mismatch at index ${i}`);
        assert.strictEqual(result.directions[i], 'short', `Direction mismatch at index ${i}`);
    }


    const firstRow = result.values.slice(0, result.cols);
    const singleResult = wasm.chande_js(high, low, close, 10, 2.0, 'short');
    assertArrayClose(firstRow, singleResult, 1e-10, "First batch row vs single mismatch");
});

test('Chande batch edge cases', () => {

    const high = new Float64Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    const low = new Float64Array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95]);
    const close = new Float64Array([8, 18, 28, 38, 48, 58, 68, 78, 88, 98]);


    const singleBatch = wasm.chande_batch_js(
        high, low, close,
        5, 5, 0,
        2.0, 2.0, 0,
        'long'
    );

    assert.strictEqual(singleBatch.rows, 1);
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.periods.length, 1);


    const largeBatch = wasm.chande_batch_js(
        high, low, close,
        5, 7, 10,
        2.0, 2.0, 0,
        'long'
    );


    assert.strictEqual(largeBatch.rows, 1);
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.periods[0], 5);
});

test('Chande warmup period validation', () => {

    const high = new Float64Array(100).fill(100);
    const low = new Float64Array(100).fill(90);
    const close = new Float64Array(100).fill(95);

    const testPeriods = [5, 10, 22, 30];

    for (const period of testPeriods) {
        const result = wasm.chande_js(high, low, close, period, 2.0, 'long');
        const expectedWarmup = period - 1;


        for (let i = 0; i < expectedWarmup; i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} for period ${period}`);
        }


        assert(!isNaN(result[expectedWarmup]), `Expected valid value at index ${expectedWarmup} for period ${period}`);
    }
});

test('Chande SIMD128 consistency', () => {


    const testCases = [
        { size: 10, period: 5 },
        { size: 100, period: 22 },
        { size: 1000, period: 50 },
        { size: 10000, period: 100 }
    ];

    for (const testCase of testCases) {
        const high = new Float64Array(testCase.size);
        const low = new Float64Array(testCase.size);
        const close = new Float64Array(testCase.size);

        for (let i = 0; i < testCase.size; i++) {
            const base = 100 + Math.sin(i * 0.1) * 10;
            high[i] = base + 5;
            low[i] = base - 5;
            close[i] = base + Math.cos(i * 0.05) * 3;
        }

        const result = wasm.chande_js(high, low, close, testCase.period, 3.0, 'long');


        assert.strictEqual(result.length, testCase.size);


        const expectedWarmup = testCase.period - 1;
        for (let i = 0; i < expectedWarmup; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }


        for (let i = expectedWarmup; i < Math.min(expectedWarmup + 10, result.length); i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
        }
    }
});

test('Chande zero-copy error handling', () => {

    assert.throws(() => {
        wasm.chande_into(0, 0, 0, 0, 10, 22, 3.0, 'long');
    }, /null pointer|invalid memory/i);


    const highPtr = wasm.chande_alloc(10);
    const lowPtr = wasm.chande_alloc(10);
    const closePtr = wasm.chande_alloc(10);
    const outPtr = wasm.chande_alloc(10);

    try {

        assert.throws(() => {
            wasm.chande_into(highPtr, lowPtr, closePtr, outPtr, 10, 0, 3.0, 'long');
        }, /invalid period/i);


        assert.throws(() => {
            wasm.chande_into(highPtr, lowPtr, closePtr, outPtr, 10, 5, 3.0, 'invalid');
        }, /invalid direction/i);
    } finally {
        wasm.chande_free(highPtr, 10);
        wasm.chande_free(lowPtr, 10);
        wasm.chande_free(closePtr, 10);
        wasm.chande_free(outPtr, 10);
    }
});

test('Chande memory management', () => {

    const sizes = [100, 1000, 10000];

    for (const size of sizes) {
        const highPtr = wasm.chande_alloc(size);
        const lowPtr = wasm.chande_alloc(size);
        const closePtr = wasm.chande_alloc(size);
        const outPtr = wasm.chande_alloc(size);

        assert(highPtr !== 0, `Failed to allocate high buffer for ${size} elements`);
        assert(lowPtr !== 0, `Failed to allocate low buffer for ${size} elements`);
        assert(closePtr !== 0, `Failed to allocate close buffer for ${size} elements`);
        assert(outPtr !== 0, `Failed to allocate output buffer for ${size} elements`);


        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        const closeMem = new Float64Array(wasm.__wasm.memory.buffer, closePtr, size);

        for (let i = 0; i < Math.min(10, size); i++) {
            highMem[i] = 100 + i;
            lowMem[i] = 90 + i;
            closeMem[i] = 95 + i;
        }


        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(highMem[i], 100 + i, `High memory corruption at index ${i}`);
            assert.strictEqual(lowMem[i], 90 + i, `Low memory corruption at index ${i}`);
            assert.strictEqual(closeMem[i], 95 + i, `Close memory corruption at index ${i}`);
        }


        wasm.chande_free(highPtr, size);
        wasm.chande_free(lowPtr, size);
        wasm.chande_free(closePtr, size);
        wasm.chande_free(outPtr, size);
    }
});

test('Chande batch zero-copy API', () => {

    const len = 50;
    const high = new Float64Array(testData.high.slice(0, len));
    const low = new Float64Array(testData.low.slice(0, len));
    const close = new Float64Array(testData.close.slice(0, len));


    const periodStart = 10, periodEnd = 20, periodStep = 10;
    const multStart = 2.0, multEnd = 3.0, multStep = 0.5;


    const expectedRows = 2 * 3;


    const highPtr = wasm.chande_alloc(len);
    const lowPtr = wasm.chande_alloc(len);
    const closePtr = wasm.chande_alloc(len);
    const outPtr = wasm.chande_alloc(expectedRows * len);

    try {

        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeMem = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        highMem.set(high);
        lowMem.set(low);
        closeMem.set(close);


        const rows = wasm.chande_batch_into(
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            len,
            periodStart, periodEnd, periodStep,
            multStart, multEnd, multStep,
            'long'
        );

        assert.strictEqual(rows, expectedRows);


        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, rows * len);


        const expected = wasm.chande_batch_js(
            high, low, close,
            periodStart, periodEnd, periodStep,
            multStart, multEnd, multStep,
            'long'
        );

        assertArrayClose(
            result,
            expected.values,
            1e-10,
            "Batch zero-copy API mismatch"
        );
    } finally {

        wasm.chande_free(highPtr, len);
        wasm.chande_free(lowPtr, len);
        wasm.chande_free(closePtr, len);
        wasm.chande_free(outPtr, expectedRows * len);
    }
});
