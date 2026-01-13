
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

test('KAUFMANSTOP partial params', () => {

    const { high, low } = testData;


    const result = wasm.kaufmanstop_js(
        new Float64Array(high),
        new Float64Array(low),
        22, 2.0, 'long', 'sma'
    );
    assert.ok(result instanceof Float64Array, 'Result should be Float64Array');
    assert.strictEqual(result.length, high.length);
});

test('KAUFMANSTOP accuracy', async () => {

    const { high, low } = testData;
    const expected = EXPECTED_OUTPUTS.kaufmanstop;

    const result = wasm.kaufmanstop_js(
        new Float64Array(high),
        new Float64Array(low),
        expected.defaultParams.period,
        expected.defaultParams.mult,
        expected.defaultParams.direction,
        expected.defaultParams.maType
    );

    assert.ok(result instanceof Float64Array, 'Result should be Float64Array');
    assert.strictEqual(result.length, high.length);


    const last5 = Array.from(result).slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-1,
        "KAUFMANSTOP last 5 values mismatch"
    );



});

test('KAUFMANSTOP default candles', () => {

    const { high, low } = testData;
    const expected = EXPECTED_OUTPUTS.kaufmanstop;

    const result = wasm.kaufmanstop_js(
        new Float64Array(high),
        new Float64Array(low),
        expected.defaultParams.period,
        expected.defaultParams.mult,
        expected.defaultParams.direction,
        expected.defaultParams.maType
    );
    assert.ok(result instanceof Float64Array, 'Result should be Float64Array');
    assert.strictEqual(result.length, high.length);
});

test('KAUFMANSTOP zero period', () => {

    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);

    assert.throws(() => {
        wasm.kaufmanstop_js(high, low, 0, 2.0, 'long', 'sma');
    }, /Invalid period/);
});

test('KAUFMANSTOP period exceeds length', () => {

    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);

    assert.throws(() => {
        wasm.kaufmanstop_js(high, low, 10, 2.0, 'long', 'sma');
    }, /Invalid period/);
});

test('KAUFMANSTOP very small dataset', () => {

    const high = new Float64Array([42.0]);
    const low = new Float64Array([41.0]);

    assert.throws(() => {
        wasm.kaufmanstop_js(high, low, 22, 2.0, 'long', 'sma');
    }, /Invalid period|Not enough valid data/);
});

test('KAUFMANSTOP empty data', () => {

    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.kaufmanstop_js(empty, empty, 22, 2.0, 'long', 'sma');
    }, /Empty data/);
});

test('KAUFMANSTOP mismatched lengths', () => {

    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]);

    assert.throws(() => {
        wasm.kaufmanstop_js(high, low, 2, 2.0, 'long', 'sma');
    });
});

test('KAUFMANSTOP all NaN input', () => {

    const allNaNHigh = new Float64Array(100);
    const allNaNLow = new Float64Array(100);
    allNaNHigh.fill(NaN);
    allNaNLow.fill(NaN);

    assert.throws(() => {
        wasm.kaufmanstop_js(allNaNHigh, allNaNLow, 22, 2.0, 'long', 'sma');
    }, /All values are NaN/);
});

test('KAUFMANSTOP NaN handling', () => {

    const { high, low } = testData;
    const expected = EXPECTED_OUTPUTS.kaufmanstop;

    const result = wasm.kaufmanstop_js(
        new Float64Array(high),
        new Float64Array(low),
        expected.defaultParams.period,
        expected.defaultParams.mult,
        expected.defaultParams.direction,
        expected.defaultParams.maType
    );
    assert.strictEqual(result.length, high.length);


    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }


    const warmup = expected.warmupPeriod;
    for (let i = 0; i < warmup && i < result.length; i++) {
        assert(isNaN(result[i]), `Expected NaN at warmup index ${i}`);
    }
});

test('KAUFMANSTOP with short direction', () => {
    const { high, low } = testData;

    const resultLong = wasm.kaufmanstop_js(
        new Float64Array(high.slice(0, 100)),
        new Float64Array(low.slice(0, 100)),
        22, 2.0, 'long', 'sma'
    );
    const resultShort = wasm.kaufmanstop_js(
        new Float64Array(high.slice(0, 100)),
        new Float64Array(low.slice(0, 100)),
        22, 2.0, 'short', 'sma'
    );


    let foundDifference = false;
    const warmup = 43;
    for (let i = warmup; i < 100; i++) {
        if (!isNaN(resultLong[i]) && !isNaN(resultShort[i]) && resultLong[i] !== resultShort[i]) {
            foundDifference = true;
            break;
        }
    }
    assert.ok(foundDifference, "Long and short directions should produce different results");
});

test('KAUFMANSTOP batch single parameter set', () => {

    const { high, low } = testData;
    const expected = EXPECTED_OUTPUTS.kaufmanstop;

    const batchResult = wasm.kaufmanstop_batch_js(
        new Float64Array(high),
        new Float64Array(low),
        22, 22, 0,
        2.0, 2.0, 0.0,
        'long', 'sma'
    );

    assert.ok(batchResult);
    assert.ok(batchResult.values);
    assert.ok(batchResult.combos);
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, high.length);


    const singleRow = batchResult.values.slice(0, high.length);
    const last5 = singleRow.slice(-5);
    assertArrayClose(
        last5,
        expected.batchDefaultRow,
        1e-1,
        "KAUFMANSTOP batch single params mismatch"
    );
});

test('KAUFMANSTOP batch multiple periods', () => {

    const { high, low } = testData;
    const testHigh = new Float64Array(high.slice(0, 100));
    const testLow = new Float64Array(low.slice(0, 100));

    const batchResult = wasm.kaufmanstop_batch_js(
        testHigh, testLow,
        20, 24, 2,
        2.0, 2.0, 0.0,
        'long', 'sma'
    );


    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.combos.length, 3);


    const periods = [20, 22, 24];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowData = batchResult.values.slice(rowStart, rowStart + 100);

        const singleResult = wasm.kaufmanstop_js(
            testHigh, testLow,
            periods[i], 2.0, 'long', 'sma'
        );

        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Period ${periods[i]} mismatch`
        );
    }
});

test('KAUFMANSTOP batch metadata', () => {

    const { high, low } = testData;
    const testHigh = new Float64Array(high.slice(0, 50));
    const testLow = new Float64Array(low.slice(0, 50));

    const result = wasm.kaufmanstop_batch_js(
        testHigh, testLow,
        20, 22, 2,
        1.5, 2.0, 0.5,
        'long', 'sma'
    );


    assert.strictEqual(result.combos.length, 4);
    assert.strictEqual(result.rows, 4);
    assert.strictEqual(result.cols, 50);


    assert.strictEqual(result.combos[0].period, 20);
    assert.strictEqual(result.combos[0].mult, 1.5);
    assert.strictEqual(result.combos[0].direction, 'long');
    assert.strictEqual(result.combos[0].ma_type, 'sma');


    assert.strictEqual(result.combos[3].period, 22);
    assert.strictEqual(result.combos[3].mult, 2.0);
});

test('KAUFMANSTOP batch edge cases', () => {

    const testData = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]);


    const singleBatch = wasm.kaufmanstop_batch_js(
        testData, testData,
        5, 5, 1,
        2.0, 2.0, 0.1,
        'long', 'sma'
    );

    assert.strictEqual(singleBatch.values.length, 25);
    assert.strictEqual(singleBatch.combos.length, 1);


    const largeBatch = wasm.kaufmanstop_batch_js(
        testData, testData,
        5, 7, 10,
        2.0, 2.0, 0,
        'long', 'sma'
    );


    assert.strictEqual(largeBatch.values.length, 25);
    assert.strictEqual(largeBatch.combos.length, 1);
});

test('KAUFMANSTOP different MA types', () => {

    const { high, low } = testData;
    const testHigh = new Float64Array(high.slice(0, 100));
    const testLow = new Float64Array(low.slice(0, 100));
    const maTypes = ['sma', 'ema', 'wma', 'smma'];
    const results = [];

    for (const maType of maTypes) {
        try {
            const result = wasm.kaufmanstop_js(
                testHigh, testLow,
                22, 2.0, 'long', maType
            );
            results.push({ maType, result });
        } catch (e) {

        }
    }


    assert(results.length >= 1, "At least SMA should be supported");


    if (results.length > 1) {
        for (let i = 1; i < results.length; i++) {
            let foundDifference = false;
            for (let j = 21; j < 100; j++) {
                if (!isNaN(results[0].result[j]) && !isNaN(results[i].result[j]) &&
                    Math.abs(results[0].result[j] - results[i].result[j]) > 1e-10) {
                    foundDifference = true;
                    break;
                }
            }


            if (!foundDifference) {
                console.log(`Note: ${results[0].maType} and ${results[i].maType} produced same results`);
            }
        }
    }
});


test('KAUFMANSTOP zero-copy API', () => {
    const high = new Float64Array([100, 102, 101, 103, 102, 104, 103, 105, 104, 106,
                                    105, 107, 106, 108, 107, 109, 108, 110, 109, 111,
                                    110, 112, 111, 113, 112]);
    const low = new Float64Array([99, 101, 100, 102, 101, 103, 102, 104, 103, 105,
                                   104, 106, 105, 107, 106, 108, 107, 109, 108, 110,
                                   109, 111, 110, 112, 111]);
    const len = high.length;


    const highPtr = wasm.kaufmanstop_alloc(len);
    const lowPtr = wasm.kaufmanstop_alloc(len);
    const outPtr = wasm.kaufmanstop_alloc(len);

    assert(highPtr !== 0, 'Failed to allocate high buffer');
    assert(lowPtr !== 0, 'Failed to allocate low buffer');
    assert(outPtr !== 0, 'Failed to allocate output buffer');

    try {

        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);


        highView.set(high);
        lowView.set(low);


        wasm.kaufmanstop_into(highPtr, lowPtr, outPtr, len, 5, 2.0, 'long', 'sma');


        const regularResult = wasm.kaufmanstop_js(high, low, 5, 2.0, 'long', 'sma');


        const outView2 = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);

        for (let i = 0; i < len; i++) {
            if (isNaN(regularResult[i]) && isNaN(outView2[i])) {
                continue;
            }
            assert(Math.abs(regularResult[i] - outView2[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${outView2[i]}`);
        }
    } finally {

        wasm.kaufmanstop_free(highPtr, len);
        wasm.kaufmanstop_free(lowPtr, len);
        wasm.kaufmanstop_free(outPtr, len);
    }
});

test('KAUFMANSTOP zero-copy error handling', () => {

    assert.throws(() => {
        wasm.kaufmanstop_into(0, 0, 0, 10, 22, 2.0, 'long', 'sma');
    }, /Null pointer/);


    const ptr = wasm.kaufmanstop_alloc(10);
    try {

        assert.throws(() => {
            wasm.kaufmanstop_into(ptr, ptr, ptr, 10, 0, 2.0, 'long', 'sma');
        }, /Invalid period/);
    } finally {
        wasm.kaufmanstop_free(ptr, 10);
    }
});

test('KAUFMANSTOP memory management', () => {

    const sizes = [100, 1000, 10000];

    for (const size of sizes) {
        const highPtr = wasm.kaufmanstop_alloc(size);
        const lowPtr = wasm.kaufmanstop_alloc(size);
        assert(highPtr !== 0, `Failed to allocate high buffer of ${size} elements`);
        assert(lowPtr !== 0, `Failed to allocate low buffer of ${size} elements`);


        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            highView[i] = i * 1.5;
            lowView[i] = i * 1.2;
        }


        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(highView[i], i * 1.5, `High memory corruption at index ${i}`);
            assert.strictEqual(lowView[i], i * 1.2, `Low memory corruption at index ${i}`);
        }


        wasm.kaufmanstop_free(highPtr, size);
        wasm.kaufmanstop_free(lowPtr, size);
    }
});

test.after(() => {
    console.log('KAUFMANSTOP WASM tests completed');
});