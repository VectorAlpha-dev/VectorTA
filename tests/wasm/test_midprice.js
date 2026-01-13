
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

test('MIDPRICE accuracy', async () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const expected = EXPECTED_OUTPUTS.midprice;

    const result = wasm.midprice_js(high, low, expected.defaultParams.period);


    assert.strictEqual(result.length, high.length);


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-9,
        "MIDPRICE last 5 values mismatch"
    );


    await compareWithRust('midprice', result, 'hl', expected.defaultParams);
});

test('MIDPRICE partial params', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);

    const result = wasm.midprice_js(high, low, 14);
    assert.strictEqual(result.length, high.length);


    assertAllNaN(result.slice(0, 13), "Expected NaN in warmup period");
    assertNoNaN(result.slice(20), "Unexpected NaN after warmup");
});

test('MIDPRICE error handling', () => {

    assert.throws(() => {
        wasm.midprice_js(new Float64Array([]), new Float64Array([]), 14);
    }, /Empty data/);


    assert.throws(() => {
        wasm.midprice_js(new Float64Array([1.0, 2.0]), new Float64Array([1.0]), 14);
    }, /Mismatched data length/);


    assert.throws(() => {
        wasm.midprice_js(new Float64Array([1.0]), new Float64Array([1.0]), 0);
    }, /Invalid period/);


    assert.throws(() => {
        wasm.midprice_js(new Float64Array([1.0, 2.0]), new Float64Array([1.0, 2.0]), 10);
    }, /Invalid period|Not enough valid data/);
});

test('MIDPRICE very small dataset', () => {

    const high = new Float64Array([42.0]);
    const low = new Float64Array([36.0]);

    assert.throws(() => {
        wasm.midprice_js(high, low, 14);
    }, /Invalid period|Not enough valid data/);
});

test('MIDPRICE all NaN values', () => {

    const high = new Float64Array([NaN, NaN, NaN]);
    const low = new Float64Array([NaN, NaN, NaN]);

    assert.throws(() => {
        wasm.midprice_js(high, low, 2);
    }, /All values are NaN/);
});


test('MIDPRICE NaN handling', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);

    const result = wasm.midprice_js(high, low, 14);


    assert.strictEqual(result.length, high.length);


    assertNoNaN(result.slice(20), "Unexpected NaN after warmup");
});

test('MIDPRICE batch single parameter', () => {

    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);

    const batchResult = wasm.midprice_batch(high, low, {
        period_range: [14, 14, 0]
    });


    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, high.length);
    assert.strictEqual(batchResult.periods.length, 1);
    assert.strictEqual(batchResult.periods[0], 14);


    const singleResult = wasm.midprice_js(high, low, 14);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('MIDPRICE batch metadata validation', () => {

    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));

    const result = wasm.midprice_batch(high, low, {
        period_range: [10, 20, 5]
    });


    assert(result.periods, 'Should have periods array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');


    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    assert.deepStrictEqual(result.periods, [10, 15, 20]);


    for (let i = 0; i < result.periods.length; i++) {
        const period = result.periods[i];
        const rowStart = i * result.cols;
        const rowData = result.values.slice(rowStart, rowStart + result.cols);


        for (let j = 0; j < period - 1; j++) {
            assert(isNaN(rowData[j]), `Expected NaN at warmup index ${j} for period ${period}`);
        }


        for (let j = period - 1; j < Math.min(period + 10, rowData.length); j++) {
            assert(!isNaN(rowData[j]), `Unexpected NaN at index ${j} for period ${period}`);
        }
    }
});

test('MIDPRICE batch multiple periods', () => {

    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));


    const batchResult = wasm.midprice_batch(high, low, {
        period_range: [10, 20, 5]
    });


    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    assert.deepStrictEqual(batchResult.periods, [10, 15, 20]);


    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);

        const singleResult = wasm.midprice_js(high, low, periods[i]);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Period ${periods[i]} mismatch`
        );
    }
});

test('MIDPRICE batch edge cases', () => {

    const high = new Float64Array([100, 110, 105, 115, 120, 125, 130, 128, 135, 140]);
    const low = new Float64Array([95, 105, 100, 110, 115, 120, 125, 123, 130, 135]);


    const singleBatch = wasm.midprice_batch(high, low, {
        period_range: [5, 5, 0]
    });

    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.periods.length, 1);
    assert.strictEqual(singleBatch.periods[0], 5);


    const largeBatch = wasm.midprice_batch(high, low, {
        period_range: [3, 5, 10]
    });


    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.periods.length, 1);
    assert.strictEqual(largeBatch.periods[0], 3);


    assert.throws(() => {
        wasm.midprice_batch(new Float64Array([]), new Float64Array([]), {
            period_range: [5, 5, 0]
        });
    }, /Empty data/);
});

test('MIDPRICE zero-copy basic', () => {
    const data = new Float64Array([
        100, 110, 105, 115, 120, 125, 130, 128, 135, 140,
        138, 142, 145, 150, 148, 152, 155, 158, 160, 162
    ]);
    const lowData = new Float64Array([
        95, 105, 100, 110, 115, 120, 125, 123, 130, 135,
        133, 137, 140, 145, 143, 147, 150, 153, 155, 157
    ]);
    const period = 5;


    const highPtr = wasm.midprice_alloc(data.length);
    const lowPtr = wasm.midprice_alloc(data.length);
    const outPtr = wasm.midprice_alloc(data.length);

    try {

        const highMemView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, data.length);
        const lowMemView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, data.length);
        highMemView.set(data);
        lowMemView.set(lowData);


        wasm.midprice_into(highPtr, lowPtr, outPtr, data.length, period);


        const outMemView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, data.length);


        const regularResult = wasm.midprice_js(data, lowData, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(outMemView[i])) {
                continue;
            }
            assert(Math.abs(regularResult[i] - outMemView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${outMemView[i]}`);
        }
    } finally {

        wasm.midprice_free(highPtr, data.length);
        wasm.midprice_free(lowPtr, data.length);
        wasm.midprice_free(outPtr, data.length);
    }
});

test('MIDPRICE zero-copy aliasing detection', () => {
    const highData = new Float64Array([100, 110, 105, 115, 120, 125, 130, 128, 135, 140]);
    const lowData = new Float64Array([95, 105, 100, 110, 115, 120, 125, 123, 130, 135]);
    const period = 3;


    const highPtr = wasm.midprice_alloc(highData.length);
    const lowPtr = wasm.midprice_alloc(lowData.length);

    try {

        const highMemView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, highData.length);
        const lowMemView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, lowData.length);
        highMemView.set(highData);
        lowMemView.set(lowData);


        wasm.midprice_into(highPtr, lowPtr, highPtr, highData.length, period);


        const highMemView2 = new Float64Array(wasm.__wasm.memory.buffer, highPtr, highData.length);
        const regularResult = wasm.midprice_js(highData, lowData, period);

        for (let i = 0; i < highData.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(highMemView2[i])) {
                continue;
            }
            assert(Math.abs(regularResult[i] - highMemView2[i]) < 1e-10,
                   `Aliasing mismatch at index ${i}`);
        }
    } finally {
        wasm.midprice_free(highPtr, highData.length);
        wasm.midprice_free(lowPtr, lowData.length);
    }
});

test('MIDPRICE zero-copy with large dataset', () => {
    const size = 10000;
    const highData = new Float64Array(size);
    const lowData = new Float64Array(size);


    for (let i = 0; i < size; i++) {
        highData[i] = 100 + i * 0.01 + Math.sin(i * 0.1) * 5;
        lowData[i] = highData[i] - 5 - Math.random() * 2;
    }

    const highPtr = wasm.midprice_alloc(size);
    const lowPtr = wasm.midprice_alloc(size);
    const outPtr = wasm.midprice_alloc(size);

    try {

        const highMemView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        const lowMemView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        highMemView.set(highData);
        lowMemView.set(lowData);

        wasm.midprice_into(highPtr, lowPtr, outPtr, size, 14);


        const outMemView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, size);


        for (let i = 0; i < 13; i++) {
            assert(isNaN(outMemView[i]), `Expected NaN at warmup index ${i}`);
        }


        for (let i = 13; i < Math.min(100, size); i++) {
            assert(!isNaN(outMemView[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.midprice_free(highPtr, size);
        wasm.midprice_free(lowPtr, size);
        wasm.midprice_free(outPtr, size);
    }
});

test('MIDPRICE zero-copy error handling', () => {

    assert.throws(() => {
        wasm.midprice_into(0, 0, 0, 10, 5);
    }, /Null pointer/i);


    const ptr1 = wasm.midprice_alloc(10);
    const ptr2 = wasm.midprice_alloc(10);
    const ptr3 = wasm.midprice_alloc(10);

    try {

        assert.throws(() => {
            wasm.midprice_into(ptr1, ptr2, ptr3, 10, 0);
        }, /Invalid period/);


        assert.throws(() => {
            wasm.midprice_into(ptr1, ptr2, ptr3, 10, 20);
        }, /Invalid period|Not enough valid data/);
    } finally {
        wasm.midprice_free(ptr1, 10);
        wasm.midprice_free(ptr2, 10);
        wasm.midprice_free(ptr3, 10);
    }
});

test('MIDPRICE batch_into basic', () => {
    const size = 50;
    const highData = new Float64Array(size);
    const lowData = new Float64Array(size);


    for (let i = 0; i < size; i++) {
        highData[i] = 100 + i * 0.5;
        lowData[i] = highData[i] - 5;
    }

    const highPtr = wasm.midprice_alloc(size);
    const lowPtr = wasm.midprice_alloc(size);


    const numRows = 3;
    const totalSize = numRows * size;
    const outPtr = wasm.midprice_alloc(totalSize);

    try {

        const highMemView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        const lowMemView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        highMemView.set(highData);
        lowMemView.set(lowData);


        const rows = wasm.midprice_batch_into(highPtr, lowPtr, outPtr, size, 10, 20, 5);
        assert.strictEqual(rows, 3);


        const outMemView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);


        const jsBatch = wasm.midprice_batch(highData, lowData, {
            period_range: [10, 20, 5]
        });

        assertArrayClose(outMemView, jsBatch.values, 1e-10, "Batch into vs JS batch mismatch");
    } finally {
        wasm.midprice_free(highPtr, size);
        wasm.midprice_free(lowPtr, size);
        wasm.midprice_free(outPtr, totalSize);
    }
});

test('MIDPRICE memory leak prevention', () => {

    const sizes = [100, 1000, 10000];

    for (const size of sizes) {
        const ptr = wasm.midprice_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);


        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }


        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5);
        }

        wasm.midprice_free(ptr, size);
    }
});

test.after(() => {
    console.log('MIDPRICE WASM tests completed');
});