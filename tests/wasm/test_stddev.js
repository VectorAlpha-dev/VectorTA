
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

test('StdDev partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.stddev_js(close, 5, 1.0);
    assert.strictEqual(result.length, close.length);
});

test('StdDev accuracy', async () => {

    const close = new Float64Array(testData.close);


    const result = wasm.stddev_js(close, 5, 1.0);

    assert.strictEqual(result.length, close.length);


    const expectedLast5 = [
        180.12506767314034,
        77.7395652441455,
        127.16225857341935,
        89.40156600773197,
        218.50034325919697,
    ];

    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        0.1,
        "StdDev last 5 values mismatch"
    );


    await compareWithRust('stddev', result, 'close', { period: 5, nbdev: 1.0 });
});

test('StdDev default candles', () => {

    const close = new Float64Array(testData.close);


    const result = wasm.stddev_js(close, 5, 1.0);
    assert.strictEqual(result.length, close.length);
});

test('StdDev zero period', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.stddev_js(inputData, 0, 1.0);
    }, /Invalid period/);
});

test('StdDev period exceeds length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.stddev_js(dataSmall, 10, 1.0);
    }, /Invalid period/);
});

test('StdDev very small dataset', () => {

    const singlePoint = new Float64Array([42.0]);

    assert.throws(() => {
        wasm.stddev_js(singlePoint, 5, 1.0);
    }, /Invalid period/);
});

test('StdDev empty input', () => {

    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.stddev_js(empty, 5, 1.0);
    });
});

test('StdDev all NaN input', () => {

    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);

    assert.throws(() => {
        wasm.stddev_js(allNaN, 3, 1.0);
    }, /All values are NaN/);
});

test('StdDev reinput', () => {

    const close = new Float64Array(testData.close);


    const firstResult = wasm.stddev_js(close, 10, 1.0);


    const secondResult = wasm.stddev_js(firstResult, 10, 1.0);

    assert.strictEqual(secondResult.length, firstResult.length);


    for (let i = 19; i < secondResult.length; i++) {
        assert(!isNaN(secondResult[i]), `Unexpected NaN at index ${i}`);
    }
});

test('StdDev NaN handling', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.stddev_js(close, 5, 1.0);
    assert.strictEqual(result.length, close.length);


    if (result.length > 20) {
        for (let i = 20; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
        }
    }
});

test('StdDev batch single parameter set', () => {

    const close = new Float64Array(testData.close);

    const batchResult = wasm.stddev_batch(close, {
        period_range: [5, 5, 0],
        nbdev_range: [1.0, 1.0, 0.0]
    });


    const singleResult = wasm.stddev_js(close, 5, 1.0);

    assert.strictEqual(batchResult.values.length, singleResult.length);



    assertArrayClose(batchResult.values, singleResult, 1e-5, "Batch vs single mismatch");
});

test('StdDev batch multiple periods', () => {

    const close = new Float64Array(testData.close.slice(0, 100));


    const batchResult = wasm.stddev_batch(close, {
        period_range: [5, 15, 5],
        nbdev_range: [1.0, 1.0, 0.0]
    });


    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);


    const periods = [5, 10, 15];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);

        const singleResult = wasm.stddev_js(close, periods[i], 1.0);

        assertArrayClose(
            rowData,
            singleResult,
            1e-5,
            `Period ${periods[i]} mismatch`
        );
    }
});

test('StdDev batch full parameter sweep', () => {

    const close = new Float64Array(testData.close.slice(0, 50));

    const batchResult = wasm.stddev_batch(close, {
        period_range: [5, 10, 5],
        nbdev_range: [1.0, 2.0, 0.5]
    });


    assert.strictEqual(batchResult.periods.length, 6);
    assert.strictEqual(batchResult.nbdevs.length, 6);
    assert.strictEqual(batchResult.rows, 6);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 6 * 50);


    const expectedCombos = [
        [5, 1.0], [5, 1.5], [5, 2.0],
        [10, 1.0], [10, 1.5], [10, 2.0]
    ];

    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(batchResult.periods[i], expectedCombos[i][0]);
        assertClose(batchResult.nbdevs[i], expectedCombos[i][1], 1e-10, `nbdev mismatch at ${i}`);
    }
});

test('StdDev zero-copy API basic', () => {

    const data = new Float64Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    const period = 5;
    const nbdev = 1.0;


    const ptr = wasm.stddev_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');


    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );


    memView.set(data);


    try {
        wasm.stddev_into(ptr, ptr, data.length, period, nbdev);


        const regularResult = wasm.stddev_js(data, period, nbdev);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue;
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {

        wasm.stddev_free(ptr, data.length);
    }
});

test('StdDev zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) * 100 + Math.random() * 10;
    }

    const ptr = wasm.stddev_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');

    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);

        wasm.stddev_into(ptr, ptr, size, 10, 1.0);


        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);


        for (let i = 0; i < 9; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }


        for (let i = 9; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.stddev_free(ptr, size);
    }
});

test('StdDev zero-copy error handling', () => {

    assert.throws(() => {
        wasm.stddev_into(0, 0, 10, 5, 1.0);
    }, /null pointer|Null pointer provided/i);


    const ptr = wasm.stddev_alloc(10);
    try {

        assert.throws(() => {
            wasm.stddev_into(ptr, ptr, 10, 0, 1.0);
        }, /Invalid period/);


        assert.throws(() => {
            wasm.stddev_into(ptr, ptr, 10, 20, 1.0);
        }, /Invalid period/);
    } finally {
        wasm.stddev_free(ptr, 10);
    }
});

test('StdDev zero-copy memory management', () => {

    const sizes = [100, 1000, 10000, 100000];

    for (const size of sizes) {
        const ptr = wasm.stddev_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);


        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 2.5;
        }


        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 2.5, `Memory corruption at index ${i}`);
        }


        wasm.stddev_free(ptr, size);
    }
});

test('StdDev batch zero-copy API', () => {
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = Math.sin(i * 0.1) * 100;
    }

    const config = {
        period_range: [5, 10, 5],
        nbdev_range: [1.0, 2.0, 1.0]
    };


    const expectedRows = 2 * 2;
    const expectedSize = expectedRows * data.length;


    const inPtr = wasm.stddev_alloc(data.length);
    const outPtr = wasm.stddev_alloc(expectedSize);

    try {

        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
        inView.set(data);


        const result = wasm.stddev_batch_into_cfg(inPtr, outPtr, data.length, config);


        assert.strictEqual(result.rows, expectedRows);
        assert.strictEqual(result.cols, data.length);
        assert.strictEqual(result.periods.length, expectedRows);
        assert.strictEqual(result.nbdevs.length, expectedRows);


        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, expectedSize);


        let hasValidValues = false;
        for (let i = 0; i < expectedSize; i++) {
            if (!isNaN(outView[i])) {
                hasValidValues = true;
                break;
            }
        }
        assert(hasValidValues, 'Expected some non-NaN values in output');

    } finally {
        wasm.stddev_free(inPtr, data.length);
        wasm.stddev_free(outPtr, expectedSize);
    }
});

test.after(() => {
    console.log('StdDev WASM tests completed');
});
