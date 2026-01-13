
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

test('ALMA partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.alma_js(close, 9, 0.85, 6.0);
    assert.strictEqual(result.length, close.length);
});

test('ALMA accuracy', async () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.alma;

    const result = wasm.alma_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.offset,
        expected.defaultParams.sigma
    );

    assert.strictEqual(result.length, close.length);


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "ALMA last 5 values mismatch"
    );


    await compareWithRust('alma', result, 'close', expected.defaultParams);
});

test('ALMA default candles', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.alma_js(close, 9, 0.85, 6.0);
    assert.strictEqual(result.length, close.length);
});

test('ALMA zero period', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.alma_js(inputData, 0, 0.85, 6.0);
    }, /Invalid period/);
});

test('ALMA period exceeds length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.alma_js(dataSmall, 10, 0.85, 6.0);
    }, /Invalid period/);
});

test('ALMA very small dataset', () => {

    const singlePoint = new Float64Array([42.0]);

    assert.throws(() => {
        wasm.alma_js(singlePoint, 9, 0.85, 6.0);
    }, /Invalid period|Not enough valid data/);
});

test('ALMA empty input', () => {

    const empty = new Float64Array([]);

    assert.throws(() => {
        wasm.alma_js(empty, 9, 0.85, 6.0);
    }, /Input data slice is empty/);
});

test('ALMA invalid sigma', () => {

    const data = new Float64Array([1.0, 2.0, 3.0]);


    assert.throws(() => {
        wasm.alma_js(data, 2, 0.85, 0.0);
    }, /Invalid sigma/);


    assert.throws(() => {
        wasm.alma_js(data, 2, 0.85, -1.0);
    }, /Invalid sigma/);
});

test('ALMA invalid offset', () => {

    const data = new Float64Array([1.0, 2.0, 3.0]);


    assert.throws(() => {
        wasm.alma_js(data, 2, NaN, 6.0);
    }, /Invalid offset/);


    assert.throws(() => {
        wasm.alma_js(data, 2, 1.5, 6.0);
    }, /Invalid offset/);


    assert.throws(() => {
        wasm.alma_js(data, 2, -0.1, 6.0);
    }, /Invalid offset/);
});

test('ALMA reinput', () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.alma;


    const firstResult = wasm.alma_js(close, 9, 0.85, 6.0);
    assert.strictEqual(firstResult.length, close.length);


    const secondResult = wasm.alma_js(firstResult, 9, 0.85, 6.0);
    assert.strictEqual(secondResult.length, firstResult.length);


    const last5 = secondResult.slice(-5);
    assertArrayClose(
        last5,
        expected.reinputLast5,
        1e-8,
        "ALMA re-input last 5 values mismatch"
    );
});

test('ALMA NaN handling', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.alma_js(close, 9, 0.85, 6.0);
    assert.strictEqual(result.length, close.length);


    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }


    assertAllNaN(result.slice(0, 8), "Expected NaN in warmup period");
});

test('ALMA all NaN input', () => {

    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);

    assert.throws(() => {
        wasm.alma_js(allNaN, 9, 0.85, 6.0);
    }, /All values are NaN/);
});

test('ALMA batch single parameter set', () => {

    const close = new Float64Array(testData.close);


    const batchResult = wasm.alma_batch(close, {
        period_range: [9, 9, 0],
        offset_range: [0.85, 0.85, 0],
        sigma_range: [6.0, 6.0, 0]
    });


    const singleResult = wasm.alma_js(close, 9, 0.85, 6.0);

    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('ALMA batch multiple periods', () => {

    const close = new Float64Array(testData.close.slice(0, 100));


    const batchResult = wasm.alma_batch(close, {
        period_range: [9, 13, 2],
        offset_range: [0.85, 0.85, 0],
        sigma_range: [6.0, 6.0, 0]
    });


    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);


    const periods = [9, 11, 13];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);

        const singleResult = wasm.alma_js(close, periods[i], 0.85, 6.0);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Period ${periods[i]} mismatch`
        );
    }
});

test('ALMA batch metadata from result', () => {

    const close = new Float64Array(20);
    close.fill(100);

    const result = wasm.alma_batch(close, {
        period_range: [9, 13, 2],
        offset_range: [0.85, 0.95, 0.05],
        sigma_range: [6.0, 7.0, 0.5]
    });


    assert.strictEqual(result.combos.length, 27);


    assert.strictEqual(result.combos[0].period, 9);
    assert.strictEqual(result.combos[0].offset, 0.85);
    assert.strictEqual(result.combos[0].sigma, 6.0);


    assert.strictEqual(result.combos[26].period, 13);
    assertClose(result.combos[26].offset, 0.95, 1e-10, "offset mismatch");
    assert.strictEqual(result.combos[26].sigma, 7.0);
});

test('ALMA batch full parameter sweep', () => {

    const close = new Float64Array(testData.close.slice(0, 50));

    const batchResult = wasm.alma_batch(close, {
        period_range: [9, 11, 2],
        offset_range: [0.85, 0.90, 0.05],
        sigma_range: [6.0, 6.0, 0]
    });


    assert.strictEqual(batchResult.combos.length, 4);
    assert.strictEqual(batchResult.rows, 4);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 4 * 50);


    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const period = batchResult.combos[combo].period;
        const offset = batchResult.combos[combo].offset;
        const sigma = batchResult.combos[combo].sigma;

        const rowStart = combo * 50;
        const rowData = batchResult.values.slice(rowStart, rowStart + 50);


        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }


        for (let i = period - 1; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('ALMA batch edge cases', () => {

    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);


    const singleBatch = wasm.alma_batch(close, {
        period_range: [5, 5, 1],
        offset_range: [0.85, 0.85, 0.1],
        sigma_range: [6.0, 6.0, 1.0]
    });

    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);


    const largeBatch = wasm.alma_batch(close, {
        period_range: [5, 7, 10],
        offset_range: [0.85, 0.85, 0],
        sigma_range: [6.0, 6.0, 0]
    });


    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);


    assert.throws(() => {
        wasm.alma_batch(new Float64Array([]), {
            period_range: [9, 9, 0],
            offset_range: [0.85, 0.85, 0],
            sigma_range: [6.0, 6.0, 0]
        });
    }, /All values are NaN/);
});


test('ALMA batch - new ergonomic API with single parameter', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.alma_batch(close, {
        period_range: [9, 9, 0],
        offset_range: [0.85, 0.85, 0],
        sigma_range: [6.0, 6.0, 0]
    });


    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');


    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.values.length, close.length);


    const combo = result.combos[0];
    assert.strictEqual(combo.period, 9);
    assert.strictEqual(combo.offset, 0.85);
    assert.strictEqual(combo.sigma, 6.0);


    const oldResult = wasm.alma_js(close, 9, 0.85, 6.0);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue;
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('ALMA batch - new API with multiple parameters', () => {

    const close = new Float64Array(testData.close.slice(0, 50));

    const result = wasm.alma_batch(close, {
        period_range: [9, 11, 2],
        offset_range: [0.85, 0.90, 0.05],
        sigma_range: [6.0, 6.0, 0]
    });


    assert.strictEqual(result.rows, 4);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 4);
    assert.strictEqual(result.values.length, 200);


    const expectedCombos = [
        { period: 9, offset: 0.85, sigma: 6.0 },
        { period: 9, offset: 0.90, sigma: 6.0 },
        { period: 11, offset: 0.85, sigma: 6.0 },
        { period: 11, offset: 0.90, sigma: 6.0 }
    ];

    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedCombos[i].period);
        assert.strictEqual(result.combos[i].offset, expectedCombos[i].offset);
        assert.strictEqual(result.combos[i].sigma, expectedCombos[i].sigma);
    }


    const secondRow = result.values.slice(result.cols, 2 * result.cols);
    assert.strictEqual(secondRow.length, 50);


    const oldResult = wasm.alma_js(close, 9, 0.85, 6.0);
    const firstRow = result.values.slice(0, result.cols);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
            continue;
        }
        assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});



test('ALMA batch - new API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));


    assert.throws(() => {
        wasm.alma_batch(close, {
            period_range: [9, 9],
            offset_range: [0.85, 0.85, 0],
            sigma_range: [6.0, 6.0, 0]
        });
    }, /Invalid config/);


    assert.throws(() => {
        wasm.alma_batch(close, {
            period_range: [9, 9, 0],
            offset_range: [0.85, 0.85, 0]

        });
    }, /Invalid config/);


    assert.throws(() => {
        wasm.alma_batch(close, {
            period_range: "invalid",
            offset_range: [0.85, 0.85, 0],
            sigma_range: [6.0, 6.0, 0]
        });
    }, /Invalid config/);
});






test('ALMA zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    const offset = 0.85;
    const sigma = 6.0;


    const ptr = wasm.alma_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');


    const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
    if (!memory || !memory.buffer) {
        console.warn('Skipping zero-copy API test: wasm memory accessor unavailable');
        wasm.alma_free(ptr, data.length);
        return;
    }
    const memView = new Float64Array(
        memory.buffer,
        ptr,
        data.length
    );


    memView.set(data);


    try {
        wasm.alma_into(ptr, ptr, data.length, period, offset, sigma);


        const regularResult = wasm.alma_js(data, period, offset, sigma);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue;
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {

        wasm.alma_free(ptr, data.length);
    }
});

test('ALMA zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }

    const ptr = wasm.alma_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');

    try {
        const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        if (!memory || !memory.buffer) {
            console.warn('Skipping zero-copy large dataset test: wasm memory accessor unavailable');
            return;
        }
        const memView = new Float64Array(memory.buffer, ptr, size);
        memView.set(data);

        wasm.alma_into(ptr, ptr, size, 9, 0.85, 6.0);


        const memory2 = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        if (!memory2 || !memory2.buffer) {
            console.warn('Skipping zero-copy large dataset verification: wasm memory accessor unavailable');
            return;
        }
        const memView2 = new Float64Array(memory2.buffer, ptr, size);


        for (let i = 0; i < 8; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }


        for (let i = 8; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.alma_free(ptr, size);
    }
});





test('ALMA SIMD128 consistency', () => {


    const testCases = [
        { size: 10, period: 5 },
        { size: 100, period: 9 },
        { size: 1000, period: 20 },
        { size: 10000, period: 50 }
    ];

    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }

        const result = wasm.alma_js(data, testCase.period, 0.85, 6.0);


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


test('ALMA zero-copy error handling', () => {

    assert.throws(() => {
        wasm.alma_into(0, 0, 10, 9, 0.85, 6.0);
    }, /null pointer|invalid memory/i);


    const ptr = wasm.alma_alloc(10);
    try {

        assert.throws(() => {
            wasm.alma_into(ptr, ptr, 10, 0, 0.85, 6.0);
        }, /Invalid period/);


        assert.throws(() => {
            wasm.alma_into(ptr, ptr, 10, 5, 0.85, 0.0);
        }, /Invalid sigma/);
    } finally {
        wasm.alma_free(ptr, 10);
    }
});


test('ALMA zero-copy memory management', () => {

    const sizes = [100, 1000, 10000, 100000];

    for (const size of sizes) {
        const ptr = wasm.alma_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);


        const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        if (!memory || !memory.buffer) {
            console.warn('Skipping zero-copy memory management: wasm memory accessor unavailable');
            wasm.alma_free(ptr, size);
            continue;
        }
        const memView = new Float64Array(memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }


        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }


        wasm.alma_free(ptr, size);
    }
});

test.after(() => {
    console.log('ALMA WASM tests completed');
});
