


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


test.skip('QQE batch single parameter set', () => {

    const close = new Float64Array(testData.close.slice(0, 100));
    const expected = EXPECTED_OUTPUTS.qqe;


    const batchResult = wasm.qqe_batch_unified_js(close, {
        rsi_period_range: [14, 14, 0],
        smoothing_factor_range: [5, 5, 0],
        fast_factor_range: [4.236, 4.236, 0]
    });


    assert.ok(batchResult.fast_values, 'Should have fast_values');
    assert.ok(batchResult.slow_values, 'Should have slow_values');
    assert.ok(batchResult.combos, 'Should have combos');
    assert.strictEqual(batchResult.rows, 1, 'Should have 1 row');
    assert.strictEqual(batchResult.cols, 100, 'Should have 100 columns');


    const singleResult = wasm.qqe_js(close, 14, 5, 4.236);
    const singleFast = singleResult.values.slice(0, singleResult.cols);
    const singleSlow = singleResult.values.slice(singleResult.cols, singleResult.cols * 2);

    assertArrayClose(batchResult.fast_values, singleFast, 1e-10, "Batch vs single fast mismatch");
    assertArrayClose(batchResult.slow_values, singleSlow, 1e-10, "Batch vs single slow mismatch");
});

test.skip('QQE batch multiple parameters', () => {

    const close = new Float64Array(testData.close.slice(0, 50));

    const batchResult = wasm.qqe_batch_unified_js(close, {
        rsi_period_range: [10, 14, 2],
        smoothing_factor_range: [3, 5, 2],
        fast_factor_range: [3.0, 4.0, 1.0]
    });


    assert.strictEqual(batchResult.combos.length, 12);
    assert.strictEqual(batchResult.rows, 12);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.fast_values.length, 12 * 50);
    assert.strictEqual(batchResult.slow_values.length, 12 * 50);


    assert.strictEqual(batchResult.combos[0].rsi_period, 10);
    assert.strictEqual(batchResult.combos[0].smoothing_factor, 3);
    assertClose(batchResult.combos[0].fast_factor, 3.0, 1e-10, "fast_factor mismatch");


    const firstRowFast = batchResult.fast_values.slice(0, 50);
    const secondRowFast = batchResult.fast_values.slice(50, 100);

    let hasDifference = false;
    for (let i = 30; i < 50; i++) {
        if (!isNaN(firstRowFast[i]) && !isNaN(secondRowFast[i])) {
            if (Math.abs(firstRowFast[i] - secondRowFast[i]) > 1e-10) {
                hasDifference = true;
                break;
            }
        }
    }
    assert.ok(hasDifference, 'Different parameters should produce different results');
});

test.skip('QQE batch edge cases', () => {

    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);


    const singleBatch = wasm.qqe_batch_unified_js(close, {
        rsi_period_range: [5, 5, 1],
        smoothing_factor_range: [3, 3, 0],
        fast_factor_range: [3.0, 3.0, 0]
    });

    assert.strictEqual(singleBatch.combos.length, 1);
    assert.strictEqual(singleBatch.fast_values.length, 15);
    assert.strictEqual(singleBatch.slow_values.length, 15);


    const largeBatch = wasm.qqe_batch_unified_js(close, {
        rsi_period_range: [5, 7, 10],
        smoothing_factor_range: [3, 3, 0],
        fast_factor_range: [3.0, 3.0, 0]
    });


    assert.strictEqual(largeBatch.combos.length, 1);
    assert.strictEqual(largeBatch.combos[0].rsi_period, 5);


    assert.throws(() => {
        wasm.qqe_batch_unified_js(new Float64Array([]), {
            rsi_period_range: [14, 14, 0],
            smoothing_factor_range: [5, 5, 0],
            fast_factor_range: [4.236, 4.236, 0]
        });
    }, /[Ee]mpty/);
});

test('QQE zero-copy API', () => {

    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
    const len = data.length;


    const inPtr = wasm.qqe_alloc(len);
    const outPtr = wasm.qqe_alloc(len);
    assert(inPtr !== 0, 'Failed to allocate input memory');
    assert(outPtr !== 0, 'Failed to allocate output memory');

    try {

        const memory = wasm.__wasm.memory.buffer;
        const inView = new Float64Array(memory, inPtr, len);
        inView.set(data);


        wasm.qqe_into(inPtr, outPtr, len, 14, 5, 4.236);


        const memory2 = wasm.__wasm.memory.buffer;
        const outView = new Float64Array(memory2, outPtr, len * 2);


        const regularResult = wasm.qqe_js(data, 14, 5, 4.236);
        const regularFast = regularResult.values.slice(0, regularResult.cols);
        const regularSlow = regularResult.values.slice(regularResult.cols, regularResult.cols * 2);

        const fast = outView.subarray(0, len);
        const slow = outView.subarray(len, len * 2);


        for (let i = 0; i < len; i++) {
            if (isNaN(regularFast[i]) && isNaN(fast[i])) continue;
            if (isNaN(regularSlow[i]) && isNaN(slow[i])) continue;

            assertClose(fast[i], regularFast[i], 1e-10,
                       `Zero-copy fast mismatch at index ${i}`);
            assertClose(slow[i], regularSlow[i], 1e-10,
                       `Zero-copy slow mismatch at index ${i}`);
        }
    } finally {

        wasm.qqe_free(inPtr, len);
        wasm.qqe_free(outPtr, len);
    }
});

test('QQE zero-copy with large dataset', () => {

    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = 50.0 + Math.sin(i * 0.01) * 10.0 + Math.random() * 2.0;
    }

    const inPtr = wasm.qqe_alloc(size);
    const outPtr = wasm.qqe_alloc(size);
    assert(inPtr !== 0, 'Failed to allocate large input buffer');
    assert(outPtr !== 0, 'Failed to allocate large output buffer');

    try {
        const memory = wasm.__wasm.memory.buffer;
        const inView = new Float64Array(memory, inPtr, size);
        inView.set(data);

        wasm.qqe_into(inPtr, outPtr, size, 14, 5, 4.236);


        const memory2 = wasm.__wasm.memory.buffer;
        const outView = new Float64Array(memory2, outPtr, size * 2);
        const fast = outView.subarray(0, size);
        const slow = outView.subarray(size, size * 2);




        const rsiStart = EXPECTED_OUTPUTS.qqe.defaultParams.rsiPeriod;
        const slowWarmup = EXPECTED_OUTPUTS.qqe.warmupPeriod;

        for (let i = 0; i < rsiStart; i++) {
            assert(isNaN(fast[i]), `Expected NaN in fast at warmup index ${i}`);
        }
        for (let i = 0; i < slowWarmup; i++) {
            assert(isNaN(slow[i]), `Expected NaN in slow at warmup index ${i}`);
        }

        for (let i = rsiStart; i < Math.min(rsiStart + 100, size); i++) {
            assert(!isNaN(fast[i]), `Unexpected NaN in fast at index ${i}`);
        }
        for (let i = slowWarmup; i < Math.min(slowWarmup + 100, size); i++) {
            assert(!isNaN(slow[i]), `Unexpected NaN in slow at index ${i}`);
        }
    } finally {
        wasm.qqe_free(inPtr, size);
        wasm.qqe_free(outPtr, size);
    }
});

test('QQE zero-copy error handling', () => {

    assert.throws(() => {
        wasm.qqe_into(0, 0, 10, 14, 5, 4.236);
    }, /null pointer|invalid memory/i);


    const ptr = wasm.qqe_alloc(20);
    const outPtr = wasm.qqe_alloc(20);
    try {

        assert.throws(() => {
            wasm.qqe_into(ptr, outPtr, 20, 0, 5, 4.236);
        }, /Invalid period/);


        assert.throws(() => {
            wasm.qqe_into(ptr, outPtr, 20, 14, 0, 4.236);
        }, /Invalid|smoothing/);
    } finally {
        wasm.qqe_free(ptr, 20);
        wasm.qqe_free(outPtr, 20);
    }
});

test('QQE memory leak prevention', () => {

    const sizes = [100, 1000, 5000];

    for (const size of sizes) {
        const ptr = wasm.qqe_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);


        const memory = wasm.__wasm.memory.buffer;
        const memView = new Float64Array(memory, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 2.5;
        }


        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 2.5, `Memory corruption at index ${i}`);
        }


        wasm.qqe_free(ptr, size);
    }
});

test.after(() => {
    console.log('QQE WASM batch and zero-copy tests completed');
});
