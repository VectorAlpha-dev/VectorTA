
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

test('HighPass2 partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.highpass_2_pole_js(close, 48, 0.707);
    assert.strictEqual(result.length, close.length);
});

test('HighPass2 accuracy', async () => {

    const close = new Float64Array(testData.close);

    const result = wasm.highpass_2_pole_js(close, 48, 0.707);

    assert.strictEqual(result.length, close.length);


    const expectedLastFive = [
        445.29073821108943,
        359.51467478973296,
        250.7236793408186,
        394.04381266217234,
        -52.65414073315134,
    ];


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-6,
        "HighPass2 last 5 values mismatch"
    );



    await compareWithRust('highpass_2_pole', result, 'close', { period: 48, k: 0.707 }, 1e-12);
});

test('HighPass2 default candles', async () => {

    const close = new Float64Array(testData.close);

    const result = wasm.highpass_2_pole_js(close, 48, 0.707);
    assert.strictEqual(result.length, close.length);



    await compareWithRust('highpass_2_pole', result, 'close', { period: 48, k: 0.707 }, 1e-12);
});

test('HighPass2 zero period', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.highpass_2_pole_js(inputData, 0, 0.707);
    });
});

test('HighPass2 period exceeds length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.highpass_2_pole_js(dataSmall, 10, 0.707);
    });
});

test('HighPass2 very small dataset', () => {

    const dataSingle = new Float64Array([42.0]);


    assert.throws(() => {
        wasm.highpass_2_pole_js(dataSingle, 2, 0.707);
    }, /Invalid period/);
});

test('HighPass2 empty input', () => {

    const dataEmpty = new Float64Array([]);

    assert.throws(() => {
        wasm.highpass_2_pole_js(dataEmpty, 48, 0.707);
    });
});

test('HighPass2 invalid k', () => {

    const data = new Float64Array([1.0, 2.0, 3.0]);


    assert.throws(() => {
        wasm.highpass_2_pole_js(data, 2, -0.5);
    });
});

test('HighPass2 all NaN', () => {

    const data = new Float64Array([NaN, NaN, NaN, NaN, NaN]);

    assert.throws(() => {
        wasm.highpass_2_pole_js(data, 3, 0.707);
    });
});

test('HighPass2 reinput', () => {

    const close = new Float64Array(testData.close);


    const firstResult = wasm.highpass_2_pole_js(close, 48, 0.707);


    const secondResult = wasm.highpass_2_pole_js(firstResult, 32, 0.707);

    assert.strictEqual(secondResult.length, firstResult.length);


    for (let i = 240; i < secondResult.length; i++) {
        assert(!isNaN(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('HighPass2 NaN handling', () => {

    const close = new Float64Array(testData.close);
    const period = 48;
    const k = 0.707;

    const result = wasm.highpass_2_pole_js(close, period, k);

    assert.strictEqual(result.length, close.length);



    for (let i = 0; i < result.length; i++) {
        assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
    }
});

test('HighPass2 batch', () => {

    const close = new Float64Array(testData.close);


    const period_start = 40;
    const period_end = 60;
    const period_step = 10;
    const k_start = 0.5;
    const k_end = 0.9;
    const k_step = 0.2;


    const config = {
        period_range: [period_start, period_end, period_step],
        k_range: [k_start, k_end, k_step]
    };

    const batch_output = wasm.highpass_2_pole_batch(close, config);
    const batch_result = batch_output.values;
    const metadata = batch_output.combos;


    assert.strictEqual(metadata.length, 9);


    assert.strictEqual(batch_result.length, 9 * close.length);


    let row_idx = 0;
    for (const period of [40, 50, 60]) {
        for (const k of [0.5, 0.7, 0.9]) {
            const individual_result = wasm.highpass_2_pole_js(close, period, k);


            const row_start = row_idx * close.length;
            const row = batch_result.slice(row_start, row_start + close.length);

            assertArrayClose(row, individual_result, 1e-9, `Period ${period}, k ${k}`);
            row_idx++;
        }
    }
});

test('HighPass2 different k values', () => {

    const close = new Float64Array(testData.close);
    const period = 48;


    for (const k of [0.1, 0.3, 0.5, 0.707, 0.9]) {
        const result = wasm.highpass_2_pole_js(close, period, k);
        assert.strictEqual(result.length, close.length);



        for (let i = 0; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for k=${k}`);
        }
    }
});

test('HighPass2 batch performance', () => {

    const close = new Float64Array(testData.close.slice(0, 1000));


    const startBatch = performance.now();
    const config = {
        period_range: [30, 70, 10],
        k_range: [0.3, 0.9, 0.2]
    };
    const batchOutput = wasm.highpass_2_pole_batch(close, config);
    const batchResult = batchOutput.values;
    const batchTime = performance.now() - startBatch;

    const startSingle = performance.now();
    const singleResults = [];
    for (let period = 30; period <= 70; period += 10) {
        for (let k = 0.3; k <= 0.9 + 1e-10; k += 0.2) {
            singleResults.push(...wasm.highpass_2_pole_js(close, period, k));
        }
    }
    const singleTime = performance.now() - startSingle;


    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);


    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});


test('HighPass2 zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const period = 5;
    const k = 0.707;


    if (!wasm.highpass_2_pole_alloc || !wasm.highpass_2_pole_into || !wasm.highpass_2_pole_free) {
        console.log('Zero-copy API not available for highpass_2_pole, skipping test');
        return;
    }


    const ptr = wasm.highpass_2_pole_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');


    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );


    memView.set(data);


    try {
        wasm.highpass_2_pole_into(ptr, ptr, data.length, period, k);


        const regularResult = wasm.highpass_2_pole_js(data, period, k);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue;
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {

        wasm.highpass_2_pole_free(ptr, data.length);
    }
});

test('HighPass2 zero-copy with large dataset', () => {

    if (!wasm.highpass_2_pole_alloc || !wasm.highpass_2_pole_into || !wasm.highpass_2_pole_free) {
        console.log('Zero-copy API not available for highpass_2_pole, skipping test');
        return;
    }

    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) * 100 + Math.random() * 10;
    }

    const ptr = wasm.highpass_2_pole_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');

    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);

        wasm.highpass_2_pole_into(ptr, ptr, size, 48, 0.707);


        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);




        for (let i = 0; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.highpass_2_pole_free(ptr, size);
    }
});


test('HighPass2 zero-copy error handling', () => {

    if (!wasm.highpass_2_pole_alloc || !wasm.highpass_2_pole_into || !wasm.highpass_2_pole_free) {
        console.log('Zero-copy API not available for highpass_2_pole, skipping test');
        return;
    }


    assert.throws(() => {
        wasm.highpass_2_pole_into(0, 0, 10, 48, 0.707);
    }, /null pointer|invalid memory/i);


    const ptr = wasm.highpass_2_pole_alloc(10);
    try {

        assert.throws(() => {
            wasm.highpass_2_pole_into(ptr, ptr, 10, 0, 0.707);
        }, /Invalid period/);


        assert.throws(() => {
            wasm.highpass_2_pole_into(ptr, ptr, 10, 5, 0.0);
        }, /Invalid k/);

        assert.throws(() => {
            wasm.highpass_2_pole_into(ptr, ptr, 10, 5, -1.0);
        }, /Invalid k/);
    } finally {
        wasm.highpass_2_pole_free(ptr, 10);
    }
});


test('HighPass2 zero-copy memory management', () => {

    if (!wasm.highpass_2_pole_alloc || !wasm.highpass_2_pole_free) {
        console.log('Zero-copy API not available for highpass_2_pole, skipping test');
        return;
    }


    const sizes = [100, 1000, 5000];

    for (const size of sizes) {
        const ptr = wasm.highpass_2_pole_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);


        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 2.5;
        }


        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 2.5, `Memory corruption at index ${i}`);
        }


        wasm.highpass_2_pole_free(ptr, size);
    }
});


test('HighPass2 SIMD128 consistency', () => {


    const testCases = [
        { size: 20, period: 5 },
        { size: 100, period: 10 },
        { size: 500, period: 48 },
        { size: 1000, period: 100 }
    ];

    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) * Math.exp(-i * 0.001) + Math.cos(i * 0.05) * 10;
        }

        const result = wasm.highpass_2_pole_js(data, testCase.period, 0.707);


        assert.strictEqual(result.length, data.length);






        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.period - 1; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }


        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(Math.abs(avgAfterWarmup) < 100, `Average value ${avgAfterWarmup} seems unreasonable`);
    }
});


test('HighPass2 batch - new ergonomic API with single parameter', () => {
    const close = new Float64Array(testData.close.slice(0, 100));

    const result = wasm.highpass_2_pole_batch(close, {
        period_range: [48, 48, 0],
        k_range: [0.707, 0.707, 0]
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
    assert.strictEqual(combo.period, 48);
    assert.strictEqual(combo.k, 0.707);


    const singleResult = wasm.highpass_2_pole_js(close, 48, 0.707);
    for (let i = 0; i < singleResult.length; i++) {
        if (isNaN(singleResult[i]) && isNaN(result.values[i])) {
            continue;
        }
        assert(Math.abs(singleResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('HighPass2 batch - edge cases', () => {
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);


    const singleBatch = wasm.highpass_2_pole_batch(close, {
        period_range: [5, 5, 1],
        k_range: [0.5, 0.5, 0.1]
    });

    assert.strictEqual(singleBatch.values.length, 12);
    assert.strictEqual(singleBatch.combos.length, 1);


    const largeBatch = wasm.highpass_2_pole_batch(close, {
        period_range: [5, 7, 10],
        k_range: [0.707, 0.707, 0]
    });


    assert.strictEqual(largeBatch.values.length, 12);
    assert.strictEqual(largeBatch.combos.length, 1);
    assert.strictEqual(largeBatch.combos[0].period, 5);


    assert.throws(() => {
        wasm.highpass_2_pole_batch(new Float64Array([]), {
            period_range: [48, 48, 0],
            k_range: [0.707, 0.707, 0]
        });
    }, /All values are NaN|Empty/);
});

test('HighPass2 first value handling', () => {

    const close = new Float64Array(testData.close.slice(0, 100));


    const closeWithNaN = new Float64Array(103);
    closeWithNaN[0] = NaN;
    closeWithNaN[1] = NaN;
    closeWithNaN[2] = NaN;
    closeWithNaN.set(close, 3);

    const result = wasm.highpass_2_pole_js(closeWithNaN, 48, 0.707);

    assert.strictEqual(result.length, closeWithNaN.length);


    assert(isNaN(result[0]));
    assert(isNaN(result[1]));
    assert(isNaN(result[2]));


    let firstValid = null;
    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i])) {
            firstValid = i;
            break;
        }
    }



    assert.strictEqual(firstValid, 3, `First valid at ${firstValid}, expected 3`);
});




