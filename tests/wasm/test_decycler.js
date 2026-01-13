
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

test('Decycler partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.decycler_js(close, 125, 0.707);
    assert.strictEqual(result.length, close.length);
});

test('Decycler accuracy', async () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.decycler;

    const result = wasm.decycler_js(
        close,
        expected.defaultParams.hp_period,
        expected.defaultParams.k
    );

    assert.strictEqual(result.length, close.length);


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "Decycler last 5 values mismatch"
    );


    await compareWithRust('decycler', result, 'close', expected.defaultParams);
});

test('Decycler default candles', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.decycler_js(close, 125, 0.707);
    assert.strictEqual(result.length, close.length);
});

test('Decycler zero period', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.decycler_js(inputData, 0, 0.707);
    }, /invalid.*period/i);
});

test('Decycler period exceeds length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.decycler_js(dataSmall, 10, 0.707);
    }, /invalid period/i);
});

test('Decycler edge case k=0', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);

    assert.throws(() => {
        wasm.decycler_js(inputData, 3, 0.0);
    }, /invalid k/i);
});

test('Decycler NaN handling', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.decycler_js(close, 125, 0.707);
    assert.strictEqual(result.length, close.length);


    const firstNonNaN = result.findIndex(v => !isNaN(v));
    if (firstNonNaN !== -1 && result.length > firstNonNaN + 127) {
        for (let i = firstNonNaN + 127; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('Decycler warmup period', () => {

    const close = new Float64Array(testData.close);
    const hp_period = 125;

    const result = wasm.decycler_js(close, hp_period, 0.707);


    const firstNonNaN = result.findIndex(v => !isNaN(v));
    if (firstNonNaN !== -1) {


        const firstInput = close.findIndex(v => !isNaN(v));
        const expectedWarmup = firstInput + 2;

        assert.strictEqual(firstNonNaN, expectedWarmup,
            `Warmup period mismatch: expected ${expectedWarmup}, got ${firstNonNaN}`);


        for (let i = 0; i < expectedWarmup; i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
        }
    }
});

test('Decycler partial NaN input', () => {


    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1.0;
    }


    data[10] = NaN;
    data[11] = NaN;


    const result = wasm.decycler_js(data, 5, 0.707);
    assert.strictEqual(result.length, data.length);



});

test('Decycler edge case k values', () => {

    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1.0;
    }


    const resultSmall = wasm.decycler_js(data, 10, 0.001);
    assert.strictEqual(resultSmall.length, data.length);


    const resultDefault = wasm.decycler_js(data, 10, 0.707);
    assert.strictEqual(resultDefault.length, data.length);


    const resultLarge = wasm.decycler_js(data, 10, 10.0);
    assert.strictEqual(resultLarge.length, data.length);



    const warmupEnd = 2;
    const checkIdx = warmupEnd + 5;
    if (data.length > checkIdx) {
        assert(resultSmall[checkIdx] !== resultDefault[checkIdx],
            "Different k values should produce different results");
        assert(resultDefault[checkIdx] !== resultLarge[checkIdx],
            "Different k values should produce different results");
    }
});

test('Decycler reinput', () => {

    const close = new Float64Array(testData.close);
    const hp_period = 30;
    const k = 0.707;


    const firstResult = wasm.decycler_js(close, hp_period, k);
    assert.strictEqual(firstResult.length, close.length);


    const secondResult = wasm.decycler_js(firstResult, hp_period, k);
    assert.strictEqual(secondResult.length, firstResult.length);


    const warmupEnd = 4;
    if (secondResult.length > warmupEnd + 10) {

        let hasValidValues = false;
        for (let i = warmupEnd; i < warmupEnd + 10; i++) {
            if (!isNaN(secondResult[i])) {
                hasValidValues = true;
                break;
            }
        }
        assert(hasValidValues, "Expected valid values after double application");
    }
});

test('Decycler all NaN input', () => {

    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);

    assert.throws(() => {
        wasm.decycler_js(allNaN, 50, 0.707);
    }, /all values are nan/i);
});

test('Decycler batch single parameter set', () => {

    const close = new Float64Array(testData.close);


    const batchResult = wasm.decycler_batch(close, {
        hp_period_range: [125, 125, 0],
        k_range: [0.707, 0.707, 0]
    });


    const singleResult = wasm.decycler_js(close, 125, 0.707);

    assert.strictEqual(batchResult.values.length, singleResult.length);

    assertArrayClose(batchResult.values, singleResult, 5e-9, "Batch vs single mismatch");
});

test('Decycler batch multiple hp_periods', () => {

    const close = new Float64Array(testData.close.slice(0, 200));


    const batchResult = wasm.decycler_batch(close, {
        hp_period_range: [100, 150, 25],
        k_range: [0.707, 0.707, 0]
    });


    assert.strictEqual(batchResult.values.length, 3 * 200);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 200);


    const hp_periods = [100, 125, 150];
    for (let i = 0; i < hp_periods.length; i++) {
        const rowStart = i * 200;
        const rowEnd = rowStart + 200;
        const rowData = batchResult.values.slice(rowStart, rowEnd);

        const singleResult = wasm.decycler_js(close, hp_periods[i], 0.707);
        assertArrayClose(rowData, singleResult, 1e-10, `Row ${i} mismatch`);
    }
});

test('Decycler batch multiple parameters', () => {

    const close = new Float64Array(testData.close.slice(0, 50));


    const batchResult = wasm.decycler_batch(close, {
        hp_period_range: [10, 20, 10],
        k_range: [0.5, 0.7, 0.1]
    });


    assert.strictEqual(batchResult.values.length, 6 * 50);
    assert.strictEqual(batchResult.rows, 6);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.combos.length, 6);


    const expectedCombos = [
        { hp_period: 10, k: 0.5 },
        { hp_period: 10, k: 0.6 },
        { hp_period: 10, k: 0.7 },
        { hp_period: 20, k: 0.5 },
        { hp_period: 20, k: 0.6 },
        { hp_period: 20, k: 0.7 }
    ];

    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(batchResult.combos[i].hp_period, expectedCombos[i].hp_period,
            `hp_period mismatch at index ${i}`);
        assertClose(batchResult.combos[i].k, expectedCombos[i].k, 0.01,
            `k value mismatch at index ${i}`);
    }



    const rowIdx = 3;
    const rowStart = rowIdx * 50;
    const rowEnd = rowStart + 50;
    const rowData = batchResult.values.slice(rowStart, rowEnd);

    const singleResult = wasm.decycler_js(close, 20, 0.5);
    assertArrayClose(rowData, singleResult, 1e-10,
        `Batch row ${rowIdx} doesn't match single calculation`);
});

test('Decycler batch invalid parameters', () => {

    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);


    assert.throws(() => {
        wasm.decycler_batch(close, {
            hp_period_range: [20, 10, 5],
            k_range: [0.707, 0.707, 0]
        });
    }, /invalid.*period|empty.*grid|invalid.*range|not enough valid data/i);




    assert.throws(() => {
        wasm.decycler_js(close, 5, -0.5);
    }, /invalid k/i);
});

test('Decycler batch edge cases', () => {

    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);


    const singleBatch = wasm.decycler_batch(close, {
        hp_period_range: [5, 5, 1],
        k_range: [0.707, 0.707, 0.1]
    });

    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);


    const largeBatch = wasm.decycler_batch(close, {
        hp_period_range: [5, 7, 10],
        k_range: [0.707, 0.707, 0]
    });


    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);


    assert.throws(() => {
        wasm.decycler_batch(new Float64Array([]), {
            hp_period_range: [125, 125, 0],
            k_range: [0.707, 0.707, 0]
        });
    }, /empty input data|empty data/i);
});


test('Decycler zero-copy API basic', () => {

    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const hp_period = 5;
    const k = 0.707;


    const ptr = wasm.decycler_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');


    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );


    memView.set(data);


    try {
        wasm.decycler_into(ptr, ptr, data.length, hp_period, k);


        const regularResult = wasm.decycler_js(data, hp_period, k);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue;
            }
            assert.strictEqual(memView[i], regularResult[i],
                             `Value mismatch at index ${i}: ${memView[i]} vs ${regularResult[i]}`);
        }
    } finally {
        wasm.decycler_free(ptr, data.length);
    }
});

test('Decycler zero-copy separate buffers', () => {

    const data = new Float64Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);

    const inPtr = wasm.decycler_alloc(data.length);
    const outPtr = wasm.decycler_alloc(data.length);

    try {
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, data.length);


        inView.set(data);


        wasm.decycler_into(inPtr, outPtr, data.length, 3, 0.707);


        for (let i = 0; i < data.length; i++) {
            assert.strictEqual(inView[i], data[i], `Input modified at index ${i}`);
        }


        const regularResult = wasm.decycler_js(data, 3, 0.707);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(outView[i])) {
                continue;
            }
            assert.strictEqual(outView[i], regularResult[i],
                             `Output mismatch at index ${i}`);
        }
    } finally {
        wasm.decycler_free(inPtr, data.length);
        wasm.decycler_free(outPtr, data.length);
    }
});

test('Decycler zero-copy error handling', () => {



    assert.throws(() => {
        wasm.decycler_into(0, 0, 10, 125, 0.707);
    }, /null pointer|invalid memory/i);


    const ptr = wasm.decycler_alloc(10);
    try {

        assert.throws(() => {
            wasm.decycler_into(ptr, ptr, 10, 0, 0.707);
        }, /invalid.*period/i);


        assert.throws(() => {
            wasm.decycler_into(ptr, ptr, 10, 5, 0.0);
        }, /invalid k/i);
    } finally {
        wasm.decycler_free(ptr, 10);
    }
});


test('Decycler batch_into API', () => {

    const data = new Float64Array(testData.close.slice(0, 100));


    const expectedSize = 2 * 2 * 100;

    const inPtr = wasm.decycler_alloc(data.length);
    const outPtr = wasm.decycler_alloc(expectedSize);

    try {
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
        inView.set(data);


        const rows = wasm.decycler_batch_into(
            inPtr, outPtr, data.length,
            10, 20, 10,
            0.5, 0.7, 0.2
        );

        assert.strictEqual(rows, 4, 'Should have 4 parameter combinations');


        const regularBatch = wasm.decycler_batch(data, {
            hp_period_range: [10, 20, 10],
            k_range: [0.5, 0.7, 0.2]
        });

        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, expectedSize);
        assertArrayClose(
            Array.from(outView),
            regularBatch.values,
            1e-10,
            "Fast batch mismatch"
        );
    } finally {
        wasm.decycler_free(inPtr, data.length);
        wasm.decycler_free(outPtr, expectedSize);
    }
});


test('Decycler zero-copy memory management', () => {

    const sizes = [100, 1000, 10000, 100000];

    for (const size of sizes) {
        const ptr = wasm.decycler_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);


        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }


        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }


        wasm.decycler_free(ptr, size);
    }
});


test('Decycler SIMD kernel consistency', () => {

    const data = new Float64Array(testData.close.slice(0, 100));
    const hp_period = 20;
    const k = 0.707;


    const autoResult = wasm.decycler_js(data, hp_period, k);



    assert.strictEqual(autoResult.length, data.length);


    const firstNonNaN = autoResult.findIndex(v => !isNaN(v));
    assert(firstNonNaN >= 2, 'Warmup period should have at least 2 NaN values');


    for (let i = firstNonNaN; i < autoResult.length; i++) {
        assert(!isNaN(autoResult[i]), `Unexpected NaN at index ${i} after warmup`);
    }
});

test.after(() => {
    console.log('Decycler WASM tests completed');
});
