
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

test('APO partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.apo_js(close, 10, 20);
    assert.strictEqual(result.length, close.length);
});

test('APO accuracy', async () => {

    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.apo;

    const result = wasm.apo_js(
        close,
        expected.defaultParams.short_period,
        expected.defaultParams.long_period
    );

    assert.strictEqual(result.length, close.length);


    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-1,
        "APO last 5 values mismatch"
    );


    await compareWithRust('apo', result, 'single', expected.defaultParams);
});

test('APO default params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.apo_js(close, 10, 20);
    assert.strictEqual(result.length, close.length);
});

test('APO zero period', () => {

    const data = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.apo_js(data, 0, 20);
    }, /Invalid period/);
});

test('APO period invalid', () => {

    const data = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.apo_js(data, 20, 10);
    }, /short_period not less than long_period/);

    assert.throws(() => {
        wasm.apo_js(data, 10, 10);
    }, /short_period not less than long_period/);
});

test('APO very small dataset', () => {

    const single_point = new Float64Array([42.0]);

    assert.throws(() => {
        wasm.apo_js(single_point, 9, 10);
    }, /Not enough valid data/);
});

test('APO reinput', () => {

    const close = new Float64Array(testData.close);


    const firstResult = wasm.apo_js(close, 10, 20);
    assert.strictEqual(firstResult.length, close.length);


    const secondResult = wasm.apo_js(
        new Float64Array(firstResult),
        5,
        15
    );
    assert.strictEqual(secondResult.length, firstResult.length);
});

test('APO NaN handling', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.apo_js(close, 10, 20);
    assert.strictEqual(result.length, close.length);


    if (result.length > 30) {
        for (let i = 30; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('APO all NaN input', () => {

    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);

    assert.throws(() => {
        wasm.apo_js(allNaN, 10, 20);
    }, /All values are NaN/);
});

test('APO batch single parameter set', () => {

    const close = new Float64Array(testData.close);


    const batchResult = wasm.apo_batch_js(
        close,
        10, 10, 0,
        20, 20, 0
    );


    const singleResult = wasm.apo_js(close, 10, 20);

    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('APO batch multiple parameters', () => {

    const close = new Float64Array(testData.close.slice(0, 100));


    const batchResult = wasm.apo_batch_js(
        close,
        5, 15, 5,
        20, 30, 10
    );


    const expectedCombos = 6;
    assert.strictEqual(batchResult.length, expectedCombos * 100);


    const firstRow = batchResult.slice(0, 100);
    const singleResult = wasm.apo_js(close, 5, 20);
    assertArrayClose(firstRow, singleResult, 1e-10, "First combination mismatch");
});

test('APO batch metadata', () => {

    const metadata = wasm.apo_batch_metadata_js(
        5, 15, 5,
        20, 30, 10
    );


    assert.strictEqual(metadata.length, 12);


    const expected = [
        5, 20,
        5, 30,
        10, 20,
        10, 30,
        15, 20,
        15, 30
    ];

    for (let i = 0; i < expected.length; i++) {
        assert.strictEqual(metadata[i], expected[i], `Metadata mismatch at index ${i}`);
    }
});

test('APO batch invalid combinations', () => {

    const close = new Float64Array(testData.close.slice(0, 50));


    const batchResult = wasm.apo_batch_js(
        close,
        10, 30, 10,
        15, 25, 10
    );


    assert.strictEqual(batchResult.length, 3 * 50);
});

test('APO batch empty result', () => {

    const close = new Float64Array([1, 2, 3, 4, 5]);


    assert.throws(() => {
        wasm.apo_batch_js(
            close,
            20, 30, 10,
            10, 15, 5
        );
    }, /Invalid period|invalid range/);
});


test('APO batch - new ergonomic API with single parameter', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.apo_batch(close, {
        short_period_range: [10, 10, 0],
        long_period_range: [20, 20, 0]
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
    assert.strictEqual(combo.short_period, 10);
    assert.strictEqual(combo.long_period, 20);


    const oldResult = wasm.apo_js(close, 10, 20);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue;
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('APO batch - new API with multiple parameters', () => {

    const close = new Float64Array(testData.close.slice(0, 50));

    const result = wasm.apo_batch(close, {
        short_period_range: [5, 15, 5],
        long_period_range: [20, 30, 10]
    });


    assert.strictEqual(result.rows, 6);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 6);
    assert.strictEqual(result.values.length, 300);


    const expectedCombos = [
        [5, 20], [5, 30],
        [10, 20], [10, 30],
        [15, 20], [15, 30]
    ];

    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].short_period, expectedCombos[i][0]);
        assert.strictEqual(result.combos[i].long_period, expectedCombos[i][1]);
    }


    const secondRow = result.values.slice(result.cols, 2 * result.cols);
    assert.strictEqual(secondRow.length, 50);


    const oldResult = wasm.apo_js(close, 5, 20);
    const firstRow = result.values.slice(0, result.cols);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
            continue;
        }
        assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('APO batch - new API matches old API results', () => {

    const close = new Float64Array(testData.close.slice(0, 100));

    const params = {
        short_period_range: [5, 15, 5],
        long_period_range: [20, 30, 5]
    };


    const oldValues = wasm.apo_batch_js(
        close,
        params.short_period_range[0], params.short_period_range[1], params.short_period_range[2],
        params.long_period_range[0], params.long_period_range[1], params.long_period_range[2]
    );


    const newResult = wasm.apo_batch(close, params);


    assert.strictEqual(oldValues.length, newResult.values.length);

    for (let i = 0; i < oldValues.length; i++) {
        if (isNaN(oldValues[i]) && isNaN(newResult.values[i])) {
            continue;
        }
        assert(Math.abs(oldValues[i] - newResult.values[i]) < 1e-10,
               `Value mismatch at index ${i}: old=${oldValues[i]}, new=${newResult.values[i]}`);
    }
});

test('APO batch - new API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));


    assert.throws(() => {
        wasm.apo_batch(close, {
            short_period_range: [10, 10]
        });
    }, /Invalid config/);


    assert.throws(() => {
        wasm.apo_batch(close, {
            short_period_range: [5, 15, 5]

        });
    }, /Invalid config/);


    assert.throws(() => {
        wasm.apo_batch(close, {
            short_period_range: "invalid",
            long_period_range: [20, 30, 5]
        });
    }, /Invalid config/);
});

test('APO edge cases', () => {

    const minData = new Float64Array(20).fill(1.0);
    const result = wasm.apo_js(minData, 10, 20);
    assert.strictEqual(result.length, minData.length);


    assert.strictEqual(result[0], 0.0);


    assert(Math.abs(result[result.length - 1]) < 1e-10, "APO should converge to 0 with constant prices");
});


test('APO fast API - basic operation', () => {
    const data = new Float64Array(testData.close);
    const len = data.length;


    const inPtr = wasm.apo_alloc(len);
    const outPtr = wasm.apo_alloc(len);
    assert(inPtr !== 0, 'Failed to allocate input memory');
    assert(outPtr !== 0, 'Failed to allocate output memory');

    try {

        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(data);


        wasm.apo_into(inPtr, outPtr, len, 10, 20);


        const resultView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const result = Array.from(resultView);


        const safeResult = wasm.apo_js(data, 10, 20);
        assertArrayClose(result, safeResult, 1e-10, "Fast API vs safe API mismatch");
    } finally {

        wasm.apo_free(inPtr, len);
        wasm.apo_free(outPtr, len);
    }
});

test('APO fast API - in-place operation (aliasing)', () => {
    const data = new Float64Array(testData.close);
    const len = data.length;


    const originalData = new Float64Array(data);


    const ptr = wasm.apo_alloc(len);
    assert(ptr !== 0, 'Failed to allocate memory');

    try {

        const wasmData = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        wasmData.set(data);


        wasm.apo_into(ptr, ptr, len, 10, 20);


        const resultView = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        const result = Array.from(resultView);


        const expectedResult = wasm.apo_js(originalData, 10, 20);
        assertArrayClose(result, expectedResult, 1e-10, "In-place operation mismatch");
    } finally {
        wasm.apo_free(ptr, len);
    }
});

test('APO fast API - error handling', () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    const len = data.length;


    assert.throws(() => {
        wasm.apo_into(0, 0, len, 10, 20);
    }, /Null pointer/);


    const inPtr = wasm.apo_alloc(len);
    const outPtr = wasm.apo_alloc(len);

    try {

        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(data);

        assert.throws(() => {
            wasm.apo_into(inPtr, outPtr, len, 0, 20);
        }, /Invalid period/);

        assert.throws(() => {
            wasm.apo_into(inPtr, outPtr, len, 20, 10);
        }, /short_period not less than long_period/);
    } finally {
        wasm.apo_free(inPtr, len);
        wasm.apo_free(outPtr, len);
    }
});

test('APO fast API - memory management', () => {
    const sizes = [100, 1000, 10000];

    sizes.forEach(size => {
        const ptr = wasm.apo_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);


        const mem = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        mem[0] = 42.0;
        mem[size - 1] = 99.0;
        assert.strictEqual(mem[0], 42.0);
        assert.strictEqual(mem[size - 1], 99.0);


        wasm.apo_free(ptr, size);
    });
});

test.after(() => {
    console.log('APO WASM tests completed');
});
