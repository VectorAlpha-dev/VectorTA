
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
    assertNoNaN
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

test('DPO partial params', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.dpo_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('DPO accuracy', async () => {

    const close = new Float64Array(testData.close);

    const result = wasm.dpo_js(close, 5);
    assert.strictEqual(result.length, close.length);


    const expectedLastFive = [
        65.3999999999287,
        131.3999999999287,
        32.599999999925785,
        98.3999999999287,
        117.99999999992724,
    ];

    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        0.1,
        "DPO last 5 values mismatch"
    );


    await compareWithRust('dpo', result, 'close', { period: 5 });
});

test('DPO default candles', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.dpo_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('DPO zero period', () => {

    const inputData = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.dpo_js(inputData, 0);
    }, /Invalid period/);
});

test('DPO period exceeds length', () => {

    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);

    assert.throws(() => {
        wasm.dpo_js(dataSmall, 10);
    }, /Invalid period/);
});

test('DPO very small dataset', () => {

    const singlePoint = new Float64Array([42.0]);

    assert.throws(() => {
        wasm.dpo_js(singlePoint, 5);
    }, /Invalid period|Not enough valid data/);
});

test('DPO nan handling', () => {

    const close = new Float64Array(testData.close);

    const result = wasm.dpo_js(close, 5);
    assert.strictEqual(result.length, close.length);



    if (result.length > 20) {
        for (let i = 20; i < result.length; i++) {
            assert.ok(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('DPO fast API (in-place)', () => {

    const close = new Float64Array(testData.close);
    const period = 5;


    const outputPtr = wasm.dpo_alloc(close.length);

    try {

        const outputInitial = new Float64Array(wasm.__wasm.memory.buffer, outputPtr, close.length);
        outputInitial.set(close);


        wasm.dpo_into(outputPtr, outputPtr, close.length, period);


        const expected = wasm.dpo_js(close, period);


        const outputResult = new Float64Array(wasm.__wasm.memory.buffer, outputPtr, close.length);

        assertArrayClose(outputResult, expected, 1e-10, "Fast API in-place mismatch");
    } finally {

        wasm.dpo_free(outputPtr, close.length);
    }
});

test('DPO fast API (separate buffers)', () => {

    const close = new Float64Array(testData.close);
    const period = 5;


    const inputPtr = wasm.dpo_alloc(close.length);
    const outputPtr = wasm.dpo_alloc(close.length);

    try {

        const input = new Float64Array(wasm.__wasm.memory.buffer, inputPtr, close.length);
        input.set(close);


        wasm.dpo_into(inputPtr, outputPtr, close.length, period);


        const expected = wasm.dpo_js(close, period);


        const outputResult = new Float64Array(wasm.__wasm.memory.buffer, outputPtr, close.length);

        assertArrayClose(outputResult, expected, 1e-10, "Fast API separate buffers mismatch");
    } finally {

        wasm.dpo_free(inputPtr, close.length);
        wasm.dpo_free(outputPtr, close.length);
    }
});

test('DPO batch single parameter', () => {

    const close = new Float64Array(testData.close);

    const config = {
        period_range: [5, 5, 0]
    };

    const batchResult = wasm.dpo_batch(close, config);


    const singleResult = wasm.dpo_js(close, 5);

    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, close.length);
    assert.strictEqual(batchResult.values.length, close.length);

    assertArrayClose(
        batchResult.values,
        singleResult,
        1e-8,
        "Batch vs single mismatch"
    );
});

test('DPO batch multiple periods', () => {

    const close = new Float64Array(testData.close.slice(0, 200));

    const config = {
        period_range: [5, 15, 5]
    };

    const batchResult = wasm.dpo_batch(close, config);


    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 200);
    assert.strictEqual(batchResult.values.length, 3 * 200);


    const periods = [5, 10, 15];
    for (let i = 0; i < periods.length; i++) {
        const period = periods[i];
        const rowStart = i * 200;
        const rowEnd = (i + 1) * 200;
        const row = batchResult.values.slice(rowStart, rowEnd);

        const expected = wasm.dpo_js(close, period);
        assertArrayClose(
            row,
            expected,
            1e-10,
            `Batch row ${i} (period=${period}) mismatch`
        );
    }
});

test('DPO all NaN input', () => {

    const allNaN = new Float64Array(100).fill(NaN);

    assert.throws(() => {
        wasm.dpo_js(allNaN, 5);
    }, /All values are NaN/);
});

test('DPO batch into (fast batch API)', () => {

    const close = new Float64Array(testData.close.slice(0, 100));
    const periodStart = 5;
    const periodEnd = 20;
    const periodStep = 5;

    const expectedRows = 4;
    const cols = close.length;


    const inputPtr = wasm.dpo_alloc(close.length);
    const outputPtr = wasm.dpo_alloc(expectedRows * cols);

    try {

        const input = new Float64Array(wasm.__wasm.memory.buffer, inputPtr, close.length);
        input.set(close);


        const rows = wasm.dpo_batch_into(
            inputPtr,
            outputPtr,
            close.length,
            periodStart,
            periodEnd,
            periodStep
        );

        assert.strictEqual(rows, expectedRows);


        const output = new Float64Array(wasm.__wasm.memory.buffer, outputPtr, rows * cols);


        const config = {
            period_range: [periodStart, periodEnd, periodStep]
        };
        const expected = wasm.dpo_batch(close, config);

        assertArrayClose(
            output,
            expected.values,
            1e-10,
            "Batch into API mismatch"
        );
    } finally {

        wasm.dpo_free(inputPtr, close.length);
        wasm.dpo_free(outputPtr, expectedRows * cols);
    }
});
