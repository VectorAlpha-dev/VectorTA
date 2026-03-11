import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('disparity_index_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.disparity_index_js(close, 14, 14, 9, 'ema');

    assert.strictEqual(result.length, close.length);
    const first = result.findIndex(v => !isNaN(v));
    assert.strictEqual(first, 34);
    for (const value of result.slice(first)) {
        assert(value >= 0 && value <= 100, `value out of range: ${value}`);
    }
});

test('disparity_index_js rejects invalid parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 64));

    assert.throws(() => {
        wasm.disparity_index_js(close, 0, 14, 9, 'ema');
    }, /Invalid ema_period/);

    assert.throws(() => {
        wasm.disparity_index_js(close, 14, 14, 9, 'bad');
    }, /Invalid smoothing_type/);
});

test('disparity_index_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 200));
    const safe = wasm.disparity_index_js(close, 14, 14, 9, 'ema');
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.disparity_index_alloc(close.length);

    try {
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        wasm.disparity_index_into(closePtr, outPtr, close.length, 14, 14, 9, 0);
        const out = Array.from(new Float64Array(memory.buffer, outPtr, close.length));
        assertArrayClose(out, safe, 1e-12, 'pointer values mismatch');
    } finally {
        wasm.disparity_index_free(outPtr, close.length);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('disparity_index_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.disparity_index_batch_js(close, {
        ema_period_range: [14, 14, 0],
        lookback_period_range: [14, 14, 0],
        smoothing_period_range: [9, 9, 0],
        smoothing_types: ['ema'],
    });
    const single = wasm.disparity_index_js(close, 14, 14, 9, 'ema');

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.values.length, close.length);
    assert.strictEqual(batch.combos[0].ema_period, 14);
    assert.strictEqual(batch.combos[0].smoothing_type, 'ema');
    assertArrayClose(batch.values, single, 1e-12, 'batch values mismatch');
});

test('disparity_index_batch_into metadata matches requested ranges', () => {
    const close = new Float64Array(testData.close.slice(0, 128));
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 3;
    const total = rows * close.length;
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.disparity_index_alloc(total);

    try {
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        const actualRows = wasm.disparity_index_batch_into(
            closePtr,
            outPtr,
            close.length,
            10,
            14,
            2,
            14,
            14,
            0,
            9,
            9,
            0,
            0,
            0,
            0,
        );
        assert.strictEqual(actualRows, rows);

        const values = Array.from(new Float64Array(memory.buffer, outPtr, total));
        const jsBatch = wasm.disparity_index_batch_js(close, {
            ema_period_range: [10, 14, 2],
            lookback_period_range: [14, 14, 0],
            smoothing_period_range: [9, 9, 0],
            smoothing_types: ['ema'],
        });
        assertArrayClose(values, jsBatch.values, 1e-12, 'batch_into values mismatch');
    } finally {
        wasm.disparity_index_free(outPtr, total);
        wasm.deallocate_f64_array(closePtr);
    }
});
