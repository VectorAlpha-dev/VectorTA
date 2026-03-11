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

function getMemory() {
    return wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
}

test('n_order_ema_js default output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.n_order_ema_js(close, 9.0, 1, 'ema', 'impulse_matched');

    assert.strictEqual(result.length, close.length);
    assert.strictEqual(result.findIndex(v => !isNaN(v)), 8);
});

test('n_order_ema_js rejects invalid parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 64));

    assert.throws(() => wasm.n_order_ema_js(close, 0.0, 1, 'ema', 'impulse_matched'), /Invalid period/);
    assert.throws(() => wasm.n_order_ema_js(close, 9.0, 0, 'ema', 'impulse_matched'), /Invalid order/);
    assert.throws(() => wasm.n_order_ema_js(close, 9.0, 1, 'bad', 'impulse_matched'), /Invalid ema_style/);
    assert.throws(() => wasm.n_order_ema_js(close, 9.0, 1, 'ema', 'bad'), /Invalid iir_style/);
});

test('n_order_ema_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 220));
    const safe = wasm.n_order_ema_js(close, 12.5, 3, 'tema', 'matched_z');
    const memory = getMemory();
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const inPtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.n_order_ema_alloc(close.length);
    try {
        new Float64Array(memory.buffer, inPtr, close.length).set(close);
        wasm.n_order_ema_into(inPtr, outPtr, close.length, 12.5, 3, 'tema', 'matched_z');
        const actual = Array.from(new Float64Array(memory.buffer, outPtr, close.length));
        assertArrayClose(actual, safe, 1e-12, 'pointer mismatch');
    } finally {
        wasm.n_order_ema_free(outPtr, close.length);
        wasm.deallocate_f64_array(inPtr);
    }
});

test('n_order_ema_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.n_order_ema_batch_js(close, {
        period_range: [15.0, 15.0, 0.0],
        order_range: [2, 2, 0],
        ema_style: 'hema',
        iir_style: 'bilinear',
    });
    const single = wasm.n_order_ema_js(close, 15.0, 2, 'hema', 'bilinear');

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].period, 15.0);
    assert.strictEqual(batch.combos[0].order, 2);
    assertArrayClose(batch.values, single, 1e-12, 'batch mismatch');
});

test('n_order_ema_batch_into metadata matches requested ranges', () => {
    const close = new Float64Array(testData.close.slice(0, 160));
    const rows = 4;
    const total = rows * close.length;
    const memory = getMemory();
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const inPtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.n_order_ema_alloc(total);
    try {
        new Float64Array(memory.buffer, inPtr, close.length).set(close);
        const actualRows = wasm.n_order_ema_batch_into(
            inPtr,
            outPtr,
            close.length,
            9.0,
            10.0,
            1.0,
            1,
            2,
            1,
            'dema',
            'all_pole',
        );
        assert.strictEqual(actualRows, rows);

        const flat = Array.from(new Float64Array(memory.buffer, outPtr, total));
        const jsBatch = wasm.n_order_ema_batch_js(close, {
            period_range: [9.0, 10.0, 1.0],
            order_range: [1, 2, 1],
            ema_style: 'dema',
            iir_style: 'all_pole',
        });
        assertArrayClose(flat, jsBatch.values, 1e-12, 'batch_into mismatch');
    } finally {
        wasm.n_order_ema_free(outPtr, total);
        wasm.deallocate_f64_array(inPtr);
    }
});
