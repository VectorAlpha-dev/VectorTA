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

test('volume_weighted_rsi_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const volume = new Float64Array(testData.volume.slice(0, 256));
    const result = wasm.volume_weighted_rsi_js(close, volume, 14);

    assert.strictEqual(result.length, close.length);
    const first = result.findIndex(v => !isNaN(v));
    assert.strictEqual(first, 13);
    for (const value of result.slice(first)) {
        assert(value >= 0 && value <= 100, `value out of range: ${value}`);
    }
});

test('volume_weighted_rsi_js rejects invalid parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 32));
    const volume = new Float64Array(testData.volume.slice(0, 32));

    assert.throws(() => {
        wasm.volume_weighted_rsi_js(close, volume, 0);
    }, /Invalid period/);

    assert.throws(() => {
        wasm.volume_weighted_rsi_js(close, volume.slice(0, 16), 14);
    }, /Input length mismatch/);
});

test('volume_weighted_rsi_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 200));
    const volume = new Float64Array(testData.volume.slice(0, 200));
    const safe = wasm.volume_weighted_rsi_js(close, volume, 14);
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const closePtr = wasm.allocate_f64_array(close.length);
    const volumePtr = wasm.allocate_f64_array(volume.length);
    const outPtr = wasm.volume_weighted_rsi_alloc(close.length);

    try {
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        new Float64Array(memory.buffer, volumePtr, volume.length).set(volume);
        wasm.volume_weighted_rsi_into(closePtr, volumePtr, outPtr, close.length, 14);
        const out = Array.from(new Float64Array(memory.buffer, outPtr, close.length));
        assertArrayClose(out, safe, 1e-12, 'pointer values mismatch');
    } finally {
        wasm.volume_weighted_rsi_free(outPtr, close.length);
        wasm.deallocate_f64_array(closePtr);
        wasm.deallocate_f64_array(volumePtr);
    }
});

test('volume_weighted_rsi_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 200));
    const volume = new Float64Array(testData.volume.slice(0, 200));
    const batch = wasm.volume_weighted_rsi_batch_js(close, volume, {
        period_range: [14, 14, 0],
    });
    const single = wasm.volume_weighted_rsi_js(close, volume, 14);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.values.length, close.length);
    assert.strictEqual(batch.combos[0].period, 14);
    assertArrayClose(batch.values, single, 1e-12, 'batch values mismatch');
});

test('volume_weighted_rsi_batch_into metadata matches requested ranges', () => {
    const close = new Float64Array(testData.close.slice(0, 128));
    const volume = new Float64Array(testData.volume.slice(0, 128));
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 3;
    const total = rows * close.length;
    const closePtr = wasm.allocate_f64_array(close.length);
    const volumePtr = wasm.allocate_f64_array(volume.length);
    const outPtr = wasm.volume_weighted_rsi_alloc(total);

    try {
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        new Float64Array(memory.buffer, volumePtr, volume.length).set(volume);
        const actualRows = wasm.volume_weighted_rsi_batch_into(
            closePtr,
            volumePtr,
            outPtr,
            close.length,
            10,
            14,
            2,
        );
        assert.strictEqual(actualRows, rows);

        const values = Array.from(new Float64Array(memory.buffer, outPtr, total));
        const jsBatch = wasm.volume_weighted_rsi_batch_js(close, volume, {
            period_range: [10, 14, 2],
        });
        assertArrayClose(values, jsBatch.values, 1e-12, 'batch_into values mismatch');
    } finally {
        wasm.volume_weighted_rsi_free(outPtr, total);
        wasm.deallocate_f64_array(closePtr);
        wasm.deallocate_f64_array(volumePtr);
    }
});
