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

test('stochastic_money_flow_index_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const volume = new Float64Array(testData.volume.slice(0, 256));
    const result = wasm.stochastic_money_flow_index_js(close, volume, 14, 3, 3, 14);

    assert.strictEqual(result.k.length, close.length);
    assert.strictEqual(result.d.length, close.length);
    assert.strictEqual(result.k.findIndex(v => !isNaN(v)), 28);
    assert.strictEqual(result.d.findIndex(v => !isNaN(v)), 30);
});

test('stochastic_money_flow_index_js rejects invalid parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 64));
    const volume = new Float64Array(testData.volume.slice(0, 64));

    assert.throws(() => {
        wasm.stochastic_money_flow_index_js(close, volume, 0, 3, 3, 14);
    }, /Invalid stoch_k_length/);
});

test('stochastic_money_flow_index_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 220));
    const volume = new Float64Array(testData.volume.slice(0, 220));
    const safe = wasm.stochastic_money_flow_index_js(close, volume, 14, 3, 3, 14);
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const closePtr = wasm.allocate_f64_array(close.length);
    const volumePtr = wasm.allocate_f64_array(volume.length);
    const outPtr = wasm.stochastic_money_flow_index_alloc(close.length);

    try {
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        new Float64Array(memory.buffer, volumePtr, volume.length).set(volume);
        wasm.stochastic_money_flow_index_into(
            closePtr,
            volumePtr,
            outPtr,
            close.length,
            14,
            3,
            3,
            14,
        );
        const flat = new Float64Array(memory.buffer, outPtr, 2 * close.length);
        const k = Array.from(flat.slice(0, close.length));
        const d = Array.from(flat.slice(close.length));
        assertArrayClose(k, safe.k, 1e-12, 'pointer k mismatch');
        assertArrayClose(d, safe.d, 1e-12, 'pointer d mismatch');
    } finally {
        wasm.stochastic_money_flow_index_free(outPtr, close.length);
        wasm.deallocate_f64_array(closePtr);
        wasm.deallocate_f64_array(volumePtr);
    }
});

test('stochastic_money_flow_index_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 220));
    const volume = new Float64Array(testData.volume.slice(0, 220));
    const batch = wasm.stochastic_money_flow_index_batch_js(close, volume, {
        stoch_k_length_range: [14, 14, 0],
        stoch_k_smooth_range: [3, 3, 0],
        stoch_d_smooth_range: [3, 3, 0],
        mfi_length_range: [14, 14, 0],
    });
    const single = wasm.stochastic_money_flow_index_js(close, volume, 14, 3, 3, 14);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].stoch_k_length, 14);
    assertArrayClose(batch.k, single.k, 1e-12, 'batch k mismatch');
    assertArrayClose(batch.d, single.d, 1e-12, 'batch d mismatch');
});

test('stochastic_money_flow_index_batch_into metadata matches requested ranges', () => {
    const close = new Float64Array(testData.close.slice(0, 180));
    const volume = new Float64Array(testData.volume.slice(0, 180));
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 3;
    const total = rows * close.length;
    const closePtr = wasm.allocate_f64_array(close.length);
    const volumePtr = wasm.allocate_f64_array(volume.length);
    const outPtr = wasm.stochastic_money_flow_index_alloc(total);

    try {
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        new Float64Array(memory.buffer, volumePtr, volume.length).set(volume);
        const actualRows = wasm.stochastic_money_flow_index_batch_into(
            closePtr,
            volumePtr,
            outPtr,
            close.length,
            10,
            14,
            2,
            3,
            3,
            0,
            3,
            3,
            0,
            14,
            14,
            0,
        );
        assert.strictEqual(actualRows, rows);

        const flat = new Float64Array(memory.buffer, outPtr, 2 * total);
        const k = Array.from(flat.slice(0, total));
        const d = Array.from(flat.slice(total));
        const jsBatch = wasm.stochastic_money_flow_index_batch_js(close, volume, {
            stoch_k_length_range: [10, 14, 2],
            stoch_k_smooth_range: [3, 3, 0],
            stoch_d_smooth_range: [3, 3, 0],
            mfi_length_range: [14, 14, 0],
        });
        assertArrayClose(k, jsBatch.k, 1e-12, 'batch_into k mismatch');
        assertArrayClose(d, jsBatch.d, 1e-12, 'batch_into d mismatch');
    } finally {
        wasm.stochastic_money_flow_index_free(outPtr, total);
        wasm.deallocate_f64_array(closePtr);
        wasm.deallocate_f64_array(volumePtr);
    }
});
