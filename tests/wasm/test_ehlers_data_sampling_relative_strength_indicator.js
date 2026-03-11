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

test('ehlers_data_sampling_relative_strength_indicator_js output contract', () => {
    const open = new Float64Array(testData.open.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.ehlers_data_sampling_relative_strength_indicator_js(open, close, 14);

    assert.strictEqual(result.ds_rsi.length, close.length);
    assert.strictEqual(result.original_rsi.length, close.length);
    assert.strictEqual(result.signal.length, close.length);
    assert.strictEqual(result.ds_rsi.findIndex(v => !isNaN(v)), 14);
    assert.strictEqual(result.original_rsi.findIndex(v => !isNaN(v)), 14);
    assert.strictEqual(result.signal.findIndex(v => !isNaN(v)), 14);
});

test('ehlers_data_sampling_relative_strength_indicator_js rejects invalid parameters', () => {
    const open = new Float64Array(testData.open.slice(0, 64));
    const close = new Float64Array(testData.close.slice(0, 64));
    assert.throws(() => {
        wasm.ehlers_data_sampling_relative_strength_indicator_js(open, close, 0);
    }, /Invalid length/);
});

test('ehlers_data_sampling_relative_strength_indicator_into pointer path matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 220));
    const close = new Float64Array(testData.close.slice(0, 220));
    const safe = wasm.ehlers_data_sampling_relative_strength_indicator_js(open, close, 14);
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const openPtr = wasm.allocate_f64_array(open.length);
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.ehlers_data_sampling_relative_strength_indicator_alloc(close.length);

    try {
        new Float64Array(memory.buffer, openPtr, open.length).set(open);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        wasm.ehlers_data_sampling_relative_strength_indicator_into(
            openPtr,
            closePtr,
            outPtr,
            close.length,
            14,
        );
        const flat = new Float64Array(memory.buffer, outPtr, 3 * close.length);
        const dsRsi = Array.from(flat.slice(0, close.length));
        const originalRsi = Array.from(flat.slice(close.length, 2 * close.length));
        const signal = Array.from(flat.slice(2 * close.length));
        assertArrayClose(dsRsi, safe.ds_rsi, 1e-12, 'pointer ds_rsi mismatch');
        assertArrayClose(originalRsi, safe.original_rsi, 1e-12, 'pointer original_rsi mismatch');
        assertArrayClose(signal, safe.signal, 1e-12, 'pointer signal mismatch');
    } finally {
        wasm.ehlers_data_sampling_relative_strength_indicator_free(outPtr, close.length);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('ehlers_data_sampling_relative_strength_indicator_batch_js single parameter set matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 220));
    const close = new Float64Array(testData.close.slice(0, 220));
    const batch = wasm.ehlers_data_sampling_relative_strength_indicator_batch_js(open, close, {
        length_range: [14, 14, 0],
    });
    const single = wasm.ehlers_data_sampling_relative_strength_indicator_js(open, close, 14);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].length, 14);
    assertArrayClose(batch.ds_rsi, single.ds_rsi, 1e-12, 'batch ds_rsi mismatch');
    assertArrayClose(batch.original_rsi, single.original_rsi, 1e-12, 'batch original_rsi mismatch');
    assertArrayClose(batch.signal, single.signal, 1e-12, 'batch signal mismatch');
});

test('ehlers_data_sampling_relative_strength_indicator_batch_into metadata matches requested ranges', () => {
    const open = new Float64Array(testData.open.slice(0, 180));
    const close = new Float64Array(testData.close.slice(0, 180));
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 3;
    const total = rows * close.length;
    const openPtr = wasm.allocate_f64_array(open.length);
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.ehlers_data_sampling_relative_strength_indicator_alloc(total);

    try {
        new Float64Array(memory.buffer, openPtr, open.length).set(open);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        const actualRows = wasm.ehlers_data_sampling_relative_strength_indicator_batch_into(
            openPtr,
            closePtr,
            outPtr,
            close.length,
            10,
            14,
            2,
        );
        assert.strictEqual(actualRows, rows);

        const flat = new Float64Array(memory.buffer, outPtr, 3 * total);
        const dsRsi = Array.from(flat.slice(0, total));
        const originalRsi = Array.from(flat.slice(total, 2 * total));
        const signal = Array.from(flat.slice(2 * total));
        const jsBatch = wasm.ehlers_data_sampling_relative_strength_indicator_batch_js(open, close, {
            length_range: [10, 14, 2],
        });
        assertArrayClose(dsRsi, jsBatch.ds_rsi, 1e-12, 'batch_into ds_rsi mismatch');
        assertArrayClose(originalRsi, jsBatch.original_rsi, 1e-12, 'batch_into original_rsi mismatch');
        assertArrayClose(signal, jsBatch.signal, 1e-12, 'batch_into signal mismatch');
    } finally {
        wasm.ehlers_data_sampling_relative_strength_indicator_free(outPtr, total);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});
