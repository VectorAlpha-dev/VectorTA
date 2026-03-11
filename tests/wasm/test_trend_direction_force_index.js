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

test('trend_direction_force_index_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.trend_direction_force_index_js(close, 10);

    assert.strictEqual(result.length, close.length);
    assert.strictEqual(result.findIndex(v => !isNaN(v)), 9);
});

test('trend_direction_force_index_js rejects invalid parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 64));

    assert.throws(() => {
        wasm.trend_direction_force_index_js(close, 0);
    }, /Invalid length/);
});

test('trend_direction_force_index_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 200));
    const safe = wasm.trend_direction_force_index_js(close, 12);
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.trend_direction_force_index_alloc(close.length);

    try {
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        wasm.trend_direction_force_index_into(closePtr, outPtr, close.length, 12);
        const actual = Array.from(new Float64Array(memory.buffer, outPtr, close.length));
        assertArrayClose(actual, safe, 1e-12, 'pointer mismatch');
    } finally {
        wasm.trend_direction_force_index_free(outPtr, close.length);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('trend_direction_force_index_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.trend_direction_force_index_batch_js(close, {
        length_range: [12, 12, 0],
    });
    const single = wasm.trend_direction_force_index_js(close, 12);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].length, 12);
    assertArrayClose(batch.values, single, 1e-12, 'batch mismatch');
});

test('trend_direction_force_index_batch_into metadata matches requested ranges', () => {
    const close = new Float64Array(testData.close.slice(0, 128));
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 3;
    const total = rows * close.length;
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.trend_direction_force_index_alloc(total);

    try {
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        const actualRows = wasm.trend_direction_force_index_batch_into(
            closePtr,
            outPtr,
            close.length,
            8,
            12,
            2,
        );
        assert.strictEqual(actualRows, rows);

        const flat = Array.from(new Float64Array(memory.buffer, outPtr, total));
        const jsBatch = wasm.trend_direction_force_index_batch_js(close, {
            length_range: [8, 12, 2],
        });
        assertArrayClose(flat, jsBatch.values, 1e-12, 'batch_into mismatch');
    } finally {
        wasm.trend_direction_force_index_free(outPtr, total);
        wasm.deallocate_f64_array(closePtr);
    }
});
