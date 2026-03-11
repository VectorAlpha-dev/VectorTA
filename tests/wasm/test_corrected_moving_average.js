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

test('corrected_moving_average_js output contract', () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    const result = wasm.corrected_moving_average_js(data, 3);
    assert.strictEqual(result.length, data.length);
    assert(isNaN(result[0]));
    assert(isNaN(result[1]));
    assert(Math.abs(result[2] - 2.0) <= 1e-12);
});

test('corrected_moving_average_js rejects invalid params', () => {
    const close = new Float64Array(testData.close.slice(0, 32));
    assert.throws(() => {
        wasm.corrected_moving_average_js(close, 0);
    }, /Invalid period/);
});

test('corrected_moving_average_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 220));
    const safe = wasm.corrected_moving_average_js(close, 17);
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const inPtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.corrected_moving_average_alloc(close.length);

    try {
        new Float64Array(memory.buffer, inPtr, close.length).set(close);
        wasm.corrected_moving_average_into(inPtr, outPtr, close.length, 17);
        const out = Array.from(new Float64Array(memory.buffer, outPtr, close.length));
        assertArrayClose(out, safe, 1e-12, 'pointer output mismatch');
    } finally {
        wasm.corrected_moving_average_free(outPtr, close.length);
        wasm.deallocate_f64_array(inPtr);
    }
});

test('corrected_moving_average_batch_js single parameter matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 220));
    const batch = wasm.corrected_moving_average_batch_js(close, {
        period_range: [17, 17, 0],
    });
    const single = wasm.corrected_moving_average_js(close, 17);
    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].period, 17);
    assertArrayClose(batch.values, single, 1e-12, 'batch output mismatch');
});

test('corrected_moving_average_batch_into matches safe batch output', () => {
    const close = new Float64Array(testData.close.slice(0, 180));
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 3;
    const total = rows * close.length;
    const inPtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.corrected_moving_average_alloc(total);

    try {
        new Float64Array(memory.buffer, inPtr, close.length).set(close);
        const actualRows = wasm.corrected_moving_average_batch_into(
            inPtr,
            outPtr,
            close.length,
            10,
            14,
            2,
        );
        assert.strictEqual(actualRows, rows);
        const flat = Array.from(new Float64Array(memory.buffer, outPtr, total));
        const jsBatch = wasm.corrected_moving_average_batch_js(close, {
            period_range: [10, 14, 2],
        });
        assertArrayClose(flat, jsBatch.values, 1e-12, 'batch_into mismatch');
    } finally {
        wasm.corrected_moving_average_free(outPtr, total);
        wasm.deallocate_f64_array(inPtr);
    }
});
