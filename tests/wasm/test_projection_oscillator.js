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

test('projection_oscillator_js output contract', () => {
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.projection_oscillator_js(high, low, close, 14, 4);

    assert.strictEqual(result.pbo.length, close.length);
    assert.strictEqual(result.signal.length, close.length);
    assert.strictEqual(result.pbo.findIndex(v => !isNaN(v)), 29);
    assert.strictEqual(result.signal.findIndex(v => !isNaN(v)), 32);
});

test('projection_oscillator_js rejects invalid parameters', () => {
    const high = new Float64Array(testData.high.slice(0, 64));
    const low = new Float64Array(testData.low.slice(0, 64));
    const close = new Float64Array(testData.close.slice(0, 64));

    assert.throws(() => {
        wasm.projection_oscillator_js(high, low, close, 0, 4);
    }, /Invalid length/);

    assert.throws(() => {
        wasm.projection_oscillator_js(high, low, close, 14, 0);
    }, /Invalid smooth_length/);
});

test('projection_oscillator_into pointer path matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const close = new Float64Array(testData.close.slice(0, 200));
    const safe = wasm.projection_oscillator_js(high, low, close, 14, 4);
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.projection_oscillator_alloc(close.length);

    try {
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        wasm.projection_oscillator_into(highPtr, lowPtr, closePtr, outPtr, close.length, 14, 4);
        const flat = new Float64Array(memory.buffer, outPtr, 2 * close.length);
        const pbo = Array.from(flat.slice(0, close.length));
        const signal = Array.from(flat.slice(close.length));
        assertArrayClose(pbo, safe.pbo, 1e-12, 'pointer pbo mismatch');
        assertArrayClose(signal, safe.signal, 1e-12, 'pointer signal mismatch');
    } finally {
        wasm.projection_oscillator_free(outPtr, close.length);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('projection_oscillator_batch_js single parameter set matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.projection_oscillator_batch_js(high, low, close, {
        length_range: [14, 14, 0],
        smooth_length_range: [4, 4, 0],
    });
    const single = wasm.projection_oscillator_js(high, low, close, 14, 4);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].length, 14);
    assert.strictEqual(batch.combos[0].smooth_length, 4);
    assertArrayClose(batch.pbo, single.pbo, 1e-12, 'batch pbo mismatch');
    assertArrayClose(batch.signal, single.signal, 1e-12, 'batch signal mismatch');
});

test('projection_oscillator_batch_into metadata matches requested ranges', () => {
    const high = new Float64Array(testData.high.slice(0, 128));
    const low = new Float64Array(testData.low.slice(0, 128));
    const close = new Float64Array(testData.close.slice(0, 128));
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 3;
    const total = rows * close.length;
    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.projection_oscillator_alloc(total);

    try {
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        const actualRows = wasm.projection_oscillator_batch_into(
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            close.length,
            10,
            14,
            2,
            4,
            4,
            0,
        );
        assert.strictEqual(actualRows, rows);

        const flat = new Float64Array(memory.buffer, outPtr, 2 * total);
        const pbo = Array.from(flat.slice(0, total));
        const signal = Array.from(flat.slice(total));
        const jsBatch = wasm.projection_oscillator_batch_js(high, low, close, {
            length_range: [10, 14, 2],
            smooth_length_range: [4, 4, 0],
        });
        assertArrayClose(pbo, jsBatch.pbo, 1e-12, 'batch_into pbo mismatch');
        assertArrayClose(signal, jsBatch.signal, 1e-12, 'batch_into signal mismatch');
    } finally {
        wasm.projection_oscillator_free(outPtr, total);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});
