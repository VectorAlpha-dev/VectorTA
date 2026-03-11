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

test('gmma_oscillator_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.gmma_oscillator_js(close, 'guppy', 3, 13, 0);

    assert.strictEqual(result.oscillator.length, close.length);
    assert.strictEqual(result.signal.length, close.length);
    assert.strictEqual(result.oscillator.findIndex(v => !isNaN(v)), 2);
    assert.strictEqual(result.signal.findIndex(v => !isNaN(v)), 0);
});

test('gmma_oscillator_js rejects invalid params', () => {
    const close = new Float64Array(testData.close.slice(0, 64));

    assert.throws(() => {
        wasm.gmma_oscillator_js(close, 'guppy', 0, 13, 0);
    }, /Invalid smooth_length/);

    assert.throws(() => {
        wasm.gmma_oscillator_js(close, 'bad_mode', 1, 13, 0);
    }, /Invalid GMMA type/);
});

test('gmma_oscillator_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 220));
    const safe = wasm.gmma_oscillator_js(close, 'super_guppy', 4, 9, 0);
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const dataPtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.gmma_oscillator_alloc(close.length);

    try {
        new Float64Array(memory.buffer, dataPtr, close.length).set(close);
        wasm.gmma_oscillator_into(dataPtr, outPtr, close.length, 'super_guppy', 4, 9, 0);
        const flat = new Float64Array(memory.buffer, outPtr, 2 * close.length);
        const oscillator = Array.from(flat.slice(0, close.length));
        const signal = Array.from(flat.slice(close.length));
        assertArrayClose(oscillator, safe.oscillator, 1e-12, 'pointer oscillator mismatch');
        assertArrayClose(signal, safe.signal, 1e-12, 'pointer signal mismatch');
    } finally {
        wasm.gmma_oscillator_free(outPtr, close.length);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('gmma_oscillator_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 220));
    const batch = wasm.gmma_oscillator_batch_js(close, {
        gmma_type: 'guppy',
        smooth_length_range: [3, 3, 0],
        signal_length_range: [13, 13, 0],
        anchor_minutes: 0,
    });
    const single = wasm.gmma_oscillator_js(close, 'guppy', 3, 13, 0);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].gmma_type, 'guppy');
    assert.strictEqual(batch.combos[0].smooth_length, 3);
    assert.strictEqual(batch.combos[0].signal_length, 13);
    assertArrayClose(batch.oscillator, single.oscillator, 1e-12, 'batch oscillator mismatch');
    assertArrayClose(batch.signal, single.signal, 1e-12, 'batch signal mismatch');
});

test('gmma_oscillator_batch_into metadata matches requested ranges', () => {
    const close = new Float64Array(testData.close.slice(0, 180));
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 9;
    const total = rows * close.length;
    const dataPtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.gmma_oscillator_alloc(total);

    try {
        new Float64Array(memory.buffer, dataPtr, close.length).set(close);
        const actualRows = wasm.gmma_oscillator_batch_into(
            dataPtr,
            outPtr,
            close.length,
            'super_guppy',
            1,
            3,
            1,
            9,
            13,
            2,
            0,
        );
        assert.strictEqual(actualRows, rows);

        const flat = new Float64Array(memory.buffer, outPtr, 2 * total);
        const oscillator = Array.from(flat.slice(0, total));
        const signal = Array.from(flat.slice(total));
        const jsBatch = wasm.gmma_oscillator_batch_js(close, {
            gmma_type: 'super_guppy',
            smooth_length_range: [1, 3, 1],
            signal_length_range: [9, 13, 2],
            anchor_minutes: 0,
        });
        assertArrayClose(oscillator, jsBatch.oscillator, 1e-12, 'batch_into oscillator mismatch');
        assertArrayClose(signal, jsBatch.signal, 1e-12, 'batch_into signal mismatch');
    } finally {
        wasm.gmma_oscillator_free(outPtr, total);
        wasm.deallocate_f64_array(dataPtr);
    }
});
