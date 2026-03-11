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

test('evasive_supertrend_js output contract', () => {
    const open = new Float64Array(testData.open.slice(0, 256));
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.evasive_supertrend_js(open, high, low, close, 10, 3.0, 1.0, 0.5);

    assert.strictEqual(result.band.length, close.length);
    assert.strictEqual(result.state.length, close.length);
    assert.strictEqual(result.noisy.length, close.length);
    assert.strictEqual(result.changed.length, close.length);
    assert(result.band.some(v => !isNaN(v)));
});

test('evasive_supertrend_js rejects invalid parameters', () => {
    const open = new Float64Array(testData.open.slice(0, 64));
    const high = new Float64Array(testData.high.slice(0, 64));
    const low = new Float64Array(testData.low.slice(0, 64));
    const close = new Float64Array(testData.close.slice(0, 64));

    assert.throws(() => {
        wasm.evasive_supertrend_js(open, high, low, close, 0, 3.0, 1.0, 0.5);
    }, /Invalid atr_length/);

    assert.throws(() => {
        wasm.evasive_supertrend_js(open, high, low, close, 10, 0.0, 1.0, 0.5);
    }, /Invalid base_multiplier/);
});

test('evasive_supertrend_into pointer path matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 220));
    const high = new Float64Array(testData.high.slice(0, 220));
    const low = new Float64Array(testData.low.slice(0, 220));
    const close = new Float64Array(testData.close.slice(0, 220));
    const safe = wasm.evasive_supertrend_js(open, high, low, close, 10, 3.0, 1.0, 0.5);
    const memory = getMemory();
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const openPtr = wasm.allocate_f64_array(open.length);
    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.evasive_supertrend_alloc(close.length);

    try {
        new Float64Array(memory.buffer, openPtr, open.length).set(open);
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);

        wasm.evasive_supertrend_into(
            openPtr, highPtr, lowPtr, closePtr, outPtr, close.length, 10, 3.0, 1.0, 0.5
        );

        const flat = new Float64Array(memory.buffer, outPtr, 4 * close.length);
        const band = Array.from(flat.slice(0, close.length));
        const state = Array.from(flat.slice(close.length, 2 * close.length));
        const noisy = Array.from(flat.slice(2 * close.length, 3 * close.length));
        const changed = Array.from(flat.slice(3 * close.length));

        assertArrayClose(band, safe.band, 1e-12, 'pointer band mismatch');
        assertArrayClose(state, safe.state, 1e-12, 'pointer state mismatch');
        assertArrayClose(noisy, safe.noisy, 1e-12, 'pointer noisy mismatch');
        assertArrayClose(changed, safe.changed, 1e-12, 'pointer changed mismatch');
    } finally {
        wasm.evasive_supertrend_free(outPtr, close.length);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('evasive_supertrend_batch_js single parameter set matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 200));
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.evasive_supertrend_batch_js(open, high, low, close, {
        atr_length_range: [10, 10, 0],
        base_multiplier_range: [3.0, 3.0, 0.0],
        noise_threshold_range: [1.0, 1.0, 0.0],
        expansion_alpha_range: [0.5, 0.5, 0.0],
    });
    const single = wasm.evasive_supertrend_js(open, high, low, close, 10, 3.0, 1.0, 0.5);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].atr_length, 10);
    assert.strictEqual(batch.combos[0].base_multiplier, 3.0);
    assert.strictEqual(batch.combos[0].noise_threshold, 1.0);
    assert.strictEqual(batch.combos[0].expansion_alpha, 0.5);
    assertArrayClose(batch.band, single.band, 1e-12, 'batch band mismatch');
    assertArrayClose(batch.state, single.state, 1e-12, 'batch state mismatch');
    assertArrayClose(batch.noisy, single.noisy, 1e-12, 'batch noisy mismatch');
    assertArrayClose(batch.changed, single.changed, 1e-12, 'batch changed mismatch');
});

test('evasive_supertrend_batch_into matches batch_js', () => {
    const open = new Float64Array(testData.open.slice(0, 160));
    const high = new Float64Array(testData.high.slice(0, 160));
    const low = new Float64Array(testData.low.slice(0, 160));
    const close = new Float64Array(testData.close.slice(0, 160));
    const memory = getMemory();
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 2;
    const total = rows * close.length;
    const openPtr = wasm.allocate_f64_array(open.length);
    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.evasive_supertrend_alloc(total);

    try {
        new Float64Array(memory.buffer, openPtr, open.length).set(open);
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);

        const actualRows = wasm.evasive_supertrend_batch_into(
            openPtr, highPtr, lowPtr, closePtr, outPtr, close.length,
            10, 12, 2,
            3.0, 3.0, 0.0,
            1.0, 1.0, 0.0,
            0.5, 0.5, 0.0,
        );
        assert.strictEqual(actualRows, rows);

        const flat = new Float64Array(memory.buffer, outPtr, 4 * total);
        const band = Array.from(flat.slice(0, total));
        const state = Array.from(flat.slice(total, 2 * total));
        const noisy = Array.from(flat.slice(2 * total, 3 * total));
        const changed = Array.from(flat.slice(3 * total));

        const jsBatch = wasm.evasive_supertrend_batch_js(open, high, low, close, {
            atr_length_range: [10, 12, 2],
            base_multiplier_range: [3.0, 3.0, 0.0],
            noise_threshold_range: [1.0, 1.0, 0.0],
            expansion_alpha_range: [0.5, 0.5, 0.0],
        });
        assertArrayClose(band, jsBatch.band, 1e-12, 'batch_into band mismatch');
        assertArrayClose(state, jsBatch.state, 1e-12, 'batch_into state mismatch');
        assertArrayClose(noisy, jsBatch.noisy, 1e-12, 'batch_into noisy mismatch');
        assertArrayClose(changed, jsBatch.changed, 1e-12, 'batch_into changed mismatch');
    } finally {
        wasm.evasive_supertrend_free(outPtr, total);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});
