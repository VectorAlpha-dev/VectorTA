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

test('stochastic_adaptive_d output contract', () => {
    const high = new Float64Array(testData.high.slice(0, 320));
    const low = new Float64Array(testData.low.slice(0, 320));
    const close = new Float64Array(testData.close.slice(0, 320));

    const result = wasm.stochastic_adaptive_d(high, low, close, 20, 9, 20, 2.0);

    assert.strictEqual(result.standard_d.length, close.length);
    assert.strictEqual(result.adaptive_d.length, close.length);
    assert.strictEqual(result.difference.length, close.length);
    assert(Array.from(result.standard_d).some(v => !isNaN(v)));
    assert(Array.from(result.adaptive_d).some(v => !isNaN(v)));
    assert(Array.from(result.difference).some(v => !isNaN(v)));
});

test('stochastic_adaptive_d_into_host pointer path matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 240));
    const low = new Float64Array(testData.low.slice(0, 240));
    const close = new Float64Array(testData.close.slice(0, 240));
    const safe = wasm.stochastic_adaptive_d(high, low, close, 14, 5, 10, 1.6);
    const outPtr = wasm.stochastic_adaptive_d_alloc(close.length);

    try {
        wasm.stochastic_adaptive_d_into_host(high, low, close, outPtr, 14, 5, 10, 1.6);
        const out = wasm.read_f64_array(outPtr, 3 * close.length);
        const standard = out.slice(0, close.length);
        const adaptive = out.slice(close.length, 2 * close.length);
        const difference = out.slice(2 * close.length, 3 * close.length);
        assertArrayClose(standard, safe.standard_d, 1e-10, 'standard mismatch');
        assertArrayClose(adaptive, safe.adaptive_d, 1e-10, 'adaptive mismatch');
        assertArrayClose(difference, safe.difference, 1e-10, 'difference mismatch');
    } finally {
        wasm.stochastic_adaptive_d_free(outPtr, close.length);
    }
});

test('stochastic_adaptive_d_batch single parameter set matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 220));
    const low = new Float64Array(testData.low.slice(0, 220));
    const close = new Float64Array(testData.close.slice(0, 220));
    const batch = wasm.stochastic_adaptive_d_batch(high, low, close, {
        k_length_range: [20, 20, 0],
        d_smoothing_range: [9, 9, 0],
        pre_smooth_range: [20, 20, 0],
        attenuation_range: [2.0, 2.0, 0.0],
    });
    const single = wasm.stochastic_adaptive_d(high, low, close, 20, 9, 20, 2.0);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].k_length, 20);
    assert.strictEqual(batch.combos[0].d_smoothing, 9);
    assert.strictEqual(batch.combos[0].pre_smooth, 20);
    assert.strictEqual(batch.combos[0].attenuation, 2.0);
    assertArrayClose(batch.standard_d, single.standard_d, 1e-10, 'batch standard mismatch');
    assertArrayClose(batch.adaptive_d, single.adaptive_d, 1e-10, 'batch adaptive mismatch');
    assertArrayClose(batch.difference, single.difference, 1e-10, 'batch difference mismatch');
});

test('stochastic_adaptive_d_batch metadata', () => {
    const high = new Float64Array(testData.high.slice(0, 180));
    const low = new Float64Array(testData.low.slice(0, 180));
    const close = new Float64Array(testData.close.slice(0, 180));
    const batch = wasm.stochastic_adaptive_d_batch(high, low, close, {
        k_length_range: [14, 16, 2],
        d_smoothing_range: [5, 7, 2],
        pre_smooth_range: [10, 12, 2],
        attenuation_range: [1.5, 2.0, 0.5],
    });

    assert.strictEqual(batch.rows, 16);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos.length, 16);
});

test('stochastic_adaptive_d_batch_into pointer path matches safe batch API', () => {
    const high = new Float64Array(testData.high.slice(0, 180));
    const low = new Float64Array(testData.low.slice(0, 180));
    const close = new Float64Array(testData.close.slice(0, 180));
    const batch = wasm.stochastic_adaptive_d_batch(high, low, close, {
        k_length_range: [14, 16, 2],
        d_smoothing_range: [5, 7, 2],
        pre_smooth_range: [10, 12, 2],
        attenuation_range: [1.5, 2.0, 0.5],
    });

    const rows = batch.rows;
    const total = rows * close.length;
    const highPtr = wasm.stochastic_adaptive_d_alloc(close.length);
    const lowPtr = wasm.stochastic_adaptive_d_alloc(close.length);
    const closePtr = wasm.stochastic_adaptive_d_alloc(close.length);
    const standardPtr = wasm.stochastic_adaptive_d_alloc(total);
    const adaptivePtr = wasm.stochastic_adaptive_d_alloc(total);
    const differencePtr = wasm.stochastic_adaptive_d_alloc(total);

    try {
        const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        if (!memory || !memory.buffer) {
            console.warn('Skipping stochastic_adaptive_d_batch_into test: wasm memory accessor unavailable');
            return;
        }
        new Float64Array(memory.buffer, highPtr, close.length).set(high);
        new Float64Array(memory.buffer, lowPtr, close.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);

        const returnedRows = wasm.stochastic_adaptive_d_batch_into(
            highPtr,
            lowPtr,
            closePtr,
            standardPtr,
            adaptivePtr,
            differencePtr,
            close.length,
            14,
            16,
            2,
            5,
            7,
            2,
            10,
            12,
            2,
            1.5,
            2.0,
            0.5,
        );

        assert.strictEqual(returnedRows, rows);
        assertArrayClose(wasm.read_f64_array(standardPtr, total), batch.standard_d, 1e-10, 'batch_into standard mismatch');
        assertArrayClose(wasm.read_f64_array(adaptivePtr, total), batch.adaptive_d, 1e-10, 'batch_into adaptive mismatch');
        assertArrayClose(wasm.read_f64_array(differencePtr, total), batch.difference, 1e-10, 'batch_into difference mismatch');
    } finally {
        wasm.stochastic_adaptive_d_free(highPtr, close.length);
        wasm.stochastic_adaptive_d_free(lowPtr, close.length);
        wasm.stochastic_adaptive_d_free(closePtr, close.length);
        wasm.stochastic_adaptive_d_free(standardPtr, total);
        wasm.stochastic_adaptive_d_free(adaptivePtr, total);
        wasm.stochastic_adaptive_d_free(differencePtr, total);
    }
});
