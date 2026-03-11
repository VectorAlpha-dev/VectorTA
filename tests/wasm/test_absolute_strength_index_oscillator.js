import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
});

test('absolute_strength_index_oscillator_js output contract and manual values', () => {
    const data = new Float64Array([10, 10]);
    const out = wasm.absolute_strength_index_oscillator_js(data, 2, 2);

    assert.strictEqual(out.oscillator.length, data.length);
    assert(Math.abs(out.oscillator[0]) <= 1e-12);
    assert(Math.abs(out.signal[0]) <= 1e-12);
    assert(Math.abs(out.histogram[0]) <= 1e-12);
    assert(Math.abs(out.oscillator[1] + (1 / 6)) <= 1e-12);
    assert(Math.abs(out.signal[1] + (2 / 9)) <= 1e-12);
    assert(Math.abs(out.histogram[1] - (1 / 18)) <= 1e-12);
});

test('absolute_strength_index_oscillator_js rejects invalid parameters', () => {
    const data = new Float64Array([1, 2, 3]);
    assert.throws(() => wasm.absolute_strength_index_oscillator_js(data, 0, 2), /Invalid ema_length/);
    assert.throws(() => wasm.absolute_strength_index_oscillator_js(data, 2, 1), /Invalid signal_length/);
});

test('absolute_strength_index_oscillator_into pointer path matches safe API', () => {
    const data = new Float64Array([10, 10, 9, 9, 10, 11]);
    const safe = wasm.absolute_strength_index_oscillator_js(data, 3, 4);

    const dataPtr = wasm.copy_f64_array(data);
    const oscillatorPtr = wasm.absolute_strength_index_oscillator_alloc(data.length);
    const signalPtr = wasm.absolute_strength_index_oscillator_alloc(data.length);
    const histogramPtr = wasm.absolute_strength_index_oscillator_alloc(data.length);

    try {
        wasm.absolute_strength_index_oscillator_into(
            dataPtr,
            oscillatorPtr,
            signalPtr,
            histogramPtr,
            data.length,
            3,
            4,
        );
        assertArrayClose(
            wasm.read_f64_array(oscillatorPtr, data.length),
            safe.oscillator,
            1e-12,
            'oscillator mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(signalPtr, data.length),
            safe.signal,
            1e-12,
            'signal mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(histogramPtr, data.length),
            safe.histogram,
            1e-12,
            'histogram mismatch',
        );
    } finally {
        wasm.absolute_strength_index_oscillator_free(oscillatorPtr, data.length);
        wasm.absolute_strength_index_oscillator_free(signalPtr, data.length);
        wasm.absolute_strength_index_oscillator_free(histogramPtr, data.length);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('absolute_strength_index_oscillator_js reset after NaN', () => {
    const data = new Float64Array([10, 10, 9, NaN, 10, 10]);
    const out = wasm.absolute_strength_index_oscillator_js(data, 2, 2);

    assert(isNaN(out.oscillator[3]));
    assert(Math.abs(out.oscillator[4]) <= 1e-12);
    assert(Math.abs(out.signal[4]) <= 1e-12);
});

test('absolute_strength_index_oscillator_batch_js single parameter set matches safe API', () => {
    const data = new Float64Array([10, 10, 9, 9, 10, 11]);
    const batch = wasm.absolute_strength_index_oscillator_batch_js(data, {
        ema_length_range: [3, 3, 0],
        signal_length_range: [4, 4, 0],
    });
    const single = wasm.absolute_strength_index_oscillator_js(data, 3, 4);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, data.length);
    assert.deepStrictEqual(batch.ema_lengths, [3]);
    assert.deepStrictEqual(batch.signal_lengths, [4]);
    assertArrayClose(batch.oscillator, single.oscillator, 1e-12, 'batch oscillator mismatch');
    assertArrayClose(batch.signal, single.signal, 1e-12, 'batch signal mismatch');
    assertArrayClose(batch.histogram, single.histogram, 1e-12, 'batch histogram mismatch');
});

test('absolute_strength_index_oscillator_batch_js metadata and batch_into', () => {
    const data = new Float64Array(Array.from({ length: 64 }, (_, i) => 95 + i * 0.2));
    const batch = wasm.absolute_strength_index_oscillator_batch_js(data, {
        ema_length_range: [3, 5, 2],
        signal_length_range: [4, 6, 2],
    });

    assert.strictEqual(batch.rows, 4);
    assert.strictEqual(batch.cols, data.length);
    assert.deepStrictEqual(batch.ema_lengths, [3, 3, 5, 5]);
    assert.deepStrictEqual(batch.signal_lengths, [4, 6, 4, 6]);

    const dataPtr = wasm.copy_f64_array(data);
    const total = batch.rows * data.length;
    const oscillatorPtr = wasm.absolute_strength_index_oscillator_alloc(total);
    const signalPtr = wasm.absolute_strength_index_oscillator_alloc(total);
    const histogramPtr = wasm.absolute_strength_index_oscillator_alloc(total);

    try {
        const rows = wasm.absolute_strength_index_oscillator_batch_into(
            dataPtr,
            oscillatorPtr,
            signalPtr,
            histogramPtr,
            data.length,
            3, 5, 2,
            4, 6, 2,
        );
        assert.strictEqual(rows, 4);
        assertArrayClose(
            wasm.read_f64_array(oscillatorPtr, total),
            batch.oscillator,
            1e-12,
            'batch_into oscillator mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(signalPtr, total),
            batch.signal,
            1e-12,
            'batch_into signal mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(histogramPtr, total),
            batch.histogram,
            1e-12,
            'batch_into histogram mismatch',
        );
    } finally {
        wasm.absolute_strength_index_oscillator_free(oscillatorPtr, total);
        wasm.absolute_strength_index_oscillator_free(signalPtr, total);
        wasm.absolute_strength_index_oscillator_free(histogramPtr, total);
        wasm.deallocate_f64_array(dataPtr);
    }
});
