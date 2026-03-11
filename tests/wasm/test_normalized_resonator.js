import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function sampleData(length) {
    const data = new Float64Array(length);
    for (let i = 0; i < length; i += 1) {
        const x = i;
        data[i] = 100.0 + x * 0.03 + Math.sin(x * 0.09) * 2.1 + Math.cos(x * 0.02) * 0.7;
    }
    return data;
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath =
        process.platform === 'win32'
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
    wasm = await import(importPath);
});

test('normalized_resonator_js output contract', () => {
    const data = sampleData(256);
    const out = wasm.normalized_resonator_js(data, 48, 0.4, 1.2, 7);

    assert.strictEqual(out.oscillator.length, data.length);
    assert.strictEqual(out.signal.length, data.length);
    assert.strictEqual(out.oscillator.findIndex(v => !isNaN(v)), 2);
    assert.strictEqual(out.signal.findIndex(v => !isNaN(v)), 2);
    assert(Number.isFinite(out.oscillator.at(-1)));
    assert(Number.isFinite(out.signal.at(-1)));
});

test('normalized_resonator_js rejects invalid parameters', () => {
    const data = sampleData(64);
    assert.throws(
        () => wasm.normalized_resonator_js(data, 1, 0.4, 1.2, 7),
        /Invalid period/,
    );
    assert.throws(
        () => wasm.normalized_resonator_js(data, 48, 0.0, 1.2, 7),
        /Invalid delta/,
    );
    assert.throws(
        () => wasm.normalized_resonator_js(data, 48, 0.4, 1.2, 0),
        /Invalid signal_length/,
    );
});

test('normalized_resonator_into pointer path matches safe API', () => {
    const data = sampleData(192);
    const safe = wasm.normalized_resonator_js(data, 48, 0.4, 1.2, 7);

    const inPtr = wasm.copy_f64_array(data);
    const oscillatorPtr = wasm.normalized_resonator_alloc(data.length);
    const signalPtr = wasm.normalized_resonator_alloc(data.length);

    try {
        wasm.normalized_resonator_into(inPtr, oscillatorPtr, signalPtr, data.length, 48, 0.4, 1.2, 7);
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
    } finally {
        wasm.normalized_resonator_free(oscillatorPtr, data.length);
        wasm.normalized_resonator_free(signalPtr, data.length);
        wasm.deallocate_f64_array(inPtr);
    }
});

test('normalized_resonator_js resets after NaN', () => {
    const data = sampleData(192);
    data[96] = Number.NaN;

    const out = wasm.normalized_resonator_js(data, 48, 0.4, 1.2, 7);
    assert(isNaN(out.oscillator[96]));
    assert(isNaN(out.signal[96]));
    assert(Number.isFinite(out.oscillator.at(-1)));
    assert(Number.isFinite(out.signal.at(-1)));
});

test('normalized_resonator_batch_js single parameter set matches safe API', () => {
    const data = sampleData(192);
    const batch = wasm.normalized_resonator_batch_js(data, {
        period_range: [48, 48, 0],
        delta_range: [0.4, 0.4, 0.0],
        lookback_mult_range: [1.2, 1.2, 0.0],
        signal_length_range: [7, 7, 0],
    });
    const safe = wasm.normalized_resonator_js(data, 48, 0.4, 1.2, 7);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, data.length);
    assertArrayClose(batch.oscillator, safe.oscillator, 1e-12, 'batch oscillator mismatch');
    assertArrayClose(batch.signal, safe.signal, 1e-12, 'batch signal mismatch');
});

test('normalized_resonator_batch_into matches batch_js', () => {
    const data = sampleData(160);
    const batch = wasm.normalized_resonator_batch_js(data, {
        period_range: [40, 44, 4],
        delta_range: [0.3, 0.5, 0.2],
        lookback_mult_range: [1.0, 1.2, 0.2],
        signal_length_range: [5, 6, 1],
    });

    const inPtr = wasm.copy_f64_array(data);
    const total = batch.rows * data.length;
    const oscillatorPtr = wasm.normalized_resonator_alloc(total);
    const signalPtr = wasm.normalized_resonator_alloc(total);

    try {
        const rows = wasm.normalized_resonator_batch_into(
            inPtr,
            oscillatorPtr,
            signalPtr,
            data.length,
            40,
            44,
            4,
            0.3,
            0.5,
            0.2,
            1.0,
            1.2,
            0.2,
            5,
            6,
            1,
        );
        assert.strictEqual(rows, 16);
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
    } finally {
        wasm.normalized_resonator_free(oscillatorPtr, total);
        wasm.normalized_resonator_free(signalPtr, total);
        wasm.deallocate_f64_array(inPtr);
    }
});

test('normalized_resonator_batch_js rejects invalid config', () => {
    const data = sampleData(64);
    assert.throws(
        () =>
            wasm.normalized_resonator_batch_js(data, {
                period_range: [1, 1, 0],
                delta_range: [0.4, 0.4, 0.0],
                lookback_mult_range: [1.2, 1.2, 0.0],
                signal_length_range: [7, 7, 0],
            }),
        /Invalid period/,
    );
});
