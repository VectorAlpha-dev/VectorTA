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
    const imported = await import(importPath);
    wasm = imported.default ?? imported;
});

test('hull_butterfly_oscillator_js output contract', () => {
    const data = new Float64Array(
        Array.from({ length: 256 }, (_, i) => 100 + i * 0.05 + Math.sin(i * 0.17) * 2.4 + Math.cos(i * 0.03) * 0.7),
    );
    const out = wasm.hull_butterfly_oscillator_js(data, 14, 2.0);

    assert.strictEqual(out.oscillator.length, data.length);
    assert.strictEqual(out.cumulative_mean.length, data.length);
    assert.strictEqual(out.signal.length, data.length);
    const firstFinite = out.oscillator.findIndex((value) => Number.isFinite(value));
    assert(firstFinite >= 15);
    for (let i = 0; i < out.signal.length; i++) {
        const value = out.signal[i];
        if (Number.isFinite(value)) {
            assert(value === -1 || value === 0 || value === 1);
        }
    }
});

test('hull_butterfly_oscillator_js rejects invalid parameters', () => {
    const data = new Float64Array([1, 2, 3, 4]);
    assert.throws(() => wasm.hull_butterfly_oscillator_js(data, 1, 2.0), /Invalid length/);
    assert.throws(() => wasm.hull_butterfly_oscillator_js(data, 14, Number.NaN), /Invalid multiplier/);
});

test('hull_butterfly_oscillator_into pointer path matches safe API', () => {
    const data = new Float64Array(
        Array.from({ length: 160 }, (_, i) => 100 + i * 0.05 + Math.sin(i * 0.17) * 2.4 + Math.cos(i * 0.03) * 0.7),
    );
    const safe = wasm.hull_butterfly_oscillator_js(data, 12, 1.75);

    const dataPtr = wasm.copy_f64_array(data);
    const oscillatorPtr = wasm.hull_butterfly_oscillator_alloc(data.length);
    const cumulativeMeanPtr = wasm.hull_butterfly_oscillator_alloc(data.length);
    const signalPtr = wasm.hull_butterfly_oscillator_alloc(data.length);

    try {
        wasm.hull_butterfly_oscillator_into(
            dataPtr,
            oscillatorPtr,
            cumulativeMeanPtr,
            signalPtr,
            data.length,
            12,
            1.75,
        );
        assertArrayClose(
            wasm.read_f64_array(oscillatorPtr, data.length),
            safe.oscillator,
            1e-12,
            'oscillator mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(cumulativeMeanPtr, data.length),
            safe.cumulative_mean,
            1e-12,
            'cumulative mean mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(signalPtr, data.length),
            safe.signal,
            1e-12,
            'signal mismatch',
        );
    } finally {
        wasm.hull_butterfly_oscillator_free(oscillatorPtr, data.length);
        wasm.hull_butterfly_oscillator_free(cumulativeMeanPtr, data.length);
        wasm.hull_butterfly_oscillator_free(signalPtr, data.length);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('hull_butterfly_oscillator_js resets after NaN', () => {
    const data = new Float64Array(
        Array.from({ length: 220 }, (_, i) => 100 + i * 0.05 + Math.sin(i * 0.17) * 2.4 + Math.cos(i * 0.03) * 0.7),
    );
    data[110] = Number.NaN;
    const out = wasm.hull_butterfly_oscillator_js(data, 12, 1.75);

    assert(isNaN(out.oscillator[110]));
    assert(isNaN(out.oscillator[111]));
    assert(isNaN(out.cumulative_mean[110]));
    assert(isNaN(out.signal[110]));
});

test('hull_butterfly_oscillator_batch_js rejects invalid config', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6]);
    assert.throws(
        () => wasm.hull_butterfly_oscillator_batch_js(data, { length_range: 'bad' }),
        /Invalid config/,
    );
});

test('hull_butterfly_oscillator_batch_js single parameter set matches safe API', () => {
    const data = new Float64Array(
        Array.from({ length: 160 }, (_, i) => 100 + i * 0.05 + Math.sin(i * 0.17) * 2.4 + Math.cos(i * 0.03) * 0.7),
    );
    const batch = wasm.hull_butterfly_oscillator_batch_js(data, {
        length_range: [12, 12, 0],
        mult_range: [1.5, 1.5, 0],
    });
    const single = wasm.hull_butterfly_oscillator_js(data, 12, 1.5);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, data.length);
    assert.deepStrictEqual(batch.lengths, [12]);
    assert.deepStrictEqual(batch.multipliers, [1.5]);
    assertArrayClose(batch.oscillator, single.oscillator, 1e-12, 'batch oscillator mismatch');
    assertArrayClose(
        batch.cumulative_mean,
        single.cumulative_mean,
        1e-12,
        'batch cumulative mean mismatch',
    );
    assertArrayClose(batch.signal, single.signal, 1e-12, 'batch signal mismatch');
});

test('hull_butterfly_oscillator_batch_js metadata and batch_into', () => {
    const data = new Float64Array(
        Array.from({ length: 120 }, (_, i) => 100 + i * 0.05 + Math.sin(i * 0.17) * 2.4 + Math.cos(i * 0.03) * 0.7),
    );
    const batch = wasm.hull_butterfly_oscillator_batch_js(data, {
        length_range: [10, 14, 2],
        mult_range: [1.0, 2.0, 0.5],
    });

    assert.strictEqual(batch.rows, 9);
    assert.strictEqual(batch.cols, data.length);
    assert.deepStrictEqual(batch.lengths, [10, 10, 10, 12, 12, 12, 14, 14, 14]);
    assert.deepStrictEqual(batch.multipliers, [1, 1.5, 2, 1, 1.5, 2, 1, 1.5, 2]);

    const dataPtr = wasm.copy_f64_array(data);
    const total = batch.rows * data.length;
    const oscillatorPtr = wasm.hull_butterfly_oscillator_alloc(total);
    const cumulativeMeanPtr = wasm.hull_butterfly_oscillator_alloc(total);
    const signalPtr = wasm.hull_butterfly_oscillator_alloc(total);

    try {
        const rows = wasm.hull_butterfly_oscillator_batch_into(
            dataPtr,
            oscillatorPtr,
            cumulativeMeanPtr,
            signalPtr,
            data.length,
            10, 14, 2,
            1.0, 2.0, 0.5,
        );
        assert.strictEqual(rows, 9);
        assertArrayClose(
            wasm.read_f64_array(oscillatorPtr, total),
            batch.oscillator,
            1e-12,
            'batch_into oscillator mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(cumulativeMeanPtr, total),
            batch.cumulative_mean,
            1e-12,
            'batch_into cumulative mean mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(signalPtr, total),
            batch.signal,
            1e-12,
            'batch_into signal mismatch',
        );
    } finally {
        wasm.hull_butterfly_oscillator_free(oscillatorPtr, total);
        wasm.hull_butterfly_oscillator_free(cumulativeMeanPtr, total);
        wasm.hull_butterfly_oscillator_free(signalPtr, total);
        wasm.deallocate_f64_array(dataPtr);
    }
});
