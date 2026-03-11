import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function sampleClose(length) {
    const out = new Float64Array(length);
    out.fill(NaN);
    let prev = 100.0;
    for (let i = 2; i < length; i += 1) {
        const x = i;
        const value = prev + Math.sin(x * 0.07) * 1.3 + Math.cos(x * 0.03) * 0.6 + x * 0.02;
        out[i] = value;
        prev = value;
    }
    return out;
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
});

test('stochastic_distance_js output contract', () => {
    const close = sampleClose(512);
    const out = wasm.stochastic_distance_js(close, 80, 12, 3, 40, -40);

    assert.strictEqual(out.oscillator.length, close.length);
    assert.strictEqual(out.signal.length, close.length);
    const firstValid = out.oscillator.findIndex(v => !isNaN(v));
    assert(firstValid >= 91);
});

test('stochastic_distance_js rejects invalid parameters', () => {
    const close = sampleClose(64);
    assert.throws(() => wasm.stochastic_distance_js(close, 0, 12, 3, 40, -40), /Invalid lookback_length/);
    assert.throws(() => wasm.stochastic_distance_js(close, 20, 12, 3, 40, 10), /Invalid os_level/);
});

test('stochastic_distance_into pointer path matches safe API', () => {
    const close = sampleClose(192);
    const safe = wasm.stochastic_distance_js(close, 50, 8, 4, 40, -40);

    const dataPtr = wasm.copy_f64_array(close);
    const oscPtr = wasm.stochastic_distance_alloc(close.length);
    const sigPtr = wasm.stochastic_distance_alloc(close.length);

    try {
        wasm.stochastic_distance_into(dataPtr, oscPtr, sigPtr, close.length, 50, 8, 4, 40, -40);
        assertArrayClose(wasm.read_f64_array(oscPtr, close.length), safe.oscillator, 1e-12, 'oscillator mismatch');
        assertArrayClose(wasm.read_f64_array(sigPtr, close.length), safe.signal, 1e-12, 'signal mismatch');
    } finally {
        wasm.stochastic_distance_free(oscPtr, close.length);
        wasm.stochastic_distance_free(sigPtr, close.length);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('stochastic_distance_js reset after NaN', () => {
    const close = sampleClose(256);
    close[120] = NaN;
    const out = wasm.stochastic_distance_js(close, 60, 10, 4, 35, -35);
    assert(isNaN(out.oscillator[120]));
    assert(isNaN(out.signal[120]));
    assert(Number.isFinite(out.oscillator.at(-1)));
});

test('stochastic_distance_batch_js single parameter set matches safe API', () => {
    const close = sampleClose(192);
    const batch = wasm.stochastic_distance_batch_js(close, {
        lookback_length_range: [50, 50, 0],
        length1_range: [8, 8, 0],
        length2_range: [4, 4, 0],
        ob_level_range: [40, 40, 0],
        os_level_range: [-40, -40, 0],
    });
    const single = wasm.stochastic_distance_js(close, 50, 8, 4, 40, -40);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assertArrayClose(batch.oscillator, single.oscillator, 1e-12, 'batch oscillator mismatch');
    assertArrayClose(batch.signal, single.signal, 1e-12, 'batch signal mismatch');
});

test('stochastic_distance_batch_js metadata and batch_into', () => {
    const close = sampleClose(160);
    const batch = wasm.stochastic_distance_batch_js(close, {
        lookback_length_range: [40, 60, 20],
        length1_range: [8, 8, 0],
        length2_range: [3, 5, 2],
        ob_level_range: [30, 40, 10],
        os_level_range: [-40, -30, 10],
    });

    assert.strictEqual(batch.rows, 16);
    assert.strictEqual(batch.cols, close.length);

    const dataPtr = wasm.copy_f64_array(close);
    const total = batch.rows * close.length;
    const oscPtr = wasm.stochastic_distance_alloc(total);
    const sigPtr = wasm.stochastic_distance_alloc(total);
    try {
        const rows = wasm.stochastic_distance_batch_into(
            dataPtr, oscPtr, sigPtr, close.length,
            40, 60, 20,
            8, 8, 0,
            3, 5, 2,
            30, 40, 10,
            -40, -30, 10,
        );
        assert.strictEqual(rows, 16);
        assertArrayClose(wasm.read_f64_array(oscPtr, total), batch.oscillator, 1e-12, 'batch_into oscillator mismatch');
        assertArrayClose(wasm.read_f64_array(sigPtr, total), batch.signal, 1e-12, 'batch_into signal mismatch');
    } finally {
        wasm.stochastic_distance_free(oscPtr, total);
        wasm.stochastic_distance_free(sigPtr, total);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('stochastic_distance_batch_js rejects invalid config', () => {
    const close = sampleClose(64);
    assert.throws(
        () => wasm.stochastic_distance_batch_js(close, {
            lookback_length_range: [0, 10, 1],
            length1_range: [8, 8, 0],
            length2_range: [4, 4, 0],
            ob_level_range: [40, 40, 0],
            os_level_range: [-40, -40, 0],
        }),
        /Invalid lookback_length/
    );
});
