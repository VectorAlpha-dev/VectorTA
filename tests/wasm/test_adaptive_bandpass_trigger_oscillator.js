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

test('adaptive_bandpass_trigger_oscillator_js output contract', () => {
    const close = sampleClose(512);
    const out = wasm.adaptive_bandpass_trigger_oscillator_js(close, 0.1, 0.07);

    assert.strictEqual(out.in_phase.length, close.length);
    assert.strictEqual(out.lead.length, close.length);
    assert(out.in_phase.findIndex(v => !isNaN(v)) >= 13);
    assert(out.lead.findIndex(v => !isNaN(v)) >= 14);
});

test('adaptive_bandpass_trigger_oscillator_js rejects invalid parameters', () => {
    const close = sampleClose(64);
    assert.throws(
        () => wasm.adaptive_bandpass_trigger_oscillator_js(close, 0.0, 0.07),
        /Invalid delta/
    );
    assert.throws(
        () => wasm.adaptive_bandpass_trigger_oscillator_js(close, 0.1, 1.0),
        /Invalid alpha/
    );
});

test('adaptive_bandpass_trigger_oscillator_into pointer path matches safe API', () => {
    const close = sampleClose(192);
    const safe = wasm.adaptive_bandpass_trigger_oscillator_js(close, 0.1, 0.07);

    const dataPtr = wasm.copy_f64_array(close);
    const inPhasePtr = wasm.adaptive_bandpass_trigger_oscillator_alloc(close.length);
    const leadPtr = wasm.adaptive_bandpass_trigger_oscillator_alloc(close.length);

    try {
        wasm.adaptive_bandpass_trigger_oscillator_into(
            dataPtr,
            inPhasePtr,
            leadPtr,
            close.length,
            0.1,
            0.07
        );
        assertArrayClose(
            wasm.read_f64_array(inPhasePtr, close.length),
            safe.in_phase,
            1e-12,
            'in_phase mismatch'
        );
        assertArrayClose(
            wasm.read_f64_array(leadPtr, close.length),
            safe.lead,
            1e-12,
            'lead mismatch'
        );
    } finally {
        wasm.adaptive_bandpass_trigger_oscillator_free(inPhasePtr, close.length);
        wasm.adaptive_bandpass_trigger_oscillator_free(leadPtr, close.length);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('adaptive_bandpass_trigger_oscillator_js reset after NaN', () => {
    const close = sampleClose(256);
    close[120] = NaN;
    const out = wasm.adaptive_bandpass_trigger_oscillator_js(close, 0.12, 0.08);
    assert(isNaN(out.in_phase[120]));
    assert(isNaN(out.lead[120]));
    assert(Number.isFinite(out.in_phase.at(-1)));
    assert(Number.isFinite(out.lead.at(-1)));
});

test('adaptive_bandpass_trigger_oscillator_batch_js single parameter set matches safe API', () => {
    const close = sampleClose(192);
    const batch = wasm.adaptive_bandpass_trigger_oscillator_batch_js(close, {
        delta_range: [0.1, 0.1, 0.0],
        alpha_range: [0.07, 0.07, 0.0],
    });
    const single = wasm.adaptive_bandpass_trigger_oscillator_js(close, 0.1, 0.07);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assertArrayClose(batch.in_phase, single.in_phase, 1e-12, 'batch in_phase mismatch');
    assertArrayClose(batch.lead, single.lead, 1e-12, 'batch lead mismatch');
});

test('adaptive_bandpass_trigger_oscillator_batch_js metadata and batch_into', () => {
    const close = sampleClose(160);
    const batch = wasm.adaptive_bandpass_trigger_oscillator_batch_js(close, {
        delta_range: [0.08, 0.12, 0.04],
        alpha_range: [0.05, 0.09, 0.02],
    });

    assert.strictEqual(batch.rows, 6);
    assert.strictEqual(batch.cols, close.length);

    const dataPtr = wasm.copy_f64_array(close);
    const total = batch.rows * close.length;
    const inPhasePtr = wasm.adaptive_bandpass_trigger_oscillator_alloc(total);
    const leadPtr = wasm.adaptive_bandpass_trigger_oscillator_alloc(total);
    try {
        const rows = wasm.adaptive_bandpass_trigger_oscillator_batch_into(
            dataPtr,
            inPhasePtr,
            leadPtr,
            close.length,
            0.08,
            0.12,
            0.04,
            0.05,
            0.09,
            0.02
        );
        assert.strictEqual(rows, 6);
        assertArrayClose(
            wasm.read_f64_array(inPhasePtr, total),
            batch.in_phase,
            1e-12,
            'batch_into in_phase mismatch'
        );
        assertArrayClose(
            wasm.read_f64_array(leadPtr, total),
            batch.lead,
            1e-12,
            'batch_into lead mismatch'
        );
    } finally {
        wasm.adaptive_bandpass_trigger_oscillator_free(inPhasePtr, total);
        wasm.adaptive_bandpass_trigger_oscillator_free(leadPtr, total);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('adaptive_bandpass_trigger_oscillator_batch_js rejects invalid config', () => {
    const close = sampleClose(64);
    assert.throws(
        () => wasm.adaptive_bandpass_trigger_oscillator_batch_js(close, {
            delta_range: [0.0, 0.1, 0.1],
            alpha_range: [0.07, 0.07, 0.0],
        }),
        /Invalid delta/
    );
});
