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

test('premier_rsi_oscillator_js output contract on constant data', () => {
    const data = new Float64Array(Array.from({ length: 8 }, () => 10));
    const out = wasm.premier_rsi_oscillator_js(data, 2, 2, 4);

    assert.strictEqual(out.values.length, data.length);
    assert(isNaN(out.values[0]));
    assert(isNaN(out.values[1]));
    assert(isNaN(out.values[2]));
    for (let i = 3; i < out.values.length; i += 1) {
        assert(Math.abs(out.values[i]) <= 1e-12);
    }
});

test('premier_rsi_oscillator_js rejects invalid parameters', () => {
    const data = new Float64Array([1, 2, 3]);
    assert.throws(() => wasm.premier_rsi_oscillator_js(data, 0, 2, 4), /Invalid rsi_length/);
    assert.throws(() => wasm.premier_rsi_oscillator_js(data, 2, 0, 4), /Invalid stoch_length/);
    assert.throws(() => wasm.premier_rsi_oscillator_js(data, 2, 2, 0), /Invalid smooth_length/);
});

test('premier_rsi_oscillator_into pointer path matches safe API', () => {
    const data = new Float64Array([10, 11, 10.5, 10, 10.2, 10.1, 10.3, 10.4]);
    const safe = wasm.premier_rsi_oscillator_js(data, 2, 2, 4);

    const dataPtr = wasm.copy_f64_array(data);
    const valuesPtr = wasm.premier_rsi_oscillator_alloc(data.length);

    try {
        wasm.premier_rsi_oscillator_into(dataPtr, valuesPtr, data.length, 2, 2, 4);
        assertArrayClose(
            wasm.read_f64_array(valuesPtr, data.length),
            safe.values,
            1e-12,
            'values mismatch',
        );
    } finally {
        wasm.premier_rsi_oscillator_free(valuesPtr, data.length);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('premier_rsi_oscillator_js reset after NaN', () => {
    const data = new Float64Array([10, 11, 10.5, 10, 10.2, NaN, 10, 10, 10.1, 10.2]);
    const out = wasm.premier_rsi_oscillator_js(data, 2, 2, 4);

    assert(isNaN(out.values[5]));
    assert(isNaN(out.values[6]));
    assert(isNaN(out.values[7]));
    assert(isNaN(out.values[8]));
    assert(Math.abs(out.values[9]) <= 1e-12);
});

test('premier_rsi_oscillator_batch_js single parameter set matches safe API', () => {
    const data = new Float64Array(Array.from({ length: 64 }, (_, i) => 95 + i * 0.2));
    const batch = wasm.premier_rsi_oscillator_batch_js(data, {
        rsi_length_range: [14, 14, 0],
        stoch_length_range: [8, 8, 0],
        smooth_length_range: [25, 25, 0],
    });
    const single = wasm.premier_rsi_oscillator_js(data, 14, 8, 25);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, data.length);
    assert.deepStrictEqual(batch.rsi_lengths, [14]);
    assert.deepStrictEqual(batch.stoch_lengths, [8]);
    assert.deepStrictEqual(batch.smooth_lengths, [25]);
    assertArrayClose(batch.values, single.values, 1e-12, 'batch values mismatch');
});

test('premier_rsi_oscillator_batch_js metadata and batch_into', () => {
    const data = new Float64Array(Array.from({ length: 64 }, (_, i) => 95 + i * 0.2));
    const batch = wasm.premier_rsi_oscillator_batch_js(data, {
        rsi_length_range: [14, 16, 2],
        stoch_length_range: [8, 10, 2],
        smooth_length_range: [25, 27, 2],
    });

    assert.strictEqual(batch.rows, 8);
    assert.strictEqual(batch.cols, data.length);
    assert.deepStrictEqual(batch.rsi_lengths, [14, 14, 14, 14, 16, 16, 16, 16]);
    assert.deepStrictEqual(batch.stoch_lengths, [8, 8, 10, 10, 8, 8, 10, 10]);
    assert.deepStrictEqual(batch.smooth_lengths, [25, 27, 25, 27, 25, 27, 25, 27]);

    const dataPtr = wasm.copy_f64_array(data);
    const total = batch.rows * data.length;
    const valuesPtr = wasm.premier_rsi_oscillator_alloc(total);

    try {
        const rows = wasm.premier_rsi_oscillator_batch_into(
            dataPtr,
            valuesPtr,
            data.length,
            14, 16, 2,
            8, 10, 2,
            25, 27, 2,
        );
        assert.strictEqual(rows, 8);
        assertArrayClose(
            wasm.read_f64_array(valuesPtr, total),
            batch.values,
            1e-12,
            'batch_into values mismatch',
        );
    } finally {
        wasm.premier_rsi_oscillator_free(valuesPtr, total);
        wasm.deallocate_f64_array(dataPtr);
    }
});
