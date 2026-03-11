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
        data[i] = 100.0 + x * 0.05 + Math.sin(x * 0.13) * 2.5 + Math.cos(x * 0.04) * 0.8;
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

test('velocity_acceleration_indicator_js output contract', () => {
    const data = sampleData(256);
    const values = wasm.velocity_acceleration_indicator_js(data, 21, 5);

    assert.strictEqual(values.length, data.length);
    assert.strictEqual(values.findIndex(v => !isNaN(v)), 4);
    assert(Number.isFinite(values.at(-1)));
});

test('velocity_acceleration_indicator_js rejects invalid parameters', () => {
    const data = sampleData(32);
    assert.throws(
        () => wasm.velocity_acceleration_indicator_js(data, 1, 5),
        /Invalid length/,
    );
    assert.throws(
        () => wasm.velocity_acceleration_indicator_js(data, 21, 0),
        /Invalid smooth_length/,
    );
});

test('velocity_acceleration_indicator_into pointer path matches safe API', () => {
    const data = sampleData(192);
    const safe = wasm.velocity_acceleration_indicator_js(data, 21, 5);

    const inPtr = wasm.copy_f64_array(data);
    const outPtr = wasm.velocity_acceleration_indicator_alloc(data.length);

    try {
        wasm.velocity_acceleration_indicator_into(inPtr, outPtr, data.length, 21, 5);
        assertArrayClose(
            wasm.read_f64_array(outPtr, data.length),
            safe,
            1e-12,
            'pointer-path mismatch',
        );
    } finally {
        wasm.velocity_acceleration_indicator_free(outPtr, data.length);
        wasm.deallocate_f64_array(inPtr);
    }
});

test('velocity_acceleration_indicator_js resets after NaN', () => {
    const data = sampleData(180);
    data[90] = Number.NaN;

    const values = wasm.velocity_acceleration_indicator_js(data, 21, 5);
    assert(isNaN(values[90]));
    assert(Number.isFinite(values.at(-1)));
});

test('velocity_acceleration_indicator_batch_js single parameter set matches safe API', () => {
    const data = sampleData(192);
    const batch = wasm.velocity_acceleration_indicator_batch_js(data, {
        length_range: [21, 21, 0],
        smooth_length_range: [5, 5, 0],
    });
    const safe = wasm.velocity_acceleration_indicator_js(data, 21, 5);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, data.length);
    assertArrayClose(batch.values, safe, 1e-12, 'batch mismatch');
});

test('velocity_acceleration_indicator_batch_into matches batch_js', () => {
    const data = sampleData(160);
    const batch = wasm.velocity_acceleration_indicator_batch_js(data, {
        length_range: [21, 23, 2],
        smooth_length_range: [4, 5, 1],
    });

    const inPtr = wasm.copy_f64_array(data);
    const total = batch.rows * data.length;
    const outPtr = wasm.velocity_acceleration_indicator_alloc(total);

    try {
        const rows = wasm.velocity_acceleration_indicator_batch_into(
            inPtr,
            outPtr,
            data.length,
            21,
            23,
            2,
            4,
            5,
            1,
        );
        assert.strictEqual(rows, batch.rows);
        assertArrayClose(
            wasm.read_f64_array(outPtr, total),
            batch.values,
            1e-12,
            'batch_into mismatch',
        );
    } finally {
        wasm.velocity_acceleration_indicator_free(outPtr, total);
        wasm.deallocate_f64_array(inPtr);
    }
});

test('velocity_acceleration_indicator_batch_js rejects invalid config', () => {
    const data = sampleData(64);
    assert.throws(
        () =>
            wasm.velocity_acceleration_indicator_batch_js(data, {
                length_range: [1, 1, 0],
                smooth_length_range: [5, 5, 0],
            }),
        /Invalid length/,
    );
});
