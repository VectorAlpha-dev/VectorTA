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

test('multi_length_stochastic_average_js output contract', () => {
    const data = new Float64Array(Array.from({ length: 256 }, (_, i) => 90 + (20 * i) / 255));
    const out = wasm.multi_length_stochastic_average_js(data, 14, 10, 'sma', 10, 'sma');

    assert.strictEqual(out.values.length, data.length);
    const firstFinite = out.values.findIndex((value) => Number.isFinite(value));
    assert(firstFinite >= 22);
    for (const value of out.values) {
        if (Number.isFinite(value)) {
            assert(value >= -1e-9 && value <= 100 + 1e-9);
        }
    }
});

test('multi_length_stochastic_average_js rejects invalid parameters', () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    assert.throws(
        () => wasm.multi_length_stochastic_average_js(data, 3, 10, 'sma', 10, 'sma'),
        /Invalid length/,
    );
    assert.throws(
        () => wasm.multi_length_stochastic_average_js(data, 14, 10, 'ema', 10, 'sma'),
        /Invalid premethod/,
    );
    assert.throws(
        () => wasm.multi_length_stochastic_average_js(data, 14, 10, 'sma', 0, 'sma'),
        /Invalid postsmooth/,
    );
});

test('multi_length_stochastic_average_into pointer path matches safe API', () => {
    const data = new Float64Array(
        Array.from({ length: 128 }, (_, i) => 100 + i * 0.04 + Math.sin(i * 0.17) * 1.8 + Math.cos(i * 0.03) * 0.4),
    );
    const safe = wasm.multi_length_stochastic_average_js(data, 12, 5, 'lsma', 4, 'sma');

    const dataPtr = wasm.copy_f64_array(data);
    const valuesPtr = wasm.multi_length_stochastic_average_alloc(data.length);

    try {
        wasm.multi_length_stochastic_average_into(
            dataPtr,
            valuesPtr,
            data.length,
            12,
            5,
            'lsma',
            4,
            'sma',
        );
        assertArrayClose(
            wasm.read_f64_array(valuesPtr, data.length),
            safe.values,
            1e-12,
            'values mismatch',
        );
    } finally {
        wasm.multi_length_stochastic_average_free(valuesPtr, data.length);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('multi_length_stochastic_average_js resets after NaN', () => {
    const data = new Float64Array(
        Array.from({ length: 220 }, (_, i) => 100 + i * 0.05 + Math.sin(i * 0.15) * 1.4),
    );
    data[110] = Number.NaN;
    const out = wasm.multi_length_stochastic_average_js(data, 12, 5, 'lsma', 4, 'sma');

    assert(isNaN(out.values[110]));
    assert(isNaN(out.values[111]));
});

test('multi_length_stochastic_average_batch_js rejects invalid config', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6]);
    assert.throws(
        () => wasm.multi_length_stochastic_average_batch_js(data, { length_range: 'bad' }),
        /Invalid config/,
    );
});

test('multi_length_stochastic_average_batch_js single parameter set matches safe API', () => {
    const data = new Float64Array(
        Array.from({ length: 128 }, (_, i) => 100 + i * 0.04 + Math.sin(i * 0.17) * 1.8 + Math.cos(i * 0.03) * 0.4),
    );
    const batch = wasm.multi_length_stochastic_average_batch_js(data, {
        length_range: [12, 12, 0],
        presmooth_range: [5, 5, 0],
        premethod: 'lsma',
        postsmooth_range: [4, 4, 0],
        postmethod: 'sma',
    });
    const single = wasm.multi_length_stochastic_average_js(data, 12, 5, 'lsma', 4, 'sma');

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, data.length);
    assert.deepStrictEqual(batch.lengths, [12]);
    assert.deepStrictEqual(batch.presmooths, [5]);
    assert.deepStrictEqual(batch.postsmooths, [4]);
    assert.deepStrictEqual(batch.premethods, ['lsma']);
    assert.deepStrictEqual(batch.postmethods, ['sma']);
    assertArrayClose(batch.values, single.values, 1e-12, 'batch values mismatch');
});

test('multi_length_stochastic_average_batch_js metadata and batch_into', () => {
    const data = new Float64Array(
        Array.from({ length: 96 }, (_, i) => 100 + i * 0.04 + Math.sin(i * 0.17) * 1.8 + Math.cos(i * 0.03) * 0.4),
    );
    const batch = wasm.multi_length_stochastic_average_batch_js(data, {
        length_range: [10, 14, 2],
        presmooth_range: [4, 6, 2],
        premethod: 'sma',
        postsmooth_range: [3, 5, 2],
        postmethod: 'lsma',
    });

    assert.strictEqual(batch.rows, 12);
    assert.strictEqual(batch.cols, data.length);
    assert.deepStrictEqual(batch.lengths, [10, 10, 10, 10, 12, 12, 12, 12, 14, 14, 14, 14]);
    assert.deepStrictEqual(batch.presmooths, [4, 4, 6, 6, 4, 4, 6, 6, 4, 4, 6, 6]);
    assert.deepStrictEqual(batch.postsmooths, [3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5]);
    assert.deepStrictEqual(batch.premethods, Array(12).fill('sma'));
    assert.deepStrictEqual(batch.postmethods, Array(12).fill('lsma'));

    const dataPtr = wasm.copy_f64_array(data);
    const total = batch.rows * data.length;
    const valuesPtr = wasm.multi_length_stochastic_average_alloc(total);

    try {
        const rows = wasm.multi_length_stochastic_average_batch_into(
            dataPtr,
            valuesPtr,
            data.length,
            10, 14, 2,
            4, 6, 2,
            'sma',
            3, 5, 2,
            'lsma',
        );
        assert.strictEqual(rows, 12);
        assertArrayClose(
            wasm.read_f64_array(valuesPtr, total),
            batch.values,
            1e-12,
            'batch_into values mismatch',
        );
    } finally {
        wasm.multi_length_stochastic_average_free(valuesPtr, total);
        wasm.deallocate_f64_array(dataPtr);
    }
});
