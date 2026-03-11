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

test('didi_index_js output contract and manual values', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6]);
    const out = wasm.didi_index_js(data, 2, 3, 4);

    assert.strictEqual(out.short.length, data.length);
    assert(out.short.slice(0, 3).every(isNaN));
    assert(out.long.slice(0, 3).every(isNaN));
    assert(Math.abs(out.short[3] - (3.5 / 3.0)) <= 1e-12);
    assert(Math.abs(out.long[3] - (2.5 / 3.0)) <= 1e-12);
    assert.strictEqual(out.crossover[3], 0);
    assert.strictEqual(out.crossunder[3], 0);
});

test('didi_index_js rejects invalid parameters', () => {
    const data = new Float64Array([1, 2, 3]);

    assert.throws(() => wasm.didi_index_js(data, 0, 2, 3), /Invalid short_length/);
    assert.throws(() => wasm.didi_index_js(data, 1, 0, 3), /Invalid medium_length/);
    assert.throws(() => wasm.didi_index_js(data, 1, 2, 0), /Invalid long_length/);
});

test('didi_index_into pointer path matches safe API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 4, 3, 2, 1]);
    const safe = wasm.didi_index_js(data, 2, 3, 4);

    const dataPtr = wasm.copy_f64_array(data);
    const shortPtr = wasm.didi_index_alloc(data.length);
    const longPtr = wasm.didi_index_alloc(data.length);
    const crossoverPtr = wasm.didi_index_alloc(data.length);
    const crossunderPtr = wasm.didi_index_alloc(data.length);

    try {
        wasm.didi_index_into(dataPtr, shortPtr, longPtr, crossoverPtr, crossunderPtr, data.length, 2, 3, 4);
        assertArrayClose(wasm.read_f64_array(shortPtr, data.length), safe.short, 1e-12, 'short mismatch');
        assertArrayClose(wasm.read_f64_array(longPtr, data.length), safe.long, 1e-12, 'long mismatch');
        assertArrayClose(wasm.read_f64_array(crossoverPtr, data.length), safe.crossover, 1e-12, 'crossover mismatch');
        assertArrayClose(wasm.read_f64_array(crossunderPtr, data.length), safe.crossunder, 1e-12, 'crossunder mismatch');
    } finally {
        wasm.didi_index_free(shortPtr, data.length);
        wasm.didi_index_free(longPtr, data.length);
        wasm.didi_index_free(crossoverPtr, data.length);
        wasm.didi_index_free(crossunderPtr, data.length);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('didi_index_js reset after NaN', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, NaN, 3, 4, 5, 6]);
    const out = wasm.didi_index_js(data, 2, 3, 4);

    assert(isNaN(out.short[5]));
    assert(isNaN(out.short[8]));
    assert(!isNaN(out.short[9]));
});

test('didi_index_batch_js single parameter set matches safe API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 4, 3, 2, 1]);
    const batch = wasm.didi_index_batch_js(data, {
        short_length_range: [2, 2, 0],
        medium_length_range: [3, 3, 0],
        long_length_range: [4, 4, 0],
    });
    const single = wasm.didi_index_js(data, 2, 3, 4);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, data.length);
    assert.deepStrictEqual(batch.short_lengths, [2]);
    assert.deepStrictEqual(batch.medium_lengths, [3]);
    assert.deepStrictEqual(batch.long_lengths, [4]);
    assertArrayClose(batch.short, single.short, 1e-12, 'batch short mismatch');
    assertArrayClose(batch.long, single.long, 1e-12, 'batch long mismatch');
    assertArrayClose(batch.crossover, single.crossover, 1e-12, 'batch crossover mismatch');
    assertArrayClose(batch.crossunder, single.crossunder, 1e-12, 'batch crossunder mismatch');
});

test('didi_index_batch_js metadata and batch_into', () => {
    const data = new Float64Array(Array.from({ length: 32 }, (_, i) => i + 1));
    const batch = wasm.didi_index_batch_js(data, {
        short_length_range: [2, 4, 2],
        medium_length_range: [3, 3, 0],
        long_length_range: [4, 6, 2],
    });

    assert.strictEqual(batch.rows, 4);
    assert.strictEqual(batch.cols, data.length);
    assert.deepStrictEqual(batch.short_lengths, [2, 2, 4, 4]);
    assert.deepStrictEqual(batch.medium_lengths, [3, 3, 3, 3]);
    assert.deepStrictEqual(batch.long_lengths, [4, 6, 4, 6]);

    const dataPtr = wasm.copy_f64_array(data);
    const total = batch.rows * data.length;
    const shortPtr = wasm.didi_index_alloc(total);
    const longPtr = wasm.didi_index_alloc(total);
    const crossoverPtr = wasm.didi_index_alloc(total);
    const crossunderPtr = wasm.didi_index_alloc(total);

    try {
        const rows = wasm.didi_index_batch_into(
            dataPtr,
            shortPtr,
            longPtr,
            crossoverPtr,
            crossunderPtr,
            data.length,
            2, 4, 2,
            3, 3, 0,
            4, 6, 2,
        );
        assert.strictEqual(rows, 4);
        assertArrayClose(wasm.read_f64_array(shortPtr, total), batch.short, 1e-12, 'batch_into short mismatch');
        assertArrayClose(wasm.read_f64_array(longPtr, total), batch.long, 1e-12, 'batch_into long mismatch');
        assertArrayClose(wasm.read_f64_array(crossoverPtr, total), batch.crossover, 1e-12, 'batch_into crossover mismatch');
        assertArrayClose(wasm.read_f64_array(crossunderPtr, total), batch.crossunder, 1e-12, 'batch_into crossunder mismatch');
    } finally {
        wasm.didi_index_free(shortPtr, total);
        wasm.didi_index_free(longPtr, total);
        wasm.didi_index_free(crossoverPtr, total);
        wasm.didi_index_free(crossunderPtr, total);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('didi_index_batch_js rejects invalid config', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);
    assert.throws(
        () => wasm.didi_index_batch_js(data, { short_length_range: [0, 4, 1] }),
        /Invalid short_length/
    );
});
