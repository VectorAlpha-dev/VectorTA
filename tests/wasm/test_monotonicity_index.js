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
        data[i] = 100.0 + x * 0.05 + Math.sin(x * 0.17) * 2.4 + Math.cos(x * 0.04) * 0.8;
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

test('monotonicity_index_js output contract', () => {
    const data = sampleData(256);
    const out = wasm.monotonicity_index_js(data, 20, 'efficiency', 5);

    assert.strictEqual(out.index.length, data.length);
    assert.strictEqual(out.cumulative_mean.length, data.length);
    assert.strictEqual(out.upper_bound.length, data.length);
    assert.strictEqual(out.index.findIndex(v => !isNaN(v)), 23);
    assert.strictEqual(out.cumulative_mean.findIndex(v => !isNaN(v)), 23);
    assert.strictEqual(out.upper_bound.findIndex(v => !isNaN(v)), 23);
    assert(Number.isFinite(out.index.at(-1)));
    assert(Number.isFinite(out.cumulative_mean.at(-1)));
    assert(Number.isFinite(out.upper_bound.at(-1)));
});

test('monotonicity_index_js rejects invalid parameters', () => {
    const data = sampleData(64);
    assert.throws(() => wasm.monotonicity_index_js(data, 1, 'efficiency', 5), /Invalid length/);
    assert.throws(() => wasm.monotonicity_index_js(data, 20, 'efficiency', 0), /Invalid index_smooth/);
    assert.throws(() => wasm.monotonicity_index_js(data, 20, 'bad', 5), /Invalid mode/);
});

test('monotonicity_index_into pointer path matches safe API', () => {
    const data = sampleData(192);
    const safe = wasm.monotonicity_index_js(data, 20, 'efficiency', 5);

    const inPtr = wasm.copy_f64_array(data);
    const indexPtr = wasm.monotonicity_index_alloc(data.length);
    const meanPtr = wasm.monotonicity_index_alloc(data.length);
    const upperPtr = wasm.monotonicity_index_alloc(data.length);

    try {
        wasm.monotonicity_index_into(inPtr, indexPtr, meanPtr, upperPtr, data.length, 20, 'efficiency', 5);
        assertArrayClose(wasm.read_f64_array(indexPtr, data.length), safe.index, 1e-12, 'index mismatch');
        assertArrayClose(
            wasm.read_f64_array(meanPtr, data.length),
            safe.cumulative_mean,
            1e-12,
            'mean mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(upperPtr, data.length),
            safe.upper_bound,
            1e-12,
            'upper mismatch',
        );
    } finally {
        wasm.monotonicity_index_free(indexPtr, data.length);
        wasm.monotonicity_index_free(meanPtr, data.length);
        wasm.monotonicity_index_free(upperPtr, data.length);
        wasm.deallocate_f64_array(inPtr);
    }
});

test('monotonicity_index_js resets after NaN', () => {
    const data = sampleData(192);
    data[96] = Number.NaN;

    const out = wasm.monotonicity_index_js(data, 20, 'efficiency', 5);
    assert(isNaN(out.index[96]));
    assert(isNaN(out.cumulative_mean[96]));
    assert(isNaN(out.upper_bound[96]));
    assert(Number.isFinite(out.index.at(-1)));
});

test('monotonicity_index_batch_js single parameter set matches safe API', () => {
    const data = sampleData(192);
    const batch = wasm.monotonicity_index_batch_js(data, {
        length_range: [20, 20, 0],
        index_smooth_range: [5, 5, 0],
        mode: 'efficiency',
    });
    const safe = wasm.monotonicity_index_js(data, 20, 'efficiency', 5);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, data.length);
    assertArrayClose(batch.index, safe.index, 1e-12, 'batch index mismatch');
    assertArrayClose(batch.cumulative_mean, safe.cumulative_mean, 1e-12, 'batch mean mismatch');
    assertArrayClose(batch.upper_bound, safe.upper_bound, 1e-12, 'batch upper mismatch');
});

test('monotonicity_index_batch_into matches batch_js', () => {
    const data = sampleData(160);
    const batch = wasm.monotonicity_index_batch_js(data, {
        length_range: [18, 20, 2],
        index_smooth_range: [4, 5, 1],
        mode: 'complexity',
    });

    const inPtr = wasm.copy_f64_array(data);
    const total = batch.rows * data.length;
    const indexPtr = wasm.monotonicity_index_alloc(total);
    const meanPtr = wasm.monotonicity_index_alloc(total);
    const upperPtr = wasm.monotonicity_index_alloc(total);

    try {
        const rows = wasm.monotonicity_index_batch_into(
            inPtr,
            indexPtr,
            meanPtr,
            upperPtr,
            data.length,
            18,
            20,
            2,
            4,
            5,
            1,
            'complexity',
        );
        assert.strictEqual(rows, 4);
        assertArrayClose(wasm.read_f64_array(indexPtr, total), batch.index, 1e-12, 'batch_into index mismatch');
        assertArrayClose(
            wasm.read_f64_array(meanPtr, total),
            batch.cumulative_mean,
            1e-12,
            'batch_into mean mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(upperPtr, total),
            batch.upper_bound,
            1e-12,
            'batch_into upper mismatch',
        );
    } finally {
        wasm.monotonicity_index_free(indexPtr, total);
        wasm.monotonicity_index_free(meanPtr, total);
        wasm.monotonicity_index_free(upperPtr, total);
        wasm.deallocate_f64_array(inPtr);
    }
});
