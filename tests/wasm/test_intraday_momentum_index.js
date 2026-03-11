import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath =
        process.platform === 'win32'
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('intraday_momentum_index_js output contract', () => {
    const open = new Float64Array(testData.open.slice(0, 512));
    const close = new Float64Array(testData.close.slice(0, 512));
    const result = wasm.intraday_momentum_index_js(open, close, 14, 6, 2.0, 20, true, 10);

    assert.strictEqual(result.imi.length, close.length);
    assert.strictEqual(result.upper_hit.length, close.length);
    assert.strictEqual(result.lower_hit.length, close.length);
    assert.strictEqual(result.signal.length, close.length);
    assert(result.imi.some(v => !isNaN(v)));
});

test('intraday_momentum_index_js rejects invalid parameters', () => {
    const open = new Float64Array(testData.open.slice(0, 128));
    const close = new Float64Array(testData.close.slice(0, 128));

    assert.throws(() => {
        wasm.intraday_momentum_index_js(open, close, 0, 6, 2.0, 20, false, 10);
    }, /Invalid length/);

    assert.throws(() => {
        wasm.intraday_momentum_index_js(open.subarray(0, 64), close, 14, 6, 2.0, 20, false, 10);
    }, /Inconsistent slice lengths/);
});

test('intraday_momentum_index_into pointer path matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const safe = wasm.intraday_momentum_index_js(open, close, 14, 6, 2.0, 20, true, 10);

    const openPtr = wasm.copy_f64_array(open);
    const closePtr = wasm.copy_f64_array(close);
    const imiPtr = wasm.intraday_momentum_index_alloc(close.length);
    const upperPtr = wasm.intraday_momentum_index_alloc(close.length);
    const lowerPtr = wasm.intraday_momentum_index_alloc(close.length);
    const signalPtr = wasm.intraday_momentum_index_alloc(close.length);

    try {
        wasm.intraday_momentum_index_into(
            openPtr,
            closePtr,
            imiPtr,
            upperPtr,
            lowerPtr,
            signalPtr,
            close.length,
            14,
            6,
            2.0,
            20,
            true,
            10,
        );
        const imi = wasm.read_f64_array(imiPtr, close.length);
        const upper = wasm.read_f64_array(upperPtr, close.length);
        const lower = wasm.read_f64_array(lowerPtr, close.length);
        const signal = wasm.read_f64_array(signalPtr, close.length);
        assertArrayClose(imi, safe.imi, 1e-10, 'imi pointer-path mismatch');
        assertArrayClose(upper, safe.upper_hit, 1e-10, 'upper pointer-path mismatch');
        assertArrayClose(lower, safe.lower_hit, 1e-10, 'lower pointer-path mismatch');
        assertArrayClose(signal, safe.signal, 1e-10, 'signal pointer-path mismatch');
    } finally {
        wasm.intraday_momentum_index_free(imiPtr, close.length);
        wasm.intraday_momentum_index_free(upperPtr, close.length);
        wasm.intraday_momentum_index_free(lowerPtr, close.length);
        wasm.intraday_momentum_index_free(signalPtr, close.length);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('intraday_momentum_index_batch_js single parameter set matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const batch = wasm.intraday_momentum_index_batch_js(open, close, {
        length_range: [14, 14, 0],
        length_ma_range: [6, 6, 0],
        mult_range: [2.0, 2.0, 0.0],
        length_bb_range: [20, 20, 0],
        apply_smoothing: false,
        low_band_range: [10, 10, 0],
    });
    const safe = wasm.intraday_momentum_index_js(open, close, 14, 6, 2.0, 20, false, 10);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assertArrayClose(batch.imi.slice(0, close.length), safe.imi, 1e-10, 'batch imi mismatch');
    assertArrayClose(
        batch.upper_hit.slice(0, close.length),
        safe.upper_hit,
        1e-10,
        'batch upper mismatch',
    );
    assertArrayClose(
        batch.lower_hit.slice(0, close.length),
        safe.lower_hit,
        1e-10,
        'batch lower mismatch',
    );
    assertArrayClose(
        batch.signal.slice(0, close.length),
        safe.signal,
        1e-10,
        'batch signal mismatch',
    );
});

test('intraday_momentum_index_batch_js metadata matches requested grid', () => {
    const open = new Float64Array(testData.open.slice(0, 192));
    const close = new Float64Array(testData.close.slice(0, 192));
    const batch = wasm.intraday_momentum_index_batch_js(open, close, {
        length_range: [12, 14, 2],
        length_ma_range: [5, 6, 1],
        mult_range: [1.5, 1.5, 0.0],
        length_bb_range: [18, 18, 0],
        apply_smoothing: true,
        low_band_range: [9, 9, 0],
    });

    assert.strictEqual(batch.rows, 4);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.imi.length, 4 * close.length);
    assert.strictEqual(batch.upper_hit.length, 4 * close.length);
    assert.strictEqual(batch.lower_hit.length, 4 * close.length);
    assert.strictEqual(batch.signal.length, 4 * close.length);
    assert.deepStrictEqual(batch.lengths, [12, 12, 14, 14]);
    assert.deepStrictEqual(batch.length_mas, [5, 6, 5, 6]);
    assert.deepStrictEqual(batch.mults, [1.5, 1.5, 1.5, 1.5]);
    assert.deepStrictEqual(batch.length_bbs, [18, 18, 18, 18]);
    assert.deepStrictEqual(batch.apply_smoothing, [true, true, true, true]);
    assert.deepStrictEqual(batch.low_bands, [9, 9, 9, 9]);
});

test('intraday_momentum_index_batch_js rejects invalid config', () => {
    const open = new Float64Array(testData.open.slice(0, 128));
    const close = new Float64Array(testData.close.slice(0, 128));

    assert.throws(() => {
        wasm.intraday_momentum_index_batch_js(open, close, {
            length_range: [0, 10, 1],
            length_ma_range: [6, 6, 0],
            mult_range: [2.0, 2.0, 0.0],
            length_bb_range: [20, 20, 0],
            apply_smoothing: false,
            low_band_range: [10, 10, 0],
        });
    }, /Invalid length/);
});
