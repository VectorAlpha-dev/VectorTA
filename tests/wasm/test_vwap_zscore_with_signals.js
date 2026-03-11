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

test('vwap_zscore_with_signals_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 512));
    const volume = new Float64Array(testData.volume.slice(0, 512));
    const result = wasm.vwap_zscore_with_signals_js(close, volume, 20, 2.5, -2.5);

    assert.strictEqual(result.zvwap.length, close.length);
    assert.strictEqual(result.support_signal.length, close.length);
    assert.strictEqual(result.resistance_signal.length, close.length);
    assert(result.zvwap.some(v => !isNaN(v)));
});

test('vwap_zscore_with_signals_js rejects invalid parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 128));
    const volume = new Float64Array(testData.volume.slice(0, 128));

    assert.throws(() => {
        wasm.vwap_zscore_with_signals_js(close, volume, 0, 2.5, -2.5);
    }, /Invalid length/);

    assert.throws(() => {
        wasm.vwap_zscore_with_signals_js(close.subarray(0, 64), volume, 20, 2.5, -2.5);
    }, /Inconsistent slice lengths/);
});

test('vwap_zscore_with_signals_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const volume = new Float64Array(testData.volume.slice(0, 256));
    const safe = wasm.vwap_zscore_with_signals_js(close, volume, 20, 2.5, -2.5);

    const closePtr = wasm.copy_f64_array(close);
    const volumePtr = wasm.copy_f64_array(volume);
    const zPtr = wasm.vwap_zscore_with_signals_alloc(close.length);
    const sPtr = wasm.vwap_zscore_with_signals_alloc(close.length);
    const rPtr = wasm.vwap_zscore_with_signals_alloc(close.length);

    try {
        wasm.vwap_zscore_with_signals_into(
            closePtr,
            volumePtr,
            zPtr,
            sPtr,
            rPtr,
            close.length,
            20,
            2.5,
            -2.5,
        );
        const zvwap = wasm.read_f64_array(zPtr, close.length);
        const support = wasm.read_f64_array(sPtr, close.length);
        const resistance = wasm.read_f64_array(rPtr, close.length);
        assertArrayClose(zvwap, safe.zvwap, 1e-10, 'zvwap pointer-path mismatch');
        assertArrayClose(support, safe.support_signal, 1e-10, 'support pointer-path mismatch');
        assertArrayClose(
            resistance,
            safe.resistance_signal,
            1e-10,
            'resistance pointer-path mismatch',
        );
    } finally {
        wasm.vwap_zscore_with_signals_free(zPtr, close.length);
        wasm.vwap_zscore_with_signals_free(sPtr, close.length);
        wasm.vwap_zscore_with_signals_free(rPtr, close.length);
        wasm.deallocate_f64_array(closePtr);
        wasm.deallocate_f64_array(volumePtr);
    }
});

test('vwap_zscore_with_signals_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const volume = new Float64Array(testData.volume.slice(0, 256));
    const batch = wasm.vwap_zscore_with_signals_batch_js(close, volume, {
        length_range: [20, 20, 0],
        upper_bottom_range: [2.5, 2.5, 0.0],
        lower_bottom_range: [-2.5, -2.5, 0.0],
    });
    const safe = wasm.vwap_zscore_with_signals_js(close, volume, 20, 2.5, -2.5);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assertArrayClose(batch.zvwap.slice(0, close.length), safe.zvwap, 1e-10, 'batch zvwap mismatch');
    assertArrayClose(
        batch.support_signal.slice(0, close.length),
        safe.support_signal,
        1e-10,
        'batch support mismatch',
    );
    assertArrayClose(
        batch.resistance_signal.slice(0, close.length),
        safe.resistance_signal,
        1e-10,
        'batch resistance mismatch',
    );
});

test('vwap_zscore_with_signals_batch_js metadata matches requested grid', () => {
    const close = new Float64Array(testData.close.slice(0, 192));
    const volume = new Float64Array(testData.volume.slice(0, 192));
    const batch = wasm.vwap_zscore_with_signals_batch_js(close, volume, {
        length_range: [18, 20, 2],
        upper_bottom_range: [2.0, 2.5, 0.5],
        lower_bottom_range: [-2.5, -2.5, 0.0],
    });

    assert.strictEqual(batch.rows, 4);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.zvwap.length, 4 * close.length);
    assert.strictEqual(batch.support_signal.length, 4 * close.length);
    assert.strictEqual(batch.resistance_signal.length, 4 * close.length);
    assert.deepStrictEqual(batch.lengths, [18, 18, 20, 20]);
    assert.deepStrictEqual(batch.upper_bottoms, [2.0, 2.5, 2.0, 2.5]);
    assert.deepStrictEqual(batch.lower_bottoms, [-2.5, -2.5, -2.5, -2.5]);
});

test('vwap_zscore_with_signals_batch_js rejects invalid config', () => {
    const close = new Float64Array(testData.close.slice(0, 128));
    const volume = new Float64Array(testData.volume.slice(0, 128));

    assert.throws(() => {
        wasm.vwap_zscore_with_signals_batch_js(close, volume, {
            length_range: [0, 10, 1],
            upper_bottom_range: [2.5, 2.5, 0.0],
            lower_bottom_range: [-2.5, -2.5, 0.0],
        });
    }, /Invalid length/);
});
