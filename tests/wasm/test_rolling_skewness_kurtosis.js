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
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('rolling_skewness_kurtosis_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.rolling_skewness_kurtosis_js(close, 50, 3);

    assert.strictEqual(result.skewness.length, close.length);
    assert.strictEqual(result.kurtosis.length, close.length);
    assert.strictEqual(result.skewness.findIndex(v => !isNaN(v)), 51);
    assert.strictEqual(result.kurtosis.findIndex(v => !isNaN(v)), 51);
});

test('rolling_skewness_kurtosis_js rejects invalid parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 64));

    assert.throws(() => {
        wasm.rolling_skewness_kurtosis_js(close, 0, 3);
    }, /Invalid length/);

    assert.throws(() => {
        wasm.rolling_skewness_kurtosis_js(close, 10, 0);
    }, /Invalid smooth_length/);
});

test('rolling_skewness_kurtosis_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 220));
    const safe = wasm.rolling_skewness_kurtosis_js(close, 40, 4);
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.rolling_skewness_kurtosis_alloc(close.length);

    try {
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        wasm.rolling_skewness_kurtosis_into(closePtr, outPtr, close.length, 40, 4);
        const flat = new Float64Array(memory.buffer, outPtr, 2 * close.length);
        const skewness = Array.from(flat.slice(0, close.length));
        const kurtosis = Array.from(flat.slice(close.length));
        assertArrayClose(skewness, safe.skewness, 1e-12, 'pointer skewness mismatch');
        assertArrayClose(kurtosis, safe.kurtosis, 1e-12, 'pointer kurtosis mismatch');
    } finally {
        wasm.rolling_skewness_kurtosis_free(outPtr, close.length);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('rolling_skewness_kurtosis_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 220));
    const batch = wasm.rolling_skewness_kurtosis_batch_js(close, {
        length_range: [50, 50, 0],
        smooth_length_range: [3, 3, 0],
    });
    const single = wasm.rolling_skewness_kurtosis_js(close, 50, 3);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].length, 50);
    assert.strictEqual(batch.combos[0].smooth_length, 3);
    assertArrayClose(batch.skewness, single.skewness, 1e-12, 'batch skewness mismatch');
    assertArrayClose(batch.kurtosis, single.kurtosis, 1e-12, 'batch kurtosis mismatch');
});

test('rolling_skewness_kurtosis_batch_into metadata matches requested ranges', () => {
    const close = new Float64Array(testData.close.slice(0, 180));
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 4;
    const total = rows * close.length;
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.rolling_skewness_kurtosis_alloc(total);

    try {
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        const actualRows = wasm.rolling_skewness_kurtosis_batch_into(
            closePtr,
            outPtr,
            close.length,
            40,
            50,
            10,
            2,
            3,
            1,
        );
        assert.strictEqual(actualRows, rows);

        const flat = new Float64Array(memory.buffer, outPtr, 2 * total);
        const skewness = Array.from(flat.slice(0, total));
        const kurtosis = Array.from(flat.slice(total));
        const jsBatch = wasm.rolling_skewness_kurtosis_batch_js(close, {
            length_range: [40, 50, 10],
            smooth_length_range: [2, 3, 1],
        });
        assertArrayClose(skewness, jsBatch.skewness, 1e-12, 'batch_into skewness mismatch');
        assertArrayClose(kurtosis, jsBatch.kurtosis, 1e-12, 'batch_into kurtosis mismatch');
    } finally {
        wasm.rolling_skewness_kurtosis_free(outPtr, total);
        wasm.deallocate_f64_array(closePtr);
    }
});
