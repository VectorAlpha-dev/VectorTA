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

test('autocorrelation_indicator_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.autocorrelation_indicator_js(close, 20, 12, false);

    assert.strictEqual(result.filtered.length, close.length);
    assert.strictEqual(result.correlations.length, 12 * close.length);
    assert.strictEqual(result.lag_count, 12);
    assert.strictEqual(result.cols, close.length);
    const lag1 = result.correlations.slice(0, close.length);
    const lag12 = result.correlations.slice(11 * close.length, 12 * close.length);
    assert.strictEqual(lag1.findIndex(v => !isNaN(v)), 20);
    assert.strictEqual(lag12.findIndex(v => !isNaN(v)), 31);
});

test('autocorrelation_indicator_js rejects invalid parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 64));

    assert.throws(() => {
        wasm.autocorrelation_indicator_js(close, 0, 8, false);
    }, /Invalid length/);

    assert.throws(() => {
        wasm.autocorrelation_indicator_js(close, 20, 0, false);
    }, /Invalid max_lag/);
});

test('autocorrelation_indicator_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 180));
    const safe = wasm.autocorrelation_indicator_js(close, 20, 8, false);
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.autocorrelation_indicator_alloc(close.length, 8);

    try {
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        wasm.autocorrelation_indicator_into(closePtr, outPtr, close.length, 20, 8, false);
        const flat = new Float64Array(memory.buffer, outPtr, close.length * 9);
        const filtered = Array.from(flat.slice(0, close.length));
        const correlations = Array.from(flat.slice(close.length));
        assertArrayClose(filtered, safe.filtered, 1e-12, 'pointer filtered mismatch');
        assertArrayClose(correlations, safe.correlations, 1e-12, 'pointer correlations mismatch');
    } finally {
        wasm.autocorrelation_indicator_free(outPtr, close.length, 8);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('autocorrelation_indicator_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 180));
    const batch = wasm.autocorrelation_indicator_batch_js(close, {
        length_range: [20, 20, 0],
        max_lag: 8,
        use_test_signal: false,
    });
    const single = wasm.autocorrelation_indicator_js(close, 20, 8, false);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.lag_count, 8);
    assert.strictEqual(batch.combos[0].length, 20);
    assertArrayClose(batch.filtered, single.filtered, 1e-12, 'batch filtered mismatch');
    assertArrayClose(batch.correlations, single.correlations, 1e-12, 'batch correlations mismatch');
});

test('autocorrelation_indicator_batch_into metadata matches requested ranges', () => {
    const close = new Float64Array(testData.close.slice(0, 180));
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 3;
    const lagCount = 6;
    const total = rows * close.length;
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.autocorrelation_indicator_alloc(total, lagCount);

    try {
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        const actualRows = wasm.autocorrelation_indicator_batch_into(
            closePtr,
            outPtr,
            close.length,
            16,
            20,
            2,
            lagCount,
            true,
        );
        assert.strictEqual(actualRows, rows);

        const flat = new Float64Array(memory.buffer, outPtr, total * (lagCount + 1));
        const filtered = Array.from(flat.slice(0, total));
        const correlations = Array.from(flat.slice(total));
        const jsBatch = wasm.autocorrelation_indicator_batch_js(close, {
            length_range: [16, 20, 2],
            max_lag: lagCount,
            use_test_signal: true,
        });
        assertArrayClose(filtered, jsBatch.filtered, 1e-12, 'batch_into filtered mismatch');
        assertArrayClose(
            correlations,
            jsBatch.correlations,
            1e-12,
            'batch_into correlations mismatch',
        );
    } finally {
        wasm.autocorrelation_indicator_free(outPtr, total, lagCount);
        wasm.deallocate_f64_array(closePtr);
    }
});
