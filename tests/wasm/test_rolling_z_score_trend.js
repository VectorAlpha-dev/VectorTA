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

test('rolling_z_score_trend_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.rolling_z_score_trend_js(close, 20);

    assert.strictEqual(result.zscore.length, close.length);
    assert.strictEqual(result.momentum.length, close.length);
    assert.strictEqual(result.zscore.findIndex(v => !isNaN(v)), 19);
    assert.strictEqual(result.momentum.findIndex(v => !isNaN(v)), 20);
});

test('rolling_z_score_trend_js rejects invalid parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 64));

    assert.throws(() => {
        wasm.rolling_z_score_trend_js(close, 0);
    }, /Invalid lookback_period/);
});

test('rolling_z_score_trend_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 220));
    const safe = wasm.rolling_z_score_trend_js(close, 20);
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.rolling_z_score_trend_alloc(close.length);

    try {
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        wasm.rolling_z_score_trend_into(closePtr, outPtr, close.length, 20);
        const flat = new Float64Array(memory.buffer, outPtr, 2 * close.length);
        const zscore = Array.from(flat.slice(0, close.length));
        const momentum = Array.from(flat.slice(close.length));
        assertArrayClose(zscore, safe.zscore, 1e-12, 'pointer zscore mismatch');
        assertArrayClose(momentum, safe.momentum, 1e-12, 'pointer momentum mismatch');
    } finally {
        wasm.rolling_z_score_trend_free(outPtr, close.length);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('rolling_z_score_trend_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 220));
    const batch = wasm.rolling_z_score_trend_batch_js(close, {
        lookback_period_range: [20, 20, 0],
    });
    const single = wasm.rolling_z_score_trend_js(close, 20);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].lookback_period, 20);
    assertArrayClose(batch.zscore, single.zscore, 1e-12, 'batch zscore mismatch');
    assertArrayClose(batch.momentum, single.momentum, 1e-12, 'batch momentum mismatch');
});

test('rolling_z_score_trend_batch_into metadata matches requested ranges', () => {
    const close = new Float64Array(testData.close.slice(0, 180));
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 3;
    const total = rows * close.length;
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.rolling_z_score_trend_alloc(total);

    try {
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        const actualRows = wasm.rolling_z_score_trend_batch_into(
            closePtr,
            outPtr,
            close.length,
            10,
            20,
            5,
        );
        assert.strictEqual(actualRows, rows);

        const flat = new Float64Array(memory.buffer, outPtr, 2 * total);
        const zscore = Array.from(flat.slice(0, total));
        const momentum = Array.from(flat.slice(total));
        const jsBatch = wasm.rolling_z_score_trend_batch_js(close, {
            lookback_period_range: [10, 20, 5],
        });
        assertArrayClose(zscore, jsBatch.zscore, 1e-12, 'batch_into zscore mismatch');
        assertArrayClose(momentum, jsBatch.momentum, 1e-12, 'batch_into momentum mismatch');
    } finally {
        wasm.rolling_z_score_trend_free(outPtr, total);
        wasm.deallocate_f64_array(closePtr);
    }
});
