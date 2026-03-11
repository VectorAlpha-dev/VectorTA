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

test('nonlinear_regression_zero_lag_moving_average_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.nonlinear_regression_zero_lag_moving_average_js(close, 15, 15);

    assert.strictEqual(result.value.length, close.length);
    assert.strictEqual(result.signal.length, close.length);
    assert.strictEqual(result.long_signal.length, close.length);
    assert.strictEqual(result.short_signal.length, close.length);
    assert.strictEqual(result.value.findIndex(v => !isNaN(v)), 42);
    assert.strictEqual(result.signal.findIndex(v => !isNaN(v)), 43);
});

test('nonlinear_regression_zero_lag_moving_average_js rejects invalid params', () => {
    const close = new Float64Array(testData.close.slice(0, 128));

    assert.throws(() => {
        wasm.nonlinear_regression_zero_lag_moving_average_js(close, 0, 15);
    }, /Invalid zlma_period/);

    assert.throws(() => {
        wasm.nonlinear_regression_zero_lag_moving_average_js(close, 15, 0);
    }, /Invalid regression_period/);
});

test('nonlinear_regression_zero_lag_moving_average_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 260));
    const safe = wasm.nonlinear_regression_zero_lag_moving_average_js(close, 17, 11);
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const dataPtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.nonlinear_regression_zero_lag_moving_average_alloc(close.length);

    try {
        new Float64Array(memory.buffer, dataPtr, close.length).set(close);
        wasm.nonlinear_regression_zero_lag_moving_average_into(
            dataPtr,
            outPtr,
            close.length,
            17,
            11
        );
        const flat = new Float64Array(memory.buffer, outPtr, 4 * close.length);
        const value = Array.from(flat.slice(0, close.length));
        const signal = Array.from(flat.slice(close.length, 2 * close.length));
        const longSignal = Array.from(flat.slice(2 * close.length, 3 * close.length));
        const shortSignal = Array.from(flat.slice(3 * close.length));
        assertArrayClose(value, safe.value, 1e-12, 'pointer value mismatch');
        assertArrayClose(signal, safe.signal, 1e-12, 'pointer signal mismatch');
        assertArrayClose(longSignal, safe.long_signal, 1e-12, 'pointer long mismatch');
        assertArrayClose(shortSignal, safe.short_signal, 1e-12, 'pointer short mismatch');
    } finally {
        wasm.nonlinear_regression_zero_lag_moving_average_free(outPtr, close.length);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('nonlinear_regression_zero_lag_moving_average_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 240));
    const batch = wasm.nonlinear_regression_zero_lag_moving_average_batch_js(close, {
        zlma_period_range: [15, 15, 0],
        regression_period_range: [15, 15, 0],
    });
    const single = wasm.nonlinear_regression_zero_lag_moving_average_js(close, 15, 15);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].zlma_period, 15);
    assert.strictEqual(batch.combos[0].regression_period, 15);
    assertArrayClose(batch.value, single.value, 1e-12, 'batch value mismatch');
    assertArrayClose(batch.signal, single.signal, 1e-12, 'batch signal mismatch');
});

test('nonlinear_regression_zero_lag_moving_average_batch_into metadata matches requested ranges', () => {
    const close = new Float64Array(testData.close.slice(0, 220));
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 9;
    const total = rows * close.length;
    const dataPtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.nonlinear_regression_zero_lag_moving_average_alloc(total);

    try {
        new Float64Array(memory.buffer, dataPtr, close.length).set(close);
        const actualRows = wasm.nonlinear_regression_zero_lag_moving_average_batch_into(
            dataPtr,
            outPtr,
            close.length,
            14,
            16,
            1,
            10,
            12,
            1
        );
        assert.strictEqual(actualRows, rows);

        const flat = new Float64Array(memory.buffer, outPtr, 4 * total);
        const value = Array.from(flat.slice(0, total));
        const signal = Array.from(flat.slice(total, 2 * total));
        const jsBatch = wasm.nonlinear_regression_zero_lag_moving_average_batch_js(close, {
            zlma_period_range: [14, 16, 1],
            regression_period_range: [10, 12, 1],
        });
        assertArrayClose(value, jsBatch.value, 1e-12, 'batch_into value mismatch');
        assertArrayClose(signal, jsBatch.signal, 1e-12, 'batch_into signal mismatch');
    } finally {
        wasm.nonlinear_regression_zero_lag_moving_average_free(outPtr, total);
        wasm.deallocate_f64_array(dataPtr);
    }
});
