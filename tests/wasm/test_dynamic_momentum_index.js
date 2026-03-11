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

test('dynamic_momentum_index_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.dynamic_momentum_index_js(close, 14, 5, 10, 30, 5);

    assert.strictEqual(result.length, close.length);
    const valid = result.filter(v => !isNaN(v));
    assert(valid.length > 0, 'expected at least one finite output');
    const first = result.findIndex(v => !isNaN(v));
    assert(first >= 13, `unexpected first valid index: ${first}`);
    for (const value of valid) {
        assert(value >= 0 && value <= 100, `value out of range: ${value}`);
    }
});

test('dynamic_momentum_index_js rejects invalid parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 32));

    assert.throws(() => {
        wasm.dynamic_momentum_index_js(close, 0, 5, 10, 30, 5);
    }, /Invalid RSI period/);

    assert.throws(() => {
        wasm.dynamic_momentum_index_js(close, 14, 5, 10, 4, 5);
    }, /Invalid limits/);
});

test('dynamic_momentum_index_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 200));
    const safe = wasm.dynamic_momentum_index_js(close, 14, 5, 10, 30, 5);
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.dynamic_momentum_index_alloc(close.length);

    try {
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        wasm.dynamic_momentum_index_into(closePtr, outPtr, close.length, 14, 5, 10, 30, 5);
        const out = Array.from(new Float64Array(memory.buffer, outPtr, close.length));
        assertArrayClose(out, safe, 1e-12, 'pointer values mismatch');
    } finally {
        wasm.dynamic_momentum_index_free(outPtr, close.length);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('dynamic_momentum_index_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.dynamic_momentum_index_batch_js(close, {
        rsi_period_range: [14, 14, 0],
        volatility_period_range: [5, 5, 0],
        volatility_sma_period_range: [10, 10, 0],
        upper_limit_range: [30, 30, 0],
        lower_limit_range: [5, 5, 0],
    });
    const single = wasm.dynamic_momentum_index_js(close, 14, 5, 10, 30, 5);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.values.length, close.length);
    assert.strictEqual(batch.combos[0].rsi_period, 14);
    assertArrayClose(batch.values, single, 1e-12, 'batch values mismatch');
});

test('dynamic_momentum_index_batch_into metadata matches requested ranges', () => {
    const close = new Float64Array(testData.close.slice(0, 128));
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 3;
    const total = rows * close.length;
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.dynamic_momentum_index_alloc(total);

    try {
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        const actualRows = wasm.dynamic_momentum_index_batch_into(
            closePtr,
            outPtr,
            close.length,
            10, 14, 2,
            5, 5, 0,
            10, 10, 0,
            30, 30, 0,
            5, 5, 0,
        );
        assert.strictEqual(actualRows, rows);

        const values = Array.from(new Float64Array(memory.buffer, outPtr, total));
        const jsBatch = wasm.dynamic_momentum_index_batch_js(close, {
            rsi_period_range: [10, 14, 2],
            volatility_period_range: [5, 5, 0],
            volatility_sma_period_range: [10, 10, 0],
            upper_limit_range: [30, 30, 0],
            lower_limit_range: [5, 5, 0],
        });
        assertArrayClose(values, jsBatch.values, 1e-12, 'batch_into values mismatch');
    } finally {
        wasm.dynamic_momentum_index_free(outPtr, total);
        wasm.deallocate_f64_array(closePtr);
    }
});
