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

test('dual_ulcer_index_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.dual_ulcer_index_js(close, 5, true, 0.1);

    assert(result.long_ulcer);
    assert(result.short_ulcer);
    assert(result.threshold);
    assert.strictEqual(result.long_ulcer.length, close.length);
    assert.strictEqual(result.short_ulcer.length, close.length);
    assert.strictEqual(result.threshold.length, close.length);

    const longFirst = result.long_ulcer.findIndex(v => !isNaN(v));
    const shortFirst = result.short_ulcer.findIndex(v => !isNaN(v));
    const thresholdFirst = result.threshold.findIndex(v => !isNaN(v));
    assert.strictEqual(longFirst, 8);
    assert.strictEqual(shortFirst, 8);
    assert.strictEqual(thresholdFirst, 8);
});

test('dual_ulcer_index_js rejects invalid parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 128));

    assert.throws(() => {
        wasm.dual_ulcer_index_js(close, 0, true, 0.1);
    }, /Invalid period/);

    assert.throws(() => {
        wasm.dual_ulcer_index_js(close, 5, false, -0.1);
    }, /Invalid threshold/);
});

test('dual_ulcer_index_into pointer path matches safe API', (t) => {
    const close = new Float64Array(testData.close.slice(0, 200));
    const safe = wasm.dual_ulcer_index_js(close, 5, true, 0.1);
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    if (!memory || !memory.buffer) {
        t.skip('raw wasm memory is not exposed by this package build');
        return;
    }

    const inPtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.dual_ulcer_index_alloc(close.length);

    try {
        new Float64Array(memory.buffer, inPtr, close.length).set(close);
        wasm.dual_ulcer_index_into(
            inPtr,
            outPtr,
            close.length,
            5,
            true,
            0.1,
        );

        const out = new Float64Array(memory.buffer, outPtr, 3 * close.length);
        const longUlcer = Array.from(out.subarray(0, close.length));
        const shortUlcer = Array.from(out.subarray(close.length, 2 * close.length));
        const threshold = Array.from(out.subarray(2 * close.length, 3 * close.length));

        assertArrayClose(longUlcer, safe.long_ulcer, 1e-10, 'pointer long_ulcer mismatch');
        assertArrayClose(shortUlcer, safe.short_ulcer, 1e-10, 'pointer short_ulcer mismatch');
        assertArrayClose(threshold, safe.threshold, 1e-10, 'pointer threshold mismatch');
    } finally {
        wasm.dual_ulcer_index_free(outPtr, close.length);
        wasm.deallocate_f64_array(inPtr);
    }
});

test('dual_ulcer_index_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.dual_ulcer_index_batch_js(close, {
        period_range: [5, 5, 0],
        threshold_range: [0.1, 0.1, 0.0],
        auto_threshold: true,
    });
    const single = wasm.dual_ulcer_index_js(close, 5, true, 0.1);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.long_ulcer.length, close.length);
    assert.strictEqual(batch.short_ulcer.length, close.length);
    assert.strictEqual(batch.threshold.length, close.length);
    assert.strictEqual(batch.combos[0].period, 5);
    assert.strictEqual(batch.combos[0].auto_threshold, true);

    assertArrayClose(batch.long_ulcer, single.long_ulcer, 1e-10, 'batch long_ulcer mismatch');
    assertArrayClose(batch.short_ulcer, single.short_ulcer, 1e-10, 'batch short_ulcer mismatch');
    assertArrayClose(batch.threshold, single.threshold, 1e-10, 'batch threshold mismatch');
});

test('dual_ulcer_index_batch_js metadata matches requested ranges', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const batch = wasm.dual_ulcer_index_batch_js(close, {
        period_range: [5, 7, 2],
        threshold_range: [0.1, 0.3, 0.2],
        auto_threshold: false,
    });

    assert.strictEqual(batch.rows, 4);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(
        batch.combos.map(combo => combo.period),
        [5, 5, 7, 7],
    );
    assert.deepStrictEqual(
        batch.combos.map(combo => combo.threshold),
        [0.1, 0.30000000000000004, 0.1, 0.30000000000000004],
    );
    assert.deepStrictEqual(
        batch.combos.map(combo => combo.auto_threshold),
        [false, false, false, false],
    );
});
