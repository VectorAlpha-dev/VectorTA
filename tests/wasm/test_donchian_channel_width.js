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

test('donchian_channel_width_js output contract', () => {
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const result = wasm.donchian_channel_width_js(high, low, 20);

    assert.strictEqual(result.length, high.length);
    assert.strictEqual(result.findIndex(v => !isNaN(v)), 19);
});

test('donchian_channel_width_js rejects invalid parameters', () => {
    const high = new Float64Array(testData.high.slice(0, 64));
    const low = new Float64Array(testData.low.slice(0, 64));

    assert.throws(() => {
        wasm.donchian_channel_width_js(high, low, 0);
    }, /Invalid period/);
});

test('donchian_channel_width_into pointer path matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const safe = wasm.donchian_channel_width_js(high, low, 20);
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const outPtr = wasm.donchian_channel_width_alloc(high.length);

    try {
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        wasm.donchian_channel_width_into(highPtr, lowPtr, outPtr, high.length, 20);
        const actual = Array.from(new Float64Array(memory.buffer, outPtr, high.length));
        assertArrayClose(actual, safe, 1e-12, 'pointer mismatch');
    } finally {
        wasm.donchian_channel_width_free(outPtr, high.length);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
    }
});

test('donchian_channel_width_batch_js single parameter set matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const batch = wasm.donchian_channel_width_batch_js(high, low, {
        period_range: [20, 20, 0],
    });
    const single = wasm.donchian_channel_width_js(high, low, 20);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, high.length);
    assert.strictEqual(batch.combos[0].period, 20);
    assertArrayClose(batch.values, single, 1e-12, 'batch mismatch');
});

test('donchian_channel_width_batch_into metadata matches requested ranges', () => {
    const high = new Float64Array(testData.high.slice(0, 128));
    const low = new Float64Array(testData.low.slice(0, 128));
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 3;
    const total = rows * high.length;
    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const outPtr = wasm.donchian_channel_width_alloc(total);

    try {
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        const actualRows = wasm.donchian_channel_width_batch_into(
            highPtr,
            lowPtr,
            outPtr,
            high.length,
            10,
            14,
            2,
        );
        assert.strictEqual(actualRows, rows);

        const flat = Array.from(new Float64Array(memory.buffer, outPtr, total));
        const jsBatch = wasm.donchian_channel_width_batch_js(high, low, {
            period_range: [10, 14, 2],
        });
        assertArrayClose(flat, jsBatch.values, 1e-12, 'batch_into mismatch');
    } finally {
        wasm.donchian_channel_width_free(outPtr, total);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
    }
});
