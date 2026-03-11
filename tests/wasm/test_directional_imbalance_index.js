import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

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

test('directional_imbalance_index_js output contract', () => {
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const result = wasm.directional_imbalance_index_js(high, low, 10, 70);

    assert.strictEqual(result.up.length, high.length);
    assert.strictEqual(result.down.length, high.length);
    assert.strictEqual(result.bulls.length, high.length);
    assert.strictEqual(result.bears.length, high.length);
    assert.strictEqual(result.upper.length, high.length);
    assert.strictEqual(result.lower.length, high.length);
    assert.strictEqual(result.up[0], 1);
    assert.strictEqual(result.down[0], 1);
    assert.strictEqual(result.bulls[0], 50);
    assert.strictEqual(result.bears[0], 50);
    assert.strictEqual(result.upper[0], high[0]);
    assert.strictEqual(result.lower[0], low[0]);
});

test('directional_imbalance_index_js rejects invalid parameters', () => {
    const high = new Float64Array(testData.high.slice(0, 128));
    const low = new Float64Array(testData.low.slice(0, 128));

    assert.throws(() => {
        wasm.directional_imbalance_index_js(high, low, 0, 70);
    }, /Invalid length/);

    assert.throws(() => {
        wasm.directional_imbalance_index_js(high, low, 10, 0);
    }, /Invalid period/);
});

test('directional_imbalance_index_into pointer path matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const safe = wasm.directional_imbalance_index_js(high, low, 7, 20);
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const outPtr = wasm.directional_imbalance_index_alloc(high.length);

    try {
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        wasm.directional_imbalance_index_into(highPtr, lowPtr, outPtr, high.length, 7, 20);
        const flat = new Float64Array(memory.buffer, outPtr, 6 * high.length);
        assertArrayClose(Array.from(flat.slice(0, high.length)), safe.up, 1e-12, 'pointer up mismatch');
        assertArrayClose(Array.from(flat.slice(high.length, 2 * high.length)), safe.down, 1e-12, 'pointer down mismatch');
        assertArrayClose(Array.from(flat.slice(2 * high.length, 3 * high.length)), safe.bulls, 1e-12, 'pointer bulls mismatch');
        assertArrayClose(Array.from(flat.slice(3 * high.length, 4 * high.length)), safe.bears, 1e-12, 'pointer bears mismatch');
        assertArrayClose(Array.from(flat.slice(4 * high.length, 5 * high.length)), safe.upper, 1e-12, 'pointer upper mismatch');
        assertArrayClose(Array.from(flat.slice(5 * high.length, 6 * high.length)), safe.lower, 1e-12, 'pointer lower mismatch');
    } finally {
        wasm.directional_imbalance_index_free(outPtr, high.length);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
    }
});

test('directional_imbalance_index_batch_js single parameter set matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const batch = wasm.directional_imbalance_index_batch_js(high, low, {
        length_range: [10, 10, 0],
        period_range: [70, 70, 0],
    });
    const single = wasm.directional_imbalance_index_js(high, low, 10, 70);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, high.length);
    assert.strictEqual(batch.combos[0].length, 10);
    assert.strictEqual(batch.combos[0].period, 70);
    assertArrayClose(batch.up, single.up, 1e-12, 'batch up mismatch');
    assertArrayClose(batch.down, single.down, 1e-12, 'batch down mismatch');
    assertArrayClose(batch.bulls, single.bulls, 1e-12, 'batch bulls mismatch');
    assertArrayClose(batch.bears, single.bears, 1e-12, 'batch bears mismatch');
});

test('directional_imbalance_index_batch_into matches batch_js', () => {
    const high = new Float64Array(testData.high.slice(0, 180));
    const low = new Float64Array(testData.low.slice(0, 180));
    const rows = 3;
    const total = rows * high.length;
    const jsBatch = wasm.directional_imbalance_index_batch_js(high, low, {
        length_range: [6, 10, 2],
        period_range: [14, 14, 0],
    });
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const outPtr = wasm.directional_imbalance_index_alloc(total);

    try {
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        const actualRows = wasm.directional_imbalance_index_batch_into(
            highPtr, lowPtr, outPtr, high.length, 6, 10, 2, 14, 14, 0
        );
        assert.strictEqual(actualRows, rows);
        const flat = new Float64Array(memory.buffer, outPtr, 6 * total);
        assertArrayClose(Array.from(flat.slice(0, total)), jsBatch.up, 1e-12, 'batch_into up mismatch');
        assertArrayClose(Array.from(flat.slice(total, 2 * total)), jsBatch.down, 1e-12, 'batch_into down mismatch');
        assertArrayClose(Array.from(flat.slice(2 * total, 3 * total)), jsBatch.bulls, 1e-12, 'batch_into bulls mismatch');
        assertArrayClose(Array.from(flat.slice(3 * total, 4 * total)), jsBatch.bears, 1e-12, 'batch_into bears mismatch');
        assertArrayClose(Array.from(flat.slice(4 * total, 5 * total)), jsBatch.upper, 1e-12, 'batch_into upper mismatch');
        assertArrayClose(Array.from(flat.slice(5 * total, 6 * total)), jsBatch.lower, 1e-12, 'batch_into lower mismatch');
    } finally {
        wasm.directional_imbalance_index_free(outPtr, total);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
    }
});
