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

function getMemory() {
    return wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
}

test('emd_trend_js output contract', () => {
    const open = new Float64Array(testData.open.slice(0, 256));
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.emd_trend_js(open, high, low, close, 'close', 'SMA', 28, 1.0);

    assert.strictEqual(result.direction.length, close.length);
    assert.strictEqual(result.average.length, close.length);
    assert.strictEqual(result.upper.length, close.length);
    assert.strictEqual(result.lower.length, close.length);
    assert(result.average.some(v => !isNaN(v)));
});

test('emd_trend_js rejects invalid parameters', () => {
    const open = new Float64Array(testData.open.slice(0, 96));
    const high = new Float64Array(testData.high.slice(0, 96));
    const low = new Float64Array(testData.low.slice(0, 96));
    const close = new Float64Array(testData.close.slice(0, 96));

    assert.throws(() => {
        wasm.emd_trend_js(open, high, low, close, 'bad', 'SMA', 28, 1.0);
    }, /Invalid source/);

    assert.throws(() => {
        wasm.emd_trend_js(open, high, low, close, 'close', 'bad', 28, 1.0);
    }, /Invalid avg_type/);
});

test('emd_trend_into pointer path matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 220));
    const high = new Float64Array(testData.high.slice(0, 220));
    const low = new Float64Array(testData.low.slice(0, 220));
    const close = new Float64Array(testData.close.slice(0, 220));
    const safe = wasm.emd_trend_js(open, high, low, close, 'close', 'SMA', 28, 1.0);
    const memory = getMemory();
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const openPtr = wasm.allocate_f64_array(open.length);
    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.emd_trend_alloc(close.length);

    try {
        new Float64Array(memory.buffer, openPtr, open.length).set(open);
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);

        wasm.emd_trend_into(
            openPtr, highPtr, lowPtr, closePtr, outPtr, close.length, 'close', 'SMA', 28, 1.0
        );

        const flat = new Float64Array(memory.buffer, outPtr, 4 * close.length);
        const direction = Array.from(flat.slice(0, close.length));
        const average = Array.from(flat.slice(close.length, 2 * close.length));
        const upper = Array.from(flat.slice(2 * close.length, 3 * close.length));
        const lower = Array.from(flat.slice(3 * close.length));

        assertArrayClose(direction, safe.direction, 1e-12, 'pointer direction mismatch');
        assertArrayClose(average, safe.average, 1e-12, 'pointer average mismatch');
        assertArrayClose(upper, safe.upper, 1e-12, 'pointer upper mismatch');
        assertArrayClose(lower, safe.lower, 1e-12, 'pointer lower mismatch');
    } finally {
        wasm.emd_trend_free(outPtr, close.length);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('emd_trend_batch_js single parameter set matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 200));
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.emd_trend_batch_js(open, high, low, close, {
        length_range: [28, 28, 0],
        mult_range: [1.0, 1.0, 0.0],
        source: 'close',
        avg_type: 'SMA',
    });
    const single = wasm.emd_trend_js(open, high, low, close, 'close', 'SMA', 28, 1.0);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].length, 28);
    assert.strictEqual(batch.combos[0].mult, 1.0);
    assert.strictEqual(batch.combos[0].source, 'close');
    assert.strictEqual(batch.combos[0].avg_type, 'SMA');
    assertArrayClose(batch.direction, single.direction, 1e-12, 'batch direction mismatch');
    assertArrayClose(batch.average, single.average, 1e-12, 'batch average mismatch');
    assertArrayClose(batch.upper, single.upper, 1e-12, 'batch upper mismatch');
    assertArrayClose(batch.lower, single.lower, 1e-12, 'batch lower mismatch');
});

test('emd_trend_batch_into matches batch_js', () => {
    const open = new Float64Array(testData.open.slice(0, 160));
    const high = new Float64Array(testData.high.slice(0, 160));
    const low = new Float64Array(testData.low.slice(0, 160));
    const close = new Float64Array(testData.close.slice(0, 160));
    const memory = getMemory();
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 4;
    const total = rows * close.length;
    const openPtr = wasm.allocate_f64_array(open.length);
    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.emd_trend_alloc(total);

    try {
        new Float64Array(memory.buffer, openPtr, open.length).set(open);
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);

        const actualRows = wasm.emd_trend_batch_into(
            openPtr, highPtr, lowPtr, closePtr, outPtr, close.length,
            28, 30, 2,
            1.0, 1.5, 0.5,
            'close', 'SMA',
        );
        assert.strictEqual(actualRows, rows);

        const flat = new Float64Array(memory.buffer, outPtr, 4 * total);
        const direction = Array.from(flat.slice(0, total));
        const average = Array.from(flat.slice(total, 2 * total));
        const upper = Array.from(flat.slice(2 * total, 3 * total));
        const lower = Array.from(flat.slice(3 * total));

        const jsBatch = wasm.emd_trend_batch_js(open, high, low, close, {
            length_range: [28, 30, 2],
            mult_range: [1.0, 1.5, 0.5],
            source: 'close',
            avg_type: 'SMA',
        });
        assertArrayClose(direction, jsBatch.direction, 1e-12, 'batch_into direction mismatch');
        assertArrayClose(average, jsBatch.average, 1e-12, 'batch_into average mismatch');
        assertArrayClose(upper, jsBatch.upper, 1e-12, 'batch_into upper mismatch');
        assertArrayClose(lower, jsBatch.lower, 1e-12, 'batch_into lower mismatch');
    } finally {
        wasm.emd_trend_free(outPtr, total);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});
