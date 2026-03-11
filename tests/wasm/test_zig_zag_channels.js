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

test('zig_zag_channels_js output contract', () => {
    const open = new Float64Array(testData.open.slice(0, 256));
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.zig_zag_channels_js(open, high, low, close, 12, true);

    assert.strictEqual(result.middle.length, close.length);
    assert.strictEqual(result.upper.length, close.length);
    assert.strictEqual(result.lower.length, close.length);
    assert(result.middle.some(v => !isNaN(v)));

    for (let i = 0; i < close.length; i++) {
        if (!isNaN(result.middle[i]) && !isNaN(result.upper[i]) && !isNaN(result.lower[i])) {
            assert(result.upper[i] >= result.middle[i]);
            assert(result.middle[i] >= result.lower[i]);
        }
    }
});

test('zig_zag_channels_js rejects invalid parameters', () => {
    const open = new Float64Array(testData.open.slice(0, 64));
    const high = new Float64Array(testData.high.slice(0, 64));
    const low = new Float64Array(testData.low.slice(0, 64));
    const close = new Float64Array(testData.close.slice(0, 64));

    assert.throws(() => {
        wasm.zig_zag_channels_js(open, high, low, close, 0, true);
    }, /Invalid length/);

    assert.throws(() => {
        wasm.zig_zag_channels_js(open, high, low, close, close.length, true);
    }, /Not enough valid data/);
});

test('zig_zag_channels_into pointer path matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 200));
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const close = new Float64Array(testData.close.slice(0, 200));
    const safe = wasm.zig_zag_channels_js(open, high, low, close, 12, true);
    const memory = getMemory();
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const openPtr = wasm.allocate_f64_array(open.length);
    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.zig_zag_channels_alloc(close.length);

    try {
        new Float64Array(memory.buffer, openPtr, open.length).set(open);
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);

        wasm.zig_zag_channels_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            close.length,
            12,
            true,
        );

        const flat = new Float64Array(memory.buffer, outPtr, 3 * close.length);
        const middle = Array.from(flat.slice(0, close.length));
        const upper = Array.from(flat.slice(close.length, 2 * close.length));
        const lower = Array.from(flat.slice(2 * close.length));

        assertArrayClose(middle, safe.middle, 1e-12, 'pointer middle mismatch');
        assertArrayClose(upper, safe.upper, 1e-12, 'pointer upper mismatch');
        assertArrayClose(lower, safe.lower, 1e-12, 'pointer lower mismatch');
    } finally {
        wasm.zig_zag_channels_free(outPtr, close.length);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('zig_zag_channels_batch_js single parameter set matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 200));
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.zig_zag_channels_batch_js(open, high, low, close, {
        length_range: [12, 12, 0],
        extend: true,
    });
    const single = wasm.zig_zag_channels_js(open, high, low, close, 12, true);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].length, 12);
    assert.strictEqual(batch.combos[0].extend, true);
    assertArrayClose(batch.middle, single.middle, 1e-12, 'batch middle mismatch');
    assertArrayClose(batch.upper, single.upper, 1e-12, 'batch upper mismatch');
    assertArrayClose(batch.lower, single.lower, 1e-12, 'batch lower mismatch');
});

test('zig_zag_channels_batch_into matches batch_js', () => {
    const open = new Float64Array(testData.open.slice(0, 128));
    const high = new Float64Array(testData.high.slice(0, 128));
    const low = new Float64Array(testData.low.slice(0, 128));
    const close = new Float64Array(testData.close.slice(0, 128));
    const memory = getMemory();
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 3;
    const total = rows * close.length;
    const openPtr = wasm.allocate_f64_array(open.length);
    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.zig_zag_channels_alloc(total);

    try {
        new Float64Array(memory.buffer, openPtr, open.length).set(open);
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);

        const actualRows = wasm.zig_zag_channels_batch_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            close.length,
            8,
            12,
            2,
            false,
        );
        assert.strictEqual(actualRows, rows);

        const flat = new Float64Array(memory.buffer, outPtr, 3 * total);
        const middle = Array.from(flat.slice(0, total));
        const upper = Array.from(flat.slice(total, 2 * total));
        const lower = Array.from(flat.slice(2 * total));

        const jsBatch = wasm.zig_zag_channels_batch_js(open, high, low, close, {
            length_range: [8, 12, 2],
            extend: false,
        });
        assertArrayClose(middle, jsBatch.middle, 1e-12, 'batch_into middle mismatch');
        assertArrayClose(upper, jsBatch.upper, 1e-12, 'batch_into upper mismatch');
        assertArrayClose(lower, jsBatch.lower, 1e-12, 'batch_into lower mismatch');
    } finally {
        wasm.zig_zag_channels_free(outPtr, total);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});
