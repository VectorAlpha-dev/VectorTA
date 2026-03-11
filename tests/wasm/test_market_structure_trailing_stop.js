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

test('market_structure_trailing_stop_js output contract', () => {
    const open = new Float64Array(testData.open.slice(0, 256));
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.market_structure_trailing_stop_js(open, high, low, close, 5, 100.0, 'All');

    assert.strictEqual(result.trailing_stop.length, close.length);
    assert.strictEqual(result.state.length, close.length);
    assert.strictEqual(result.structure.length, close.length);
    assert(result.trailing_stop.some(v => !isNaN(v)));

    for (const value of result.state) {
        if (!isNaN(value)) {
            assert(value === -1 || value === 0 || value === 1);
        }
    }
});

test('market_structure_trailing_stop_js rejects invalid parameters', () => {
    const open = new Float64Array(testData.open.slice(0, 64));
    const high = new Float64Array(testData.high.slice(0, 64));
    const low = new Float64Array(testData.low.slice(0, 64));
    const close = new Float64Array(testData.close.slice(0, 64));

    assert.throws(() => {
        wasm.market_structure_trailing_stop_js(open, high, low, close, 0, 100.0, 'CHoCH');
    }, /Invalid length/);

    assert.throws(() => {
        wasm.market_structure_trailing_stop_js(open, high, low, close, 5, -1.0, 'CHoCH');
    }, /Invalid increment_factor/);

    assert.throws(() => {
        wasm.market_structure_trailing_stop_js(open, high, low, close, 5, 100.0, 'bad');
    }, /Invalid reset_on/);
});

test('market_structure_trailing_stop_into pointer path matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 220));
    const high = new Float64Array(testData.high.slice(0, 220));
    const low = new Float64Array(testData.low.slice(0, 220));
    const close = new Float64Array(testData.close.slice(0, 220));
    const safe = wasm.market_structure_trailing_stop_js(open, high, low, close, 5, 100.0, 'All');
    const memory = getMemory();
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const openPtr = wasm.allocate_f64_array(open.length);
    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.market_structure_trailing_stop_alloc(close.length);

    try {
        new Float64Array(memory.buffer, openPtr, open.length).set(open);
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);

        wasm.market_structure_trailing_stop_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            close.length,
            5,
            100.0,
            'All',
        );

        const flat = new Float64Array(memory.buffer, outPtr, 3 * close.length);
        const trailingStop = Array.from(flat.slice(0, close.length));
        const state = Array.from(flat.slice(close.length, 2 * close.length));
        const structure = Array.from(flat.slice(2 * close.length));

        assertArrayClose(trailingStop, safe.trailing_stop, 1e-12, 'pointer trailing_stop mismatch');
        assertArrayClose(state, safe.state, 1e-12, 'pointer state mismatch');
        assertArrayClose(structure, safe.structure, 1e-12, 'pointer structure mismatch');
    } finally {
        wasm.market_structure_trailing_stop_free(outPtr, close.length);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('market_structure_trailing_stop_batch_js single parameter set matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 200));
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.market_structure_trailing_stop_batch_js(open, high, low, close, {
        length_range: [5, 5, 0],
        increment_factor_range: [100.0, 100.0, 0.0],
        reset_on: 'All',
    });
    const single = wasm.market_structure_trailing_stop_js(open, high, low, close, 5, 100.0, 'All');

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].length, 5);
    assert.strictEqual(batch.combos[0].increment_factor, 100.0);
    assert.strictEqual(batch.combos[0].reset_on, 'All');
    assertArrayClose(batch.trailing_stop, single.trailing_stop, 1e-12, 'batch trailing_stop mismatch');
    assertArrayClose(batch.state, single.state, 1e-12, 'batch state mismatch');
    assertArrayClose(batch.structure, single.structure, 1e-12, 'batch structure mismatch');
});

test('market_structure_trailing_stop_batch_into matches batch_js', () => {
    const open = new Float64Array(testData.open.slice(0, 160));
    const high = new Float64Array(testData.high.slice(0, 160));
    const low = new Float64Array(testData.low.slice(0, 160));
    const close = new Float64Array(testData.close.slice(0, 160));
    const memory = getMemory();
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 3;
    const total = rows * close.length;
    const openPtr = wasm.allocate_f64_array(open.length);
    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.market_structure_trailing_stop_alloc(total);

    try {
        new Float64Array(memory.buffer, openPtr, open.length).set(open);
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);

        const actualRows = wasm.market_structure_trailing_stop_batch_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            close.length,
            5,
            9,
            2,
            100.0,
            100.0,
            0.0,
            'All',
        );
        assert.strictEqual(actualRows, rows);

        const flat = new Float64Array(memory.buffer, outPtr, 3 * total);
        const trailingStop = Array.from(flat.slice(0, total));
        const state = Array.from(flat.slice(total, 2 * total));
        const structure = Array.from(flat.slice(2 * total));

        const jsBatch = wasm.market_structure_trailing_stop_batch_js(open, high, low, close, {
            length_range: [5, 9, 2],
            increment_factor_range: [100.0, 100.0, 0.0],
            reset_on: 'All',
        });
        assertArrayClose(trailingStop, jsBatch.trailing_stop, 1e-12, 'batch_into trailing_stop mismatch');
        assertArrayClose(state, jsBatch.state, 1e-12, 'batch_into state mismatch');
        assertArrayClose(structure, jsBatch.structure, 1e-12, 'batch_into structure mismatch');
    } finally {
        wasm.market_structure_trailing_stop_free(outPtr, total);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});
