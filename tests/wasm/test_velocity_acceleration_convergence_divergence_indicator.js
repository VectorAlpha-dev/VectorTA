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

function makeHlcc4(limit) {
    const high = testData.high.slice(0, limit);
    const low = testData.low.slice(0, limit);
    const close = testData.close.slice(0, limit);
    return new Float64Array(close.map((c, i) => (high[i] + low[i] + 2 * c) / 4));
}

test('velocity_acceleration_convergence_divergence_indicator_js output contract', () => {
    const data = makeHlcc4(256);
    const result = wasm.velocity_acceleration_convergence_divergence_indicator_js(data, 21, 5);

    assert.strictEqual(result.vacd.length, data.length);
    assert.strictEqual(result.signal.length, data.length);
    assert.strictEqual(result.vacd.findIndex(v => !isNaN(v)), 4);
    assert.strictEqual(result.signal.findIndex(v => !isNaN(v)), 4);
});

test('velocity_acceleration_convergence_divergence_indicator_js rejects invalid params', () => {
    const data = makeHlcc4(64);

    assert.throws(() => {
        wasm.velocity_acceleration_convergence_divergence_indicator_js(data, 1, 5);
    }, /Invalid length/);
});

test('velocity_acceleration_convergence_divergence_indicator_into pointer path matches safe API', () => {
    const data = makeHlcc4(220);
    const safe = wasm.velocity_acceleration_convergence_divergence_indicator_js(data, 21, 5);
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const dataPtr = wasm.allocate_f64_array(data.length);
    const outPtr = wasm.velocity_acceleration_convergence_divergence_indicator_alloc(data.length);

    try {
        new Float64Array(memory.buffer, dataPtr, data.length).set(data);
        wasm.velocity_acceleration_convergence_divergence_indicator_into(
            dataPtr,
            outPtr,
            data.length,
            21,
            5,
        );
        const flat = new Float64Array(memory.buffer, outPtr, 2 * data.length);
        const vacd = Array.from(flat.slice(0, data.length));
        const signal = Array.from(flat.slice(data.length));
        assertArrayClose(vacd, safe.vacd, 1e-12, 'pointer vacd mismatch');
        assertArrayClose(signal, safe.signal, 1e-12, 'pointer signal mismatch');
    } finally {
        wasm.velocity_acceleration_convergence_divergence_indicator_free(outPtr, data.length);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('velocity_acceleration_convergence_divergence_indicator_batch_js single parameter set matches safe API', () => {
    const data = makeHlcc4(220);
    const batch = wasm.velocity_acceleration_convergence_divergence_indicator_batch_js(data, {
        length_range: [21, 21, 0],
        smooth_length_range: [5, 5, 0],
    });
    const single = wasm.velocity_acceleration_convergence_divergence_indicator_js(data, 21, 5);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, data.length);
    assert.strictEqual(batch.combos[0].length, 21);
    assert.strictEqual(batch.combos[0].smooth_length, 5);
    assertArrayClose(batch.vacd, single.vacd, 1e-12, 'batch vacd mismatch');
    assertArrayClose(batch.signal, single.signal, 1e-12, 'batch signal mismatch');
});

test('velocity_acceleration_convergence_divergence_indicator_batch_into metadata matches requested ranges', () => {
    const data = makeHlcc4(180);
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 6;
    const total = rows * data.length;
    const dataPtr = wasm.allocate_f64_array(data.length);
    const outPtr = wasm.velocity_acceleration_convergence_divergence_indicator_alloc(total);

    try {
        new Float64Array(memory.buffer, dataPtr, data.length).set(data);
        const actualRows = wasm.velocity_acceleration_convergence_divergence_indicator_batch_into(
            dataPtr,
            outPtr,
            data.length,
            21,
            25,
            2,
            3,
            5,
            2,
        );
        assert.strictEqual(actualRows, rows);

        const flat = new Float64Array(memory.buffer, outPtr, 2 * total);
        const vacd = Array.from(flat.slice(0, total));
        const signal = Array.from(flat.slice(total));
        const jsBatch = wasm.velocity_acceleration_convergence_divergence_indicator_batch_js(data, {
            length_range: [21, 25, 2],
            smooth_length_range: [3, 5, 2],
        });
        assertArrayClose(vacd, jsBatch.vacd, 1e-12, 'batch_into vacd mismatch');
        assertArrayClose(signal, jsBatch.signal, 1e-12, 'batch_into signal mismatch');
    } finally {
        wasm.velocity_acceleration_convergence_divergence_indicator_free(outPtr, total);
        wasm.deallocate_f64_array(dataPtr);
    }
});
