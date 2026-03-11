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

test('goertzel_cycle_composite_wave_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 320));
    const result = wasm.goertzel_cycle_composite_wave_js(close, {
        max_period: 32,
        start_at_cycle: 1,
        use_top_cycles: 2,
    });

    assert.strictEqual(result.length, close.length);
    assert.strictEqual(result.findIndex(v => !isNaN(v)), 160);
});

test('goertzel_cycle_composite_wave_js rejects invalid parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 128));
    assert.throws(() => {
        wasm.goertzel_cycle_composite_wave_js(close, {
            max_period: 1,
            start_at_cycle: 1,
            use_top_cycles: 2,
        });
    }, /Invalid parameter max_period/);
});

test('goertzel_cycle_composite_wave_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 360));
    const safe = wasm.goertzel_cycle_composite_wave_js(close, {
        max_period: 32,
        start_at_cycle: 1,
        use_top_cycles: 2,
        bar_to_calculate: 1,
    });
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.goertzel_cycle_composite_wave_alloc(close.length);

    try {
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        wasm.goertzel_cycle_composite_wave_into(closePtr, outPtr, close.length, 32, 1, 2, 1);
        const flat = new Float64Array(memory.buffer, outPtr, close.length);
        assertArrayClose(Array.from(flat), safe, 1e-12, 'pointer mismatch');
    } finally {
        wasm.goertzel_cycle_composite_wave_free(outPtr, close.length);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('goertzel_cycle_composite_wave_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 360));
    const batch = wasm.goertzel_cycle_composite_wave_batch_js(close, {
        max_period_range: [32, 32, 0],
        start_at_cycle_range: [1, 1, 0],
        use_top_cycles_range: [2, 2, 0],
    });
    const single = wasm.goertzel_cycle_composite_wave_js(close, {
        max_period: 32,
        start_at_cycle: 1,
        use_top_cycles: 2,
    });

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].max_period, 32);
    assertArrayClose(batch.values, single, 1e-12, 'batch mismatch');
});

test('goertzel_cycle_composite_wave_batch_into metadata matches requested ranges', () => {
    const close = new Float64Array(testData.close.slice(0, 360));
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 3;
    const total = rows * close.length;
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.goertzel_cycle_composite_wave_alloc(total);

    try {
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        const actualRows = wasm.goertzel_cycle_composite_wave_batch_into(
            closePtr,
            outPtr,
            close.length,
            24,
            28,
            2,
            1,
            1,
            0,
            2,
            2,
            0,
        );
        assert.strictEqual(actualRows, rows);

        const flat = new Float64Array(memory.buffer, outPtr, total);
        const jsBatch = wasm.goertzel_cycle_composite_wave_batch_js(close, {
            max_period_range: [24, 28, 2],
            start_at_cycle_range: [1, 1, 0],
            use_top_cycles_range: [2, 2, 0],
        });
        assertArrayClose(Array.from(flat), jsBatch.values, 1e-12, 'batch_into mismatch');
    } finally {
        wasm.goertzel_cycle_composite_wave_free(outPtr, total);
        wasm.deallocate_f64_array(closePtr);
    }
});
