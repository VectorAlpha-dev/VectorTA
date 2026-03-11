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

test('fvg_positioning_average_js output contract', () => {
    const open = new Float64Array(testData.open.slice(0, 256));
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.fvg_positioning_average_js(open, high, low, close, 30, 'Bar Count', 0.25);

    assert.strictEqual(result.bull_average.length, close.length);
    assert.strictEqual(result.bear_average.length, close.length);
    assert.strictEqual(result.bull_mid.length, close.length);
    assert.strictEqual(result.bear_mid.length, close.length);
    assert(result.bull_average.some(v => !isNaN(v)) || result.bear_average.some(v => !isNaN(v)));
});

test('fvg_positioning_average_js rejects invalid parameters', () => {
    const open = new Float64Array(testData.open.slice(0, 64));
    const high = new Float64Array(testData.high.slice(0, 64));
    const low = new Float64Array(testData.low.slice(0, 64));
    const close = new Float64Array(testData.close.slice(0, 64));

    assert.throws(() => {
        wasm.fvg_positioning_average_js(open, high, low, close, 0, 'Bar Count', 0.25);
    }, /Invalid lookback/);

    assert.throws(() => {
        wasm.fvg_positioning_average_js(open, high, low, close, 30, 'bad', 0.25);
    }, /Invalid lookback_type/);

    assert.throws(() => {
        wasm.fvg_positioning_average_js(open, high, low, close, 30, 'Bar Count', -1.0);
    }, /Invalid atr_multiplier/);
});

test('fvg_positioning_average_into pointer path matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 220));
    const high = new Float64Array(testData.high.slice(0, 220));
    const low = new Float64Array(testData.low.slice(0, 220));
    const close = new Float64Array(testData.close.slice(0, 220));
    const safe = wasm.fvg_positioning_average_js(open, high, low, close, 30, 'Bar Count', 0.25);
    const memory = getMemory();
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const openPtr = wasm.allocate_f64_array(open.length);
    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.fvg_positioning_average_alloc(close.length);

    try {
        new Float64Array(memory.buffer, openPtr, open.length).set(open);
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);

        wasm.fvg_positioning_average_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            close.length,
            30,
            'Bar Count',
            0.25,
        );

        const flat = new Float64Array(memory.buffer, outPtr, 4 * close.length);
        const bullAverage = Array.from(flat.slice(0, close.length));
        const bearAverage = Array.from(flat.slice(close.length, 2 * close.length));
        const bullMid = Array.from(flat.slice(2 * close.length, 3 * close.length));
        const bearMid = Array.from(flat.slice(3 * close.length));

        assertArrayClose(bullAverage, safe.bull_average, 1e-12, 'pointer bull_average mismatch');
        assertArrayClose(bearAverage, safe.bear_average, 1e-12, 'pointer bear_average mismatch');
        assertArrayClose(bullMid, safe.bull_mid, 1e-12, 'pointer bull_mid mismatch');
        assertArrayClose(bearMid, safe.bear_mid, 1e-12, 'pointer bear_mid mismatch');
    } finally {
        wasm.fvg_positioning_average_free(outPtr, close.length);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('fvg_positioning_average_batch_js single parameter set matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 200));
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.fvg_positioning_average_batch_js(open, high, low, close, {
        lookback_range: [30, 30, 0],
        atr_multiplier_range: [0.25, 0.25, 0.0],
        lookback_type: 'Bar Count',
    });
    const single = wasm.fvg_positioning_average_js(open, high, low, close, 30, 'Bar Count', 0.25);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].lookback, 30);
    assert.strictEqual(batch.combos[0].lookback_type, 'Bar Count');
    assert.strictEqual(batch.combos[0].atr_multiplier, 0.25);
    assertArrayClose(batch.bull_average, single.bull_average, 1e-12, 'batch bull_average mismatch');
    assertArrayClose(batch.bear_average, single.bear_average, 1e-12, 'batch bear_average mismatch');
    assertArrayClose(batch.bull_mid, single.bull_mid, 1e-12, 'batch bull_mid mismatch');
    assertArrayClose(batch.bear_mid, single.bear_mid, 1e-12, 'batch bear_mid mismatch');
});

test('fvg_positioning_average_batch_into matches batch_js', () => {
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
    const outPtr = wasm.fvg_positioning_average_alloc(total);

    try {
        new Float64Array(memory.buffer, openPtr, open.length).set(open);
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);

        const actualRows = wasm.fvg_positioning_average_batch_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            close.length,
            30,
            34,
            2,
            0.25,
            0.25,
            0.0,
            'Bar Count',
        );
        assert.strictEqual(actualRows, rows);

        const flat = new Float64Array(memory.buffer, outPtr, 4 * total);
        const bullAverage = Array.from(flat.slice(0, total));
        const bearAverage = Array.from(flat.slice(total, 2 * total));
        const bullMid = Array.from(flat.slice(2 * total, 3 * total));
        const bearMid = Array.from(flat.slice(3 * total));

        const jsBatch = wasm.fvg_positioning_average_batch_js(open, high, low, close, {
            lookback_range: [30, 34, 2],
            atr_multiplier_range: [0.25, 0.25, 0.0],
            lookback_type: 'Bar Count',
        });
        assertArrayClose(bullAverage, jsBatch.bull_average, 1e-12, 'batch_into bull_average mismatch');
        assertArrayClose(bearAverage, jsBatch.bear_average, 1e-12, 'batch_into bear_average mismatch');
        assertArrayClose(bullMid, jsBatch.bull_mid, 1e-12, 'batch_into bull_mid mismatch');
        assertArrayClose(bearMid, jsBatch.bear_mid, 1e-12, 'batch_into bear_mid mismatch');
    } finally {
        wasm.fvg_positioning_average_free(outPtr, total);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});
