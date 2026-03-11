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

test('reversal_signals_js output contract', () => {
    const open = new Float64Array(testData.open.slice(0, 256));
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const volume = new Float64Array(testData.volume.slice(0, 256));
    const result = wasm.reversal_signals_js(open, high, low, close, volume, 12, 3, true, 50, 'EMA', 33);

    assert.strictEqual(result.buy_signal.length, close.length);
    assert.strictEqual(result.sell_signal.length, close.length);
    assert.strictEqual(result.stepped_ma.length, close.length);
    assert.strictEqual(result.state.length, close.length);
    assert(result.stepped_ma.some(v => !isNaN(v)));
});

test('reversal_signals_js rejects invalid parameters', () => {
    const open = new Float64Array(testData.open.slice(0, 64));
    const high = new Float64Array(testData.high.slice(0, 64));
    const low = new Float64Array(testData.low.slice(0, 64));
    const close = new Float64Array(testData.close.slice(0, 64));
    const volume = new Float64Array(testData.volume.slice(0, 64));

    assert.throws(() => {
        wasm.reversal_signals_js(open, high, low, close, volume, 0, 3, true, 50, 'EMA', 33);
    }, /Invalid lookback_period/);

    assert.throws(() => {
        wasm.reversal_signals_js(open, high, low, close, volume, 12, 3, true, 50, 'BAD', 33);
    }, /Invalid trend_ma_type/);
});

test('reversal_signals_into pointer path matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 220));
    const high = new Float64Array(testData.high.slice(0, 220));
    const low = new Float64Array(testData.low.slice(0, 220));
    const close = new Float64Array(testData.close.slice(0, 220));
    const volume = new Float64Array(testData.volume.slice(0, 220));
    const safe = wasm.reversal_signals_js(open, high, low, close, volume, 12, 3, true, 50, 'EMA', 33);
    const memory = getMemory();
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const openPtr = wasm.allocate_f64_array(open.length);
    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const closePtr = wasm.allocate_f64_array(close.length);
    const volumePtr = wasm.allocate_f64_array(volume.length);
    const outPtr = wasm.reversal_signals_alloc(close.length);

    try {
        new Float64Array(memory.buffer, openPtr, open.length).set(open);
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        new Float64Array(memory.buffer, volumePtr, volume.length).set(volume);

        wasm.reversal_signals_into(
            openPtr, highPtr, lowPtr, closePtr, volumePtr, outPtr, close.length,
            12, 3, true, 50, 'EMA', 33
        );

        const flat = new Float64Array(memory.buffer, outPtr, 4 * close.length);
        const buySignal = Array.from(flat.slice(0, close.length));
        const sellSignal = Array.from(flat.slice(close.length, 2 * close.length));
        const steppedMa = Array.from(flat.slice(2 * close.length, 3 * close.length));
        const state = Array.from(flat.slice(3 * close.length));

        assertArrayClose(buySignal, safe.buy_signal, 1e-12, 'pointer buy_signal mismatch');
        assertArrayClose(sellSignal, safe.sell_signal, 1e-12, 'pointer sell_signal mismatch');
        assertArrayClose(steppedMa, safe.stepped_ma, 1e-12, 'pointer stepped_ma mismatch');
        assertArrayClose(state, safe.state, 1e-12, 'pointer state mismatch');
    } finally {
        wasm.reversal_signals_free(outPtr, close.length);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
        wasm.deallocate_f64_array(volumePtr);
    }
});

test('reversal_signals_batch_js single parameter set matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 200));
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const close = new Float64Array(testData.close.slice(0, 200));
    const volume = new Float64Array(testData.volume.slice(0, 200));
    const batch = wasm.reversal_signals_batch_js(open, high, low, close, volume, {
        lookback_period_range: [12, 12, 0],
        confirmation_period_range: [3, 3, 0],
        trend_ma_period_range: [50, 50, 0],
        ma_step_period_range: [33, 33, 0],
        use_volume_confirmation: true,
        trend_ma_type: 'EMA',
    });
    const single = wasm.reversal_signals_js(open, high, low, close, volume, 12, 3, true, 50, 'EMA', 33);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].lookback_period, 12);
    assert.strictEqual(batch.combos[0].confirmation_period, 3);
    assert.strictEqual(batch.combos[0].trend_ma_period, 50);
    assert.strictEqual(batch.combos[0].ma_step_period, 33);
    assertArrayClose(batch.buy_signal, single.buy_signal, 1e-12, 'batch buy_signal mismatch');
    assertArrayClose(batch.sell_signal, single.sell_signal, 1e-12, 'batch sell_signal mismatch');
    assertArrayClose(batch.stepped_ma, single.stepped_ma, 1e-12, 'batch stepped_ma mismatch');
    assertArrayClose(batch.state, single.state, 1e-12, 'batch state mismatch');
});

test('reversal_signals_batch_into matches batch_js', () => {
    const open = new Float64Array(testData.open.slice(0, 160));
    const high = new Float64Array(testData.high.slice(0, 160));
    const low = new Float64Array(testData.low.slice(0, 160));
    const close = new Float64Array(testData.close.slice(0, 160));
    const volume = new Float64Array(testData.volume.slice(0, 160));
    const memory = getMemory();
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 4;
    const total = rows * close.length;
    const openPtr = wasm.allocate_f64_array(open.length);
    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const closePtr = wasm.allocate_f64_array(close.length);
    const volumePtr = wasm.allocate_f64_array(volume.length);
    const outPtr = wasm.reversal_signals_alloc(total);

    try {
        new Float64Array(memory.buffer, openPtr, open.length).set(open);
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        new Float64Array(memory.buffer, volumePtr, volume.length).set(volume);

        const actualRows = wasm.reversal_signals_batch_into(
            openPtr, highPtr, lowPtr, closePtr, volumePtr, outPtr, close.length,
            10, 12, 2,
            2, 3, 1,
            30, 30, 0,
            20, 20, 0,
            false,
            'SMA'
        );
        assert.strictEqual(actualRows, rows);

        const flat = new Float64Array(memory.buffer, outPtr, 4 * total);
        const buySignal = Array.from(flat.slice(0, total));
        const sellSignal = Array.from(flat.slice(total, 2 * total));
        const steppedMa = Array.from(flat.slice(2 * total, 3 * total));
        const state = Array.from(flat.slice(3 * total));

        const jsBatch = wasm.reversal_signals_batch_js(open, high, low, close, volume, {
            lookback_period_range: [10, 12, 2],
            confirmation_period_range: [2, 3, 1],
            trend_ma_period_range: [30, 30, 0],
            ma_step_period_range: [20, 20, 0],
            use_volume_confirmation: false,
            trend_ma_type: 'SMA',
        });
        assertArrayClose(buySignal, jsBatch.buy_signal, 1e-12, 'batch_into buy_signal mismatch');
        assertArrayClose(sellSignal, jsBatch.sell_signal, 1e-12, 'batch_into sell_signal mismatch');
        assertArrayClose(steppedMa, jsBatch.stepped_ma, 1e-12, 'batch_into stepped_ma mismatch');
        assertArrayClose(state, jsBatch.state, 1e-12, 'batch_into state mismatch');
    } finally {
        wasm.reversal_signals_free(outPtr, total);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
        wasm.deallocate_f64_array(volumePtr);
    }
});
