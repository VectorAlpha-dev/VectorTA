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

test('candle_strength_oscillator_js output contract', () => {
    const open = new Float64Array(testData.open.slice(0, 256));
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.candle_strength_oscillator_js(open, high, low, close, 50, false, 50, 'bollinger');

    assert.strictEqual(result.strength.length, close.length);
    assert.strictEqual(result.highs.length, close.length);
    assert.strictEqual(result.lows.length, close.length);
    assert.strictEqual(result.mid.length, close.length);
    assert.strictEqual(result.long_signal.length, close.length);
    assert.strictEqual(result.short_signal.length, close.length);
    assert.strictEqual(result.strength.findIndex(v => !isNaN(v)), 55);
    assert.strictEqual(result.highs.findIndex(v => !isNaN(v)), 104);
    assert.strictEqual(result.lows.findIndex(v => !isNaN(v)), 104);
    assert.strictEqual(result.mid.findIndex(v => !isNaN(v)), 104);
});

test('candle_strength_oscillator_js rejects invalid parameters', () => {
    const open = new Float64Array(testData.open.slice(0, 128));
    const high = new Float64Array(testData.high.slice(0, 128));
    const low = new Float64Array(testData.low.slice(0, 128));
    const close = new Float64Array(testData.close.slice(0, 128));

    assert.throws(() => {
        wasm.candle_strength_oscillator_js(open, high, low, close, 0, false, 50, 'bollinger');
    }, /Invalid period/);

    assert.throws(() => {
        wasm.candle_strength_oscillator_js(open, high, low, close, 50, true, 0, 'bollinger');
    }, /Invalid atr_length/);

    assert.throws(() => {
        wasm.candle_strength_oscillator_js(open, high, low, close, 50, false, 50, 'bad');
    }, /Invalid mode/);
});

test('candle_strength_oscillator_into pointer path matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 200));
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const close = new Float64Array(testData.close.slice(0, 200));
    const safe = wasm.candle_strength_oscillator_js(open, high, low, close, 30, true, 14, 'donchian');
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const openPtr = wasm.allocate_f64_array(open.length);
    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.candle_strength_oscillator_alloc(close.length);

    try {
        new Float64Array(memory.buffer, openPtr, open.length).set(open);
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        wasm.candle_strength_oscillator_into(
            openPtr, highPtr, lowPtr, closePtr, outPtr, close.length, 30, true, 14, 'donchian'
        );
        const flat = new Float64Array(memory.buffer, outPtr, 6 * close.length);
        const strength = Array.from(flat.slice(0, close.length));
        const highs = Array.from(flat.slice(close.length, 2 * close.length));
        const lows = Array.from(flat.slice(2 * close.length, 3 * close.length));
        const mid = Array.from(flat.slice(3 * close.length, 4 * close.length));
        const longSignal = Array.from(flat.slice(4 * close.length, 5 * close.length));
        const shortSignal = Array.from(flat.slice(5 * close.length, 6 * close.length));
        assertArrayClose(strength, safe.strength, 1e-12, 'pointer strength mismatch');
        assertArrayClose(highs, safe.highs, 1e-12, 'pointer highs mismatch');
        assertArrayClose(lows, safe.lows, 1e-12, 'pointer lows mismatch');
        assertArrayClose(mid, safe.mid, 1e-12, 'pointer mid mismatch');
        assertArrayClose(longSignal, safe.long_signal, 1e-12, 'pointer long mismatch');
        assertArrayClose(shortSignal, safe.short_signal, 1e-12, 'pointer short mismatch');
    } finally {
        wasm.candle_strength_oscillator_free(outPtr, close.length);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('candle_strength_oscillator_batch_js single parameter set matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 200));
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.candle_strength_oscillator_batch_js(open, high, low, close, {
        period_range: [30, 30, 0],
        atr_enabled: true,
        atr_length_range: [14, 14, 0],
        mode: 'bollinger',
    });
    const single = wasm.candle_strength_oscillator_js(open, high, low, close, 30, true, 14, 'bollinger');

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].period, 30);
    assert.strictEqual(batch.combos[0].atr_enabled, true);
    assert.strictEqual(batch.combos[0].atr_length, 14);
    assert.strictEqual(batch.combos[0].mode, 'bollinger');
    assertArrayClose(batch.strength, single.strength, 1e-12, 'batch strength mismatch');
    assertArrayClose(batch.highs, single.highs, 1e-12, 'batch highs mismatch');
    assertArrayClose(batch.lows, single.lows, 1e-12, 'batch lows mismatch');
    assertArrayClose(batch.mid, single.mid, 1e-12, 'batch mid mismatch');
});

test('candle_strength_oscillator_batch_into matches batch_js', () => {
    const open = new Float64Array(testData.open.slice(0, 180));
    const high = new Float64Array(testData.high.slice(0, 180));
    const low = new Float64Array(testData.low.slice(0, 180));
    const close = new Float64Array(testData.close.slice(0, 180));
    const rows = 3;
    const total = rows * close.length;
    const jsBatch = wasm.candle_strength_oscillator_batch_js(open, high, low, close, {
        period_range: [30, 34, 2],
        atr_enabled: true,
        atr_length_range: [14, 14, 0],
        mode: 'donchian',
    });
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const openPtr = wasm.allocate_f64_array(open.length);
    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.candle_strength_oscillator_alloc(total);

    try {
        new Float64Array(memory.buffer, openPtr, open.length).set(open);
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);
        const actualRows = wasm.candle_strength_oscillator_batch_into(
            openPtr, highPtr, lowPtr, closePtr, outPtr, close.length,
            30, 34, 2, true, 14, 14, 0, 'donchian'
        );
        assert.strictEqual(actualRows, rows);
        const flat = new Float64Array(memory.buffer, outPtr, 6 * total);
        assertArrayClose(Array.from(flat.slice(0, total)), jsBatch.strength, 1e-12, 'batch_into strength mismatch');
        assertArrayClose(Array.from(flat.slice(total, 2 * total)), jsBatch.highs, 1e-12, 'batch_into highs mismatch');
        assertArrayClose(Array.from(flat.slice(2 * total, 3 * total)), jsBatch.lows, 1e-12, 'batch_into lows mismatch');
        assertArrayClose(Array.from(flat.slice(3 * total, 4 * total)), jsBatch.mid, 1e-12, 'batch_into mid mismatch');
        assertArrayClose(Array.from(flat.slice(4 * total, 5 * total)), jsBatch.long_signal, 1e-12, 'batch_into long mismatch');
        assertArrayClose(Array.from(flat.slice(5 * total, 6 * total)), jsBatch.short_signal, 1e-12, 'batch_into short mismatch');
    } finally {
        wasm.candle_strength_oscillator_free(outPtr, total);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});
