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

test('cyberpunk_value_trend_analyzer_js output contract', () => {
    const open = new Float64Array(testData.open.slice(0, 256));
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.cyberpunk_value_trend_analyzer_js(open, high, low, close, 30, 75);

    assert.strictEqual(result.value_trend.length, close.length);
    assert.strictEqual(result.sell_signal.length, close.length);
    assert(result.value_trend.some(v => !isNaN(v)));
});

test('cyberpunk_value_trend_analyzer_js rejects invalid parameters', () => {
    const open = new Float64Array(testData.open.slice(0, 120));
    const high = new Float64Array(testData.high.slice(0, 120));
    const low = new Float64Array(testData.low.slice(0, 120));
    const close = new Float64Array(testData.close.slice(0, 120));

    assert.throws(() => {
        wasm.cyberpunk_value_trend_analyzer_js(open, high, low, close, 0, 75);
    }, /Invalid entry_level/);

    assert.throws(() => {
        wasm.cyberpunk_value_trend_analyzer_js(open, high, low, close, 30, 101);
    }, /Invalid exit_level/);
});

test('cyberpunk_value_trend_analyzer_into pointer path matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 220));
    const high = new Float64Array(testData.high.slice(0, 220));
    const low = new Float64Array(testData.low.slice(0, 220));
    const close = new Float64Array(testData.close.slice(0, 220));
    const safe = wasm.cyberpunk_value_trend_analyzer_js(open, high, low, close, 30, 75);
    const memory = getMemory();
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const openPtr = wasm.allocate_f64_array(open.length);
    const highPtr = wasm.allocate_f64_array(high.length);
    const lowPtr = wasm.allocate_f64_array(low.length);
    const closePtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.cyberpunk_value_trend_analyzer_alloc(close.length);

    try {
        new Float64Array(memory.buffer, openPtr, open.length).set(open);
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);

        wasm.cyberpunk_value_trend_analyzer_into(
            openPtr, highPtr, lowPtr, closePtr, outPtr, close.length, 30, 75
        );

        const flat = new Float64Array(memory.buffer, outPtr, 6 * close.length);
        const outputs = [
            Array.from(flat.slice(0, close.length)),
            Array.from(flat.slice(close.length, 2 * close.length)),
            Array.from(flat.slice(2 * close.length, 3 * close.length)),
            Array.from(flat.slice(3 * close.length, 4 * close.length)),
            Array.from(flat.slice(4 * close.length, 5 * close.length)),
            Array.from(flat.slice(5 * close.length)),
        ];
        const expected = [
            safe.value_trend,
            safe.value_trend_lag,
            safe.deviation_index,
            safe.overbought_signal,
            safe.buy_signal,
            safe.sell_signal,
        ];
        for (let i = 0; i < outputs.length; i += 1) {
            assertArrayClose(outputs[i], expected[i], 1e-12, `pointer output ${i} mismatch`);
        }
    } finally {
        wasm.cyberpunk_value_trend_analyzer_free(outPtr, close.length);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('cyberpunk_value_trend_analyzer_batch_js single parameter set matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 200));
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.cyberpunk_value_trend_analyzer_batch_js(open, high, low, close, {
        entry_level_range: [30, 30, 0],
        exit_level_range: [75, 75, 0],
    });
    const single = wasm.cyberpunk_value_trend_analyzer_js(open, high, low, close, 30, 75);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].entry_level, 30);
    assert.strictEqual(batch.combos[0].exit_level, 75);
    assertArrayClose(batch.value_trend, single.value_trend, 1e-12, 'batch value_trend mismatch');
    assertArrayClose(batch.sell_signal, single.sell_signal, 1e-12, 'batch sell_signal mismatch');
});

test('cyberpunk_value_trend_analyzer_batch_into matches batch_js', () => {
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
    const outPtr = wasm.cyberpunk_value_trend_analyzer_alloc(total);

    try {
        new Float64Array(memory.buffer, openPtr, open.length).set(open);
        new Float64Array(memory.buffer, highPtr, high.length).set(high);
        new Float64Array(memory.buffer, lowPtr, low.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);

        const actualRows = wasm.cyberpunk_value_trend_analyzer_batch_into(
            openPtr, highPtr, lowPtr, closePtr, outPtr, close.length,
            25, 35, 10,
            70, 80, 10,
        );
        assert.strictEqual(actualRows, rows);

        const flat = new Float64Array(memory.buffer, outPtr, 6 * total);
        const valueTrend = Array.from(flat.slice(0, total));
        const valueTrendLag = Array.from(flat.slice(total, 2 * total));
        const deviationIndex = Array.from(flat.slice(2 * total, 3 * total));
        const overboughtSignal = Array.from(flat.slice(3 * total, 4 * total));
        const buySignal = Array.from(flat.slice(4 * total, 5 * total));
        const sellSignal = Array.from(flat.slice(5 * total));

        const jsBatch = wasm.cyberpunk_value_trend_analyzer_batch_js(open, high, low, close, {
            entry_level_range: [25, 35, 10],
            exit_level_range: [70, 80, 10],
        });
        assertArrayClose(valueTrend, jsBatch.value_trend, 1e-12, 'batch_into value_trend mismatch');
        assertArrayClose(valueTrendLag, jsBatch.value_trend_lag, 1e-12, 'batch_into lag mismatch');
        assertArrayClose(deviationIndex, jsBatch.deviation_index, 1e-12, 'batch_into deviation mismatch');
        assertArrayClose(overboughtSignal, jsBatch.overbought_signal, 1e-12, 'batch_into overbought mismatch');
        assertArrayClose(buySignal, jsBatch.buy_signal, 1e-12, 'batch_into buy mismatch');
        assertArrayClose(sellSignal, jsBatch.sell_signal, 1e-12, 'batch_into sell mismatch');
    } finally {
        wasm.cyberpunk_value_trend_analyzer_free(outPtr, total);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});
