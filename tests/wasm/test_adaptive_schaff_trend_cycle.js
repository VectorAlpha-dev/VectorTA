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

test('adaptive_schaff_trend_cycle output contract', () => {
    const high = new Float64Array(testData.high.slice(0, 320));
    const low = new Float64Array(testData.low.slice(0, 320));
    const close = new Float64Array(testData.close.slice(0, 320));

    const result = wasm.adaptive_schaff_trend_cycle(high, low, close, 55, 12, 0.45, 26, 50);

    assert.strictEqual(result.stc.length, close.length);
    assert.strictEqual(result.histogram.length, close.length);
    assert(Array.from(result.stc).some(v => !isNaN(v)));
    assert(Array.from(result.histogram).some(v => !isNaN(v)));
});

test('adaptive_schaff_trend_cycle_into_host pointer path matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 240));
    const low = new Float64Array(testData.low.slice(0, 240));
    const close = new Float64Array(testData.close.slice(0, 240));
    const safe = wasm.adaptive_schaff_trend_cycle(high, low, close, 40, 10, 0.38, 20, 42);
    const outPtr = wasm.adaptive_schaff_trend_cycle_alloc(close.length);

    try {
        wasm.adaptive_schaff_trend_cycle_into_host(
            high, low, close, outPtr, 40, 10, 0.38, 20, 42
        );
        const out = wasm.read_f64_array(outPtr, 2 * close.length);
        const stc = out.slice(0, close.length);
        const histogram = out.slice(close.length, 2 * close.length);
        assertArrayClose(stc, safe.stc, 1e-10, 'stc mismatch');
        assertArrayClose(histogram, safe.histogram, 1e-10, 'histogram mismatch');
    } finally {
        wasm.adaptive_schaff_trend_cycle_free(outPtr, close.length);
    }
});

test('adaptive_schaff_trend_cycle_batch single parameter set matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 220));
    const low = new Float64Array(testData.low.slice(0, 220));
    const close = new Float64Array(testData.close.slice(0, 220));
    const batch = wasm.adaptive_schaff_trend_cycle_batch(high, low, close, {
        adaptive_length_range: [55, 55, 0],
        stc_length_range: [12, 12, 0],
        smoothing_factor_range: [0.45, 0.45, 0.0],
        fast_length_range: [26, 26, 0],
        slow_length_range: [50, 50, 0],
    });
    const single = wasm.adaptive_schaff_trend_cycle(high, low, close, 55, 12, 0.45, 26, 50);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].adaptive_length, 55);
    assert.strictEqual(batch.combos[0].stc_length, 12);
    assert.strictEqual(batch.combos[0].smoothing_factor, 0.45);
    assert.strictEqual(batch.combos[0].fast_length, 26);
    assert.strictEqual(batch.combos[0].slow_length, 50);
    assertArrayClose(batch.stc, single.stc, 1e-10, 'batch stc mismatch');
    assertArrayClose(batch.histogram, single.histogram, 1e-10, 'batch histogram mismatch');
});

test('adaptive_schaff_trend_cycle_batch metadata', () => {
    const high = new Float64Array(testData.high.slice(0, 180));
    const low = new Float64Array(testData.low.slice(0, 180));
    const close = new Float64Array(testData.close.slice(0, 180));
    const batch = wasm.adaptive_schaff_trend_cycle_batch(high, low, close, {
        adaptive_length_range: [30, 32, 2],
        stc_length_range: [9, 11, 2],
        smoothing_factor_range: [0.4, 0.5, 0.1],
        fast_length_range: [18, 20, 2],
        slow_length_range: [40, 42, 2],
    });

    assert.strictEqual(batch.rows, 32);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos.length, 32);
});

test('adaptive_schaff_trend_cycle_batch_into pointer path matches safe batch API', () => {
    const high = new Float64Array(testData.high.slice(0, 180));
    const low = new Float64Array(testData.low.slice(0, 180));
    const close = new Float64Array(testData.close.slice(0, 180));
    const batch = wasm.adaptive_schaff_trend_cycle_batch(high, low, close, {
        adaptive_length_range: [30, 32, 2],
        stc_length_range: [9, 11, 2],
        smoothing_factor_range: [0.4, 0.5, 0.1],
        fast_length_range: [18, 20, 2],
        slow_length_range: [40, 42, 2],
    });

    const rows = batch.rows;
    const total = rows * close.length;
    const highPtr = wasm.adaptive_schaff_trend_cycle_alloc(close.length);
    const lowPtr = wasm.adaptive_schaff_trend_cycle_alloc(close.length);
    const closePtr = wasm.adaptive_schaff_trend_cycle_alloc(close.length);
    const stcPtr = wasm.adaptive_schaff_trend_cycle_alloc(total);
    const histogramPtr = wasm.adaptive_schaff_trend_cycle_alloc(total);

    try {
        const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        if (!memory || !memory.buffer) {
            console.warn('Skipping adaptive_schaff_trend_cycle_batch_into test: wasm memory accessor unavailable');
            return;
        }
        new Float64Array(memory.buffer, highPtr, close.length).set(high);
        new Float64Array(memory.buffer, lowPtr, close.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);

        const returnedRows = wasm.adaptive_schaff_trend_cycle_batch_into(
            highPtr,
            lowPtr,
            closePtr,
            stcPtr,
            histogramPtr,
            close.length,
            30,
            32,
            2,
            9,
            11,
            2,
            0.4,
            0.5,
            0.1,
            18,
            20,
            2,
            40,
            42,
            2,
        );

        assert.strictEqual(returnedRows, rows);
        assertArrayClose(wasm.read_f64_array(stcPtr, total), batch.stc, 1e-10, 'batch_into stc mismatch');
        assertArrayClose(wasm.read_f64_array(histogramPtr, total), batch.histogram, 1e-10, 'batch_into histogram mismatch');
    } finally {
        wasm.adaptive_schaff_trend_cycle_free(highPtr, close.length);
        wasm.adaptive_schaff_trend_cycle_free(lowPtr, close.length);
        wasm.adaptive_schaff_trend_cycle_free(closePtr, close.length);
        wasm.adaptive_schaff_trend_cycle_free(stcPtr, total);
        wasm.adaptive_schaff_trend_cycle_free(histogramPtr, total);
    }
});
