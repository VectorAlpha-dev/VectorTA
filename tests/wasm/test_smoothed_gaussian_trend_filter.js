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

test('smoothed_gaussian_trend_filter output contract', () => {
    const high = new Float64Array(testData.high.slice(0, 320));
    const low = new Float64Array(testData.low.slice(0, 320));
    const close = new Float64Array(testData.close.slice(0, 320));

    const result = wasm.smoothed_gaussian_trend_filter(high, low, close, 15, 3, 22, 7);

    assert.strictEqual(result.filter.length, close.length);
    assert.strictEqual(result.supertrend.length, close.length);
    assert.strictEqual(result.trend.length, close.length);
    assert.strictEqual(result.ranging.length, close.length);
    assert(Array.from(result.filter).some(v => !isNaN(v)));
    assert(Array.from(result.supertrend).some(v => !isNaN(v)));
});

test('smoothed_gaussian_trend_filter_into_host pointer path matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 240));
    const low = new Float64Array(testData.low.slice(0, 240));
    const close = new Float64Array(testData.close.slice(0, 240));
    const safe = wasm.smoothed_gaussian_trend_filter(high, low, close, 18, 4, 20, 5);
    const outPtr = wasm.smoothed_gaussian_trend_filter_alloc(close.length);

    try {
        wasm.smoothed_gaussian_trend_filter_into_host(high, low, close, outPtr, 18, 4, 20, 5);
        const out = wasm.read_f64_array(outPtr, 4 * close.length);
        const filter = out.slice(0, close.length);
        const supertrend = out.slice(close.length, 2 * close.length);
        const trend = out.slice(2 * close.length, 3 * close.length);
        const ranging = out.slice(3 * close.length, 4 * close.length);
        assertArrayClose(filter, safe.filter, 1e-10, 'filter mismatch');
        assertArrayClose(supertrend, safe.supertrend, 1e-10, 'supertrend mismatch');
        assertArrayClose(trend, safe.trend, 1e-10, 'trend mismatch');
        assertArrayClose(ranging, safe.ranging, 1e-10, 'ranging mismatch');
    } finally {
        wasm.smoothed_gaussian_trend_filter_free(outPtr, close.length);
    }
});

test('smoothed_gaussian_trend_filter_batch single parameter set matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 220));
    const low = new Float64Array(testData.low.slice(0, 220));
    const close = new Float64Array(testData.close.slice(0, 220));
    const batch = wasm.smoothed_gaussian_trend_filter_batch(high, low, close, {
        gaussian_length_range: [15, 15, 0],
        poles_range: [3, 3, 0],
        smoothing_length_range: [22, 22, 0],
        linreg_offset_range: [7, 7, 0],
    });
    const single = wasm.smoothed_gaussian_trend_filter(high, low, close, 15, 3, 22, 7);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].gaussian_length, 15);
    assert.strictEqual(batch.combos[0].poles, 3);
    assert.strictEqual(batch.combos[0].smoothing_length, 22);
    assert.strictEqual(batch.combos[0].linreg_offset, 7);
    assertArrayClose(batch.filter, single.filter, 1e-10, 'batch filter mismatch');
    assertArrayClose(batch.supertrend, single.supertrend, 1e-10, 'batch supertrend mismatch');
    assertArrayClose(batch.trend, single.trend, 1e-10, 'batch trend mismatch');
    assertArrayClose(batch.ranging, single.ranging, 1e-10, 'batch ranging mismatch');
});

test('smoothed_gaussian_trend_filter_batch metadata', () => {
    const high = new Float64Array(testData.high.slice(0, 180));
    const low = new Float64Array(testData.low.slice(0, 180));
    const close = new Float64Array(testData.close.slice(0, 180));
    const batch = wasm.smoothed_gaussian_trend_filter_batch(high, low, close, {
        gaussian_length_range: [15, 17, 2],
        poles_range: [2, 4, 2],
        smoothing_length_range: [20, 22, 2],
        linreg_offset_range: [5, 7, 2],
    });

    assert.strictEqual(batch.rows, 16);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos.length, 16);
});

test('smoothed_gaussian_trend_filter_batch_into pointer path matches safe batch API', () => {
    const high = new Float64Array(testData.high.slice(0, 180));
    const low = new Float64Array(testData.low.slice(0, 180));
    const close = new Float64Array(testData.close.slice(0, 180));
    const batch = wasm.smoothed_gaussian_trend_filter_batch(high, low, close, {
        gaussian_length_range: [15, 17, 2],
        poles_range: [2, 4, 2],
        smoothing_length_range: [20, 22, 2],
        linreg_offset_range: [5, 7, 2],
    });

    const rows = batch.rows;
    const total = rows * close.length;
    const highPtr = wasm.smoothed_gaussian_trend_filter_alloc(close.length);
    const lowPtr = wasm.smoothed_gaussian_trend_filter_alloc(close.length);
    const closePtr = wasm.smoothed_gaussian_trend_filter_alloc(close.length);
    const filterPtr = wasm.smoothed_gaussian_trend_filter_alloc(total);
    const supertrendPtr = wasm.smoothed_gaussian_trend_filter_alloc(total);
    const trendPtr = wasm.smoothed_gaussian_trend_filter_alloc(total);
    const rangingPtr = wasm.smoothed_gaussian_trend_filter_alloc(total);

    try {
        const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        if (!memory || !memory.buffer) {
            console.warn('Skipping smoothed_gaussian_trend_filter_batch_into test: wasm memory accessor unavailable');
            return;
        }
        new Float64Array(memory.buffer, highPtr, close.length).set(high);
        new Float64Array(memory.buffer, lowPtr, close.length).set(low);
        new Float64Array(memory.buffer, closePtr, close.length).set(close);

        const returnedRows = wasm.smoothed_gaussian_trend_filter_batch_into(
            highPtr,
            lowPtr,
            closePtr,
            filterPtr,
            supertrendPtr,
            trendPtr,
            rangingPtr,
            close.length,
            15,
            17,
            2,
            2,
            4,
            2,
            20,
            22,
            2,
            5,
            7,
            2,
        );

        assert.strictEqual(returnedRows, rows);
        assertArrayClose(wasm.read_f64_array(filterPtr, total), batch.filter, 1e-10, 'batch_into filter mismatch');
        assertArrayClose(wasm.read_f64_array(supertrendPtr, total), batch.supertrend, 1e-10, 'batch_into supertrend mismatch');
        assertArrayClose(wasm.read_f64_array(trendPtr, total), batch.trend, 1e-10, 'batch_into trend mismatch');
        assertArrayClose(wasm.read_f64_array(rangingPtr, total), batch.ranging, 1e-10, 'batch_into ranging mismatch');
    } finally {
        wasm.smoothed_gaussian_trend_filter_free(highPtr, close.length);
        wasm.smoothed_gaussian_trend_filter_free(lowPtr, close.length);
        wasm.smoothed_gaussian_trend_filter_free(closePtr, close.length);
        wasm.smoothed_gaussian_trend_filter_free(filterPtr, total);
        wasm.smoothed_gaussian_trend_filter_free(supertrendPtr, total);
        wasm.smoothed_gaussian_trend_filter_free(trendPtr, total);
        wasm.smoothed_gaussian_trend_filter_free(rangingPtr, total);
    }
});
