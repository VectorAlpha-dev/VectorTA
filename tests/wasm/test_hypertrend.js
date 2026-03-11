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

test('hypertrend output contract', () => {
    const high = new Float64Array(testData.high.slice(0, 320));
    const low = new Float64Array(testData.low.slice(0, 320));
    const source = new Float64Array(testData.close.slice(0, 320));

    const result = wasm.hypertrend(high, low, source, 5.0, 14.0, 80.0);

    assert.strictEqual(result.upper.length, source.length);
    assert.strictEqual(result.average.length, source.length);
    assert.strictEqual(result.lower.length, source.length);
    assert.strictEqual(result.trend.length, source.length);
    assert.strictEqual(result.changed.length, source.length);
    assert(Array.from(result.average).some(v => !isNaN(v)));
    for (let i = 0; i < source.length; i++) {
        const u = result.upper[i];
        const l = result.lower[i];
        assert((isNaN(u) && isNaN(l)) || u >= l);
    }
});

test('hypertrend_into_host pointer path matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 240));
    const low = new Float64Array(testData.low.slice(0, 240));
    const source = new Float64Array(testData.close.slice(0, 240));
    const safe = wasm.hypertrend(high, low, source, 4.5, 10.0, 60.0);
    const outPtr = wasm.hypertrend_alloc(source.length);

    try {
        wasm.hypertrend_into_host(high, low, source, outPtr, 4.5, 10.0, 60.0);
        const out = wasm.read_f64_array(outPtr, 5 * source.length);
        const upper = out.slice(0, source.length);
        const average = out.slice(source.length, 2 * source.length);
        const lower = out.slice(2 * source.length, 3 * source.length);
        const trend = out.slice(3 * source.length, 4 * source.length);
        const changed = out.slice(4 * source.length, 5 * source.length);
        assertArrayClose(upper, safe.upper, 1e-10, 'upper mismatch');
        assertArrayClose(average, safe.average, 1e-10, 'average mismatch');
        assertArrayClose(lower, safe.lower, 1e-10, 'lower mismatch');
        assertArrayClose(trend, safe.trend, 1e-10, 'trend mismatch');
        assertArrayClose(changed, safe.changed, 1e-10, 'changed mismatch');
    } finally {
        wasm.hypertrend_free(outPtr, source.length);
    }
});

test('hypertrend_batch single parameter set matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 240));
    const low = new Float64Array(testData.low.slice(0, 240));
    const source = new Float64Array(testData.close.slice(0, 240));
    const batch = wasm.hypertrend_batch(high, low, source, {
        factor_range: [5.0, 5.0, 0.0],
        slope_range: [14.0, 14.0, 0.0],
        width_percent_range: [80.0, 80.0, 0.0],
    });
    const single = wasm.hypertrend(high, low, source, 5.0, 14.0, 80.0);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, source.length);
    assert.strictEqual(batch.combos[0].factor, 5.0);
    assert.strictEqual(batch.combos[0].slope, 14.0);
    assert.strictEqual(batch.combos[0].width_percent, 80.0);
    assertArrayClose(batch.upper, single.upper, 1e-10, 'batch upper mismatch');
    assertArrayClose(batch.average, single.average, 1e-10, 'batch average mismatch');
    assertArrayClose(batch.lower, single.lower, 1e-10, 'batch lower mismatch');
    assertArrayClose(batch.trend, single.trend, 1e-10, 'batch trend mismatch');
    assertArrayClose(batch.changed, single.changed, 1e-10, 'batch changed mismatch');
});

test('hypertrend_batch metadata', () => {
    const high = new Float64Array(testData.high.slice(0, 180));
    const low = new Float64Array(testData.low.slice(0, 180));
    const source = new Float64Array(testData.close.slice(0, 180));
    const batch = wasm.hypertrend_batch(high, low, source, {
        factor_range: [4.0, 5.0, 1.0],
        slope_range: [10.0, 12.0, 2.0],
        width_percent_range: [60.0, 80.0, 20.0],
    });

    assert.strictEqual(batch.rows, 8);
    assert.strictEqual(batch.cols, source.length);
    assert.strictEqual(batch.combos.length, 8);
    assert.deepStrictEqual(
        batch.combos.map(combo => combo.factor),
        [4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0],
    );
});

test('hypertrend_batch_into pointer path matches safe batch API', () => {
    const high = new Float64Array(testData.high.slice(0, 180));
    const low = new Float64Array(testData.low.slice(0, 180));
    const source = new Float64Array(testData.close.slice(0, 180));
    const batch = wasm.hypertrend_batch(high, low, source, {
        factor_range: [4.0, 5.0, 1.0],
        slope_range: [10.0, 12.0, 2.0],
        width_percent_range: [60.0, 80.0, 20.0],
    });

    const rows = batch.rows;
    const total = rows * source.length;
    const highPtr = wasm.hypertrend_alloc(source.length);
    const lowPtr = wasm.hypertrend_alloc(source.length);
    const sourcePtr = wasm.hypertrend_alloc(source.length);
    const upperPtr = wasm.hypertrend_alloc(total);
    const averagePtr = wasm.hypertrend_alloc(total);
    const lowerPtr = wasm.hypertrend_alloc(total);
    const trendPtr = wasm.hypertrend_alloc(total);
    const changedPtr = wasm.hypertrend_alloc(total);

    try {
        const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        if (!memory || !memory.buffer) {
            console.warn('Skipping hypertrend_batch_into test: wasm memory accessor unavailable');
            return;
        }
        new Float64Array(memory.buffer, highPtr, source.length).set(high);
        new Float64Array(memory.buffer, lowPtr, source.length).set(low);
        new Float64Array(memory.buffer, sourcePtr, source.length).set(source);

        const returnedRows = wasm.hypertrend_batch_into(
            highPtr,
            lowPtr,
            sourcePtr,
            upperPtr,
            averagePtr,
            lowerPtr,
            trendPtr,
            changedPtr,
            source.length,
            4.0,
            5.0,
            1.0,
            10.0,
            12.0,
            2.0,
            60.0,
            80.0,
            20.0,
        );

        assert.strictEqual(returnedRows, rows);
        assertArrayClose(wasm.read_f64_array(upperPtr, total), batch.upper, 1e-10, 'batch_into upper mismatch');
        assertArrayClose(wasm.read_f64_array(averagePtr, total), batch.average, 1e-10, 'batch_into average mismatch');
        assertArrayClose(wasm.read_f64_array(lowerPtr, total), batch.lower, 1e-10, 'batch_into lower mismatch');
        assertArrayClose(wasm.read_f64_array(trendPtr, total), batch.trend, 1e-10, 'batch_into trend mismatch');
        assertArrayClose(wasm.read_f64_array(changedPtr, total), batch.changed, 1e-10, 'batch_into changed mismatch');
    } finally {
        wasm.hypertrend_free(highPtr, source.length);
        wasm.hypertrend_free(lowPtr, source.length);
        wasm.hypertrend_free(sourcePtr, source.length);
        wasm.hypertrend_free(upperPtr, total);
        wasm.hypertrend_free(averagePtr, total);
        wasm.hypertrend_free(lowerPtr, total);
        wasm.hypertrend_free(trendPtr, total);
        wasm.hypertrend_free(changedPtr, total);
    }
});
