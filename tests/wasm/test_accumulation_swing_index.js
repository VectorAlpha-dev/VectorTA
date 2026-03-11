import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function manualAccumulationSwingIndex(open, high, low, close, dailyLimit) {
    const n = close.length;
    const out = new Float64Array(n);
    out.fill(NaN);

    let first = -1;
    for (let i = 0; i < n; i++) {
        if (
            Number.isFinite(open[i]) &&
            Number.isFinite(high[i]) &&
            Number.isFinite(low[i]) &&
            Number.isFinite(close[i])
        ) {
            first = i;
            break;
        }
    }
    if (first === -1) {
        return out;
    }

    out[first] = 0;
    let accum = 0;
    let prevOpen = open[first];
    let prevClose = close[first];

    for (let i = first + 1; i < n; i++) {
        const o = open[i];
        const h = high[i];
        const l = low[i];
        const c = close[i];
        if (
            Number.isFinite(o) &&
            Number.isFinite(h) &&
            Number.isFinite(l) &&
            Number.isFinite(c) &&
            Number.isFinite(prevOpen) &&
            Number.isFinite(prevClose)
        ) {
            const absHighClose = Math.abs(h - prevClose);
            const absLowClose = Math.abs(l - prevClose);
            const absCloseOpen = Math.abs(prevClose - prevOpen);
            const k = absHighClose >= absLowClose ? absHighClose : absLowClose;
            const barRange = h - l;
            let r;
            if (absHighClose >= absLowClose) {
                if (absHighClose >= barRange) {
                    r = absHighClose - 0.5 * absLowClose + 0.25 * absCloseOpen;
                } else {
                    r = barRange + 0.25 * absCloseOpen;
                }
            } else if (absLowClose >= barRange) {
                r = absLowClose - 0.5 * absHighClose + 0.25 * absCloseOpen;
            } else {
                r = barRange + 0.25 * absCloseOpen;
            }
            if (r !== 0) {
                accum += (
                    50
                    * (((c - prevClose) + 0.5 * (c - o) + 0.25 * (prevClose - prevOpen)) / r)
                    * k
                    / dailyLimit
                );
            }
        }
        out[i] = accum;
        prevOpen = o;
        prevClose = c;
    }

    return out;
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('accumulation_swing_index_js matches manual reference', () => {
    const n = 160;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const result = wasm.accumulation_swing_index_js(open, high, low, close, 10000.0);
    const expected = manualAccumulationSwingIndex(open, high, low, close, 10000.0);
    assertArrayClose(result, expected, 1e-12, 'ASI mismatch');
});

test('accumulation_swing_index_batch_js first row matches single', () => {
    const n = 96;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const batch = wasm.accumulation_swing_index_batch_js(open, high, low, close, {
        daily_limit_range: [10000.0, 12000.0, 2000.0],
    });
    const single = wasm.accumulation_swing_index_js(open, high, low, close, 10000.0);
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, n);
    assert.deepStrictEqual(batch.daily_limits, [10000.0, 12000.0]);
    assertArrayClose(batch.values.slice(0, n), single, 1e-12, 'batch mismatch');
});

test('accumulation_swing_index_into pointer path matches safe API', () => {
    const n = 128;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const safe = wasm.accumulation_swing_index_js(open, high, low, close, 10000.0);

    const openPtr = wasm.allocate_f64_array(n);
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const closePtr = wasm.allocate_f64_array(n);
    const outPtr = wasm.accumulation_swing_index_alloc(n);

    try {
        wasm.write_f64_array(openPtr, open);
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        wasm.write_f64_array(closePtr, close);
        wasm.accumulation_swing_index_into(openPtr, highPtr, lowPtr, closePtr, outPtr, n, 10000.0);
        const got = wasm.read_f64_array(outPtr, n);
        assertArrayClose(got, safe, 1e-12, 'pointer mismatch');
    } finally {
        wasm.accumulation_swing_index_free(outPtr, n);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('accumulation_swing_index_batch_into returns row count', () => {
    const n = 96;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const openPtr = wasm.allocate_f64_array(n);
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const closePtr = wasm.allocate_f64_array(n);
    const rows = 2;
    const outPtr = wasm.accumulation_swing_index_alloc(rows * n);

    try {
        wasm.write_f64_array(openPtr, open);
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        wasm.write_f64_array(closePtr, close);
        const returnedRows = wasm.accumulation_swing_index_batch_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            n,
            10000.0,
            12000.0,
            2000.0,
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.accumulation_swing_index_free(outPtr, rows * n);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('accumulation_swing_index_js rejects invalid daily_limit', () => {
    const n = 64;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    assert.throws(() => {
        wasm.accumulation_swing_index_js(open, high, low, close, 0.0);
    }, /Invalid daily_limit/);
});
