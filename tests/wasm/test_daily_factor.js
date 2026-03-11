import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function manualReference(open, high, low, close, thresholdLevel = 0.35) {
    const n = close.length;
    const value = new Float64Array(n);
    const ema = new Float64Array(n);
    const signal = new Float64Array(n);
    value.fill(NaN);
    ema.fill(NaN);
    signal.fill(NaN);
    const alpha = 2 / 15;
    let prevOpen = NaN;
    let prevHigh = NaN;
    let prevLow = NaN;
    let prevClose = NaN;
    let prevEma = NaN;
    let hasPrev = false;

    for (let i = 0; i < n; i++) {
        if (![open[i], high[i], low[i], close[i]].every(Number.isFinite)) {
            continue;
        }
        ema[i] = Number.isFinite(prevEma) ? prevEma + alpha * (close[i] - prevEma) : close[i];
        if (hasPrev) {
            const spread = prevHigh - prevLow;
            value[i] = spread !== 0 ? Math.abs(prevOpen - prevClose) / spread : 0;
        } else {
            value[i] = 0;
        }

        if (value[i] > thresholdLevel && close[i] > ema[i]) {
            signal[i] = 2;
        } else if (value[i] > thresholdLevel && close[i] < ema[i]) {
            signal[i] = -2;
        } else if (close[i] > ema[i]) {
            signal[i] = 1;
        } else if (close[i] < ema[i]) {
            signal[i] = -1;
        } else {
            signal[i] = 0;
        }

        prevOpen = open[i];
        prevHigh = high[i];
        prevLow = low[i];
        prevClose = close[i];
        prevEma = ema[i];
        hasPrev = true;
    }

    return { value, ema, signal };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('daily_factor_js matches manual reference', () => {
    const n = 192;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const result = wasm.daily_factor_js(open, high, low, close, 0.35);
    const expected = manualReference(open, high, low, close, 0.35);
    assertArrayClose(result.value, expected.value, 1e-12, 'value mismatch');
    assertArrayClose(result.ema, expected.ema, 1e-12, 'ema mismatch');
    assertArrayClose(result.signal, expected.signal, 1e-12, 'signal mismatch');
});

test('daily_factor_batch_js first row matches single', () => {
    const n = 144;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const batch = wasm.daily_factor_batch_js(open, high, low, close, {
        threshold_level_range: [0.35, 0.45, 0.10],
    });
    const single = wasm.daily_factor_js(open, high, low, close, 0.35);
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, n);
    assertArrayClose(batch.threshold_levels, [0.35, 0.45], 1e-12, 'threshold grid mismatch');
    assertArrayClose(batch.value.slice(0, n), single.value, 1e-12, 'batch value mismatch');
    assertArrayClose(batch.ema.slice(0, n), single.ema, 1e-12, 'batch ema mismatch');
    assertArrayClose(batch.signal.slice(0, n), single.signal, 1e-12, 'batch signal mismatch');
});

test('daily_factor_into pointer path matches safe API', () => {
    const n = 160;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const safe = wasm.daily_factor_js(open, high, low, close, 0.35);
    const openPtr = wasm.allocate_f64_array(n);
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const closePtr = wasm.allocate_f64_array(n);
    const valuePtr = wasm.daily_factor_alloc(n);
    const emaPtr = wasm.daily_factor_alloc(n);
    const signalPtr = wasm.daily_factor_alloc(n);
    try {
        wasm.write_f64_array(openPtr, open);
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        wasm.write_f64_array(closePtr, close);
        wasm.daily_factor_into(openPtr, highPtr, lowPtr, closePtr, valuePtr, emaPtr, signalPtr, n, 0.35);
        const gotValue = wasm.read_f64_array(valuePtr, n);
        const gotEma = wasm.read_f64_array(emaPtr, n);
        const gotSignal = wasm.read_f64_array(signalPtr, n);
        assertArrayClose(gotValue, safe.value, 1e-12, 'pointer value mismatch');
        assertArrayClose(gotEma, safe.ema, 1e-12, 'pointer ema mismatch');
        assertArrayClose(gotSignal, safe.signal, 1e-12, 'pointer signal mismatch');
    } finally {
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
        wasm.daily_factor_free(valuePtr, n);
        wasm.daily_factor_free(emaPtr, n);
        wasm.daily_factor_free(signalPtr, n);
    }
});

test('daily_factor_batch_into returns row count', () => {
    const n = 128;
    const rows = 2;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const openPtr = wasm.allocate_f64_array(n);
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const closePtr = wasm.allocate_f64_array(n);
    const valuePtr = wasm.daily_factor_alloc(rows * n);
    const emaPtr = wasm.daily_factor_alloc(rows * n);
    const signalPtr = wasm.daily_factor_alloc(rows * n);
    try {
        wasm.write_f64_array(openPtr, open);
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        wasm.write_f64_array(closePtr, close);
        const returnedRows = wasm.daily_factor_batch_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            valuePtr,
            emaPtr,
            signalPtr,
            n,
            0.35,
            0.45,
            0.10,
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
        wasm.daily_factor_free(valuePtr, rows * n);
        wasm.daily_factor_free(emaPtr, rows * n);
        wasm.daily_factor_free(signalPtr, rows * n);
    }
});

test('daily_factor_js rejects invalid threshold_level', () => {
    const n = 64;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    assert.throws(
        () => wasm.daily_factor_js(open, high, low, close, 1.5),
        /Invalid threshold_level/,
    );
});
