import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

const DAY_MS = 86_400_000;

function priceRef(high, low, close, useClose) {
    if (useClose) {
        return close;
    }
    if (Number.isFinite(high) && Number.isFinite(low) && Number.isFinite(close)) {
        return (high + low + close) / 3;
    }
    return NaN;
}

function periodId(tsMs, sessionMode) {
    const sec = Math.trunc(tsMs / 1000);
    if (sessionMode === '4_hours') return Math.trunc(Math.trunc(sec / 3600) / 4);
    if (sessionMode === 'daily') return Math.trunc(sec / 86400);
    if (sessionMode === 'weekly') return Math.trunc((Math.trunc(sec / 86400) + 3) / 7);
    return 0;
}

function rollingStdIgnoreNa(values, period) {
    const out = new Float64Array(values.length);
    out.fill(NaN);
    const buf = new Float64Array(period);
    let head = 0;
    let count = 0;
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < values.length; i++) {
        const value = values[i];
        if (Number.isFinite(value)) {
            if (count === period) {
                const old = buf[head];
                sum -= old;
                sumSq -= old * old;
            } else {
                count += 1;
            }
            buf[head] = value;
            sum += value;
            sumSq += value * value;
            head = (head + 1) % period;
        }
        if (count === period) {
            const mean = sum / period;
            const variance = Math.max(sumSq / period - mean * mean, 0);
            out[i] = Math.sqrt(variance);
        }
    }
    return out;
}

function manualVdo(
    timestamps,
    high,
    low,
    close,
    volume,
    sessionMode,
    rollingPeriod,
    rollingDays,
    useClose,
    deviationMode,
    zWindow,
    pctVolLookback,
    pctMinSigma,
    absVolLookback,
) {
    const n = close.length;
    const residAbs = new Float64Array(n);
    const residPct = new Float64Array(n);
    residAbs.fill(NaN);
    residPct.fill(NaN);

    if (sessionMode === 'rolling_bars') {
        const queue = Array.from({ length: rollingPeriod }, () => [0, 0]);
        let head = 0;
        let count = 0;
        let sumPv = 0;
        let sumVol = 0;
        for (let i = 0; i < n; i++) {
            const pr = priceRef(high[i], low[i], close[i], useClose);
            const pv = Number.isFinite(pr) && Number.isFinite(volume[i]) ? pr * volume[i] : 0;
            const vv = Number.isFinite(pr) && Number.isFinite(volume[i]) ? volume[i] : 0;
            if (count === rollingPeriod) {
                const [oldPv, oldVol] = queue[head];
                sumPv -= oldPv;
                sumVol -= oldVol;
            } else {
                count += 1;
            }
            queue[head] = [pv, vv];
            sumPv += pv;
            sumVol += vv;
            head = (head + 1) % rollingPeriod;
            const vwap = sumVol !== 0 ? sumPv / sumVol : NaN;
            if (Number.isFinite(pr) && Number.isFinite(vwap)) {
                residAbs[i] = pr - vwap;
                if (vwap !== 0) residPct[i] = 100 * (pr / vwap - 1);
            }
        }
    } else if (sessionMode === 'rolling_days') {
        const queue = [];
        let sumPv = 0;
        let sumVol = 0;
        const daysMs = rollingDays * DAY_MS;
        for (let i = 0; i < n; i++) {
            const pr = priceRef(high[i], low[i], close[i], useClose);
            const pv = Number.isFinite(pr) && Number.isFinite(volume[i]) ? pr * volume[i] : 0;
            const vv = Number.isFinite(pr) && Number.isFinite(volume[i]) ? volume[i] : 0;
            queue.push([timestamps[i], pv, vv]);
            sumPv += pv;
            sumVol += vv;
            const cutoff = timestamps[i] - daysMs;
            while (queue.length > 0 && queue[0][0] < cutoff) {
                const [, oldPv, oldVol] = queue.shift();
                sumPv -= oldPv;
                sumVol -= oldVol;
            }
            const vwap = sumVol !== 0 ? sumPv / sumVol : NaN;
            if (Number.isFinite(pr) && Number.isFinite(vwap)) {
                residAbs[i] = pr - vwap;
                if (vwap !== 0) residPct[i] = 100 * (pr / vwap - 1);
            }
        }
    } else {
        let sumPv = 0;
        let sumVol = 0;
        let lastId = null;
        for (let i = 0; i < n; i++) {
            const currentId = periodId(timestamps[i], sessionMode);
            if (currentId !== lastId) {
                lastId = currentId;
                sumPv = 0;
                sumVol = 0;
            }
            const pr = priceRef(high[i], low[i], close[i], useClose);
            if (Number.isFinite(pr) && Number.isFinite(volume[i])) {
                sumPv += pr * volume[i];
                sumVol += volume[i];
            }
            const vwap = sumVol !== 0 ? sumPv / sumVol : NaN;
            if (Number.isFinite(pr) && Number.isFinite(vwap)) {
                residAbs[i] = pr - vwap;
                if (vwap !== 0) residPct[i] = 100 * (pr / vwap - 1);
            }
        }
    }

    if (deviationMode === 'absolute') {
        const std1 = rollingStdIgnoreNa(residAbs, absVolLookback);
        for (let i = 0; i < std1.length; i++) if (Number.isFinite(std1[i])) std1[i] = Math.max(std1[i], 1.0);
        return { osc: residAbs, std1, std2: Float64Array.from(std1, v => v * 2), std3: Float64Array.from(std1, v => v * 3) };
    }
    if (deviationMode === 'percent') {
        const std1 = rollingStdIgnoreNa(residPct, pctVolLookback);
        for (let i = 0; i < std1.length; i++) if (Number.isFinite(std1[i])) std1[i] = Math.max(std1[i], pctMinSigma);
        return { osc: residPct, std1, std2: Float64Array.from(std1, v => v * 2), std3: Float64Array.from(std1, v => v * 3) };
    }

    const osc = new Float64Array(n);
    osc.fill(NaN);
    const std1 = new Float64Array(n);
    std1.fill(1);
    const buf = new Float64Array(zWindow);
    let head = 0;
    let count = 0;
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
        const value = residAbs[i];
        if (Number.isFinite(value)) {
            if (count === zWindow) {
                const old = buf[head];
                sum -= old;
                sumSq -= old * old;
            } else {
                count += 1;
            }
            buf[head] = value;
            sum += value;
            sumSq += value * value;
            head = (head + 1) % zWindow;
        }
        if (count === zWindow && Number.isFinite(value)) {
            const mean = sum / zWindow;
            const std = Math.sqrt(Math.max(sumSq / zWindow - mean * mean, 0));
            if (std !== 0) osc[i] = (value - mean) / std;
        }
    }
    return { osc, std1, std2: Float64Array.from(std1, v => v * 2), std3: Float64Array.from(std1, v => v * 3) };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('vwap_deviation_oscillator_js matches manual reference', () => {
    const n = 160;
    const timestamps = new Float64Array(testData.timestamp.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const volume = new Float64Array(testData.volume.slice(0, n));
    const result = wasm.vwap_deviation_oscillator_js(
        timestamps, high, low, close, volume,
        'rolling_bars', 20, 30, true, 'percent', 50, 25, 0.1, 100,
    );
    const expected = manualVdo(
        timestamps, high, low, close, volume,
        'rolling_bars', 20, 30, true, 'percent', 50, 25, 0.1, 100,
    );
    assertArrayClose(result.osc, expected.osc, 1e-12, 'osc mismatch');
    assertArrayClose(result.std1, expected.std1, 1e-12, 'std1 mismatch');
    assertArrayClose(result.std2, expected.std2, 1e-12, 'std2 mismatch');
    assertArrayClose(result.std3, expected.std3, 1e-12, 'std3 mismatch');
});

test('vwap_deviation_oscillator_batch_js first row matches single', () => {
    const n = 96;
    const timestamps = new Float64Array(testData.timestamp.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const volume = new Float64Array(testData.volume.slice(0, n));
    const batch = wasm.vwap_deviation_oscillator_batch_js(timestamps, high, low, close, volume, {
        session_mode: 'rolling_bars',
        use_close: false,
        deviation_mode: 'zscore',
        rolling_period_range: [20, 22, 2],
        rolling_days_range: [30, 30, 0],
        z_window_range: [25, 25, 0],
        pct_vol_lookback_range: [100, 100, 0],
        pct_min_sigma_range: [0.1, 0.1, 0.0],
        abs_vol_lookback_range: [100, 100, 0],
    });
    const single = wasm.vwap_deviation_oscillator_js(
        timestamps, high, low, close, volume,
        'rolling_bars', 20, 30, false, 'zscore', 25, 100, 0.1, 100,
    );
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, n);
    assertArrayClose(batch.osc.slice(0, n), single.osc, 1e-12, 'batch osc mismatch');
});

test('vwap_deviation_oscillator_into pointer path matches safe API', () => {
    const n = 128;
    const timestamps = new Float64Array(testData.timestamp.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const volume = new Float64Array(testData.volume.slice(0, n));
    const safe = wasm.vwap_deviation_oscillator_js(
        timestamps, high, low, close, volume,
        'daily', 20, 30, false, 'absolute', 50, 100, 0.1, 20,
    );

    const tsPtr = wasm.allocate_f64_array(n);
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const closePtr = wasm.allocate_f64_array(n);
    const volumePtr = wasm.allocate_f64_array(n);
    const outPtr = wasm.vwap_deviation_oscillator_alloc(4 * n);

    try {
        wasm.write_f64_array(tsPtr, timestamps);
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        wasm.write_f64_array(closePtr, close);
        wasm.write_f64_array(volumePtr, volume);
        wasm.vwap_deviation_oscillator_into(
            tsPtr, highPtr, lowPtr, closePtr, volumePtr, outPtr, n,
            'daily', 20, 30, false, 'absolute', 50, 100, 0.1, 20,
        );
        const flat = wasm.read_f64_array(outPtr, 4 * n);
        assertArrayClose(flat.slice(0, n), safe.osc, 1e-12, 'pointer osc mismatch');
        assertArrayClose(flat.slice(n, 2 * n), safe.std1, 1e-12, 'pointer std1 mismatch');
    } finally {
        wasm.vwap_deviation_oscillator_free(outPtr, 4 * n);
        wasm.deallocate_f64_array(tsPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
        wasm.deallocate_f64_array(volumePtr);
    }
});

test('vwap_deviation_oscillator_batch_into returns row count', () => {
    const n = 96;
    const timestamps = new Float64Array(testData.timestamp.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const volume = new Float64Array(testData.volume.slice(0, n));
    const tsPtr = wasm.allocate_f64_array(n);
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const closePtr = wasm.allocate_f64_array(n);
    const volumePtr = wasm.allocate_f64_array(n);
    const rows = 2;
    const oscPtr = wasm.vwap_deviation_oscillator_alloc(rows * n);
    const std1Ptr = wasm.vwap_deviation_oscillator_alloc(rows * n);
    const std2Ptr = wasm.vwap_deviation_oscillator_alloc(rows * n);
    const std3Ptr = wasm.vwap_deviation_oscillator_alloc(rows * n);
    try {
        wasm.write_f64_array(tsPtr, timestamps);
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        wasm.write_f64_array(closePtr, close);
        wasm.write_f64_array(volumePtr, volume);
        const gotRows = wasm.vwap_deviation_oscillator_batch_into(
            tsPtr, highPtr, lowPtr, closePtr, volumePtr,
            oscPtr, std1Ptr, std2Ptr, std3Ptr,
            n, 'rolling_bars', false, 'zscore',
            20, 22, 2,
            30, 30, 0,
            25, 25, 0,
            100, 100, 0,
            0.1, 0.1, 0.0,
            100, 100, 0,
        );
        assert.strictEqual(gotRows, rows);
    } finally {
        wasm.vwap_deviation_oscillator_free(oscPtr, rows * n);
        wasm.vwap_deviation_oscillator_free(std1Ptr, rows * n);
        wasm.vwap_deviation_oscillator_free(std2Ptr, rows * n);
        wasm.vwap_deviation_oscillator_free(std3Ptr, rows * n);
        wasm.deallocate_f64_array(tsPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
        wasm.deallocate_f64_array(volumePtr);
    }
});

test('vwap_deviation_oscillator_js rejects invalid params', () => {
    const n = 64;
    const timestamps = new Float64Array(testData.timestamp.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const volume = new Float64Array(testData.volume.slice(0, n));

    assert.throws(() => {
        wasm.vwap_deviation_oscillator_js(
            timestamps, high, low, close, volume,
            'rolling_bars', 20, 30, false, 'zscore', 4, 100, 0.1, 100,
        );
    }, /Invalid z_window/);
});
