import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function ema(close, period) {
    const out = new Float64Array(close.length);
    out.fill(NaN);
    const alpha = 2 / (period + 1);
    let prev = NaN;
    for (let i = 0; i < close.length; i++) {
        const x = close[i];
        if (!Number.isFinite(x)) {
            prev = NaN;
            continue;
        }
        prev = Number.isFinite(prev) ? prev + alpha * (x - prev) : x;
        out[i] = prev;
    }
    return out;
}

function manualReference(
    high,
    low,
    close,
    period = 14,
    calculationMethod = 'normalized',
    normalizedBarsBack = 120,
    rawRollingPeriod = 50,
    rawThresholdPercentile = 95.0,
    thresholdLevel = 80.0,
) {
    const n = close.length;
    const ma = ema(close, period);
    const bull = new Float64Array(n);
    const bear = new Float64Array(n);
    const value = new Float64Array(n);
    const upper = new Float64Array(n);
    const lower = new Float64Array(n);
    const bullishSignal = new Float64Array(n);
    const bearishSignal = new Float64Array(n);
    const zeroCrossUp = new Float64Array(n);
    const zeroCrossDown = new Float64Array(n);
    bull.fill(NaN);
    bear.fill(NaN);
    value.fill(NaN);
    upper.fill(NaN);
    lower.fill(NaN);
    bullishSignal.fill(NaN);
    bearishSignal.fill(NaN);
    zeroCrossUp.fill(NaN);
    zeroCrossDown.fill(NaN);

    for (let i = 0; i < n; i++) {
        if ([high[i], low[i], ma[i]].every(Number.isFinite)) {
            bull[i] = high[i] - ma[i];
            bear[i] = ma[i] - low[i];
        }
    }

    if (calculationMethod === 'normalized') {
        for (let i = 0; i < n; i++) {
            upper[i] = thresholdLevel;
            lower[i] = -thresholdLevel;
            const left = Math.max(0, i - normalizedBarsBack + 1);
            const bullWindow = Array.from(bull.slice(left, i + 1)).filter(Number.isFinite);
            const bearWindow = Array.from(bear.slice(left, i + 1)).filter(Number.isFinite);
            if (!Number.isFinite(bull[i]) || !Number.isFinite(bear[i]) || bullWindow.length === 0 || bearWindow.length === 0) {
                continue;
            }
            const bullMin = Math.min(...bullWindow);
            const bullMax = Math.max(...bullWindow);
            const bearMin = Math.min(...bearWindow);
            const bearMax = Math.max(...bearWindow);
            const bullRange = bullMax - bullMin;
            const bearRange = bearMax - bearMin;
            if (bullRange > 0 && bearRange > 0) {
                const normBull = ((bull[i] - bullMin) / bullRange - 0.5) * 100;
                const normBear = ((bear[i] - bearMin) / bearRange - 0.5) * 100;
                value[i] = normBull - normBear;
            }
        }
    } else {
        for (let i = 0; i < n; i++) {
            if (Number.isFinite(bull[i]) && Number.isFinite(bear[i])) {
                value[i] = bull[i] - bear[i];
            }
        }
        for (let i = 0; i < n; i++) {
            const left = Math.max(0, i - rawRollingPeriod + 1);
            const window = Array.from(value.slice(left, i + 1)).filter(Number.isFinite);
            if (!window.length) continue;
            const lo = Math.min(...window);
            const hi = Math.max(...window);
            const range = hi - lo;
            upper[i] = lo + range * (rawThresholdPercentile / 100);
            lower[i] = lo + range * ((100 - rawThresholdPercentile) / 100);
        }
    }

    let prevTotal = NaN;
    for (let i = 0; i < n; i++) {
        if (Number.isFinite(value[i]) && Number.isFinite(upper[i]) && Number.isFinite(lower[i])) {
            bullishSignal[i] = value[i] > upper[i] ? 1 : 0;
            bearishSignal[i] = value[i] < lower[i] ? 1 : 0;
            zeroCrossUp[i] = Number.isFinite(prevTotal) && value[i] > 0 && prevTotal <= 0 ? 1 : 0;
            zeroCrossDown[i] = Number.isFinite(prevTotal) && value[i] < 0 && prevTotal >= 0 ? 1 : 0;
            prevTotal = value[i];
        }
    }

    return { value, bull, bear, ma, upper, lower, bullishSignal, bearishSignal, zeroCrossUp, zeroCrossDown };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('bulls_v_bears_js matches manual reference', () => {
    const n = 192;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const result = wasm.bulls_v_bears_js(high, low, close, 14, 'ema', 'normalized', 120, 50, 95.0, 80.0);
    const expected = manualReference(high, low, close);
    assertArrayClose(result.value, expected.value, 1e-12, 'value mismatch');
    assertArrayClose(result.bull, expected.bull, 1e-12, 'bull mismatch');
    assertArrayClose(result.bear, expected.bear, 1e-12, 'bear mismatch');
    assertArrayClose(result.ma, expected.ma, 1e-12, 'ma mismatch');
    assertArrayClose(result.upper, expected.upper, 1e-12, 'upper mismatch');
    assertArrayClose(result.lower, expected.lower, 1e-12, 'lower mismatch');
    assertArrayClose(result.bullish_signal, expected.bullishSignal, 1e-12, 'bullish signal mismatch');
    assertArrayClose(result.bearish_signal, expected.bearishSignal, 1e-12, 'bearish signal mismatch');
    assertArrayClose(result.zero_cross_up, expected.zeroCrossUp, 1e-12, 'zero cross up mismatch');
    assertArrayClose(result.zero_cross_down, expected.zeroCrossDown, 1e-12, 'zero cross down mismatch');
});

test('bulls_v_bears_batch_js first row matches single', () => {
    const n = 144;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const batch = wasm.bulls_v_bears_batch_js(high, low, close, {
        period_range: [14, 16, 2],
        normalized_bars_back_range: [120, 120, 0],
        raw_rolling_period_range: [50, 50, 0],
        raw_threshold_percentile_range: [95.0, 95.0, 0.0],
        threshold_level_range: [80.0, 80.0, 0.0],
        calculation_method: 'raw',
    });
    const single = wasm.bulls_v_bears_js(high, low, close, 14, 'ema', 'raw', 120, 50, 95.0, 80.0);
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, n);
    assertArrayClose(batch.periods, [14, 16], 1e-12, 'period grid mismatch');
    assertArrayClose(batch.value.slice(0, n), single.value, 1e-12, 'batch value mismatch');
    assertArrayClose(batch.upper.slice(0, n), single.upper, 1e-12, 'batch upper mismatch');
    assertArrayClose(batch.lower.slice(0, n), single.lower, 1e-12, 'batch lower mismatch');
});

test('bulls_v_bears_into pointer path matches safe API', () => {
    const n = 160;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const safe = wasm.bulls_v_bears_js(high, low, close, 14, 'ema', 'raw', 120, 50, 95.0, 80.0);
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const closePtr = wasm.allocate_f64_array(n);
    const valuePtr = wasm.bulls_v_bears_alloc(n);
    const bullPtr = wasm.bulls_v_bears_alloc(n);
    const bearPtr = wasm.bulls_v_bears_alloc(n);
    const maPtr = wasm.bulls_v_bears_alloc(n);
    const upperPtr = wasm.bulls_v_bears_alloc(n);
    const lowerPtr = wasm.bulls_v_bears_alloc(n);
    const bullishPtr = wasm.bulls_v_bears_alloc(n);
    const bearishPtr = wasm.bulls_v_bears_alloc(n);
    const upPtr = wasm.bulls_v_bears_alloc(n);
    const downPtr = wasm.bulls_v_bears_alloc(n);
    try {
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        wasm.write_f64_array(closePtr, close);
        wasm.bulls_v_bears_into(
            highPtr, lowPtr, closePtr, valuePtr, bullPtr, bearPtr, maPtr, upperPtr, lowerPtr,
            bullishPtr, bearishPtr, upPtr, downPtr, n, 14, 'ema', 'raw', 120, 50, 95.0, 80.0,
        );
        assertArrayClose(wasm.read_f64_array(valuePtr, n), safe.value, 1e-12, 'pointer value mismatch');
        assertArrayClose(wasm.read_f64_array(upperPtr, n), safe.upper, 1e-12, 'pointer upper mismatch');
        assertArrayClose(wasm.read_f64_array(lowerPtr, n), safe.lower, 1e-12, 'pointer lower mismatch');
    } finally {
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
        for (const ptr of [valuePtr, bullPtr, bearPtr, maPtr, upperPtr, lowerPtr, bullishPtr, bearishPtr, upPtr, downPtr]) {
            wasm.bulls_v_bears_free(ptr, n);
        }
    }
});

test('bulls_v_bears_batch_into returns row count', () => {
    const n = 128;
    const rows = 2;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const closePtr = wasm.allocate_f64_array(n);
    const ptrs = Array.from({ length: 10 }, () => wasm.bulls_v_bears_alloc(rows * n));
    try {
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        wasm.write_f64_array(closePtr, close);
        const returnedRows = wasm.bulls_v_bears_batch_into(
            highPtr, lowPtr, closePtr,
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7], ptrs[8], ptrs[9],
            n, 14, 16, 2, 120, 120, 0, 50, 50, 0, 95.0, 95.0, 0.0, 80.0, 80.0, 0.0, 'ema', 'raw',
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
        for (const ptr of ptrs) {
            wasm.bulls_v_bears_free(ptr, rows * n);
        }
    }
});

test('bulls_v_bears_js rejects invalid threshold level', () => {
    const n = 64;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    assert.throws(
        () => wasm.bulls_v_bears_js(high, low, close, 14, 'ema', 'normalized', 120, 50, 95.0, 120.0),
        /Invalid threshold_level/,
    );
});
