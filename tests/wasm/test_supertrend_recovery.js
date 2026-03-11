import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function trendData(size) {
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        const base = 100 + i * 0.8;
        high[i] = base + 1.2 + (i % 3) * 0.1;
        low[i] = base - 1.0 - (i % 2) * 0.1;
        close[i] = base + ((i % 5) - 2) * 0.05;
    }
    return { high, low, close };
}

function reversalData() {
    const close = new Float64Array([
        100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 102.0, 99.0, 96.0, 93.0, 91.0, 90.0,
        91.0, 93.0, 96.0, 100.0, 105.0, 109.0, 112.0, 111.0, 109.0, 107.0, 104.0, 101.0, 98.0,
        96.0, 95.0, 96.0, 98.0,
    ]);
    const high = new Float64Array(close.length);
    const low = new Float64Array(close.length);
    for (let i = 0; i < close.length; i++) {
        high[i] = close[i] + 1.0;
        low[i] = close[i] - 1.0;
    }
    return { high, low, close };
}

function recoveryData() {
    const close = new Float64Array([
        100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 102.0, 97.0, 92.0, 88.0, 90.0, 93.0, 96.0,
        99.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0,
        90.0, 89.0, 88.0,
    ]);
    const high = new Float64Array(close.length);
    const low = new Float64Array(close.length);
    for (let i = 0; i < close.length; i++) {
        high[i] = close[i] + 0.9;
        low[i] = close[i] - 0.9;
    }
    return { high, low, close };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath =
        process.platform === 'win32' ? 'file:///' + wasmPath.replace(/\\/g, '/') : wasmPath;
    wasm = await import(importPath);
});

test('SuperTrend Recovery reversal behavior', () => {
    const { high, low, close } = reversalData();
    const result = wasm.supertrend_recovery_js(high, low, close, 4, 1.5, 5.0, 1.0);
    const changedIdx = [];
    for (let i = 0; i < result.changed.length; i++) {
        if (result.changed[i] === 1) {
            changedIdx.push(i);
        }
    }
    assert(changedIdx.length > 0);
    const first = changedIdx[0];
    assert(Number.isFinite(result.band[first]));
    assert(Number.isFinite(result.switch_price[first]));
    assert(result.trend[first] === 1 || result.trend[first] === -1);
});

test('SuperTrend Recovery invalid alpha', () => {
    const { high, low, close } = trendData(160);
    assert.throws(() => {
        wasm.supertrend_recovery_js(high, low, close, 10, 3.0, 0.0, 1.0);
    }, /invalid alpha_percent/i);
});

test('SuperTrend Recovery NaN gap restart', () => {
    const { high, low, close } = trendData(170);
    high[120] = Number.NaN;
    low[120] = Number.NaN;
    close[120] = Number.NaN;
    const result = wasm.supertrend_recovery_js(high, low, close, 10, 3.0, 5.0, 1.0);
    const restartEnd = Math.min(result.band.length, 120 + 10);
    for (let i = 120; i < restartEnd; i++) {
        assert(isNaN(result.band[i]));
        assert(isNaN(result.switch_price[i]));
        assert(isNaN(result.trend[i]));
        assert(isNaN(result.changed[i]));
    }
});

test('SuperTrend Recovery recovery logic changes band', () => {
    const { high, low, close } = recoveryData();
    const recovered = wasm.supertrend_recovery_js(high, low, close, 4, 3.0, 100.0, 0.0);
    const baseline = wasm.supertrend_recovery_js(high, low, close, 4, 3.0, 0.1, 1000.0);
    let found = false;
    for (let i = 0; i < recovered.band.length; i++) {
        if (
            Number.isFinite(recovered.band[i]) &&
            Number.isFinite(baseline.band[i]) &&
            recovered.trend[i] === baseline.trend[i]
        ) {
            if (recovered.trend[i] === 1 && recovered.band[i] > baseline.band[i]) {
                found = true;
                break;
            }
            if (recovered.trend[i] === -1 && recovered.band[i] < baseline.band[i]) {
                found = true;
                break;
            }
        }
    }
    assert(found);
});

test('SuperTrend Recovery fast API matches safe API', () => {
    const { high, low, close } = trendData(160);
    const len = close.length;
    const highPtr = wasm.supertrend_recovery_alloc(len);
    const lowPtr = wasm.supertrend_recovery_alloc(len);
    const closePtr = wasm.supertrend_recovery_alloc(len);
    const bandPtr = wasm.supertrend_recovery_alloc(len);
    const switchPtr = wasm.supertrend_recovery_alloc(len);
    const trendPtr = wasm.supertrend_recovery_alloc(len);
    const changedPtr = wasm.supertrend_recovery_alloc(len);

    try {
        new Float64Array(wasm.__wasm.memory.buffer, highPtr, len).set(high);
        new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len).set(low);
        new Float64Array(wasm.__wasm.memory.buffer, closePtr, len).set(close);

        wasm.supertrend_recovery_into(
            highPtr,
            lowPtr,
            closePtr,
            bandPtr,
            switchPtr,
            trendPtr,
            changedPtr,
            len,
            10,
            3.0,
            5.0,
            1.0,
        );

        const safe = wasm.supertrend_recovery_js(high, low, close, 10, 3.0, 5.0, 1.0);
        assertArrayClose(
            new Float64Array(wasm.__wasm.memory.buffer, bandPtr, len),
            safe.band,
            1e-12,
            'band mismatch',
        );
        assertArrayClose(
            new Float64Array(wasm.__wasm.memory.buffer, switchPtr, len),
            safe.switch_price,
            1e-12,
            'switch mismatch',
        );
        assertArrayClose(
            new Float64Array(wasm.__wasm.memory.buffer, trendPtr, len),
            safe.trend,
            1e-12,
            'trend mismatch',
        );
        assertArrayClose(
            new Float64Array(wasm.__wasm.memory.buffer, changedPtr, len),
            safe.changed,
            1e-12,
            'changed mismatch',
        );
    } finally {
        wasm.supertrend_recovery_free(highPtr, len);
        wasm.supertrend_recovery_free(lowPtr, len);
        wasm.supertrend_recovery_free(closePtr, len);
        wasm.supertrend_recovery_free(bandPtr, len);
        wasm.supertrend_recovery_free(switchPtr, len);
        wasm.supertrend_recovery_free(trendPtr, len);
        wasm.supertrend_recovery_free(changedPtr, len);
    }
});

test('SuperTrend Recovery batch matches single', () => {
    const { high, low, close } = trendData(170);
    const result = wasm.supertrend_recovery_batch(high, low, close, {
        atr_length_range: [4, 5, 1],
        multiplier_range: [1.5, 2.0, 0.5],
        alpha_percent_range: [5.0, 10.0, 5.0],
        threshold_atr_range: [0.5, 1.0, 0.5],
    });

    assert.strictEqual(result.rows, 16);
    assert.strictEqual(result.cols, close.length);

    for (let row = 0; row < result.rows; row++) {
        const combo = result.combos[row];
        const single = wasm.supertrend_recovery_js(
            high,
            low,
            close,
            combo.atr_length,
            combo.multiplier,
            combo.alpha_percent,
            combo.threshold_atr,
        );
        const start = row * result.cols;
        const end = start + result.cols;
        assertArrayClose(result.band.slice(start, end), single.band, 1e-12, `band row ${row}`);
        assertArrayClose(
            result.switch_price.slice(start, end),
            single.switch_price,
            1e-12,
            `switch row ${row}`,
        );
        assertArrayClose(
            result.trend.slice(start, end),
            single.trend,
            1e-12,
            `trend row ${row}`,
        );
        assertArrayClose(
            result.changed.slice(start, end),
            single.changed,
            1e-12,
            `changed row ${row}`,
        );
    }
});
