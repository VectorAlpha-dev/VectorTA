import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function sma(data, period) {
    const out = new Float64Array(data.length);
    out.fill(NaN);
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
        sum += data[i];
        if (i + 1 >= period) {
            if (i + 1 > period) {
                sum -= data[i - period];
            }
            out[i] = sum / period;
        }
    }
    return out;
}

function ema(data, period) {
    const out = new Float64Array(data.length);
    out.fill(NaN);
    const alpha = 2 / (period + 1);
    let seed = 0;
    let current = NaN;
    for (let i = 0; i < data.length; i++) {
        seed += data[i];
        if (i + 1 < period) continue;
        if (i + 1 === period) {
            current = seed / period;
        } else {
            current = alpha * data[i] + (1 - alpha) * current;
        }
        out[i] = current;
    }
    return out;
}

function wmaWindow(values) {
    let num = 0;
    let den = 0;
    for (let i = 0; i < values.length; i++) {
        const w = i + 1;
        num += values[i] * w;
        den += w;
    }
    return num / den;
}

function hma(data, period) {
    const out = new Float64Array(data.length);
    const diff = new Float64Array(data.length);
    out.fill(NaN);
    diff.fill(NaN);
    const half = Math.floor(period / 2);
    const sqrtLen = Math.floor(Math.sqrt(period));
    for (let i = 0; i < data.length; i++) {
        if (i + 1 >= period && i + 1 >= half) {
            diff[i] = 2 * wmaWindow(Array.from(data.slice(i + 1 - half, i + 1))) - wmaWindow(Array.from(data.slice(i + 1 - period, i + 1)));
        }
        if (i + 1 >= period + sqrtLen - 1) {
            const window = Array.from(diff.slice(i + 1 - sqrtLen, i + 1));
            if (window.every(Number.isFinite)) {
                out[i] = wmaWindow(window);
            }
        }
    }
    return out;
}

function stddev4(data, period) {
    const out = new Float64Array(data.length);
    out.fill(NaN);
    for (let i = period - 1; i < data.length; i++) {
        const window = Array.from(data.slice(i + 1 - period, i + 1));
        const mean = window.reduce((a, b) => a + b, 0) / period;
        const variance = window.reduce((a, b) => {
            const d = b - mean;
            return a + d * d;
        }, 0) / period;
        out[i] = Math.sqrt(variance) * 4;
    }
    return out;
}

function arrayEma(temp, period) {
    const alpha = 2 / (period + 1);
    let current = temp[temp.length - 1];
    for (let idx = temp.length - 2; idx >= 0; idx--) {
        current = alpha * temp[idx] + (1 - alpha) * current;
    }
    return current;
}

function arraySma(temp, period) {
    return Array.from(temp.slice(0, period)).reduce((a, b) => a + b, 0) / period;
}

function manualReference(data, maType = 'ema', smoothingWindow = 7, slowLength = 30, fastLength = 14, resolution = 50) {
    const n = data.length;
    const slowMa = maType === 'ema' ? ema(data, slowLength) : sma(data, slowLength);
    const fastMa = maType === 'ema' ? ema(data, fastLength) : sma(data, fastLength);
    const price = hma(data, smoothingWindow);
    const stdev = stddev4(data, smoothingWindow);
    const value = new Float64Array(n);
    const forecast = new Float64Array(n);
    const upper = new Float64Array(n);
    const lower = new Float64Array(n);
    const direction = new Float64Array(n);
    for (const arr of [value, forecast, upper, lower, direction]) arr.fill(NaN);

    for (let i = 0; i < n; i++) {
        if (Number.isFinite(slowMa[i]) && Number.isFinite(fastMa[i])) {
            direction[i] = fastMa[i] > slowMa[i] ? -1 : 1;
        }
        if (i > 0 && Number.isFinite(price[i]) && Number.isFinite(price[i - 1]) && Number.isFinite(stdev[i])) {
            forecast[i] = price[i] + (price[i] - price[i - 1]);
            upper[i] = forecast[i] + stdev[i];
            lower[i] = forecast[i] - stdev[i];
        }
        if (i < 2 * slowLength || !Number.isFinite(direction[i]) || !Number.isFinite(upper[i]) || !Number.isFinite(lower[i])) {
            continue;
        }
        const memory = [];
        for (let j = 0; j <= 2 * slowLength; j++) memory.push(data[i - j]);
        const step = (upper[i] - lower[i]) / (resolution - 1);
        let hits = 0;
        for (let k = 0; k < resolution; k++) {
            const possibility = lower[i] + step * k;
            const temp = [possibility, ...memory];
            const slowFuture = maType === 'ema' ? arrayEma(temp, slowLength) : arraySma(temp, slowLength);
            const fastFuture = maType === 'ema' ? arrayEma(temp, fastLength) : arraySma(temp, fastLength);
            const crossed = direction[i] < 0 ? slowFuture > fastFuture : slowFuture <= fastFuture;
            if (crossed) hits += 1;
        }
        value[i] = 100 * hits / resolution;
    }

    return { value, slow_ma: slowMa, fast_ma: fastMa, forecast, upper, lower, direction };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('moving_average_cross_probability_js matches manual reference', () => {
    const data = new Float64Array(testData.close.slice(0, 220));
    const result = wasm.moving_average_cross_probability_js(data, 'ema', 7, 30, 14, 50);
    const expected = manualReference(data);
    for (const key of ['value', 'slow_ma', 'fast_ma', 'forecast', 'upper', 'lower', 'direction']) {
        assertArrayClose(result[key], expected[key], 1e-8, `${key} mismatch`);
    }
});

test('moving_average_cross_probability_batch_js first row matches single', () => {
    const data = new Float64Array(testData.close.slice(0, 180));
    const batch = wasm.moving_average_cross_probability_batch_js(data, {
        smoothing_window_range: [7, 8, 1],
        slow_length_range: [30, 30, 0],
        fast_length_range: [14, 14, 0],
        resolution_range: [50, 50, 0],
        ma_type: 'ema',
    });
    const single = wasm.moving_average_cross_probability_js(data, 'ema', 7, 30, 14, 50);
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, data.length);
    assertArrayClose(batch.smoothing_windows, [7, 8], 0, 'smoothing window grid mismatch');
    assertArrayClose(batch.value.slice(0, data.length), single.value, 1e-8, 'batch value mismatch');
    assertArrayClose(batch.slow_ma.slice(0, data.length), single.slow_ma, 1e-8, 'batch slow_ma mismatch');
});

test('moving_average_cross_probability_into pointer path matches safe API', () => {
    const data = new Float64Array(testData.close.slice(0, 180));
    const safe = wasm.moving_average_cross_probability_js(data, 'sma', 8, 26, 11, 40);
    const dataPtr = wasm.allocate_f64_array(data.length);
    const valuePtr = wasm.moving_average_cross_probability_alloc(data.length);
    const slowPtr = wasm.moving_average_cross_probability_alloc(data.length);
    const fastPtr = wasm.moving_average_cross_probability_alloc(data.length);
    const forecastPtr = wasm.moving_average_cross_probability_alloc(data.length);
    const upperPtr = wasm.moving_average_cross_probability_alloc(data.length);
    const lowerPtr = wasm.moving_average_cross_probability_alloc(data.length);
    const directionPtr = wasm.moving_average_cross_probability_alloc(data.length);
    try {
        wasm.write_f64_array(dataPtr, data);
        wasm.moving_average_cross_probability_into(
            dataPtr, valuePtr, slowPtr, fastPtr, forecastPtr, upperPtr, lowerPtr, directionPtr,
            data.length, 'sma', 8, 26, 11, 40,
        );
        assertArrayClose(wasm.read_f64_array(valuePtr, data.length), safe.value, 1e-8, 'value pointer mismatch');
        assertArrayClose(wasm.read_f64_array(directionPtr, data.length), safe.direction, 1e-8, 'direction pointer mismatch');
    } finally {
        wasm.deallocate_f64_array(dataPtr);
        for (const ptr of [valuePtr, slowPtr, fastPtr, forecastPtr, upperPtr, lowerPtr, directionPtr]) {
            wasm.moving_average_cross_probability_free(ptr, data.length);
        }
    }
});

test('moving_average_cross_probability_batch_into returns row count', () => {
    const data = new Float64Array(testData.close.slice(0, 160));
    const rows = 2;
    const dataPtr = wasm.allocate_f64_array(data.length);
    const ptrs = Array.from({ length: 7 }, () => wasm.moving_average_cross_probability_alloc(rows * data.length));
    try {
        wasm.write_f64_array(dataPtr, data);
        const returnedRows = wasm.moving_average_cross_probability_batch_into(
            dataPtr, ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6],
            data.length, 7, 8, 1, 30, 30, 0, 14, 14, 0, 50, 50, 0, 'ema',
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.deallocate_f64_array(dataPtr);
        for (const ptr of ptrs) {
            wasm.moving_average_cross_probability_free(ptr, rows * data.length);
        }
    }
});

test('moving_average_cross_probability_js rejects invalid length order', () => {
    const data = new Float64Array(testData.close.slice(0, 96));
    assert.throws(
        () => wasm.moving_average_cross_probability_js(data, 'ema', 7, 10, 14, 50),
        /Invalid length order/,
    );
});
