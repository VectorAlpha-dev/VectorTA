import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function firstValidIndex(high, low, close, source) {
    for (let i = 0; i < close.length; i++) {
        if (
            Number.isFinite(high[i])
            && Number.isFinite(low[i])
            && Number.isFinite(close[i])
            && Number.isFinite(source[i])
        ) {
            return i;
        }
    }
    return -1;
}

function avgFinite(a, b) {
    return Number.isFinite(a) && Number.isFinite(b) ? 0.5 * (a + b) : NaN;
}

function diffFinite(a, b) {
    return Number.isFinite(a) && Number.isFinite(b) ? a - b : NaN;
}

function gaussianValue(x, bandwidth) {
    return Math.exp(-0.5 * (x / bandwidth) ** 2) / Math.sqrt(2 * Math.PI);
}

function rollingMidpoint(high, low, length, first) {
    const out = new Float64Array(high.length);
    out.fill(NaN);
    const warm = first + length - 1;
    for (let i = first; i < high.length; i++) {
        if (!Number.isFinite(high[i]) || !Number.isFinite(low[i])) {
            continue;
        }
        const start = Math.max(first, i + 1 - length);
        let hh = -Infinity;
        let ll = Infinity;
        for (let j = start; j <= i; j++) {
            hh = Math.max(hh, high[j]);
            ll = Math.min(ll, low[j]);
        }
        if (i >= warm) {
            out[i] = 0.5 * (hh + ll);
        }
    }
    return out;
}

function chebyshevSeries(data, length, ripple) {
    const out = new Float64Array(data.length);
    out.fill(NaN);
    const a = Math.cosh((1 / length) * Math.acosh(1 / (1 - ripple)));
    const b = Math.sinh((1 / length) * Math.asinh(1 / ripple));
    const c = (a - b) / (a + b);
    const oneMinusC = 1 - c;
    for (let i = 0; i < data.length; i++) {
        if (!Number.isFinite(data[i])) {
            continue;
        }
        const prev = i > 0 && Number.isFinite(out[i - 1]) ? out[i - 1] : 0;
        out[i] = oneMinusC * data[i] + c * prev;
    }
    return out;
}

function gaussianKernelSeries(data, size, h, r) {
    const out = new Float64Array(data.length);
    out.fill(NaN);
    const weights = [];
    for (let i = 0; i <= size; i++) {
        weights.push(gaussianValue((i * i) / (h * h * r), r));
    }
    for (let i = size; i < data.length; i++) {
        let sum = 0;
        let weightSum = 0;
        let ok = true;
        for (let j = 0; j <= size; j++) {
            const value = data[i - j];
            if (!Number.isFinite(value)) {
                ok = false;
                break;
            }
            sum += value * weights[j];
            weightSum += weights[j];
        }
        if (ok && weightSum !== 0) {
            out[i] = sum / weightSum;
        }
    }
    return out;
}

function smoothSeries(data, length, extra) {
    const cheb = chebyshevSeries(data, length, 0.5);
    return extra ? gaussianKernelSeries(cheb, 4, 2.0, 1.0) : cheb;
}

function wmaSeries(data, length) {
    const out = new Float64Array(data.length);
    out.fill(NaN);
    const denom = (length * (length + 1)) / 2;
    for (let i = length - 1; i < data.length; i++) {
        let sum = 0;
        let ok = true;
        for (let j = 0; j < length; j++) {
            const value = data[i + 1 - length + j];
            if (!Number.isFinite(value)) {
                ok = false;
                break;
            }
            sum += value * (j + 1);
        }
        if (ok) {
            out[i] = sum / denom;
        }
    }
    return out;
}

function shiftBack(data, shift) {
    const out = new Float64Array(data.length);
    out.fill(NaN);
    if (shift === 0) {
        out.set(data);
        return out;
    }
    for (let i = shift; i < data.length; i++) {
        out[i] = data[i - shift];
    }
    return out;
}

function rollingRmsWindow(data, window) {
    const out = new Float64Array(data.length);
    out.fill(NaN);
    const queue = [];
    let sumSq = 0;
    for (let i = 0; i < data.length; i++) {
        if (!Number.isFinite(data[i])) {
            queue.length = 0;
            sumSq = 0;
            continue;
        }
        const sq = data[i] * data[i];
        queue.push(sq);
        sumSq += sq;
        if (queue.length > window) {
            sumSq -= queue.shift();
        }
        if (queue.length === window && window > 1) {
            out[i] = Math.sqrt(sumSq / (window - 1));
        }
    }
    return out;
}

function rmsAll(signal, gate) {
    const out = new Float64Array(signal.length);
    out.fill(NaN);
    let sumSq = 0;
    let count = 0;
    for (let i = 0; i < signal.length; i++) {
        if (Number.isFinite(signal[i])) {
            sumSq += signal[i] * signal[i];
            count += 1;
        }
        if (Number.isFinite(gate[i]) && count !== 0) {
            out[i] = Math.sqrt(sumSq / count);
        }
    }
    return out;
}

function normalizeValue(value, minLevel, maxLevel, mode, clamp) {
    if (mode === 'disabled') {
        return value;
    }
    if (
        !Number.isFinite(value)
        || !Number.isFinite(minLevel)
        || !Number.isFinite(maxLevel)
        || minLevel === maxLevel
    ) {
        return NaN;
    }
    let scaled = (value - minLevel) / (maxLevel - minLevel);
    if (clamp) {
        scaled = Math.min(Math.max(scaled, 0), 1);
    }
    return (scaled - 0.5) * 200;
}

function map2(a, b, fn) {
    const out = new Float64Array(a.length);
    out.fill(NaN);
    for (let i = 0; i < a.length; i++) {
        out[i] = fn(a[i], b[i]);
    }
    return out;
}

function manualIchimokuOscillator(
    high,
    low,
    close,
    source,
    conversionPeriods,
    basePeriods,
    laggingSpanPeriods,
    displacement,
    maLength,
    smoothingLength,
    extraSmoothing,
    normalize,
    windowSize,
    clamp,
    topBand,
    midBand,
) {
    const first = firstValidIndex(high, low, close, source);
    if (first === -1) {
        throw new Error('No valid data');
    }
    const conversionRaw = rollingMidpoint(high, low, conversionPeriods, first);
    const baseRaw = rollingMidpoint(high, low, basePeriods, first);
    const spanBRaw = rollingMidpoint(high, low, laggingSpanPeriods, first);
    const kumoAInput = map2(conversionRaw, baseRaw, avgFinite);
    const kumoA = smoothSeries(kumoAInput, smoothingLength, extraSmoothing);
    const kumoB = smoothSeries(spanBRaw, smoothingLength, extraSmoothing);
    const kumoCenter = map2(kumoA, kumoB, avgFinite);
    const kumoACentered = map2(kumoA, kumoCenter, diffFinite);
    const kumoBCentered = map2(kumoB, kumoCenter, diffFinite);

    const shift = displacement - 1;
    const kumoCenterOffset = shiftBack(kumoCenter, shift);
    const kumoAOffset = shiftBack(kumoA, shift);
    const kumoBOffset = shiftBack(kumoB, shift);
    const kumoAOffsetCentered = map2(kumoAOffset, kumoCenterOffset, diffFinite);
    const kumoBOffsetCentered = map2(kumoBOffset, kumoCenterOffset, diffFinite);

    const chikouRaw = shiftBack(source, displacement + 1);
    const chikouInput = map2(source, chikouRaw, diffFinite);
    const signalInput = map2(source, kumoCenterOffset, diffFinite);
    const conversionInput = map2(conversionRaw, kumoCenterOffset, diffFinite);
    const baseInput = map2(baseRaw, kumoCenterOffset, diffFinite);

    const chikou = smoothSeries(chikouInput, smoothingLength, extraSmoothing);
    const signal = smoothSeries(signalInput, smoothingLength, extraSmoothing);
    const conversion = smoothSeries(conversionInput, smoothingLength, extraSmoothing);
    const base = smoothSeries(baseInput, smoothingLength, extraSmoothing);
    const ma = wmaSeries(signal, maLength);

    let dev;
    if (normalize === 'all') {
        dev = rmsAll(signal, kumoAOffset);
    } else if (normalize === 'window') {
        dev = rollingRmsWindow(signal, windowSize);
    } else {
        dev = new Float64Array(close.length);
        dev.fill(0);
    }

    const maxLevel = Float64Array.from(dev, (v) => (Number.isFinite(v) ? v * topBand : NaN));
    const minLevel = Float64Array.from(dev, (v) => (Number.isFinite(v) ? -v * topBand : NaN));
    const highLevel = Float64Array.from(dev, (v) => (Number.isFinite(v) ? v * midBand : NaN));
    const lowLevel = Float64Array.from(dev, (v) => (Number.isFinite(v) ? -v * midBand : NaN));

    const out = {
        signal: new Float64Array(close.length),
        ma: new Float64Array(close.length),
        conversion: new Float64Array(close.length),
        base: new Float64Array(close.length),
        chikou: new Float64Array(close.length),
        current_kumo_a: new Float64Array(close.length),
        current_kumo_b: new Float64Array(close.length),
        future_kumo_a: new Float64Array(close.length),
        future_kumo_b: new Float64Array(close.length),
        max_level: new Float64Array(close.length),
        high_level: new Float64Array(close.length),
        low_level: new Float64Array(close.length),
        min_level: new Float64Array(close.length),
    };
    for (const values of Object.values(out)) {
        values.fill(NaN);
    }

    for (let i = 0; i < close.length; i++) {
        out.signal[i] = normalizeValue(signal[i], minLevel[i], maxLevel[i], normalize, clamp);
        out.ma[i] = normalizeValue(ma[i], minLevel[i], maxLevel[i], normalize, clamp);
        out.conversion[i] = normalizeValue(conversion[i], minLevel[i], maxLevel[i], normalize, clamp);
        out.base[i] = normalizeValue(base[i], minLevel[i], maxLevel[i], normalize, clamp);
        out.chikou[i] = normalizeValue(chikou[i], minLevel[i], maxLevel[i], normalize, clamp);
        out.future_kumo_a[i] = normalizeValue(kumoACentered[i], minLevel[i], maxLevel[i], normalize, clamp);
        out.future_kumo_b[i] = normalizeValue(kumoBCentered[i], minLevel[i], maxLevel[i], normalize, clamp);
        if (i >= shift) {
            out.current_kumo_a[i] = normalizeValue(
                kumoAOffsetCentered[i],
                minLevel[i - shift],
                maxLevel[i - shift],
                normalize,
                clamp,
            );
            out.current_kumo_b[i] = normalizeValue(
                kumoBOffsetCentered[i],
                minLevel[i - shift],
                maxLevel[i - shift],
                normalize,
                clamp,
            );
        }
        out.max_level[i] = normalizeValue(maxLevel[i], minLevel[i], maxLevel[i], normalize, clamp);
        out.high_level[i] = normalizeValue(highLevel[i], minLevel[i], maxLevel[i], normalize, clamp);
        out.low_level[i] = normalizeValue(lowLevel[i], minLevel[i], maxLevel[i], normalize, clamp);
        out.min_level[i] = normalizeValue(minLevel[i], minLevel[i], maxLevel[i], normalize, clamp);
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

test('ichimoku_oscillator_js matches manual reference', () => {
    const n = 192;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const source = new Float64Array(testData.close.slice(0, n));
    const result = wasm.ichimoku_oscillator_js(
        high, low, close, source,
        9, 26, 52, 26, 12, 3, true, 'window', 20, true, 2.0, 1.5,
    );
    const expected = manualIchimokuOscillator(
        high, low, close, source,
        9, 26, 52, 26, 12, 3, true, 'window', 20, true, 2.0, 1.5,
    );
    for (const key of ['signal', 'ma', 'conversion', 'base', 'chikou', 'current_kumo_a', 'future_kumo_b', 'max_level']) {
        assertArrayClose(result[key], expected[key], 1e-12, `${key} mismatch`);
    }
});

test('ichimoku_oscillator_batch_js first row matches single', () => {
    const n = 160;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const source = new Float64Array(testData.close.slice(0, n));
    const batch = wasm.ichimoku_oscillator_batch_js(high, low, close, source, {
        conversion_periods_range: [9, 11, 2],
        base_periods_range: [26, 26, 0],
        lagging_span_periods_range: [52, 52, 0],
        displacement_range: [26, 26, 0],
        ma_length_range: [12, 12, 0],
        smoothing_length_range: [3, 3, 0],
        window_size_range: [20, 20, 0],
        top_band_range: [2.0, 2.0, 0.0],
        mid_band_range: [1.5, 1.5, 0.0],
        extra_smoothing: true,
        normalize: 'window',
        clamp: true,
    });
    const single = wasm.ichimoku_oscillator_js(
        high, low, close, source,
        9, 26, 52, 26, 12, 3, true, 'window', 20, true, 2.0, 1.5,
    );
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, n);
    for (const key of ['signal', 'current_kumo_a', 'future_kumo_b']) {
        assertArrayClose(batch[key].slice(0, n), single[key], 1e-12, `${key} batch mismatch`);
    }
});

test('ichimoku_oscillator_into pointer path matches safe API', () => {
    const n = 160;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const source = new Float64Array(testData.close.slice(0, n));
    const safe = wasm.ichimoku_oscillator_js(
        high, low, close, source,
        9, 26, 52, 26, 12, 3, true, 'window', 20, true, 2.0, 1.5,
    );

    const inputPtrs = {
        high: wasm.allocate_f64_array(n),
        low: wasm.allocate_f64_array(n),
        close: wasm.allocate_f64_array(n),
        source: wasm.allocate_f64_array(n),
    };
    const outputKeys = [
        'signal', 'ma', 'conversion', 'base', 'chikou',
        'current_kumo_a', 'current_kumo_b', 'future_kumo_a', 'future_kumo_b',
        'max_level', 'high_level', 'low_level', 'min_level',
    ];
    const outputPtrs = Object.fromEntries(
        outputKeys.map((key) => [key, wasm.ichimoku_oscillator_alloc(n)]),
    );

    try {
        wasm.write_f64_array(inputPtrs.high, high);
        wasm.write_f64_array(inputPtrs.low, low);
        wasm.write_f64_array(inputPtrs.close, close);
        wasm.write_f64_array(inputPtrs.source, source);
        wasm.ichimoku_oscillator_into(
            inputPtrs.high,
            inputPtrs.low,
            inputPtrs.close,
            inputPtrs.source,
            outputPtrs.signal,
            outputPtrs.ma,
            outputPtrs.conversion,
            outputPtrs.base,
            outputPtrs.chikou,
            outputPtrs.current_kumo_a,
            outputPtrs.current_kumo_b,
            outputPtrs.future_kumo_a,
            outputPtrs.future_kumo_b,
            outputPtrs.max_level,
            outputPtrs.high_level,
            outputPtrs.low_level,
            outputPtrs.min_level,
            n,
            9, 26, 52, 26, 12, 3, true, 'window', 20, true, 2.0, 1.5,
        );
        for (const key of outputKeys) {
            const got = wasm.read_f64_array(outputPtrs[key], n);
            assertArrayClose(got, safe[key], 1e-12, `${key} pointer mismatch`);
        }
    } finally {
        for (const ptr of Object.values(outputPtrs)) {
            wasm.ichimoku_oscillator_free(ptr, n);
        }
        for (const ptr of Object.values(inputPtrs)) {
            wasm.deallocate_f64_array(ptr);
        }
    }
});

test('ichimoku_oscillator_batch_into returns row count', () => {
    const n = 128;
    const rows = 2;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const source = new Float64Array(testData.close.slice(0, n));
    const inputPtrs = {
        high: wasm.allocate_f64_array(n),
        low: wasm.allocate_f64_array(n),
        close: wasm.allocate_f64_array(n),
        source: wasm.allocate_f64_array(n),
    };
    const outputKeys = [
        'signal', 'ma', 'conversion', 'base', 'chikou',
        'current_kumo_a', 'current_kumo_b', 'future_kumo_a', 'future_kumo_b',
        'max_level', 'high_level', 'low_level', 'min_level',
    ];
    const outputPtrs = Object.fromEntries(
        outputKeys.map((key) => [key, wasm.ichimoku_oscillator_alloc(rows * n)]),
    );

    try {
        wasm.write_f64_array(inputPtrs.high, high);
        wasm.write_f64_array(inputPtrs.low, low);
        wasm.write_f64_array(inputPtrs.close, close);
        wasm.write_f64_array(inputPtrs.source, source);
        const returnedRows = wasm.ichimoku_oscillator_batch_into(
            inputPtrs.high,
            inputPtrs.low,
            inputPtrs.close,
            inputPtrs.source,
            outputPtrs.signal,
            outputPtrs.ma,
            outputPtrs.conversion,
            outputPtrs.base,
            outputPtrs.chikou,
            outputPtrs.current_kumo_a,
            outputPtrs.current_kumo_b,
            outputPtrs.future_kumo_a,
            outputPtrs.future_kumo_b,
            outputPtrs.max_level,
            outputPtrs.high_level,
            outputPtrs.low_level,
            outputPtrs.min_level,
            n,
            9, 11, 2,
            26, 26, 0,
            52, 52, 0,
            26, 26, 0,
            12, 12, 0,
            3, 3, 0,
            20, 20, 0,
            2.0, 2.0, 0.0,
            1.5, 1.5, 0.0,
            true,
            'window',
            true,
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        for (const ptr of Object.values(outputPtrs)) {
            wasm.ichimoku_oscillator_free(ptr, rows * n);
        }
        for (const ptr of Object.values(inputPtrs)) {
            wasm.deallocate_f64_array(ptr);
        }
    }
});

test('ichimoku_oscillator_js rejects invalid period', () => {
    const n = 96;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const source = new Float64Array(testData.close.slice(0, n));
    assert.throws(() => {
        wasm.ichimoku_oscillator_js(
            high, low, close, source,
            0, 26, 52, 26, 12, 3, true, 'window', 20, true, 2.0, 1.5,
        );
    }, /Invalid period/);
});

test('ichimoku_oscillator_batch_js rejects invalid config', () => {
    const n = 96;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const source = new Float64Array(testData.close.slice(0, n));
    assert.throws(() => {
        wasm.ichimoku_oscillator_batch_js(high, low, close, source, {
            conversion_periods_range: [9, 11],
            base_periods_range: [26, 26, 0],
            lagging_span_periods_range: [52, 52, 0],
            displacement_range: [26, 26, 0],
            ma_length_range: [12, 12, 0],
            smoothing_length_range: [3, 3, 0],
            window_size_range: [20, 20, 0],
            top_band_range: [2.0, 2.0, 0.0],
            mid_band_range: [1.5, 1.5, 0.0],
            extra_smoothing: true,
            normalize: 'window',
            clamp: true,
        });
    }, /Invalid config/);
});
