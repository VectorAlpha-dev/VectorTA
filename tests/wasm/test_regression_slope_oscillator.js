import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function manualReference(data, minRange = 10, maxRange = 100, step = 5, signalLine = 7) {
    const n = data.length;
    const out = {
        value: new Float64Array(n),
        signal: new Float64Array(n),
        bullish_reversal: new Float64Array(n),
        bearish_reversal: new Float64Array(n),
    };
    for (const key of Object.keys(out)) out[key].fill(NaN);
    const lengths = [];
    for (let value = minRange; value <= maxRange; value += step) lengths.push(value);

    for (let idx = maxRange - 1; idx < n; idx++) {
        const slopes = [];
        let valid = true;
        for (const length of lengths) {
            const start = idx + 1 - length;
            let sumX = 0;
            let sumY = 0;
            let sumXSqr = 0;
            let sumXY = 0;
            for (let j = 0; j < length; j++) {
                const sample = data[start + j];
                if (!Number.isFinite(sample) || sample <= 0) {
                    valid = false;
                    break;
                }
                const x = j + 1;
                const y = Math.log(sample);
                sumX += x;
                sumY += y;
                sumXSqr += x * x;
                sumXY += x * y;
            }
            if (!valid) break;
            slopes.push((length * sumXY - sumX * sumY) / (length * sumXSqr - sumX * sumX));
        }
        if (valid) {
            out.value[idx] = slopes.reduce((a, b) => a + b, 0) / slopes.length;
        }
    }

    const queue = [];
    let runningSum = 0;
    for (let idx = 0; idx < n; idx++) {
        const value = out.value[idx];
        if (Number.isFinite(value)) {
            queue.push(value);
            runningSum += value;
            if (queue.length > signalLine) {
                runningSum -= queue.shift();
            }
        }
        if (queue.length === signalLine) {
            out.signal[idx] = runningSum / signalLine;
        }
        if (Number.isFinite(value) && Number.isFinite(out.signal[idx])) {
            const prevValue = idx > 0 ? out.value[idx - 1] : NaN;
            const prevSignal = idx > 0 ? out.signal[idx - 1] : NaN;
            out.bullish_reversal[idx] = Number.isFinite(prevValue)
                && Number.isFinite(prevSignal)
                && value > out.signal[idx]
                && prevValue <= prevSignal
                && value < 0
                ? 1
                : 0;
            out.bearish_reversal[idx] = Number.isFinite(prevValue)
                && Number.isFinite(prevSignal)
                && value < out.signal[idx]
                && prevValue >= prevSignal
                && value > 0
                ? 1
                : 0;
        }
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

test('regression_slope_oscillator_js matches manual reference', () => {
    const data = new Float64Array(testData.close.slice(0, 220));
    const result = wasm.regression_slope_oscillator_js(data, 10, 30, 5, 7);
    const expected = manualReference(data, 10, 30, 5, 7);
    for (const key of ['value', 'signal', 'bullish_reversal', 'bearish_reversal']) {
        assertArrayClose(result[key], expected[key], 1e-10, `${key} mismatch`);
    }
});

test('regression_slope_oscillator_batch_js first row matches single', () => {
    const data = new Float64Array(testData.close.slice(0, 180));
    const batch = wasm.regression_slope_oscillator_batch_js(data, {
        min_range_range: [10, 15, 5],
        max_range_range: [30, 30, 0],
        step_range: [5, 5, 0],
        signal_line_range: [7, 7, 0],
    });
    const single = wasm.regression_slope_oscillator_js(data, 10, 30, 5, 7);
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, data.length);
    assertArrayClose(batch.min_ranges, [10, 15], 0, 'min_ranges mismatch');
    assertArrayClose(batch.max_ranges, [30, 30], 0, 'max_ranges mismatch');
    assertArrayClose(batch.steps, [5, 5], 0, 'steps mismatch');
    assertArrayClose(batch.signal_lines, [7, 7], 0, 'signal_lines mismatch');
    for (const key of ['value', 'signal', 'bullish_reversal', 'bearish_reversal']) {
        assertArrayClose(batch[key].slice(0, data.length), single[key], 1e-10, `${key} batch mismatch`);
    }
});

test('regression_slope_oscillator_into pointer path matches safe API', () => {
    const data = new Float64Array(testData.close.slice(0, 180));
    const safe = wasm.regression_slope_oscillator_js(data, 10, 30, 5, 7);
    const dataPtr = wasm.allocate_f64_array(data.length);
    const valuePtr = wasm.regression_slope_oscillator_alloc(data.length);
    const signalPtr = wasm.regression_slope_oscillator_alloc(data.length);
    const bullishPtr = wasm.regression_slope_oscillator_alloc(data.length);
    const bearishPtr = wasm.regression_slope_oscillator_alloc(data.length);
    try {
        wasm.write_f64_array(dataPtr, data);
        wasm.regression_slope_oscillator_into(
            dataPtr, valuePtr, signalPtr, bullishPtr, bearishPtr, data.length, 10, 30, 5, 7,
        );
        assertArrayClose(wasm.read_f64_array(valuePtr, data.length), safe.value, 1e-10, 'value pointer mismatch');
        assertArrayClose(wasm.read_f64_array(signalPtr, data.length), safe.signal, 1e-10, 'signal pointer mismatch');
        assertArrayClose(wasm.read_f64_array(bullishPtr, data.length), safe.bullish_reversal, 1e-10, 'bullish pointer mismatch');
        assertArrayClose(wasm.read_f64_array(bearishPtr, data.length), safe.bearish_reversal, 1e-10, 'bearish pointer mismatch');
    } finally {
        wasm.deallocate_f64_array(dataPtr);
        wasm.regression_slope_oscillator_free(valuePtr, data.length);
        wasm.regression_slope_oscillator_free(signalPtr, data.length);
        wasm.regression_slope_oscillator_free(bullishPtr, data.length);
        wasm.regression_slope_oscillator_free(bearishPtr, data.length);
    }
});

test('regression_slope_oscillator_batch_into returns row count', () => {
    const data = new Float64Array(testData.close.slice(0, 128));
    const rows = 2;
    const dataPtr = wasm.allocate_f64_array(data.length);
    const valuePtr = wasm.regression_slope_oscillator_alloc(rows * data.length);
    const signalPtr = wasm.regression_slope_oscillator_alloc(rows * data.length);
    const bullishPtr = wasm.regression_slope_oscillator_alloc(rows * data.length);
    const bearishPtr = wasm.regression_slope_oscillator_alloc(rows * data.length);
    try {
        wasm.write_f64_array(dataPtr, data);
        const returnedRows = wasm.regression_slope_oscillator_batch_into(
            dataPtr, valuePtr, signalPtr, bullishPtr, bearishPtr, data.length,
            10, 15, 5, 30, 30, 0, 5, 5, 0, 7, 7, 0,
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.deallocate_f64_array(dataPtr);
        wasm.regression_slope_oscillator_free(valuePtr, rows * data.length);
        wasm.regression_slope_oscillator_free(signalPtr, rows * data.length);
        wasm.regression_slope_oscillator_free(bullishPtr, rows * data.length);
        wasm.regression_slope_oscillator_free(bearishPtr, rows * data.length);
    }
});

test('regression_slope_oscillator_js rejects invalid range config', () => {
    const data = new Float64Array(testData.close.slice(0, 96));
    assert.throws(
        () => wasm.regression_slope_oscillator_js(data, 20, 10, 5, 7),
        /Invalid range config/,
    );
});
