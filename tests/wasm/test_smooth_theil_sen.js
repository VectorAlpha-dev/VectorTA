import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function weights(size) {
    const center = (size - 1) * 0.5;
    const width = (size - 0.5) * Math.pow(1024, -0.5) + 0.5;
    const values = Array.from({ length: size }, (_, i) =>
        Math.exp(-Math.pow((i - center) / width, 2) / 4) / Math.sqrt(2 * Math.PI),
    );
    const sum = values.reduce((a, b) => a + b, 0);
    return values.map((v) => v / sum);
}

function estimate(values, style) {
    if (style === 'mean') {
        return values.reduce((a, b) => a + b, 0) / values.length;
    }
    const sorted = [...values].sort((a, b) => a - b);
    if (style === 'median') {
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) * 0.5;
    }
    return sorted.reduce((sum, v, i) => sum + v * weights(sorted.length)[i], 0);
}

function manualReference(
    data,
    length = 25,
    offset = 0,
    multiplier = 2.0,
    slopeStyle = 'smooth_median',
    residualStyle = 'smooth_median',
    deviationStyle = 'mad',
    madStyle = 'smooth_median',
    includePredictionInDeviation = false,
) {
    const n = data.length;
    const out = {
        value: new Float64Array(n),
        upper: new Float64Array(n),
        lower: new Float64Array(n),
        slope: new Float64Array(n),
        intercept: new Float64Array(n),
        deviation: new Float64Array(n),
    };
    for (const key of Object.keys(out)) out[key].fill(NaN);
    const warmup = length + offset - 1;

    for (let idx = warmup; idx < n; idx++) {
        const base = idx - offset;
        const start = base + 1 - length;
        const end = includePredictionInDeviation ? idx : base;
        let finite = true;
        for (let i = start; i <= end; i++) {
            if (!Number.isFinite(data[i])) {
                finite = false;
                break;
            }
        }
        if (!finite) continue;

        const slopes = [];
        for (let i = 0; i < length - 1; i++) {
            const valueI = data[base - i];
            for (let j = i + 1; j < length; j++) {
                slopes.push((data[base - j] - valueI) / (j - i));
            }
        }
        const beta1 = estimate(slopes, slopeStyle);
        const residuals = [];
        for (let j = 0; j < length; j++) residuals.push(data[base - j] - beta1 * j);
        const beta0 = estimate(residuals, residualStyle);
        const predicted = beta0 - beta1 * offset;

        let dev;
        if (deviationStyle === 'mad') {
            const errors = [];
            const pointStart = includePredictionInDeviation ? -offset : 0;
            for (let point = pointStart; point < length; point++) {
                const sourceIdx = idx - offset - point;
                const predictedPoint = beta0 + beta1 * point;
                errors.push(Math.abs(data[sourceIdx] - predictedPoint));
            }
            dev = estimate(errors, madStyle) * multiplier;
        } else {
            let sq = 0;
            let count = 0;
            const pointStart = includePredictionInDeviation ? -offset : 0;
            for (let point = pointStart; point < length; point++) {
                const sourceIdx = idx - offset - point;
                const predictedPoint = beta0 + beta1 * point;
                const diff = data[sourceIdx] - predictedPoint;
                sq += diff * diff;
                count += 1;
            }
            dev = Math.sqrt(sq / count) * multiplier;
        }

        out.value[idx] = predicted;
        out.upper[idx] = predicted + dev;
        out.lower[idx] = predicted - dev;
        out.slope[idx] = beta1;
        out.intercept[idx] = beta0;
        out.deviation[idx] = dev;
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

test('smooth_theil_sen_js matches manual reference', () => {
    const data = new Float64Array(testData.close.slice(0, 192));
    const result = wasm.smooth_theil_sen_js(
        data, 21, 2, 1.75, 'smooth_median', 'median', 'mad', 'smooth_median', true,
    );
    const expected = manualReference(
        data, 21, 2, 1.75, 'smooth_median', 'median', 'mad', 'smooth_median', true,
    );
    for (const key of ['value', 'upper', 'lower', 'slope', 'intercept', 'deviation']) {
        assertArrayClose(result[key], expected[key], 1e-10, `${key} mismatch`);
    }
});

test('smooth_theil_sen_batch_js first row matches single', () => {
    const data = new Float64Array(testData.close.slice(0, 144));
    const batch = wasm.smooth_theil_sen_batch_js(data, {
        length_range: [21, 23, 2],
        offset_range: [1, 1, 0],
        multiplier_range: [1.5, 1.5, 0.0],
        slope_style: 'mean',
        residual_style: 'smooth_median',
        deviation_style: 'rmsd',
        mad_style: 'median',
        include_prediction_in_deviation: false,
    });
    const single = wasm.smooth_theil_sen_js(
        data, 21, 1, 1.5, 'mean', 'smooth_median', 'rmsd', 'median', false,
    );
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, data.length);
    assertArrayClose(batch.lengths, [21, 23], 0, 'length grid mismatch');
    assertArrayClose(batch.offsets, [1, 1], 0, 'offset grid mismatch');
    assertArrayClose(batch.multipliers, [1.5, 1.5], 1e-12, 'multiplier grid mismatch');
    for (const key of ['value', 'upper', 'lower', 'slope', 'intercept', 'deviation']) {
        assertArrayClose(batch[key].slice(0, data.length), single[key], 1e-10, `${key} batch mismatch`);
    }
});

test('smooth_theil_sen_into pointer path matches safe API', () => {
    const data = new Float64Array(testData.close.slice(0, 160));
    const safe = wasm.smooth_theil_sen_js(
        data, 21, 2, 1.5, 'smooth_median', 'median', 'mad', 'smooth_median', true,
    );
    const dataPtr = wasm.allocate_f64_array(data.length);
    const valuePtr = wasm.smooth_theil_sen_alloc(data.length);
    const upperPtr = wasm.smooth_theil_sen_alloc(data.length);
    const lowerPtr = wasm.smooth_theil_sen_alloc(data.length);
    const slopePtr = wasm.smooth_theil_sen_alloc(data.length);
    const interceptPtr = wasm.smooth_theil_sen_alloc(data.length);
    const deviationPtr = wasm.smooth_theil_sen_alloc(data.length);
    try {
        wasm.write_f64_array(dataPtr, data);
        wasm.smooth_theil_sen_into(
            dataPtr, valuePtr, upperPtr, lowerPtr, slopePtr, interceptPtr, deviationPtr,
            data.length, 21, 2, 1.5, 'smooth_median', 'median', 'mad', 'smooth_median', true,
        );
        assertArrayClose(wasm.read_f64_array(valuePtr, data.length), safe.value, 1e-10, 'value pointer mismatch');
        assertArrayClose(wasm.read_f64_array(upperPtr, data.length), safe.upper, 1e-10, 'upper pointer mismatch');
        assertArrayClose(wasm.read_f64_array(lowerPtr, data.length), safe.lower, 1e-10, 'lower pointer mismatch');
        assertArrayClose(wasm.read_f64_array(slopePtr, data.length), safe.slope, 1e-10, 'slope pointer mismatch');
        assertArrayClose(wasm.read_f64_array(interceptPtr, data.length), safe.intercept, 1e-10, 'intercept pointer mismatch');
        assertArrayClose(wasm.read_f64_array(deviationPtr, data.length), safe.deviation, 1e-10, 'deviation pointer mismatch');
    } finally {
        wasm.deallocate_f64_array(dataPtr);
        wasm.smooth_theil_sen_free(valuePtr, data.length);
        wasm.smooth_theil_sen_free(upperPtr, data.length);
        wasm.smooth_theil_sen_free(lowerPtr, data.length);
        wasm.smooth_theil_sen_free(slopePtr, data.length);
        wasm.smooth_theil_sen_free(interceptPtr, data.length);
        wasm.smooth_theil_sen_free(deviationPtr, data.length);
    }
});

test('smooth_theil_sen_batch_into returns row count', () => {
    const data = new Float64Array(testData.close.slice(0, 128));
    const rows = 2;
    const dataPtr = wasm.allocate_f64_array(data.length);
    const valuePtr = wasm.smooth_theil_sen_alloc(rows * data.length);
    const upperPtr = wasm.smooth_theil_sen_alloc(rows * data.length);
    const lowerPtr = wasm.smooth_theil_sen_alloc(rows * data.length);
    const slopePtr = wasm.smooth_theil_sen_alloc(rows * data.length);
    const interceptPtr = wasm.smooth_theil_sen_alloc(rows * data.length);
    const deviationPtr = wasm.smooth_theil_sen_alloc(rows * data.length);
    try {
        wasm.write_f64_array(dataPtr, data);
        const returnedRows = wasm.smooth_theil_sen_batch_into(
            dataPtr, valuePtr, upperPtr, lowerPtr, slopePtr, interceptPtr, deviationPtr,
            data.length, 21, 23, 2, 1, 1, 0, 1.5, 1.5, 0.0, 'mean', 'smooth_median', 'rmsd', 'median', false,
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.deallocate_f64_array(dataPtr);
        wasm.smooth_theil_sen_free(valuePtr, rows * data.length);
        wasm.smooth_theil_sen_free(upperPtr, rows * data.length);
        wasm.smooth_theil_sen_free(lowerPtr, rows * data.length);
        wasm.smooth_theil_sen_free(slopePtr, rows * data.length);
        wasm.smooth_theil_sen_free(interceptPtr, rows * data.length);
        wasm.smooth_theil_sen_free(deviationPtr, rows * data.length);
    }
});

test('smooth_theil_sen_js rejects invalid style', () => {
    const data = new Float64Array(testData.close.slice(0, 64));
    assert.throws(
        () => wasm.smooth_theil_sen_js(
            data, 21, 0, 2.0, 'bad_style', 'smooth_median', 'mad', 'smooth_median', false,
        ),
        /Invalid stat style/,
    );
});
