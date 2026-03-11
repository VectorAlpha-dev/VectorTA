import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function nz(arr, idx) {
    if (idx < 0) {
        return 0;
    }
    const value = arr[idx];
    return Number.isFinite(value) ? value : 0;
}

function manualReference(source, high, low, smoothPeriod = 10) {
    const n = source.length;
    const out = new Float64Array(n);
    const smooth = new Float64Array(n);
    const detrender = new Float64Array(n);
    const q1 = new Float64Array(n);
    const i1 = new Float64Array(n);
    const range1 = new Float64Array(n);
    out.fill(NaN);

    const periodMult = 0.075 * smoothPeriod + 0.54;
    let i2Prev = 0;
    let q2Prev = 0;
    let rePrev = 0;
    let imPrev = 0;
    let periodPrev = 0;
    let snrPrev = 0;
    let validCount = 0;

    for (let i = 0; i < n; i++) {
        if (!(Number.isFinite(source[i]) && Number.isFinite(high[i]) && Number.isFinite(low[i]))) {
            continue;
        }

        range1[i] = 0.1 * (high[i] - low[i]) + 0.9 * (i > 0 ? range1[i - 1] : 0);

        if (validCount > 5) {
            smooth[i] = (4 * source[i] + 3 * nz(source, i - 1) + 2 * nz(source, i - 2) + nz(source, i - 3)) / 10;
            detrender[i] = (
                0.0962 * smooth[i]
                + 0.5769 * nz(smooth, i - 2)
                - 0.5769 * nz(smooth, i - 4)
                - 0.0962 * nz(smooth, i - 6)
            ) * periodMult;
            i1[i] = nz(detrender, i - 3);
            q1[i] = (
                0.0962 * detrender[i]
                + 0.5769 * nz(detrender, i - 2)
                - 0.5769 * nz(detrender, i - 4)
                - 0.0962 * nz(detrender, i - 6)
            ) * periodMult;

            const ji = (
                0.0962 * i1[i]
                + 0.5769 * nz(i1, i - 2)
                - 0.5769 * nz(i1, i - 4)
                - 0.0962 * nz(i1, i - 6)
            ) * periodMult;
            const jq = (
                0.0962 * q1[i]
                + 0.5769 * nz(q1, i - 2)
                - 0.5769 * nz(q1, i - 4)
                - 0.0962 * nz(q1, i - 6)
            ) * periodMult;

            const i2 = 0.2 * (i1[i] - jq) + 0.8 * i2Prev;
            const q2 = 0.2 * (q1[i] + ji) + 0.8 * q2Prev;
            const re = 0.2 * (i2 * i2Prev + q2 * q2Prev) + 0.8 * rePrev;
            const im = 0.2 * (i2 * q2Prev - q2 * i2Prev) + 0.8 * imPrev;

            let period = periodPrev;
            if (re !== 0 && im !== 0) {
                const angle = Math.atan2(im, re);
                if (angle !== 0) {
                    period = (2 * Math.PI) / Math.abs(angle);
                }
            }
            if (periodPrev !== 0) {
                period = Math.min(period, 1.5 * periodPrev);
                period = Math.max(period, 0.67 * periodPrev);
            }
            period = Math.min(Math.max(period, 6), 50);
            period = 0.2 * period + 0.8 * periodPrev;

            const power = i1[i] * i1[i] + q1[i] * q1[i];
            const noise = range1[i] * range1[i];
            let snr = snrPrev;
            if (power > 0 && noise > 0) {
                const snrRaw = 10 * Math.log(power / noise) / Math.log(10) + 6;
                snr = 0.25 * snrRaw + 0.75 * snrPrev;
            }

            i2Prev = i2;
            q2Prev = q2;
            rePrev = re;
            imPrev = im;
            periodPrev = period;
            snrPrev = snr;
        }

        validCount += 1;
        if (validCount > 6) {
            out[i] = snrPrev;
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

test('l2_ehlers_signal_to_noise_js matches manual reference', () => {
    const n = 180;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const source = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        source[i] = (high[i] + low[i]) * 0.5;
    }
    const result = wasm.l2_ehlers_signal_to_noise_js(source, high, low, 10);
    const expected = manualReference(source, high, low, 10);
    assertArrayClose(result, expected, 1e-12, 'snr mismatch');
});

test('l2_ehlers_signal_to_noise_batch_js first row matches single', () => {
    const n = 160;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const source = new Float64Array(testData.close.slice(0, n));
    const batch = wasm.l2_ehlers_signal_to_noise_batch_js(source, high, low, {
        smooth_period_range: [10, 12, 2],
    });
    const single = wasm.l2_ehlers_signal_to_noise_js(source, high, low, 10);
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, n);
    assert.deepStrictEqual(batch.combos.map((combo) => combo.smooth_period), [10, 12]);
    assertArrayClose(batch.values.slice(0, n), single, 1e-12, 'batch first row mismatch');
});

test('l2_ehlers_signal_to_noise_into pointer path matches safe API', () => {
    const n = 160;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const source = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        source[i] = (high[i] + low[i]) * 0.5;
    }
    const safe = wasm.l2_ehlers_signal_to_noise_js(source, high, low, 10);

    const sourcePtr = wasm.allocate_f64_array(n);
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const outPtr = wasm.l2_ehlers_signal_to_noise_alloc(n);

    try {
        wasm.write_f64_array(sourcePtr, source);
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        wasm.l2_ehlers_signal_to_noise_into(sourcePtr, highPtr, lowPtr, outPtr, n, 10);
        assertArrayClose(wasm.read_f64_array(outPtr, n), safe, 1e-12, 'pointer mismatch');
    } finally {
        wasm.deallocate_f64_array(sourcePtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.l2_ehlers_signal_to_noise_free(outPtr, n);
    }
});

test('l2_ehlers_signal_to_noise_batch_into returns row count', () => {
    const n = 128;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const source = new Float64Array(testData.close.slice(0, n));
    const sourcePtr = wasm.allocate_f64_array(n);
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const rows = 2;
    const total = rows * n;
    const outPtr = wasm.l2_ehlers_signal_to_noise_alloc(total);

    try {
        wasm.write_f64_array(sourcePtr, source);
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        const returnedRows = wasm.l2_ehlers_signal_to_noise_batch_into(
            sourcePtr,
            highPtr,
            lowPtr,
            outPtr,
            n,
            10,
            12,
            2,
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.deallocate_f64_array(sourcePtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.l2_ehlers_signal_to_noise_free(outPtr, total);
    }
});

test('l2_ehlers_signal_to_noise_js rejects invalid period', () => {
    const n = 64;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const source = new Float64Array(testData.close.slice(0, n));
    assert.throws(
        () => wasm.l2_ehlers_signal_to_noise_js(source, high, low, 0),
        /Invalid smooth_period/,
    );
});
