import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const MAX_LOOKBACK = 75;

let wasm;
let testData;

function nz(value) {
    return Number.isFinite(value) ? value : 0;
}

function ringGet(buf, center, off) {
    const n = buf.length;
    let idx = center + n - (off % n);
    if (idx >= n) {
        idx -= n;
    }
    return buf[idx];
}

function median3(a, b, c) {
    if (!(Number.isFinite(a) && Number.isFinite(b) && Number.isFinite(c))) {
        return NaN;
    }
    return (a + b + c) - Math.min(a, b, c) - Math.max(a, b, c);
}

function manualReference(data, alpha = 0.07, cutoff = 8.0) {
    const out = new Float64Array(data.length);
    out.fill(NaN);
    const srcRing = new Float64Array(MAX_LOOKBACK + 1);
    const smoothRing = new Float64Array(3);
    const cRing = new Float64Array(7);
    const dpRing = new Float64Array(5);
    srcRing.fill(NaN);
    smoothRing.fill(NaN);
    cRing.fill(NaN);
    dpRing.fill(NaN);

    let srcIdx = 0;
    let smoothIdx = 0;
    let cIdx = 0;
    let dpIdx = 0;
    let prevIp = NaN;
    let prevP = NaN;
    let prevQ1 = NaN;
    let prevI1 = NaN;
    const f3Hist = [NaN, NaN, NaN];
    let validCount = 0;

    const coefC = (1 - 0.5 * alpha) * (1 - 0.5 * alpha);
    const oneMinusAlpha = 1 - alpha;
    const a1 = Math.exp(-Math.PI / cutoff);
    const b1 = 2 * a1 * Math.cos(1.738 * Math.PI / cutoff);
    const c1 = a1 * a1;
    const coef2 = b1 + c1;
    const coef3 = -(c1 + b1 * c1);
    const coef4 = c1 * c1;
    const coef1 = 1 - coef2 - coef3 - coef4;

    for (let i = 0; i < data.length; i++) {
        const source = data[i];
        if (!Number.isFinite(source)) {
            continue;
        }

        srcRing[srcIdx] = source;
        const src0 = ringGet(srcRing, srcIdx, 0);
        const src1 = ringGet(srcRing, srcIdx, 1);
        const src2 = ringGet(srcRing, srcIdx, 2);
        const src3 = ringGet(srcRing, srcIdx, 3);

        const smooth = Number.isFinite(src0) && Number.isFinite(src1) && Number.isFinite(src2) && Number.isFinite(src3)
            ? (src0 + 2 * src1 + 2 * src2 + src3) / 6
            : NaN;
        smoothRing[smoothIdx] = smooth;

        const smooth1 = nz(ringGet(smoothRing, smoothIdx, 1));
        const smooth2 = nz(ringGet(smoothRing, smoothIdx, 2));
        const cPrev1 = nz(ringGet(cRing, cIdx, 1));
        const cPrev2 = nz(ringGet(cRing, cIdx, 2));
        const cMain = Number.isFinite(smooth)
            ? coefC * (smooth - 2 * smooth1 + smooth2)
                + 2 * oneMinusAlpha * cPrev1
                - oneMinusAlpha * oneMinusAlpha * cPrev2
            : NaN;
        const cFallback = Number.isFinite(src0) && Number.isFinite(src1) && Number.isFinite(src2)
            ? (src0 - 2 * src1 + src2) / 4
            : NaN;
        const cValue = Number.isFinite(cMain) ? cMain : cFallback;
        cRing[cIdx] = cValue;

        const q1 = Number.isFinite(cValue)
            ? (
                0.0962 * cValue
                + 0.5769 * nz(ringGet(cRing, cIdx, 2))
                - 0.5769 * nz(ringGet(cRing, cIdx, 4))
                - 0.0962 * nz(ringGet(cRing, cIdx, 6))
            ) * (0.5 + 0.08 * nz(prevIp))
            : NaN;
        const i1 = nz(ringGet(cRing, cIdx, 3));

        let dpRaw = 0;
        if (Number.isFinite(q1) && Number.isFinite(prevQ1) && q1 !== 0 && prevQ1 !== 0) {
            const prevI1Nz = nz(prevI1);
            const prevQ1Nz = nz(prevQ1);
            const numer = (i1 / q1) - (prevI1Nz / prevQ1Nz);
            const denom = 1 + i1 * prevI1Nz / (q1 * prevQ1Nz);
            dpRaw = numer / denom;
        }
        const dp = dpRaw < 0.1 ? 0.1 : dpRaw > 1.1 ? 1.1 : dpRaw;
        dpRing[dpIdx] = dp;

        const mdInner = median3(
            ringGet(dpRing, dpIdx, 2),
            ringGet(dpRing, dpIdx, 3),
            ringGet(dpRing, dpIdx, 4),
        );
        const md = median3(dp, ringGet(dpRing, dpIdx, 1), mdInner);
        const dc = md === 0 ? 15 : (2 * Math.PI / md) + 0.5;
        const ip = 0.33 * dc + 0.67 * nz(prevIp);
        const p = 0.15 * ip + 0.85 * nz(prevP);

        const pr = Number.isFinite(p) ? Math.round(Math.abs(p - 1)) : NaN;
        let v1 = 0;
        if (Number.isFinite(pr)) {
            if (pr >= 1 && pr <= MAX_LOOKBACK) {
                const past = ringGet(srcRing, srcIdx, pr);
                v1 = Number.isFinite(past) ? source - past : NaN;
            }
        }

        const rawF3 = Number.isFinite(v1)
            ? coef1 * v1 + coef2 * nz(f3Hist[0]) + coef3 * nz(f3Hist[1]) + coef4 * nz(f3Hist[2])
            : NaN;
        const f3 = Number.isFinite(rawF3) ? rawF3 : v1;

        prevQ1 = q1;
        prevI1 = i1;
        prevIp = ip;
        prevP = p;
        f3Hist[2] = f3Hist[1];
        f3Hist[1] = f3Hist[0];
        f3Hist[0] = f3;

        validCount += 1;
        srcIdx = (srcIdx + 1) % srcRing.length;
        smoothIdx = (smoothIdx + 1) % smoothRing.length;
        cIdx = (cIdx + 1) % cRing.length;
        dpIdx = (dpIdx + 1) % dpRing.length;

        if (validCount > MAX_LOOKBACK) {
            out[i] = f3;
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

test('ehlers_smoothed_adaptive_momentum_js matches manual reference', () => {
    const n = 220;
    const source = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        source[i] = (testData.high[i] + testData.low[i]) * 0.5;
    }
    const result = wasm.ehlers_smoothed_adaptive_momentum_js(source, 0.07, 8.0);
    const expected = manualReference(source, 0.07, 8.0);
    assertArrayClose(result, expected, 1e-12, 'esam mismatch');
});

test('ehlers_smoothed_adaptive_momentum_batch_js first row matches single', () => {
    const n = 180;
    const source = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        source[i] = (testData.high[i] + testData.low[i]) * 0.5;
    }
    const batch = wasm.ehlers_smoothed_adaptive_momentum_batch_js(source, {
        alpha_range: [0.07, 0.09, 0.02],
        cutoff_range: [8.0, 8.0, 0.0],
    });
    const single = wasm.ehlers_smoothed_adaptive_momentum_js(source, 0.07, 8.0);
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, n);
    assertArrayClose(batch.values.slice(0, n), single, 1e-12, 'batch first row mismatch');
});

test('ehlers_smoothed_adaptive_momentum_into pointer path matches safe API', () => {
    const n = 180;
    const source = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        source[i] = (testData.high[i] + testData.low[i]) * 0.5;
    }
    const safe = wasm.ehlers_smoothed_adaptive_momentum_js(source, 0.07, 8.0);
    const inPtr = wasm.allocate_f64_array(n);
    const outPtr = wasm.ehlers_smoothed_adaptive_momentum_alloc(n);
    try {
        wasm.write_f64_array(inPtr, source);
        wasm.ehlers_smoothed_adaptive_momentum_into(inPtr, outPtr, n, 0.07, 8.0);
        assertArrayClose(wasm.read_f64_array(outPtr, n), safe, 1e-12, 'pointer mismatch');
    } finally {
        wasm.deallocate_f64_array(inPtr);
        wasm.ehlers_smoothed_adaptive_momentum_free(outPtr, n);
    }
});

test('ehlers_smoothed_adaptive_momentum_batch_into returns row count', () => {
    const n = 180;
    const source = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        source[i] = (testData.high[i] + testData.low[i]) * 0.5;
    }
    const inPtr = wasm.allocate_f64_array(n);
    const rows = 2;
    const outPtr = wasm.ehlers_smoothed_adaptive_momentum_alloc(rows * n);
    try {
        wasm.write_f64_array(inPtr, source);
        const returnedRows = wasm.ehlers_smoothed_adaptive_momentum_batch_into(
            inPtr,
            outPtr,
            n,
            0.07, 0.09, 0.02,
            8.0, 8.0, 0.0,
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.deallocate_f64_array(inPtr);
        wasm.ehlers_smoothed_adaptive_momentum_free(outPtr, rows * n);
    }
});

test('ehlers_smoothed_adaptive_momentum_js rejects invalid cutoff', () => {
    const source = new Float64Array(testData.close.slice(0, 128));
    assert.throws(
        () => wasm.ehlers_smoothed_adaptive_momentum_js(source, 0.07, 0.0),
        /Invalid cutoff/,
    );
});
