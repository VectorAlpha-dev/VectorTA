import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function averageRanks(values) {
    const order = Array.from(values.keys()).sort((a, b) => values[a] - values[b]);
    const ranks = new Float64Array(values.length);
    let start = 0;
    while (start < values.length) {
        const current = values[order[start]];
        let end = start + 1;
        while (end < values.length && values[order[end]] === current) {
            end += 1;
        }
        const avgRank = (start + 1 + end) * 0.5;
        for (let i = start; i < end; i++) {
            ranks[order[i]] = avgRank;
        }
        start = end;
    }
    return ranks;
}

function pearsonCorrelation(x, y) {
    const n = x.length;
    let sumX = 0;
    let sumY = 0;
    for (let i = 0; i < n; i++) {
        sumX += x[i];
        sumY += y[i];
    }
    const meanX = sumX / n;
    const meanY = sumY / n;
    let cov = 0;
    let varX = 0;
    let varY = 0;
    for (let i = 0; i < n; i++) {
        const dx = x[i] - meanX;
        const dy = y[i] - meanY;
        cov += dx * dy;
        varX += dx * dx;
        varY += dy * dy;
    }
    const denom = Math.sqrt(varX * varY);
    return denom === 0 || !Number.isFinite(denom) ? NaN : cov / denom;
}

function manualReference(main, compare, lookback = 30, smoothingLength = 3) {
    const n = main.length;
    const raw = new Float64Array(n);
    const smoothed = new Float64Array(n);
    raw.fill(NaN);
    smoothed.fill(NaN);
    const mainReturns = new Float64Array(n);
    const compareReturns = new Float64Array(n);
    mainReturns.fill(NaN);
    compareReturns.fill(NaN);

    for (let i = 1; i < n; i++) {
        if (
            Number.isFinite(main[i - 1]) &&
            Number.isFinite(main[i]) &&
            Number.isFinite(compare[i - 1]) &&
            Number.isFinite(compare[i])
        ) {
            mainReturns[i] = main[i] - main[i - 1];
            compareReturns[i] = compare[i] - compare[i - 1];
        }
    }

    for (let i = lookback; i < n; i++) {
        const mainWindow = mainReturns.slice(i + 1 - lookback, i + 1);
        const compareWindow = compareReturns.slice(i + 1 - lookback, i + 1);
        if (![...mainWindow].every(Number.isFinite) || ![...compareWindow].every(Number.isFinite)) {
            continue;
        }
        raw[i] = pearsonCorrelation(averageRanks(mainWindow), averageRanks(compareWindow));
    }

    for (let i = smoothingLength - 1; i < n; i++) {
        let sum = 0;
        let valid = true;
        for (let j = i + 1 - smoothingLength; j <= i; j++) {
            if (!Number.isFinite(raw[j])) {
                valid = false;
                break;
            }
            sum += raw[j];
        }
        if (valid) {
            smoothed[i] = sum / smoothingLength;
        }
    }

    return { raw, smoothed };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('spearman_correlation_js matches manual reference', () => {
    const main = new Float64Array(testData.close.slice(0, 180));
    const compare = new Float64Array(testData.open.slice(0, 180));
    const result = wasm.spearman_correlation_js(main, compare, 30, 3);
    const expected = manualReference(main, compare, 30, 3);
    assertArrayClose(result.raw, expected.raw, 1e-12, 'raw mismatch');
    assertArrayClose(result.smoothed, expected.smoothed, 1e-12, 'smoothed mismatch');
});

test('spearman_correlation_batch_js first row matches single', () => {
    const main = new Float64Array(testData.close.slice(0, 160));
    const compare = new Float64Array(testData.open.slice(0, 160));
    const batch = wasm.spearman_correlation_batch_js(main, compare, {
        lookback_range: [30, 32, 2],
        smoothing_length_range: [3, 3, 0],
    });
    const single = wasm.spearman_correlation_js(main, compare, 30, 3);
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, main.length);
    assert.deepStrictEqual(batch.lookback, [30, 32]);
    assertArrayClose(batch.raw.slice(0, main.length), single.raw, 1e-12, 'batch raw mismatch');
    assertArrayClose(
        batch.smoothed.slice(0, main.length),
        single.smoothed,
        1e-12,
        'batch smoothed mismatch',
    );
});

test('spearman_correlation_into pointer path matches safe API', () => {
    const n = 160;
    const main = new Float64Array(testData.close.slice(0, n));
    const compare = new Float64Array(testData.open.slice(0, n));
    const safe = wasm.spearman_correlation_js(main, compare, 30, 3);

    const mainPtr = wasm.allocate_f64_array(n);
    const comparePtr = wasm.allocate_f64_array(n);
    const rawPtr = wasm.spearman_correlation_alloc(n);
    const smoothedPtr = wasm.spearman_correlation_alloc(n);

    try {
        wasm.write_f64_array(mainPtr, main);
        wasm.write_f64_array(comparePtr, compare);
        wasm.spearman_correlation_into(mainPtr, comparePtr, rawPtr, smoothedPtr, n, 30, 3);
        assertArrayClose(wasm.read_f64_array(rawPtr, n), safe.raw, 1e-12, 'pointer raw mismatch');
        assertArrayClose(
            wasm.read_f64_array(smoothedPtr, n),
            safe.smoothed,
            1e-12,
            'pointer smoothed mismatch',
        );
    } finally {
        wasm.spearman_correlation_free(rawPtr, n);
        wasm.spearman_correlation_free(smoothedPtr, n);
        wasm.deallocate_f64_array(mainPtr);
        wasm.deallocate_f64_array(comparePtr);
    }
});

test('spearman_correlation_batch_into returns row count', () => {
    const n = 128;
    const rows = 2;
    const main = new Float64Array(testData.close.slice(0, n));
    const compare = new Float64Array(testData.open.slice(0, n));
    const mainPtr = wasm.allocate_f64_array(n);
    const comparePtr = wasm.allocate_f64_array(n);
    const rawPtr = wasm.spearman_correlation_alloc(rows * n);
    const smoothedPtr = wasm.spearman_correlation_alloc(rows * n);

    try {
        wasm.write_f64_array(mainPtr, main);
        wasm.write_f64_array(comparePtr, compare);
        const returnedRows = wasm.spearman_correlation_batch_into(
            mainPtr,
            comparePtr,
            rawPtr,
            smoothedPtr,
            n,
            30,
            32,
            2,
            3,
            3,
            0,
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.spearman_correlation_free(rawPtr, rows * n);
        wasm.spearman_correlation_free(smoothedPtr, rows * n);
        wasm.deallocate_f64_array(mainPtr);
        wasm.deallocate_f64_array(comparePtr);
    }
});

test('spearman_correlation_js rejects invalid lookback', () => {
    const main = new Float64Array(testData.close.slice(0, 32));
    const compare = new Float64Array(testData.open.slice(0, 32));
    assert.throws(() => {
        wasm.spearman_correlation_js(main, compare, 0, 3);
    }, /Invalid lookback/);
});
