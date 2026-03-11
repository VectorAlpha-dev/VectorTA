import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function manualVqi(open, high, low, close, fastLength, slowLength) {
    const n = close.length;
    const vqiSum = new Float64Array(n);
    const fastSma = new Float64Array(n);
    const slowSma = new Float64Array(n);
    fastSma.fill(NaN);
    slowSma.fill(NaN);

    let prevClose = NaN;
    let prevVqiT = 0.0;
    let cumulative = 0.0;

    for (let i = 0; i < n; i++) {
        const rng = high[i] - low[i];
        let tr = NaN;
        if (Number.isFinite(high[i]) && Number.isFinite(low[i])) {
            if (Number.isFinite(prevClose)) {
                tr = Math.max(Math.abs(rng), Math.abs(high[i] - prevClose), Math.abs(low[i] - prevClose));
            } else {
                tr = Math.abs(rng);
            }
        }

        let vqiT = prevVqiT;
        if (
            Number.isFinite(prevClose) &&
            Number.isFinite(open[i]) &&
            Number.isFinite(high[i]) &&
            Number.isFinite(low[i]) &&
            Number.isFinite(close[i]) &&
            Number.isFinite(tr) &&
            tr !== 0 &&
            Number.isFinite(rng) &&
            rng !== 0
        ) {
            vqiT = 0.5 * (((close[i] - prevClose) / tr) + ((close[i] - open[i]) / rng));
        }

        const raw = Number.isFinite(prevClose) && Number.isFinite(open[i]) && Number.isFinite(close[i])
            ? Math.abs(vqiT) * 0.5 * ((close[i] - prevClose) + (close[i] - open[i]))
            : 0.0;

        cumulative += raw;
        vqiSum[i] = cumulative;
        prevVqiT = vqiT;
        if (Number.isFinite(close[i])) {
            prevClose = close[i];
        }
    }

    for (let i = fastLength - 1; i < n; i++) {
        let sum = 0;
        for (let j = i - fastLength + 1; j <= i; j++) {
            sum += vqiSum[j];
        }
        fastSma[i] = sum / fastLength;
    }

    for (let i = slowLength - 1; i < n; i++) {
        let sum = 0;
        for (let j = i - slowLength + 1; j <= i; j++) {
            sum += vqiSum[j];
        }
        slowSma[i] = sum / slowLength;
    }

    return { vqiSum, fastSma, slowSma };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('volatility_quality_index_js matches manual reference', () => {
    const n = 256;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    const result = wasm.volatility_quality_index_js(open, high, low, close, 9, 21);
    const expected = manualVqi(open, high, low, close, 9, 21);

    assertArrayClose(result.vqi_sum, expected.vqiSum, 5e-12, 'vqi_sum mismatch');
    assertArrayClose(result.fast_sma, expected.fastSma, 5e-12, 'fast_sma mismatch');
    assertArrayClose(result.slow_sma, expected.slowSma, 5e-12, 'slow_sma mismatch');
    assert.strictEqual(result.vqi_sum.length, n);
    assert.ok(isNaN(result.fast_sma[0]));
    assert.ok(isNaN(result.slow_sma[0]));
});

test('volatility_quality_index_batch_js first row matches single', () => {
    const n = 128;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    const batch = wasm.volatility_quality_index_batch_js(open, high, low, close, {
        fast_length_range: [9, 11, 2],
        slow_length_range: [20, 20, 0],
    });
    const single = wasm.volatility_quality_index_js(open, high, low, close, 9, 20);

    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, n);
    assert.strictEqual(batch.combos.length, 2);
    assert.strictEqual(batch.combos[0].fast_length, 9);
    assert.strictEqual(batch.combos[0].slow_length, 20);
    assertArrayClose(batch.vqi_sum.slice(0, n), single.vqi_sum, 5e-12, 'batch vqi_sum mismatch');
    assertArrayClose(batch.fast_sma.slice(0, n), single.fast_sma, 5e-12, 'batch fast_sma mismatch');
    assertArrayClose(batch.slow_sma.slice(0, n), single.slow_sma, 5e-12, 'batch slow_sma mismatch');
});

test('volatility_quality_index_into pointer path matches safe API', () => {
    const n = 128;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const safe = wasm.volatility_quality_index_js(open, high, low, close, 9, 21);

    const openPtr = wasm.allocate_f64_array(n);
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const closePtr = wasm.allocate_f64_array(n);
    const outPtr = wasm.volatility_quality_index_alloc(3 * n);

    try {
        wasm.write_f64_array(openPtr, open);
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        wasm.write_f64_array(closePtr, close);
        wasm.volatility_quality_index_into(openPtr, highPtr, lowPtr, closePtr, outPtr, n, 9, 21);
        const flat = wasm.read_f64_array(outPtr, 3 * n);
        assertArrayClose(flat.slice(0, n), safe.vqi_sum, 5e-12, 'pointer vqi_sum mismatch');
        assertArrayClose(flat.slice(n, 2 * n), safe.fast_sma, 5e-12, 'pointer fast_sma mismatch');
        assertArrayClose(flat.slice(2 * n, 3 * n), safe.slow_sma, 5e-12, 'pointer slow_sma mismatch');
    } finally {
        wasm.volatility_quality_index_free(outPtr, 3 * n);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('volatility_quality_index_batch_into returns row count', () => {
    const n = 96;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    const openPtr = wasm.allocate_f64_array(n);
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const closePtr = wasm.allocate_f64_array(n);
    const rows = 2;
    const vqiPtr = wasm.volatility_quality_index_alloc(rows * n);
    const fastPtr = wasm.volatility_quality_index_alloc(rows * n);
    const slowPtr = wasm.volatility_quality_index_alloc(rows * n);

    try {
        wasm.write_f64_array(openPtr, open);
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        wasm.write_f64_array(closePtr, close);
        const returnedRows = wasm.volatility_quality_index_batch_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            vqiPtr,
            fastPtr,
            slowPtr,
            n,
            9,
            11,
            2,
            20,
            20,
            0,
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.volatility_quality_index_free(vqiPtr, rows * n);
        wasm.volatility_quality_index_free(fastPtr, rows * n);
        wasm.volatility_quality_index_free(slowPtr, rows * n);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('volatility_quality_index_js rejects invalid length', () => {
    const n = 64;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    assert.throws(() => {
        wasm.volatility_quality_index_js(open, high, low, close, 0, 21);
    }, /Invalid fast_length/);
});

test('volatility_quality_index_batch_js rejects invalid config', () => {
    const n = 64;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    assert.throws(() => {
        wasm.volatility_quality_index_batch_js(open, high, low, close, {
            fast_length_range: [9, 11],
            slow_length_range: [20, 20, 0],
        });
    }, /Invalid config/);
});
