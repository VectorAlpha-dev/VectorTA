import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function manualTtf(high, low, length) {
    const out = new Float64Array(high.length);
    out.fill(NaN);

    let first = -1;
    for (let i = 0; i < high.length; i++) {
        if (Number.isFinite(high[i]) && Number.isFinite(low[i])) {
            first = i;
            break;
        }
    }
    if (first === -1 || high.length - first < length) {
        return out;
    }

    const warm = first + length - 1;
    const hhSeries = new Float64Array(high.length);
    const llSeries = new Float64Array(low.length);
    hhSeries.fill(NaN);
    llSeries.fill(NaN);

    for (let i = warm; i < high.length; i++) {
        const start = i + 1 - length;
        let hh = -Infinity;
        let ll = Infinity;
        for (let j = start; j <= i; j++) {
            hh = Math.max(hh, high[j]);
            ll = Math.min(ll, low[j]);
        }
        hhSeries[i] = hh;
        llSeries[i] = ll;
        const histHh = i >= length && Number.isFinite(hhSeries[i - length]) ? hhSeries[i - length] : 0.0;
        const histLl = i >= length && Number.isFinite(llSeries[i - length]) ? llSeries[i - length] : 0.0;
        const buyPower = hh - histLl;
        const sellPower = histHh - ll;
        const denom = buyPower + sellPower;
        if (Number.isFinite(denom) && denom !== 0) {
            out[i] = 200 * (buyPower - sellPower) / denom;
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

test('trend_trigger_factor_js matches manual reference', () => {
    const n = 160;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const result = wasm.trend_trigger_factor_js(high, low, 15);
    const expected = manualTtf(high, low, 15);
    assertArrayClose(result, expected, 1e-12, 'manual mismatch');
});

test('trend_trigger_factor_batch_js first row matches single', () => {
    const n = 128;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const batch = wasm.trend_trigger_factor_batch_js(high, low, {
        length_range: [15, 17, 2],
    });
    const single = wasm.trend_trigger_factor_js(high, low, 15);
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, n);
    assert.deepStrictEqual(batch.lengths, [15, 17]);
    assertArrayClose(batch.values.slice(0, n), single, 1e-12, 'batch mismatch');
});

test('trend_trigger_factor_into pointer path matches safe API', () => {
    const n = 128;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const safe = wasm.trend_trigger_factor_js(high, low, 15);

    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const outPtr = wasm.trend_trigger_factor_alloc(n);

    try {
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        wasm.trend_trigger_factor_into(highPtr, lowPtr, outPtr, n, 15);
        const got = wasm.read_f64_array(outPtr, n);
        assertArrayClose(got, safe, 1e-12, 'pointer mismatch');
    } finally {
        wasm.trend_trigger_factor_free(outPtr, n);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
    }
});

test('trend_trigger_factor_batch_into returns row count', () => {
    const n = 96;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const rows = 2;
    const outPtr = wasm.trend_trigger_factor_alloc(rows * n);

    try {
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        const returnedRows = wasm.trend_trigger_factor_batch_into(
            highPtr,
            lowPtr,
            outPtr,
            n,
            15,
            17,
            2,
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.trend_trigger_factor_free(outPtr, rows * n);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
    }
});

test('trend_trigger_factor_js rejects invalid length', () => {
    const n = 64;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    assert.throws(() => {
        wasm.trend_trigger_factor_js(high, low, 0);
    }, /Invalid length/);
});

test('trend_trigger_factor_batch_js rejects invalid config', () => {
    const n = 64;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    assert.throws(() => {
        wasm.trend_trigger_factor_batch_js(high, low, {
            length_range: [15, 17],
        });
    }, /Invalid config/);
});
