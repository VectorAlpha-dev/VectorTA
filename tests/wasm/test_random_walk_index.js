import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function manualRandomWalkIndex(high, low, close, length) {
    const n = close.length;
    const outHigh = new Float64Array(n);
    const outLow = new Float64Array(n);
    outHigh.fill(NaN);
    outLow.fill(NaN);

    let first = -1;
    for (let i = 0; i < n; i++) {
        if (Number.isFinite(high[i]) && Number.isFinite(low[i]) && Number.isFinite(close[i])) {
            first = i;
            break;
        }
    }
    if (first === -1) {
        return { high: outHigh, low: outLow };
    }

    const warm = first + length - 1;
    const alpha = 1 / length;
    const sqrtLength = Math.sqrt(length);
    let prevClose = close[first];
    let sumTr = high[first] - low[first];
    let atr = NaN;

    if (length === 1) {
        atr = sumTr;
        const denom = atr * sqrtLength;
        if (Number.isFinite(denom) && denom !== 0) {
            outHigh[first] = high[first] / denom;
            outLow[first] = -low[first] / denom;
        }
    }

    for (let i = first + 1; i < n; i++) {
        const tr = Math.max(
            high[i] - low[i],
            Math.abs(high[i] - prevClose),
            Math.abs(low[i] - prevClose),
        );
        if (i <= warm) {
            sumTr += tr;
            if (i === warm) {
                atr = sumTr / length;
            }
        } else {
            atr = atr + alpha * (tr - atr);
        }

        if (i >= warm) {
            const histLow = i >= length && Number.isFinite(low[i - length]) ? low[i - length] : 0;
            const histHigh = i >= length && Number.isFinite(high[i - length]) ? high[i - length] : 0;
            const denom = atr * sqrtLength;
            if (Number.isFinite(denom) && denom !== 0) {
                outHigh[i] = (high[i] - histLow) / denom;
                outLow[i] = (histHigh - low[i]) / denom;
            }
        }
        prevClose = close[i];
    }

    return { high: outHigh, low: outLow };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('random_walk_index_js matches manual reference', () => {
    const n = 160;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const result = wasm.random_walk_index_js(high, low, close, 14);
    const expected = manualRandomWalkIndex(high, low, close, 14);
    assertArrayClose(result.high, expected.high, 1e-12, 'high mismatch');
    assertArrayClose(result.low, expected.low, 1e-12, 'low mismatch');
});

test('random_walk_index_batch_js first row matches single', () => {
    const n = 96;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const batch = wasm.random_walk_index_batch_js(high, low, close, {
        length_range: [14, 16, 2],
    });
    const single = wasm.random_walk_index_js(high, low, close, 14);
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, n);
    assert.deepStrictEqual(batch.lengths, [14, 16]);
    assertArrayClose(batch.high.slice(0, n), single.high, 1e-12, 'batch high mismatch');
    assertArrayClose(batch.low.slice(0, n), single.low, 1e-12, 'batch low mismatch');
});

test('random_walk_index_into pointer path matches safe API', () => {
    const n = 128;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const safe = wasm.random_walk_index_js(high, low, close, 14);

    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const closePtr = wasm.allocate_f64_array(n);
    const outHighPtr = wasm.random_walk_index_alloc(n);
    const outLowPtr = wasm.random_walk_index_alloc(n);

    try {
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        wasm.write_f64_array(closePtr, close);
        wasm.random_walk_index_into(highPtr, lowPtr, closePtr, outHighPtr, outLowPtr, n, 14);
        const gotHigh = wasm.read_f64_array(outHighPtr, n);
        const gotLow = wasm.read_f64_array(outLowPtr, n);
        assertArrayClose(gotHigh, safe.high, 1e-12, 'pointer high mismatch');
        assertArrayClose(gotLow, safe.low, 1e-12, 'pointer low mismatch');
    } finally {
        wasm.random_walk_index_free(outHighPtr, n);
        wasm.random_walk_index_free(outLowPtr, n);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('random_walk_index_batch_into returns row count', () => {
    const n = 96;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const closePtr = wasm.allocate_f64_array(n);
    const rows = 2;
    const outHighPtr = wasm.random_walk_index_alloc(rows * n);
    const outLowPtr = wasm.random_walk_index_alloc(rows * n);

    try {
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        wasm.write_f64_array(closePtr, close);
        const returnedRows = wasm.random_walk_index_batch_into(
            highPtr,
            lowPtr,
            closePtr,
            outHighPtr,
            outLowPtr,
            n,
            14,
            16,
            2,
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.random_walk_index_free(outHighPtr, rows * n);
        wasm.random_walk_index_free(outLowPtr, rows * n);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('random_walk_index_js rejects invalid length', () => {
    const n = 64;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    assert.throws(() => {
        wasm.random_walk_index_js(high, low, close, 0);
    }, /Invalid length/);
});
