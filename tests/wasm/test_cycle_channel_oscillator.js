import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function manualRma(values, length) {
    const out = new Float64Array(values.length);
    out.fill(NaN);
    let total = 0;
    let count = 0;
    let current = NaN;
    for (let i = 0; i < values.length; i++) {
        const value = values[i];
        if (!Number.isFinite(value)) {
            continue;
        }
        if (count < length) {
            total += value;
            count += 1;
            if (count === length) {
                current = total / length;
                out[i] = current;
            }
        } else {
            current = current + (value - current) / length;
            count += 1;
            out[i] = current;
        }
    }
    return out;
}

function manualAtr(high, low, close, length) {
    const tr = new Float64Array(close.length);
    tr.fill(NaN);
    let prevClose = null;
    for (let i = 0; i < close.length; i++) {
        if (!(Number.isFinite(high[i]) && Number.isFinite(low[i]) && Number.isFinite(close[i]))) {
            continue;
        }
        if (prevClose === null) {
            tr[i] = high[i] - low[i];
        } else {
            tr[i] = Math.max(
                high[i] - low[i],
                Math.abs(high[i] - prevClose),
                Math.abs(low[i] - prevClose),
            );
        }
        prevClose = close[i];
    }
    return manualRma(tr, length);
}

function delayedOrCurrent(values, idx, delay, current) {
    if (idx >= delay && Number.isFinite(values[idx - delay])) {
        return values[idx - delay];
    }
    return current;
}

function manualCycleChannelOscillator(source, high, low, close, shortCycleLength, mediumCycleLength, mediumMultiplier) {
    const shortPeriod = Math.floor(shortCycleLength / 2);
    const mediumPeriod = Math.floor(mediumCycleLength / 2);
    const shortDelay = Math.floor(shortPeriod / 2);
    const mediumDelay = Math.floor(mediumPeriod / 2);
    const shortMa = manualRma(source, shortPeriod);
    const mediumMa = manualRma(source, mediumPeriod);
    const mediumAtr = manualAtr(high, low, close, mediumPeriod);
    const fast = new Float64Array(source.length);
    const slow = new Float64Array(source.length);
    fast.fill(NaN);
    slow.fill(NaN);

    for (let i = 0; i < source.length; i++) {
        if (!(Number.isFinite(source[i]) && Number.isFinite(high[i]) && Number.isFinite(low[i]) && Number.isFinite(close[i]))) {
            continue;
        }
        const mediumCenter = delayedOrCurrent(mediumMa, i, mediumDelay, source[i]);
        const shortCenter = delayedOrCurrent(shortMa, i, shortDelay, source[i]);
        const offset = mediumMultiplier * mediumAtr[i];
        const denom = 2 * offset;
        if (Number.isFinite(denom) && denom !== 0) {
            const mediumBottom = mediumCenter - offset;
            fast[i] = (source[i] - mediumBottom) / denom;
            slow[i] = (shortCenter - mediumBottom) / denom;
        }
    }

    return { fast, slow };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('cycle_channel_oscillator_js matches manual reference', () => {
    const n = 160;
    const source = new Float64Array(testData.close.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const result = wasm.cycle_channel_oscillator_js(source, high, low, close, 10, 30, 1.0, 3.0);
    const expected = manualCycleChannelOscillator(source, high, low, close, 10, 30, 3.0);
    assertArrayClose(result.fast, expected.fast, 1e-12, 'fast mismatch');
    assertArrayClose(result.slow, expected.slow, 1e-12, 'slow mismatch');
});

test('cycle_channel_oscillator_batch_js first row matches single', () => {
    const n = 96;
    const source = new Float64Array(testData.close.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const batch = wasm.cycle_channel_oscillator_batch_js(source, high, low, close, {
        short_cycle_length_range: [10, 10, 0],
        medium_cycle_length_range: [30, 32, 2],
        short_multiplier_range: [1.0, 1.0, 0.0],
        medium_multiplier_range: [3.0, 3.0, 0.0],
    });
    const single = wasm.cycle_channel_oscillator_js(source, high, low, close, 10, 30, 1.0, 3.0);
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, n);
    assert.deepStrictEqual(batch.short_cycle_lengths, [10, 10]);
    assert.deepStrictEqual(batch.medium_cycle_lengths, [30, 32]);
    assertArrayClose(batch.fast.slice(0, n), single.fast, 1e-12, 'batch fast mismatch');
    assertArrayClose(batch.slow.slice(0, n), single.slow, 1e-12, 'batch slow mismatch');
});

test('cycle_channel_oscillator_into pointer path matches safe API', () => {
    const n = 128;
    const source = new Float64Array(testData.close.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const safe = wasm.cycle_channel_oscillator_js(source, high, low, close, 10, 30, 1.0, 3.0);

    const sourcePtr = wasm.allocate_f64_array(n);
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const closePtr = wasm.allocate_f64_array(n);
    const outFastPtr = wasm.cycle_channel_oscillator_alloc(n);
    const outSlowPtr = wasm.cycle_channel_oscillator_alloc(n);

    try {
        wasm.write_f64_array(sourcePtr, source);
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        wasm.write_f64_array(closePtr, close);
        wasm.cycle_channel_oscillator_into(
            sourcePtr, highPtr, lowPtr, closePtr, outFastPtr, outSlowPtr, n, 10, 30, 1.0, 3.0,
        );
        const gotFast = wasm.read_f64_array(outFastPtr, n);
        const gotSlow = wasm.read_f64_array(outSlowPtr, n);
        assertArrayClose(gotFast, safe.fast, 1e-12, 'pointer fast mismatch');
        assertArrayClose(gotSlow, safe.slow, 1e-12, 'pointer slow mismatch');
    } finally {
        wasm.cycle_channel_oscillator_free(outFastPtr, n);
        wasm.cycle_channel_oscillator_free(outSlowPtr, n);
        wasm.deallocate_f64_array(sourcePtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('cycle_channel_oscillator_batch_into returns row count', () => {
    const n = 96;
    const rows = 2;
    const source = new Float64Array(testData.close.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    const sourcePtr = wasm.allocate_f64_array(n);
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const closePtr = wasm.allocate_f64_array(n);
    const outFastPtr = wasm.cycle_channel_oscillator_alloc(rows * n);
    const outSlowPtr = wasm.cycle_channel_oscillator_alloc(rows * n);

    try {
        wasm.write_f64_array(sourcePtr, source);
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        wasm.write_f64_array(closePtr, close);
        const returnedRows = wasm.cycle_channel_oscillator_batch_into(
            sourcePtr,
            highPtr,
            lowPtr,
            closePtr,
            outFastPtr,
            outSlowPtr,
            n,
            10, 10, 0,
            30, 32, 2,
            1.0, 1.0, 0.0,
            3.0, 3.0, 0.0,
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.cycle_channel_oscillator_free(outFastPtr, rows * n);
        wasm.cycle_channel_oscillator_free(outSlowPtr, rows * n);
        wasm.deallocate_f64_array(sourcePtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('cycle_channel_oscillator_js rejects invalid length', () => {
    const n = 64;
    const source = new Float64Array(testData.close.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    assert.throws(() => {
        wasm.cycle_channel_oscillator_js(source, high, low, close, 1, 30, 1.0, 3.0);
    }, /Invalid cycle length/);
});
