import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function manualVwma(price, volume, length) {
    const out = new Float64Array(price.length);
    out.fill(NaN);
    for (let i = length - 1; i < price.length; i++) {
        let weighted = 0;
        let total = 0;
        let valid = true;
        for (let j = i + 1 - length; j <= i; j++) {
            if (!Number.isFinite(price[j]) || !Number.isFinite(volume[j])) {
                valid = false;
                break;
            }
            weighted += price[j] * volume[j];
            total += volume[j];
        }
        if (valid && total !== 0) {
            out[i] = weighted / total;
        }
    }
    return out;
}

function manualSma(values, length) {
    const out = new Float64Array(values.length);
    out.fill(NaN);
    for (let i = length - 1; i < values.length; i++) {
        let sum = 0;
        let valid = true;
        for (let j = i + 1 - length; j <= i; j++) {
            if (!Number.isFinite(values[j])) {
                valid = false;
                break;
            }
            sum += values[j];
        }
        if (valid) {
            out[i] = sum / length;
        }
    }
    return out;
}

function manualPmarp(price, volume, maLength = 20, lookback = 60, signalLength = 10) {
    const ma = manualVwma(price, volume, maLength);
    const n = price.length;
    const pmar = new Float64Array(n);
    const pmarp = new Float64Array(n);
    const pmarHigh = new Float64Array(n);
    const pmarLow = new Float64Array(n);
    const scaledPmar = new Float64Array(n);
    pmar.fill(NaN);
    pmarp.fill(NaN);
    pmarHigh.fill(NaN);
    pmarLow.fill(NaN);
    scaledPmar.fill(NaN);

    let hi = 1.0;
    let lo = 1.0;
    let seen = false;
    for (let i = 0; i < n; i++) {
        if (Number.isFinite(price[i]) && Number.isFinite(ma[i]) && ma[i] !== 0) {
            pmar[i] = price[i] / ma[i];
            hi = Math.max(hi, pmar[i]);
            lo = Math.min(lo, pmar[i]);
            seen = true;
        }
        if (seen) {
            pmarHigh[i] = hi;
            pmarLow[i] = lo;
            if (Number.isFinite(pmar[i])) {
                if (pmar[i] >= 1.0) {
                    scaledPmar[i] = hi === 1.0 ? 50.0 : (((pmar[i] - 1.0) * (100.0 / (hi - 1.0))) / 2.0) + 50.0;
                } else {
                    scaledPmar[i] = lo === 1.0 ? 50.0 : ((pmar[i] - lo) * (100.0 / (1.0 - lo))) / 2.0;
                }
            }
        }
    }

    for (let i = maLength; i < n; i++) {
        if (!Number.isFinite(pmar[i])) {
            continue;
        }
        const current = Math.abs(pmar[i]);
        const win = Math.min(i, lookback);
        let count = 0;
        for (let offset = 1; offset <= win; offset++) {
            const prev = Math.abs(pmar[i - offset]);
            if (!(Number.isFinite(prev) && prev > current)) {
                count += 1;
            }
        }
        pmarp[i] = (count / win) * 100.0;
    }

    const plotline = pmarp;
    const signal = manualSma(plotline, signalLength);
    return { pmar, pmarp, plotline, signal, pmarHigh, pmarLow, scaledPmar };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('price_moving_average_ratio_percentile_js matches manual reference', () => {
    const n = 160;
    const price = new Float64Array(testData.close.slice(0, n));
    const volume = new Float64Array(testData.volume.slice(0, n));
    const result = wasm.price_moving_average_ratio_percentile_js(
        price,
        volume,
        20,
        'vwma',
        60,
        10,
        'sma',
        'pmarp',
    );
    const expected = manualPmarp(price, volume, 20, 60, 10);
    assertArrayClose(result.pmar, expected.pmar, 5e-12, 'pmar mismatch');
    assertArrayClose(result.pmarp, expected.pmarp, 5e-12, 'pmarp mismatch');
    assertArrayClose(result.plotline, expected.plotline, 5e-12, 'plotline mismatch');
    assertArrayClose(result.signal, expected.signal, 5e-12, 'signal mismatch');
    assertArrayClose(result.pmar_high, expected.pmarHigh, 5e-12, 'pmar_high mismatch');
    assertArrayClose(result.pmar_low, expected.pmarLow, 5e-12, 'pmar_low mismatch');
    assertArrayClose(result.scaled_pmar, expected.scaledPmar, 5e-12, 'scaled_pmar mismatch');
});

test('price_moving_average_ratio_percentile_batch_js first row matches single', () => {
    const n = 128;
    const price = new Float64Array(testData.close.slice(0, n));
    const volume = new Float64Array(testData.volume.slice(0, n));
    const batch = wasm.price_moving_average_ratio_percentile_batch_js(price, volume, {
        ma_length_range: [20, 22, 2],
        pmarp_lookback_range: [60, 60, 0],
        signal_ma_length_range: [10, 10, 0],
        ma_type: 'vwma',
        signal_ma_type: 'sma',
        line_mode: 'pmarp',
    });
    const single = wasm.price_moving_average_ratio_percentile_js(
        price,
        volume,
        20,
        'vwma',
        60,
        10,
        'sma',
        'pmarp',
    );
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, n);
    assert.deepStrictEqual(batch.ma_lengths, [20, 22]);
    assertArrayClose(batch.plotline.slice(0, n), single.plotline, 1e-12, 'batch plotline mismatch');
    assertArrayClose(batch.signal.slice(0, n), single.signal, 1e-12, 'batch signal mismatch');
});

test('price_moving_average_ratio_percentile_into pointer path matches safe API', () => {
    const n = 128;
    const price = new Float64Array(testData.close.slice(0, n));
    const volume = new Float64Array(testData.volume.slice(0, n));
    const safe = wasm.price_moving_average_ratio_percentile_js(
        price,
        volume,
        20,
        'vwma',
        60,
        10,
        'sma',
        'pmarp',
    );

    const pricePtr = wasm.allocate_f64_array(n);
    const volumePtr = wasm.allocate_f64_array(n);
    const pmarPtr = wasm.price_moving_average_ratio_percentile_alloc(n);
    const pmarpPtr = wasm.price_moving_average_ratio_percentile_alloc(n);
    const plotlinePtr = wasm.price_moving_average_ratio_percentile_alloc(n);
    const signalPtr = wasm.price_moving_average_ratio_percentile_alloc(n);
    const pmarHighPtr = wasm.price_moving_average_ratio_percentile_alloc(n);
    const pmarLowPtr = wasm.price_moving_average_ratio_percentile_alloc(n);
    const scaledPmarPtr = wasm.price_moving_average_ratio_percentile_alloc(n);

    try {
        wasm.write_f64_array(pricePtr, price);
        wasm.write_f64_array(volumePtr, volume);
        wasm.price_moving_average_ratio_percentile_into(
            pricePtr,
            volumePtr,
            pmarPtr,
            pmarpPtr,
            plotlinePtr,
            signalPtr,
            pmarHighPtr,
            pmarLowPtr,
            scaledPmarPtr,
            n,
            20,
            'vwma',
            60,
            10,
            'sma',
            'pmarp',
        );
        assertArrayClose(wasm.read_f64_array(pmarPtr, n), safe.pmar, 1e-12, 'pointer pmar mismatch');
        assertArrayClose(wasm.read_f64_array(pmarpPtr, n), safe.pmarp, 1e-12, 'pointer pmarp mismatch');
        assertArrayClose(wasm.read_f64_array(plotlinePtr, n), safe.plotline, 1e-12, 'pointer plotline mismatch');
        assertArrayClose(wasm.read_f64_array(signalPtr, n), safe.signal, 1e-12, 'pointer signal mismatch');
        assertArrayClose(wasm.read_f64_array(pmarHighPtr, n), safe.pmar_high, 1e-12, 'pointer pmar_high mismatch');
        assertArrayClose(wasm.read_f64_array(pmarLowPtr, n), safe.pmar_low, 1e-12, 'pointer pmar_low mismatch');
        assertArrayClose(wasm.read_f64_array(scaledPmarPtr, n), safe.scaled_pmar, 1e-12, 'pointer scaled_pmar mismatch');
    } finally {
        wasm.price_moving_average_ratio_percentile_free(pmarPtr, n);
        wasm.price_moving_average_ratio_percentile_free(pmarpPtr, n);
        wasm.price_moving_average_ratio_percentile_free(plotlinePtr, n);
        wasm.price_moving_average_ratio_percentile_free(signalPtr, n);
        wasm.price_moving_average_ratio_percentile_free(pmarHighPtr, n);
        wasm.price_moving_average_ratio_percentile_free(pmarLowPtr, n);
        wasm.price_moving_average_ratio_percentile_free(scaledPmarPtr, n);
        wasm.deallocate_f64_array(pricePtr);
        wasm.deallocate_f64_array(volumePtr);
    }
});

test('price_moving_average_ratio_percentile_batch_into returns row count', () => {
    const n = 96;
    const price = new Float64Array(testData.close.slice(0, n));
    const volume = new Float64Array(testData.volume.slice(0, n));
    const pricePtr = wasm.allocate_f64_array(n);
    const volumePtr = wasm.allocate_f64_array(n);
    const rows = 2;
    const total = rows * n;
    const pmarPtr = wasm.price_moving_average_ratio_percentile_alloc(total);
    const pmarpPtr = wasm.price_moving_average_ratio_percentile_alloc(total);
    const plotlinePtr = wasm.price_moving_average_ratio_percentile_alloc(total);
    const signalPtr = wasm.price_moving_average_ratio_percentile_alloc(total);
    const pmarHighPtr = wasm.price_moving_average_ratio_percentile_alloc(total);
    const pmarLowPtr = wasm.price_moving_average_ratio_percentile_alloc(total);
    const scaledPmarPtr = wasm.price_moving_average_ratio_percentile_alloc(total);

    try {
        wasm.write_f64_array(pricePtr, price);
        wasm.write_f64_array(volumePtr, volume);
        const returnedRows = wasm.price_moving_average_ratio_percentile_batch_into(
            pricePtr,
            volumePtr,
            pmarPtr,
            pmarpPtr,
            plotlinePtr,
            signalPtr,
            pmarHighPtr,
            pmarLowPtr,
            scaledPmarPtr,
            n,
            20,
            22,
            2,
            60,
            60,
            0,
            10,
            10,
            0,
            'vwma',
            'sma',
            'pmarp',
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.price_moving_average_ratio_percentile_free(pmarPtr, total);
        wasm.price_moving_average_ratio_percentile_free(pmarpPtr, total);
        wasm.price_moving_average_ratio_percentile_free(plotlinePtr, total);
        wasm.price_moving_average_ratio_percentile_free(signalPtr, total);
        wasm.price_moving_average_ratio_percentile_free(pmarHighPtr, total);
        wasm.price_moving_average_ratio_percentile_free(pmarLowPtr, total);
        wasm.price_moving_average_ratio_percentile_free(scaledPmarPtr, total);
        wasm.deallocate_f64_array(pricePtr);
        wasm.deallocate_f64_array(volumePtr);
    }
});

test('price_moving_average_ratio_percentile_js rejects invalid period', () => {
    const n = 64;
    const price = new Float64Array(testData.close.slice(0, n));
    const volume = new Float64Array(testData.volume.slice(0, n));
    assert.throws(() => {
        wasm.price_moving_average_ratio_percentile_js(
            price,
            volume,
            0,
            'vwma',
            60,
            10,
            'sma',
            'pmarp',
        );
    }, /Invalid period/);
});
