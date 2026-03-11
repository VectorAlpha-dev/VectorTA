import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function manualVzo(close, volume, length, intradaySmoothing, noiseFilter) {
    const alpha = 2 / (length + 1);
    const beta = 1 - alpha;
    const smoothAlpha = 2 / (noiseFilter + 1);
    const smoothBeta = 1 - smoothAlpha;
    const out = new Float64Array(close.length);
    out.fill(NaN);

    let firstValid = -1;
    for (let i = 0; i < volume.length; i++) {
        if (Number.isFinite(volume[i])) {
            firstValid = i;
            break;
        }
    }
    if (firstValid === -1) {
        return out;
    }

    let prevClose = NaN;
    let emaDirection = 0.0;
    let emaTotal = 0.0;
    let smooth = 0.0;
    let smoothValid = false;

    for (let i = firstValid; i < close.length; i++) {
        let raw = NaN;
        if (Number.isFinite(volume[i])) {
            const directed = Number.isFinite(close[i]) && Number.isFinite(prevClose) && close[i] > prevClose
                ? volume[i]
                : -volume[i];
            emaDirection = beta * emaDirection + alpha * directed;
            emaTotal = beta * emaTotal + alpha * volume[i];
            if (emaTotal !== 0) {
                raw = 100 * emaDirection / emaTotal;
            }
        } else if (emaTotal !== 0) {
            raw = 100 * emaDirection / emaTotal;
        }

        if (Number.isFinite(close[i])) {
            prevClose = close[i];
        }

        if (intradaySmoothing) {
            if (Number.isFinite(raw)) {
                smooth = smoothBeta * smooth + smoothAlpha * raw;
                smoothValid = true;
                out[i] = smooth;
            } else if (smoothValid) {
                out[i] = smooth;
            }
        } else {
            out[i] = raw;
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

test('volume_zone_oscillator_js matches manual reference', () => {
    const n = 160;
    const close = new Float64Array(testData.close.slice(0, n));
    const volume = new Float64Array(testData.volume.slice(0, n));
    const result = wasm.volume_zone_oscillator_js(close, volume, 14, true, 4);
    const expected = manualVzo(close, volume, 14, true, 4);
    assertArrayClose(result, expected, 1e-12, 'manual mismatch');
    assert.strictEqual(result.length, n);
    assert.ok(isNaN(result[0]) || Number.isFinite(result[0]));
});

test('volume_zone_oscillator_batch_js first row matches single', () => {
    const n = 128;
    const close = new Float64Array(testData.close.slice(0, n));
    const volume = new Float64Array(testData.volume.slice(0, n));
    const batch = wasm.volume_zone_oscillator_batch_js(close, volume, {
        length_range: [14, 16, 2],
        intraday_smoothing: true,
        noise_filter_range: [4, 4, 0],
    });
    const single = wasm.volume_zone_oscillator_js(close, volume, 14, true, 4);
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, n);
    assert.strictEqual(batch.combos.length, 2);
    assert.strictEqual(batch.combos[0].length, 14);
    assert.strictEqual(batch.combos[0].intraday_smoothing, true);
    assert.strictEqual(batch.combos[0].noise_filter, 4);
    assertArrayClose(batch.values.slice(0, n), single, 1e-12, 'batch mismatch');
});

test('volume_zone_oscillator_into pointer path matches safe API', () => {
    const n = 128;
    const close = new Float64Array(testData.close.slice(0, n));
    const volume = new Float64Array(testData.volume.slice(0, n));
    const safe = wasm.volume_zone_oscillator_js(close, volume, 14, true, 4);

    const closePtr = wasm.allocate_f64_array(n);
    const volumePtr = wasm.allocate_f64_array(n);
    const outPtr = wasm.volume_zone_oscillator_alloc(n);

    try {
        wasm.write_f64_array(closePtr, close);
        wasm.write_f64_array(volumePtr, volume);
        wasm.volume_zone_oscillator_into(closePtr, volumePtr, outPtr, n, 14, true, 4);
        const flat = wasm.read_f64_array(outPtr, n);
        assertArrayClose(flat, safe, 1e-12, 'pointer mismatch');
    } finally {
        wasm.volume_zone_oscillator_free(outPtr, n);
        wasm.deallocate_f64_array(closePtr);
        wasm.deallocate_f64_array(volumePtr);
    }
});

test('volume_zone_oscillator_batch_into returns row count', () => {
    const n = 96;
    const close = new Float64Array(testData.close.slice(0, n));
    const volume = new Float64Array(testData.volume.slice(0, n));
    const closePtr = wasm.allocate_f64_array(n);
    const volumePtr = wasm.allocate_f64_array(n);
    const rows = 2;
    const outPtr = wasm.volume_zone_oscillator_alloc(rows * n);

    try {
        wasm.write_f64_array(closePtr, close);
        wasm.write_f64_array(volumePtr, volume);
        const returnedRows = wasm.volume_zone_oscillator_batch_into(
            closePtr,
            volumePtr,
            outPtr,
            n,
            14,
            16,
            2,
            true,
            4,
            4,
            0,
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.volume_zone_oscillator_free(outPtr, rows * n);
        wasm.deallocate_f64_array(closePtr);
        wasm.deallocate_f64_array(volumePtr);
    }
});

test('volume_zone_oscillator_js rejects invalid length', () => {
    const n = 64;
    const close = new Float64Array(testData.close.slice(0, n));
    const volume = new Float64Array(testData.volume.slice(0, n));
    assert.throws(() => {
        wasm.volume_zone_oscillator_js(close, volume, 1, true, 4);
    }, /Invalid length/);
});

test('volume_zone_oscillator_batch_js rejects invalid config', () => {
    const n = 64;
    const close = new Float64Array(testData.close.slice(0, n));
    const volume = new Float64Array(testData.volume.slice(0, n));
    assert.throws(() => {
        wasm.volume_zone_oscillator_batch_js(close, volume, {
            length_range: [14, 16],
            intraday_smoothing: true,
            noise_filter_range: [4, 4, 0],
        });
    }, /Invalid config/);
});
