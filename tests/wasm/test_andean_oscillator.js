import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function manualAndean(open, close, length = 50, signalLength = 9) {
    const bull = new Float64Array(open.length);
    const bear = new Float64Array(open.length);
    const signal = new Float64Array(open.length);
    bull.fill(NaN);
    bear.fill(NaN);
    signal.fill(NaN);

    const alpha = 2 / (length + 1);
    const signalAlpha = 2 / (signalLength + 1);
    let initialized = false;
    let up1 = NaN;
    let up2 = NaN;
    let dn1 = NaN;
    let dn2 = NaN;
    let sig = NaN;

    for (let i = 0; i < open.length; i++) {
        const o = open[i];
        const c = close[i];
        if (!Number.isFinite(o) || !Number.isFinite(c)) {
            continue;
        }

        const c2 = c * c;
        const o2 = o * o;

        if (!initialized) {
            up1 = c;
            up2 = c2;
            dn1 = c;
            dn2 = c2;
            sig = 0;
            bull[i] = 0;
            bear[i] = 0;
            signal[i] = 0;
            initialized = true;
            continue;
        }

        up1 = Math.max(c, o, up1 - (up1 - c) * alpha);
        up2 = Math.max(c2, o2, up2 - (up2 - c2) * alpha);
        dn1 = Math.min(c, o, dn1 + (c - dn1) * alpha);
        dn2 = Math.min(c2, o2, dn2 + (c2 - dn2) * alpha);

        bull[i] = Math.sqrt(Math.max(0, dn2 - dn1 * dn1));
        bear[i] = Math.sqrt(Math.max(0, up2 - up1 * up1));
        const src = Math.max(bull[i], bear[i]);
        sig = signalAlpha * src + (1 - signalAlpha) * sig;
        signal[i] = sig;
    }

    return { bull, bear, signal };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('andean_oscillator_js matches manual reference', () => {
    const open = new Float64Array(testData.open.slice(0, 160));
    const close = new Float64Array(testData.close.slice(0, 160));
    const result = wasm.andean_oscillator_js(open, close, 50, 9);
    const expected = manualAndean(open, close, 50, 9);
    assertArrayClose(result.bull, expected.bull, 1e-12, 'bull mismatch');
    assertArrayClose(result.bear, expected.bear, 1e-12, 'bear mismatch');
    assertArrayClose(result.signal, expected.signal, 1e-12, 'signal mismatch');
});

test('andean_oscillator_batch_js first row matches single', () => {
    const open = new Float64Array(testData.open.slice(0, 128));
    const close = new Float64Array(testData.close.slice(0, 128));
    const batch = wasm.andean_oscillator_batch_js(open, close, {
        length_range: [50, 52, 2],
        signal_length_range: [9, 11, 2],
    });
    const single = wasm.andean_oscillator_js(open, close, 50, 9);
    assert.strictEqual(batch.rows, 4);
    assert.strictEqual(batch.cols, 128);
    assert.deepStrictEqual(Array.from(batch.lengths), [50, 50, 52, 52]);
    assert.deepStrictEqual(Array.from(batch.signal_lengths), [9, 11, 9, 11]);
    assertArrayClose(batch.bull.slice(0, 128), single.bull, 1e-12, 'batch bull mismatch');
    assertArrayClose(batch.bear.slice(0, 128), single.bear, 1e-12, 'batch bear mismatch');
    assertArrayClose(batch.signal.slice(0, 128), single.signal, 1e-12, 'batch signal mismatch');
});

test('andean_oscillator_into pointer path matches safe API', () => {
    const n = 144;
    const open = new Float64Array(testData.open.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const safe = wasm.andean_oscillator_js(open, close, 50, 9);
    const openPtr = wasm.allocate_f64_array(n);
    const closePtr = wasm.allocate_f64_array(n);
    const outBullPtr = wasm.andean_oscillator_alloc(n);
    const outBearPtr = wasm.andean_oscillator_alloc(n);
    const outSignalPtr = wasm.andean_oscillator_alloc(n);
    try {
        wasm.write_f64_array(openPtr, open);
        wasm.write_f64_array(closePtr, close);
        wasm.andean_oscillator_into(openPtr, closePtr, outBullPtr, outBearPtr, outSignalPtr, n, 50, 9);
        const bull = wasm.read_f64_array(outBullPtr, n);
        const bear = wasm.read_f64_array(outBearPtr, n);
        const signal = wasm.read_f64_array(outSignalPtr, n);
        assertArrayClose(bull, safe.bull, 1e-12, 'pointer bull mismatch');
        assertArrayClose(bear, safe.bear, 1e-12, 'pointer bear mismatch');
        assertArrayClose(signal, safe.signal, 1e-12, 'pointer signal mismatch');
    } finally {
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(closePtr);
        wasm.andean_oscillator_free(outBullPtr, n);
        wasm.andean_oscillator_free(outBearPtr, n);
        wasm.andean_oscillator_free(outSignalPtr, n);
    }
});

test('andean_oscillator_batch_into returns row count', () => {
    const n = 128;
    const rows = 4;
    const open = new Float64Array(testData.open.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const openPtr = wasm.allocate_f64_array(n);
    const closePtr = wasm.allocate_f64_array(n);
    const outBullPtr = wasm.andean_oscillator_alloc(rows * n);
    const outBearPtr = wasm.andean_oscillator_alloc(rows * n);
    const outSignalPtr = wasm.andean_oscillator_alloc(rows * n);
    try {
        wasm.write_f64_array(openPtr, open);
        wasm.write_f64_array(closePtr, close);
        const returnedRows = wasm.andean_oscillator_batch_into(
            openPtr,
            closePtr,
            outBullPtr,
            outBearPtr,
            outSignalPtr,
            n,
            50,
            52,
            2,
            9,
            11,
            2,
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(closePtr);
        wasm.andean_oscillator_free(outBullPtr, rows * n);
        wasm.andean_oscillator_free(outBearPtr, rows * n);
        wasm.andean_oscillator_free(outSignalPtr, rows * n);
    }
});

test('andean_oscillator_js rejects invalid length', () => {
    const open = new Float64Array(testData.open.slice(0, 64));
    const close = new Float64Array(testData.close.slice(0, 64));
    assert.throws(
        () => wasm.andean_oscillator_js(open, close, 0, 9),
        /Invalid length/,
    );
});
