import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('impulse_macd_js output contract', () => {
    const high = new Float64Array(testData.high.slice(0, 720));
    const low = new Float64Array(testData.low.slice(0, 720));
    const close = new Float64Array(testData.close.slice(0, 720));

    const result = wasm.impulse_macd_js(high, low, close, 34, 9);

    assert.strictEqual(result.impulse_macd.length, close.length);
    assert.strictEqual(result.impulse_histo.length, close.length);
    assert.strictEqual(result.signal.length, close.length);
    assert(Array.from(result.impulse_histo.slice(0, 8)).every(v => isNaN(v)));
    assert(Array.from(result.signal.slice(0, 8)).every(v => isNaN(v)));
    assert(Array.from(result.impulse_macd).some(v => !isNaN(v)));
    assert(Array.from(result.impulse_histo.slice(8)).some(v => !isNaN(v)));
    assert(Array.from(result.signal.slice(8)).some(v => !isNaN(v)));
});

test('impulse_macd_into_host pointer path matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 320));
    const low = new Float64Array(testData.low.slice(0, 320));
    const close = new Float64Array(testData.close.slice(0, 320));
    const safe = wasm.impulse_macd_js(high, low, close, 34, 9);
    const outPtr = wasm.impulse_macd_alloc(close.length);

    try {
        wasm.impulse_macd_into_host(high, low, close, outPtr, 34, 9);
        const flat = wasm.read_f64_array(outPtr, close.length * 3);
        const impulseMacd = flat.slice(0, close.length);
        const impulseHisto = flat.slice(close.length, close.length * 2);
        const signal = flat.slice(close.length * 2);
        assertArrayClose(impulseMacd, safe.impulse_macd, 1e-10, 'impulse_macd pointer-path mismatch');
        assertArrayClose(impulseHisto, safe.impulse_histo, 1e-10, 'impulse_histo pointer-path mismatch');
        assertArrayClose(signal, safe.signal, 1e-10, 'signal pointer-path mismatch');
    } finally {
        wasm.impulse_macd_free(outPtr, close.length);
    }
});

test('impulse_macd_batch_js single parameter set matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 320));
    const low = new Float64Array(testData.low.slice(0, 320));
    const close = new Float64Array(testData.close.slice(0, 320));
    const batch = wasm.impulse_macd_batch_js(high, low, close, {
        length_ma_range: [34, 34, 0],
        length_signal_range: [9, 9, 0],
    });
    const single = wasm.impulse_macd_js(high, low, close, 34, 9);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.impulse_macd.length, close.length);
    assert.strictEqual(batch.impulse_histo.length, close.length);
    assert.strictEqual(batch.signal.length, close.length);
    assert.strictEqual(batch.combos[0].length_ma, 34);
    assert.strictEqual(batch.combos[0].length_signal, 9);
    assertArrayClose(batch.impulse_macd, single.impulse_macd, 1e-10, 'impulse_macd batch mismatch');
    assertArrayClose(batch.impulse_histo, single.impulse_histo, 1e-10, 'impulse_histo batch mismatch');
    assertArrayClose(batch.signal, single.signal, 1e-10, 'signal batch mismatch');
});
