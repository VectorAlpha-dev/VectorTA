import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function mixedData(size) {
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        const x = i;
        const c = 100 + 0.18 * x + Math.sin(x * 0.23) * 5.5 + Math.cos(x * 0.07) * 2.0;
        close[i] = c;
        high[i] = c + 1.1 + (i % 3) * 0.05;
        low[i] = c - 1.0 - (i % 2) * 0.05;
    }
    return { high, low, close };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath =
        process.platform === 'win32' ? 'file:///' + wasmPath.replace(/\\/g, '/') : wasmPath;
    wasm = await import(importPath);
});

test('Standardized PSAR Oscillator outputs are present', () => {
    const { high, low, close } = mixedData(220);
    const result = wasm.standardized_psar_oscillator_js(
        high, low, close, 0.02, 0.0005, 0.2, 21, 40, 3, 15, 1, true, true,
    );
    assert(result.oscillator.some(Number.isFinite));
    assert(result.ma.some(Number.isFinite));
});

test('Standardized PSAR Oscillator invalid start throws', () => {
    const { high, low, close } = mixedData(64);
    assert.throws(() => {
        wasm.standardized_psar_oscillator_js(
            high, low, close, 0.0, 0.0005, 0.2, 21, 40, 3, 15, 1, true, true,
        );
    }, /invalid start/i);
});

test('Standardized PSAR Oscillator NaN gap restart', () => {
    const { high, low, close } = mixedData(220);
    high[120] = Number.NaN;
    low[120] = Number.NaN;
    close[120] = Number.NaN;
    const result = wasm.standardized_psar_oscillator_js(
        high, low, close, 0.02, 0.0005, 0.2, 12, 10, 3, 15, 1, true, true,
    );
    const restartEnd = Math.min(result.oscillator.length, 120 + Math.max(12, 2));
    for (let i = 120; i < restartEnd; i++) {
        assert(isNaN(result.oscillator[i]));
    }
});

test('Standardized PSAR Oscillator fast API matches safe API', () => {
    const { high, low, close } = mixedData(180);
    const len = close.length;
    const highPtr = wasm.standardized_psar_oscillator_alloc(len);
    const lowPtr = wasm.standardized_psar_oscillator_alloc(len);
    const closePtr = wasm.standardized_psar_oscillator_alloc(len);
    const oscPtr = wasm.standardized_psar_oscillator_alloc(len);
    const maPtr = wasm.standardized_psar_oscillator_alloc(len);
    const br1Ptr = wasm.standardized_psar_oscillator_alloc(len);
    const br2Ptr = wasm.standardized_psar_oscillator_alloc(len);
    const rb1Ptr = wasm.standardized_psar_oscillator_alloc(len);
    const rb2Ptr = wasm.standardized_psar_oscillator_alloc(len);
    const bw1Ptr = wasm.standardized_psar_oscillator_alloc(len);
    const bw2Ptr = wasm.standardized_psar_oscillator_alloc(len);
    try {
        new Float64Array(wasm.__wasm.memory.buffer, highPtr, len).set(high);
        new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len).set(low);
        new Float64Array(wasm.__wasm.memory.buffer, closePtr, len).set(close);
        wasm.standardized_psar_oscillator_into(
            highPtr, lowPtr, closePtr, oscPtr, maPtr, br1Ptr, br2Ptr, rb1Ptr, rb2Ptr, bw1Ptr, bw2Ptr,
            len, 0.02, 0.0005, 0.2, 21, 40, 3, 15, 1, true, true,
        );
        const safe = wasm.standardized_psar_oscillator_js(
            high, low, close, 0.02, 0.0005, 0.2, 21, 40, 3, 15, 1, true, true,
        );
        assertArrayClose(
            new Float64Array(wasm.__wasm.memory.buffer, oscPtr, len),
            safe.oscillator,
            1e-12,
        );
    } finally {
        for (const ptr of [highPtr, lowPtr, closePtr, oscPtr, maPtr, br1Ptr, br2Ptr, rb1Ptr, rb2Ptr, bw1Ptr, bw2Ptr]) {
            wasm.standardized_psar_oscillator_free(ptr, len);
        }
    }
});

test('Standardized PSAR Oscillator batch returns rows', () => {
    const { high, low, close } = mixedData(180);
    const result = wasm.standardized_psar_oscillator_batch(high, low, close, {
        start_range: [0.02, 0.03, 0.01],
        increment_range: [0.0005, 0.0005, 0.0],
        maximum_range: [0.2, 0.2, 0.0],
        standardization_length_range: [10, 11, 1],
        wma_length_range: [8, 8, 0],
        wma_lag_range: [2, 2, 0],
        pivot_left_range: [6, 6, 0],
        pivot_right_range: [1, 1, 0],
        plot_bullish: true,
        plot_bearish: true,
    });
    assert.strictEqual(result.rows, 4);
    assert.strictEqual(result.cols, close.length);
});
