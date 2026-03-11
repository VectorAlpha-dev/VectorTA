import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function trendData(size, slope) {
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        const c = 100 + slope * i;
        close[i] = c;
        high[i] = c + 1.0;
        low[i] = c - 1.0;
    }
    return { high, low, close };
}

function mixedData(size) {
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        const x = i;
        const c = 100 + 0.35 * x + Math.sin(x * 0.27) * 4.0 + Math.cos(x * 0.11) * 1.5;
        close[i] = c;
        high[i] = c + 1.2 + (i % 3) * 0.1;
        low[i] = c - 1.1 - (i % 2) * 0.1;
    }
    return { high, low, close };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath =
        process.platform === 'win32' ? 'file:///' + wasmPath.replace(/\\/g, '/') : wasmPath;
    wasm = await import(importPath);
});

test('Geometric Bias Oscillator increasing trend reaches hundred', () => {
    const { high, low, close } = trendData(160, 1.0);
    const result = wasm.geometric_bias_oscillator_js(high, low, close, 20, 2.0, 14, 3);
    const start = Math.max(20, 14) + 3 - 2;
    for (let i = start; i < result.length; i++) {
        assert(Math.abs(result[i] - 100.0) <= 1e-12);
    }
});

test('Geometric Bias Oscillator decreasing trend reaches negative hundred', () => {
    const { high, low, close } = trendData(160, -1.0);
    const result = wasm.geometric_bias_oscillator_js(high, low, close, 20, 2.0, 14, 3);
    const start = Math.max(20, 14) + 3 - 2;
    for (let i = start; i < result.length; i++) {
        assert(Math.abs(result[i] + 100.0) <= 1e-12);
    }
});

test('Geometric Bias Oscillator invalid length', () => {
    const { high, low, close } = mixedData(64);
    assert.throws(() => {
        wasm.geometric_bias_oscillator_js(high, low, close, 9, 2.0, 14, 1);
    }, /invalid length/i);
});

test('Geometric Bias Oscillator NaN gap restart', () => {
    const { high, low, close } = mixedData(180);
    high[120] = Number.NaN;
    low[120] = Number.NaN;
    close[120] = Number.NaN;
    const result = wasm.geometric_bias_oscillator_js(high, low, close, 18, 2.0, 10, 3);
    const restartEnd = Math.min(result.length, 120 + Math.max(18, 10) + 3 - 1);
    for (let i = 120; i < restartEnd; i++) {
        assert(isNaN(result[i]));
    }
});

test('Geometric Bias Oscillator fast API matches safe API', () => {
    const { high, low, close } = mixedData(200);
    const len = close.length;
    const highPtr = wasm.geometric_bias_oscillator_alloc(len);
    const lowPtr = wasm.geometric_bias_oscillator_alloc(len);
    const closePtr = wasm.geometric_bias_oscillator_alloc(len);
    const outPtr = wasm.geometric_bias_oscillator_alloc(len);

    try {
        new Float64Array(wasm.__wasm.memory.buffer, highPtr, len).set(high);
        new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len).set(low);
        new Float64Array(wasm.__wasm.memory.buffer, closePtr, len).set(close);

        wasm.geometric_bias_oscillator_into(
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            len,
            24,
            2.4,
            12,
            4,
        );

        const safe = wasm.geometric_bias_oscillator_js(high, low, close, 24, 2.4, 12, 4);
        assertArrayClose(
            new Float64Array(wasm.__wasm.memory.buffer, outPtr, len),
            safe,
            1e-12,
            'fast API mismatch',
        );
    } finally {
        wasm.geometric_bias_oscillator_free(highPtr, len);
        wasm.geometric_bias_oscillator_free(lowPtr, len);
        wasm.geometric_bias_oscillator_free(closePtr, len);
        wasm.geometric_bias_oscillator_free(outPtr, len);
    }
});

test('Geometric Bias Oscillator batch matches single', () => {
    const { high, low, close } = mixedData(170);
    const result = wasm.geometric_bias_oscillator_batch(high, low, close, {
        length_range: [10, 12, 2],
        multiplier_range: [1.5, 2.0, 0.5],
        atr_length_range: [5, 6, 1],
        smooth_range: [1, 2, 1],
    });

    assert.strictEqual(result.rows, 16);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, result.rows * result.cols);

    for (let row = 0; row < result.rows; row++) {
        const combo = result.combos[row];
        const single = wasm.geometric_bias_oscillator_js(
            high,
            low,
            close,
            combo.length,
            combo.multiplier,
            combo.atr_length,
            combo.smooth,
        );
        const start = row * result.cols;
        const end = start + result.cols;
        assertArrayClose(
            result.values.slice(start, end),
            single,
            1e-12,
            `batch row mismatch at row ${row}`,
        );
    }
});
