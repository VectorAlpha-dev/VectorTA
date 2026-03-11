import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function trendData(size) {
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        const base = 100 + i * 0.9;
        high[i] = base + 1 + (i % 3) * 0.1;
        low[i] = base - 1 - (i % 2) * 0.1;
        close[i] = base;
    }
    return { high, low, close };
}

function reversalData() {
    const close = new Float64Array([
        100.0, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0, 96.5, 96.0, 95.5, 97.0, 99.0, 101.0, 104.0,
        108.0, 111.0, 113.0, 112.0, 111.0, 109.0, 107.0, 105.0, 103.0, 101.0, 99.0, 97.0, 95.0,
        94.0, 93.5, 93.0,
    ]);
    const high = new Float64Array(close.length);
    const low = new Float64Array(close.length);
    for (let i = 0; i < close.length; i++) {
        high[i] = close[i] + 0.8;
        low[i] = close[i] - 0.8;
    }
    return { high, low, close };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath =
        process.platform === 'win32' ? 'file:///' + wasmPath.replace(/\\/g, '/') : wasmPath;
    wasm = await import(importPath);
});

test('Statistical Trailing Stop reversal behavior', () => {
    const { high, low, close } = reversalData();
    const result = wasm.statistical_trailing_stop_js(high, low, close, 3, 10, 'level0');

    const changedIdx = [];
    for (let i = 0; i < result.changed.length; i++) {
        if (result.changed[i] === 1) {
            changedIdx.push(i);
        }
    }
    assert(changedIdx.length > 0);
    assert(Number.isFinite(result.anchor[changedIdx[0]]));
    assert.strictEqual(result.bias[changedIdx[0]], 1);
});

test('Statistical Trailing Stop invalid base level', () => {
    const { high, low, close } = trendData(160);
    assert.throws(() => {
        wasm.statistical_trailing_stop_js(high, low, close, 10, 100, 'bad');
    }, /invalid base level/i);
});

test('Statistical Trailing Stop NaN gap restart', () => {
    const { high, low, close } = trendData(170);
    high[120] = Number.NaN;
    low[120] = Number.NaN;
    close[120] = Number.NaN;
    const result = wasm.statistical_trailing_stop_js(high, low, close, 10, 100, 'level2');

    const restartEnd = Math.min(result.level.length, 120 + 10 + 100 + 1);
    for (let i = 120; i < restartEnd; i++) {
        assert(isNaN(result.level[i]));
        assert(isNaN(result.bias[i]));
        assert(isNaN(result.changed[i]));
    }
});

test('Statistical Trailing Stop fast API matches safe API', () => {
    const { high, low, close } = trendData(160);
    const len = close.length;
    const highPtr = wasm.statistical_trailing_stop_alloc(len);
    const lowPtr = wasm.statistical_trailing_stop_alloc(len);
    const closePtr = wasm.statistical_trailing_stop_alloc(len);
    const levelPtr = wasm.statistical_trailing_stop_alloc(len);
    const anchorPtr = wasm.statistical_trailing_stop_alloc(len);
    const biasPtr = wasm.statistical_trailing_stop_alloc(len);
    const changedPtr = wasm.statistical_trailing_stop_alloc(len);

    try {
        new Float64Array(wasm.__wasm.memory.buffer, highPtr, len).set(high);
        new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len).set(low);
        new Float64Array(wasm.__wasm.memory.buffer, closePtr, len).set(close);

        wasm.statistical_trailing_stop_into(
            highPtr,
            lowPtr,
            closePtr,
            levelPtr,
            anchorPtr,
            biasPtr,
            changedPtr,
            len,
            10,
            100,
            'level2',
        );

        const safe = wasm.statistical_trailing_stop_js(high, low, close, 10, 100, 'level2');
        assertArrayClose(
            new Float64Array(wasm.__wasm.memory.buffer, levelPtr, len),
            safe.level,
            1e-12,
            'level mismatch',
        );
        assertArrayClose(
            new Float64Array(wasm.__wasm.memory.buffer, anchorPtr, len),
            safe.anchor,
            1e-12,
            'anchor mismatch',
        );
        assertArrayClose(
            new Float64Array(wasm.__wasm.memory.buffer, biasPtr, len),
            safe.bias,
            1e-12,
            'bias mismatch',
        );
        assertArrayClose(
            new Float64Array(wasm.__wasm.memory.buffer, changedPtr, len),
            safe.changed,
            1e-12,
            'changed mismatch',
        );
    } finally {
        wasm.statistical_trailing_stop_free(highPtr, len);
        wasm.statistical_trailing_stop_free(lowPtr, len);
        wasm.statistical_trailing_stop_free(closePtr, len);
        wasm.statistical_trailing_stop_free(levelPtr, len);
        wasm.statistical_trailing_stop_free(anchorPtr, len);
        wasm.statistical_trailing_stop_free(biasPtr, len);
        wasm.statistical_trailing_stop_free(changedPtr, len);
    }
});

test('Statistical Trailing Stop batch matches single', () => {
    const { high, low, close } = trendData(170);
    const result = wasm.statistical_trailing_stop_batch(high, low, close, {
        data_length_range: [9, 10, 1],
        normalization_length_range: [10, 11, 1],
        base_level_range: ['level1', 'level2', 1],
    });

    assert.strictEqual(result.rows, 8);
    assert.strictEqual(result.cols, close.length);

    for (let row = 0; row < result.rows; row++) {
        const combo = result.combos[row];
        const single = wasm.statistical_trailing_stop_js(
            high,
            low,
            close,
            combo.data_length,
            combo.normalization_length,
            combo.base_level,
        );
        const start = row * result.cols;
        const end = start + result.cols;
        assertArrayClose(result.level.slice(start, end), single.level, 1e-12, `level row ${row}`);
        assertArrayClose(
            result.anchor.slice(start, end),
            single.anchor,
            1e-12,
            `anchor row ${row}`,
        );
        assertArrayClose(result.bias.slice(start, end), single.bias, 1e-12, `bias row ${row}`);
        assertArrayClose(
            result.changed.slice(start, end),
            single.changed,
            1e-12,
            `changed row ${row}`,
        );
    }
});
