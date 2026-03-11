import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function linearData(size) {
    const out = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        out[i] = i;
    }
    return out;
}

function constantData(size, value) {
    const out = new Float64Array(size);
    out.fill(value);
    return out;
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath =
        process.platform === 'win32'
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
    wasm = await import(importPath);
});

test('Adaptive MACD constant series flattens to zero', () => {
    const data = constantData(24, 100.0);
    const result = wasm.adaptive_macd_js(data, 6, 5, 10, 4);

    assert(result.macd.slice(0, 5).every(v => isNaN(v)));
    assert(result.signal.slice(0, 5).every(v => isNaN(v)));
    assert(result.hist.slice(0, 5).every(v => isNaN(v)));
    assertArrayClose(result.macd.slice(5), new Float64Array(19), 1e-12, 'macd zero mismatch');
    assertArrayClose(result.signal.slice(5), new Float64Array(19), 1e-12, 'signal zero mismatch');
    assertArrayClose(result.hist.slice(5), new Float64Array(19), 1e-12, 'hist zero mismatch');
});

test('Adaptive MACD invalid period', () => {
    const data = linearData(10);
    assert.throws(() => {
        wasm.adaptive_macd_js(data, 1, 3, 6, 3);
    }, /invalid period/i);
});

test('Adaptive MACD NaN gap semantics', () => {
    const data = new Float64Array([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, Number.NaN, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
    ]);
    const result = wasm.adaptive_macd_js(data, 4, 3, 6, 3);

    assert(result.macd.slice(0, 3).every(v => isNaN(v)));
    assert(!isNaN(result.macd[3]));
    assert(result.macd.slice(6, 10).every(v => isNaN(v)));
    assert(!isNaN(result.macd[10]));
    assert(isNaN(result.hist[6]));
});

test('Adaptive MACD fast API matches safe API', () => {
    const data = linearData(28);
    const len = data.length;
    const inPtr = wasm.adaptive_macd_alloc(len);
    const macdPtr = wasm.adaptive_macd_alloc(len);
    const signalPtr = wasm.adaptive_macd_alloc(len);
    const histPtr = wasm.adaptive_macd_alloc(len);

    try {
        new Float64Array(wasm.__wasm.memory.buffer, inPtr, len).set(data);
        wasm.adaptive_macd_into(inPtr, macdPtr, signalPtr, histPtr, len, 5, 4, 8, 3);

        const safe = wasm.adaptive_macd_js(data, 5, 4, 8, 3);
        const macd = new Float64Array(wasm.__wasm.memory.buffer, macdPtr, len);
        const signal = new Float64Array(wasm.__wasm.memory.buffer, signalPtr, len);
        const hist = new Float64Array(wasm.__wasm.memory.buffer, histPtr, len);

        assertArrayClose(new Float64Array(macd), safe.macd, 1e-12, 'fast api macd mismatch');
        assertArrayClose(new Float64Array(signal), safe.signal, 1e-12, 'fast api signal mismatch');
        assertArrayClose(new Float64Array(hist), safe.hist, 1e-12, 'fast api hist mismatch');
    } finally {
        wasm.adaptive_macd_free(inPtr, len);
        wasm.adaptive_macd_free(macdPtr, len);
        wasm.adaptive_macd_free(signalPtr, len);
        wasm.adaptive_macd_free(histPtr, len);
    }
});

test('Adaptive MACD batch matches single', () => {
    const data = linearData(26);
    const result = wasm.adaptive_macd_batch(data, {
        length_range: [4, 5, 1],
        fast_period_range: [3, 4, 1],
        slow_period_range: [6, 7, 1],
        signal_period_range: [3, 3, 0],
    });

    assert.strictEqual(result.rows, 8);
    assert.strictEqual(result.cols, data.length);
    assert.strictEqual(result.macd.length, result.rows * result.cols);
    assert.strictEqual(result.signal.length, result.rows * result.cols);
    assert.strictEqual(result.hist.length, result.rows * result.cols);

    for (let row = 0; row < result.rows; row++) {
        const combo = result.combos[row];
        const single = wasm.adaptive_macd_js(
            data,
            combo.length,
            combo.fast_period,
            combo.slow_period,
            combo.signal_period,
        );
        const start = row * result.cols;
        const end = start + result.cols;
        assertArrayClose(result.macd.slice(start, end), single.macd, 1e-12, `macd row ${row}`);
        assertArrayClose(result.signal.slice(start, end), single.signal, 1e-12, `signal row ${row}`);
        assertArrayClose(result.hist.slice(start, end), single.hist, 1e-12, `hist row ${row}`);
    }
});
