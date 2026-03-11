import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function referenceFmDemodulator(open, close, period) {
    if (open.length === 0 || close.length === 0) {
        throw new Error('empty');
    }
    if (open.length !== close.length) {
        throw new Error('length mismatch');
    }
    if (period === 0 || period > open.length) {
        throw new Error('Invalid period');
    }

    let first = -1;
    for (let i = 0; i < open.length; i++) {
        if (!Number.isNaN(open[i]) && !Number.isNaN(close[i])) {
            first = i;
            break;
        }
    }
    if (first === -1) {
        throw new Error('All values are NaN');
    }
    const needed = Math.max(1, period - 2);
    if (open.length - first < needed) {
        throw new Error('Not enough valid data');
    }

    const out = new Float64Array(open.length);
    out.fill(NaN);
    const a1 = Math.exp(-1.414 * Math.PI / period);
    const b1 = 2 * a1 * Math.cos(1.414 * Math.PI / period);
    const c2 = b1;
    const c3 = -(a1 * a1);
    const c1 = 1 - c2 - c3;
    const warmupBars = Math.max(0, period - 3);
    let prevHl = 0;
    let ss1 = 0;
    let ss2 = 0;
    let validCount = 0;

    for (let i = first; i < open.length; i++) {
        if (Number.isNaN(open[i]) || Number.isNaN(close[i])) {
            prevHl = 0;
            ss1 = 0;
            ss2 = 0;
            validCount = 0;
            continue;
        }
        const derivative = close[i] - open[i];
        const hl = Math.max(-1, Math.min(1, 10 * derivative));
        const value = validCount < 3
            ? derivative
            : c1 * (hl + prevHl) * 0.5 + c2 * ss1 + c3 * ss2;
        prevHl = hl;
        ss2 = ss1;
        ss1 = value;
        validCount += 1;
        if (validCount > warmupBars) {
            out[i] = value;
        }
    }
    return out;
}

function assertArrayClose(actual, expected, tol = 1e-12) {
    assert.strictEqual(actual.length, expected.length);
    for (let i = 0; i < actual.length; i++) {
        const a = actual[i];
        const e = expected[i];
        if (Number.isNaN(a) || Number.isNaN(e)) {
            assert(Number.isNaN(a) && Number.isNaN(e), `NaN mismatch at index ${i}: ${a} vs ${e}`);
        } else {
            assert(Math.abs(a - e) <= tol, `Mismatch at index ${i}: ${a} vs ${e}`);
        }
    }
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
});

test('Ehlers FM Demodulator matches local reference', () => {
    const open = new Float64Array([1, 1, 1, 1, 1, 1, 1, 1]);
    const close = new Float64Array([1.1, 0.9, 1.2, 0.8, 1.05, 0.95, 1.15, 0.85]);
    const result = wasm.ehlers_fm_demodulator_js(open, close, 5);
    const expected = referenceFmDemodulator(open, close, 5);
    assertArrayClose(result, expected);
});

test('Ehlers FM Demodulator small period reference case', () => {
    const open = new Float64Array([1, 1, 1, 1, 1, 1]);
    const close = new Float64Array([1.1, 0.9, 1.2, 0.8, 1.05, 0.95]);
    const result = wasm.ehlers_fm_demodulator_js(open, close, 3);
    const expected = new Float64Array([
        0.10000000000000009,
        -0.09999999999999998,
        0.19999999999999996,
        0.013357467332142794,
        -0.26250860145491506,
        -0.011431966667873657,
    ]);
    assertArrayClose(result, expected);
});

test('Ehlers FM Demodulator batch matches single', () => {
    const open = new Float64Array([1, 1, 1, 1, 1, 1, 1, 1]);
    const close = new Float64Array([1.1, 0.9, 1.2, 0.8, 1.05, 0.95, 1.15, 0.85]);
    const batch = wasm.ehlers_fm_demodulator_batch(open, close, {
        period_range: [5, 7, 1],
    });
    assert.strictEqual(batch.rows, 3);
    assert.strictEqual(batch.cols, open.length);
    assert.strictEqual(batch.combos.length, 3);
    for (let row = 0; row < batch.rows; row++) {
        const period = batch.combos[row].period;
        const single = wasm.ehlers_fm_demodulator_js(open, close, period);
        const start = row * batch.cols;
        const end = start + batch.cols;
        assertArrayClose(batch.values.slice(start, end), single);
    }
});

test('Ehlers FM Demodulator zero-copy API matches regular API', () => {
    const open = new Float64Array([1, 1, 1, 1, 1, 1, 1, 1]);
    const close = new Float64Array([1.1, 0.9, 1.2, 0.8, 1.05, 0.95, 1.15, 0.85]);
    const expected = wasm.ehlers_fm_demodulator_js(open, close, 5);
    const len = open.length;
    const openPtr = wasm.ehlers_fm_demodulator_alloc(len);
    const closePtr = wasm.ehlers_fm_demodulator_alloc(len);
    const outPtr = wasm.ehlers_fm_demodulator_alloc(len);
    try {
        const mem = wasm.__wasm.memory.buffer;
        new Float64Array(mem, openPtr, len).set(open);
        new Float64Array(mem, closePtr, len).set(close);
        wasm.ehlers_fm_demodulator_into(openPtr, closePtr, outPtr, len, 5);
        const out = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        assertArrayClose(new Float64Array(out), expected);
    } finally {
        wasm.ehlers_fm_demodulator_free(openPtr, len);
        wasm.ehlers_fm_demodulator_free(closePtr, len);
        wasm.ehlers_fm_demodulator_free(outPtr, len);
    }
});

test('Ehlers FM Demodulator batch_into matches batch API', () => {
    const open = new Float64Array([1, 1, 1, 1, 1, 1, 1, 1]);
    const close = new Float64Array([1.1, 0.9, 1.2, 0.8, 1.05, 0.95, 1.15, 0.85]);
    const batch = wasm.ehlers_fm_demodulator_batch(open, close, {
        period_range: [5, 7, 1],
    });
    const len = open.length;
    const rowsExpected = 3;
    const openPtr = wasm.ehlers_fm_demodulator_alloc(len);
    const closePtr = wasm.ehlers_fm_demodulator_alloc(len);
    const outPtr = wasm.ehlers_fm_demodulator_alloc(len * rowsExpected);
    try {
        new Float64Array(wasm.__wasm.memory.buffer, openPtr, len).set(open);
        new Float64Array(wasm.__wasm.memory.buffer, closePtr, len).set(close);
        const rows = wasm.ehlers_fm_demodulator_batch_into(
            openPtr,
            closePtr,
            outPtr,
            len,
            5,
            7,
            1
        );
        assert.strictEqual(rows, rowsExpected);
        const out = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len * rowsExpected);
        assertArrayClose(new Float64Array(out), batch.values);
    } finally {
        wasm.ehlers_fm_demodulator_free(openPtr, len);
        wasm.ehlers_fm_demodulator_free(closePtr, len);
        wasm.ehlers_fm_demodulator_free(outPtr, len * rowsExpected);
    }
});

test('Ehlers FM Demodulator invalid inputs throw', () => {
    const open = new Float64Array([1, 1, 1, 1]);
    const close = new Float64Array([1.1, 0.9, 1.2, 0.8]);
    assert.throws(
        () => wasm.ehlers_fm_demodulator_js(open, close, 0),
        /invalid period/
    );
    assert.throws(
        () => wasm.ehlers_fm_demodulator_js(open, new Float64Array([1.1, 0.9]), 3),
        /length mismatch/
    );
    const allNaN = new Float64Array(8);
    allNaN.fill(NaN);
    assert.throws(
        () => wasm.ehlers_fm_demodulator_js(allNaN, allNaN, 5),
        /all values are NaN/
    );
});
