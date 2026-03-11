import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function nz(array, idx) {
    if (idx < 0) {
        return 0;
    }
    const value = array[idx];
    return Number.isFinite(value) ? value : 0;
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

function manualMesa(source, length) {
    const n = source.length;
    const out = new Float64Array(n);
    const hp = new Float64Array(n);
    const filt = new Float64Array(n);
    out.fill(NaN);
    hp.fill(NaN);
    filt.fill(NaN);

    const pi = 3.14;
    const alpha1 = (Math.cos(0.707 * 2 * pi / 48) + Math.sin(0.707 * 2 * pi / 48) - 1) / Math.cos(0.707 * 2 * pi / 48);
    const hpCoef = (1 - alpha1 / 2) * (1 - alpha1 / 2);
    const hpFeedback1 = 2 * (1 - alpha1);
    const hpFeedback2 = -((1 - alpha1) * (1 - alpha1));
    const a1 = Math.exp(-1.414 * pi / 10);
    const b1 = 2 * a1 * Math.cos(1.414 * pi / 10);
    const c2 = b1;
    const c3 = -(a1 * a1);
    const c1 = 1 - c2 - c3;

    for (let i = 0; i < n; i++) {
        if (!Number.isFinite(source[i])) {
            continue;
        }
        hp[i] = hpCoef * (source[i] - 2 * nz(source, i - 1) + nz(source, i - 2));
        hp[i] += hpFeedback1 * nz(hp, i - 1) + hpFeedback2 * nz(hp, i - 2);
        filt[i] = c1 * hp[i] + c2 * nz(filt, i - 1) + c3 * nz(filt, i - 2);
    }

    for (let i = 0; i < n; i++) {
        if (!Number.isFinite(filt[i])) {
            continue;
        }
        let highest = filt[i];
        let lowest = filt[i];
        for (let count = 0; count < length; count++) {
            const value = nz(filt, i - count);
            highest = Math.max(highest, value);
            lowest = Math.min(lowest, value);
        }
        const denom = highest - lowest;
        if (!Number.isFinite(denom) || denom === 0) {
            continue;
        }
        const stoc = (filt[i] - lowest) / denom;
        if (Number.isFinite(stoc)) {
            out[i] = c1 * stoc + c2 * nz(out, i - 1) + c3 * nz(out, i - 2);
        }
    }

    return out;
}

function manualReference(source, l1 = 48, l2 = 21, l3 = 9, l4 = 6, trig = 2) {
    const mesa1 = manualMesa(source, l1);
    const mesa2 = manualMesa(source, l2);
    const mesa3 = manualMesa(source, l3);
    const mesa4 = manualMesa(source, l4);
    return {
        mesa_1: mesa1,
        mesa_2: mesa2,
        mesa_3: mesa3,
        mesa_4: mesa4,
        trigger_1: manualSma(mesa1, trig),
        trigger_2: manualSma(mesa2, trig),
        trigger_3: manualSma(mesa3, trig),
        trigger_4: manualSma(mesa4, trig),
    };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('mesa_stochastic_multi_length_js matches manual reference', () => {
    const n = 180;
    const source = new Float64Array(testData.close.slice(0, n));
    const result = wasm.mesa_stochastic_multi_length_js(source, 48, 21, 9, 6, 2);
    const expected = manualReference(source, 48, 21, 9, 6, 2);
    for (const key of Object.keys(expected)) {
        assertArrayClose(result[key], expected[key], 1e-12, `${key} mismatch`);
    }
});

test('mesa_stochastic_multi_length_batch_js first row matches single', () => {
    const n = 160;
    const source = new Float64Array(testData.close.slice(0, n));
    const batch = wasm.mesa_stochastic_multi_length_batch_js(source, {
        length_1_range: [48, 50, 2],
        length_2_range: [21, 21, 0],
        length_3_range: [9, 9, 0],
        length_4_range: [6, 6, 0],
        trigger_length_range: [2, 2, 0],
    });
    const single = wasm.mesa_stochastic_multi_length_js(source, 48, 21, 9, 6, 2);
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, n);
    assert.deepStrictEqual(batch.length_1, [48, 50]);
    assertArrayClose(batch.mesa_1.slice(0, n), single.mesa_1, 1e-12, 'batch mesa_1 mismatch');
    assertArrayClose(batch.trigger_1.slice(0, n), single.trigger_1, 1e-12, 'batch trigger_1 mismatch');
});

test('mesa_stochastic_multi_length_into pointer path matches safe API', () => {
    const n = 160;
    const source = new Float64Array(testData.close.slice(0, n));
    const safe = wasm.mesa_stochastic_multi_length_js(source, 48, 21, 9, 6, 2);

    const sourcePtr = wasm.allocate_f64_array(n);
    const mesa1Ptr = wasm.mesa_stochastic_multi_length_alloc(n);
    const mesa2Ptr = wasm.mesa_stochastic_multi_length_alloc(n);
    const mesa3Ptr = wasm.mesa_stochastic_multi_length_alloc(n);
    const mesa4Ptr = wasm.mesa_stochastic_multi_length_alloc(n);
    const trig1Ptr = wasm.mesa_stochastic_multi_length_alloc(n);
    const trig2Ptr = wasm.mesa_stochastic_multi_length_alloc(n);
    const trig3Ptr = wasm.mesa_stochastic_multi_length_alloc(n);
    const trig4Ptr = wasm.mesa_stochastic_multi_length_alloc(n);

    try {
        wasm.write_f64_array(sourcePtr, source);
        wasm.mesa_stochastic_multi_length_into(
            sourcePtr,
            mesa1Ptr,
            mesa2Ptr,
            mesa3Ptr,
            mesa4Ptr,
            trig1Ptr,
            trig2Ptr,
            trig3Ptr,
            trig4Ptr,
            n,
            48,
            21,
            9,
            6,
            2,
        );
        assertArrayClose(wasm.read_f64_array(mesa1Ptr, n), safe.mesa_1, 1e-12, 'pointer mesa_1 mismatch');
        assertArrayClose(wasm.read_f64_array(mesa2Ptr, n), safe.mesa_2, 1e-12, 'pointer mesa_2 mismatch');
        assertArrayClose(wasm.read_f64_array(mesa3Ptr, n), safe.mesa_3, 1e-12, 'pointer mesa_3 mismatch');
        assertArrayClose(wasm.read_f64_array(mesa4Ptr, n), safe.mesa_4, 1e-12, 'pointer mesa_4 mismatch');
        assertArrayClose(wasm.read_f64_array(trig1Ptr, n), safe.trigger_1, 1e-12, 'pointer trigger_1 mismatch');
        assertArrayClose(wasm.read_f64_array(trig2Ptr, n), safe.trigger_2, 1e-12, 'pointer trigger_2 mismatch');
        assertArrayClose(wasm.read_f64_array(trig3Ptr, n), safe.trigger_3, 1e-12, 'pointer trigger_3 mismatch');
        assertArrayClose(wasm.read_f64_array(trig4Ptr, n), safe.trigger_4, 1e-12, 'pointer trigger_4 mismatch');
    } finally {
        wasm.mesa_stochastic_multi_length_free(mesa1Ptr, n);
        wasm.mesa_stochastic_multi_length_free(mesa2Ptr, n);
        wasm.mesa_stochastic_multi_length_free(mesa3Ptr, n);
        wasm.mesa_stochastic_multi_length_free(mesa4Ptr, n);
        wasm.mesa_stochastic_multi_length_free(trig1Ptr, n);
        wasm.mesa_stochastic_multi_length_free(trig2Ptr, n);
        wasm.mesa_stochastic_multi_length_free(trig3Ptr, n);
        wasm.mesa_stochastic_multi_length_free(trig4Ptr, n);
        wasm.deallocate_f64_array(sourcePtr);
    }
});

test('mesa_stochastic_multi_length_batch_into returns row count', () => {
    const n = 128;
    const source = new Float64Array(testData.close.slice(0, n));
    const sourcePtr = wasm.allocate_f64_array(n);
    const rows = 2;
    const total = rows * n;
    const mesa1Ptr = wasm.mesa_stochastic_multi_length_alloc(total);
    const mesa2Ptr = wasm.mesa_stochastic_multi_length_alloc(total);
    const mesa3Ptr = wasm.mesa_stochastic_multi_length_alloc(total);
    const mesa4Ptr = wasm.mesa_stochastic_multi_length_alloc(total);
    const trig1Ptr = wasm.mesa_stochastic_multi_length_alloc(total);
    const trig2Ptr = wasm.mesa_stochastic_multi_length_alloc(total);
    const trig3Ptr = wasm.mesa_stochastic_multi_length_alloc(total);
    const trig4Ptr = wasm.mesa_stochastic_multi_length_alloc(total);

    try {
        wasm.write_f64_array(sourcePtr, source);
        const returnedRows = wasm.mesa_stochastic_multi_length_batch_into(
            sourcePtr,
            mesa1Ptr,
            mesa2Ptr,
            mesa3Ptr,
            mesa4Ptr,
            trig1Ptr,
            trig2Ptr,
            trig3Ptr,
            trig4Ptr,
            n,
            48,
            50,
            2,
            21,
            21,
            0,
            9,
            9,
            0,
            6,
            6,
            0,
            2,
            2,
            0,
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.mesa_stochastic_multi_length_free(mesa1Ptr, total);
        wasm.mesa_stochastic_multi_length_free(mesa2Ptr, total);
        wasm.mesa_stochastic_multi_length_free(mesa3Ptr, total);
        wasm.mesa_stochastic_multi_length_free(mesa4Ptr, total);
        wasm.mesa_stochastic_multi_length_free(trig1Ptr, total);
        wasm.mesa_stochastic_multi_length_free(trig2Ptr, total);
        wasm.mesa_stochastic_multi_length_free(trig3Ptr, total);
        wasm.mesa_stochastic_multi_length_free(trig4Ptr, total);
        wasm.deallocate_f64_array(sourcePtr);
    }
});

test('mesa_stochastic_multi_length_js rejects invalid period', () => {
    const n = 64;
    const source = new Float64Array(testData.close.slice(0, n));
    assert.throws(() => {
        wasm.mesa_stochastic_multi_length_js(source, 0, 21, 9, 6, 2);
    }, /Invalid period/);
});
