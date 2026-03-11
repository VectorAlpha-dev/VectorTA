import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function referenceLco(data, period) {
    const out = new Float64Array(data.length);
    out.fill(NaN);
    let first = -1;
    for (let i = 0; i < data.length; i++) {
        if (!Number.isNaN(data[i])) {
            first = i;
            break;
        }
    }
    if (first === -1) {
        throw new Error('All values are NaN');
    }
    if (period === 0 || period > data.length) {
        throw new Error('Invalid period');
    }
    if (data.length - first <= period + 1) {
        throw new Error('Not enough valid data');
    }

    const varX = (period * period - 1) / 12;
    for (let end = first + period + 1; end < data.length; end++) {
        const start = end + 1 - period;
        let sum = 0;
        let sum2 = 0;
        let weighted = 0;
        let hasNaN = false;
        for (let i = 0; i < period; i++) {
            const value = data[start + i];
            if (Number.isNaN(value)) {
                hasNaN = true;
                break;
            }
            sum += value;
            sum2 += value * value;
            weighted += (i + 1) * value;
        }
        if (hasNaN || varX <= 0) {
            continue;
        }
        const centered = weighted - ((period + 1) * 0.5) * sum;
        const mean = sum / period;
        const varY = sum2 / period - mean * mean;
        if (varY <= 0) {
            continue;
        }
        const corr = (centered / period) / Math.sqrt(varY * varX);
        out[end] = Math.max(-1, Math.min(1, corr));
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

test('LCO matches local reference', () => {
    const data = new Float64Array([
        12.0, 14.0, 13.0, 15.0, 16.5, 18.0, 17.0, 19.5,
        21.0, 20.5, 22.5, 24.0, 23.0, 25.5, 26.0, 28.0,
    ]);
    const result = wasm.linear_correlation_oscillator_js(data, 5);
    const expected = referenceLco(data, 5);
    assertArrayClose(result, expected);
});

test('LCO increasing and decreasing sequences hit +/-1', () => {
    const inc = Float64Array.from({ length: 24 }, (_, i) => i);
    const dec = Float64Array.from({ length: 24 }, (_, i) => -i);
    const incResult = wasm.linear_correlation_oscillator_js(inc, 6);
    const decResult = wasm.linear_correlation_oscillator_js(dec, 6);
    for (let i = 0; i < 7; i++) {
        assert(Number.isNaN(incResult[i]));
        assert(Number.isNaN(decResult[i]));
    }
    for (let i = 7; i < incResult.length; i++) {
        assert(Math.abs(incResult[i] - 1) <= 1e-12);
        assert(Math.abs(decResult[i] + 1) <= 1e-12);
    }
});

test('LCO batch matches single', () => {
    const data = new Float64Array([
        12.0, 14.0, 13.0, 15.0, 16.5, 18.0, 17.0, 19.5,
        21.0, 20.5, 22.5, 24.0, 23.0, 25.5, 26.0, 28.0,
    ]);
    const batch = wasm.linear_correlation_oscillator_batch(data, {
        period_range: [5, 7, 1],
    });
    assert.strictEqual(batch.rows, 3);
    assert.strictEqual(batch.cols, data.length);
    assert.strictEqual(batch.combos.length, 3);
    for (let row = 0; row < batch.rows; row++) {
        const period = batch.combos[row].period;
        const single = wasm.linear_correlation_oscillator_js(data, period);
        const start = row * batch.cols;
        const end = start + batch.cols;
        assertArrayClose(batch.values.slice(start, end), single);
    }
});

test('LCO zero-copy API matches regular API', () => {
    const data = new Float64Array([
        12.0, 14.0, 13.0, 15.0, 16.5, 18.0, 17.0, 19.5,
        21.0, 20.5, 22.5, 24.0, 23.0, 25.5, 26.0, 28.0,
    ]);
    const expected = wasm.linear_correlation_oscillator_js(data, 5);
    const ptr = wasm.linear_correlation_oscillator_alloc(data.length);
    try {
        const mem = new Float64Array(wasm.__wasm.memory.buffer, ptr, data.length);
        mem.set(data);
        wasm.linear_correlation_oscillator_into(ptr, ptr, data.length, 5);
        assertArrayClose(new Float64Array(mem), expected);
    } finally {
        wasm.linear_correlation_oscillator_free(ptr, data.length);
    }
});

test('LCO invalid inputs throw', () => {
    assert.throws(
        () => wasm.linear_correlation_oscillator_js(new Float64Array([1, 2, 3]), 0),
        /Invalid period/
    );
    assert.throws(
        () => wasm.linear_correlation_oscillator_js(new Float64Array([1, 2, 3, 4]), 4),
        /Not enough valid data|Invalid period/
    );
    const allNaN = new Float64Array(10);
    allNaN.fill(NaN);
    assert.throws(
        () => wasm.linear_correlation_oscillator_js(allNaN, 5),
        /All values are NaN/
    );
});
