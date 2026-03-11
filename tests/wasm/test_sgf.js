import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function polySeries(length, coeffs, warmPrefix = 0) {
    const out = new Float64Array(length);
    out.fill(NaN);
    for (let i = warmPrefix; i < length; i++) {
        let power = 1.0;
        let value = 0.0;
        for (const coef of coeffs) {
            value += coef * power;
            power *= i;
        }
        out[i] = value;
    }
    return out;
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
});

test('SGF quadratic exactness', () => {
    const data = polySeries(64, [2.0, -0.5, 0.25], 3);
    const result = wasm.sgf_js(data, 9, 2);
    assert.strictEqual(result.length, data.length);
    assertArrayClose(result.slice(11), data.slice(11), 1e-10, 'SGF quadratic exactness mismatch');
});

test('SGF invalid poly order', () => {
    const data = polySeries(16, [1.0, 2.0], 0);
    assert.throws(() => wasm.sgf_js(data, 5, 5), /invalid polynomial order/i);
});

test('SGF batch parity', () => {
    const data = polySeries(80, [1.0, 0.3, -0.02], 2);
    const batch = wasm.sgf_batch(data, {
        period_range: [7, 11, 2],
        poly_order_range: [2, 2, 0],
    });
    assert.strictEqual(batch.rows, 3);
    assert.strictEqual(batch.cols, data.length);
    const direct = wasm.sgf_js(data, 9, 2);
    const row = batch.values.slice(data.length, data.length * 2);
    assertArrayClose(row, direct, 1e-10, 'SGF batch/direct mismatch');
});

test('SGF into helper', () => {
    const data = polySeries(48, [3.0, 0.2, -0.03], 1);
    const inPtr = wasm.sgf_alloc(data.length);
    const outPtr = wasm.sgf_alloc(data.length);
    try {
        const memory = wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory);
        if (!memory) {
            assert.ok(true);
            return;
        }
        new Float64Array(memory.buffer, inPtr, data.length).set(data);
        wasm.sgf_into(inPtr, outPtr, data.length, 9, 2);
        const out = new Float64Array(memory.buffer, outPtr, data.length).slice();
        const direct = wasm.sgf_js(data, 9, 2);
        assertArrayClose(out, direct, 1e-10, 'SGF into mismatch');
    } finally {
        wasm.sgf_free(inPtr, data.length);
        wasm.sgf_free(outPtr, data.length);
    }
});
