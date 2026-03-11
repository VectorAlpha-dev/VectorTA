import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function quadraticData(size) {
    const out = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        out[i] = i * i;
    }
    return out;
}

function cubicData(size) {
    const out = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        const x = i;
        out[i] = x * x * x - 2 * x * x + 3 * x + 1;
    }
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

test('Polynomial Regression Extrapolation quadratic exactness', () => {
    const data = quadraticData(24);
    const result = wasm.polynomial_regression_extrapolation_js(data, 5, 2, 2);

    const expected = new Float64Array(data.length);
    expected.fill(Number.NaN);
    for (let idx = 4; idx < data.length; idx++) {
        const x = idx + 2;
        expected[idx] = x * x;
    }

    assertArrayClose(result, expected, 1e-9, 'quadratic extrapolation mismatch');
});

test('Polynomial Regression Extrapolation invalid params', () => {
    const data = quadraticData(16);

    assert.throws(() => {
        wasm.polynomial_regression_extrapolation_js(data, 4, 1, 9);
    }, /Invalid degree/);

    assert.throws(() => {
        wasm.polynomial_regression_extrapolation_js(data, 3, 1, 3);
    }, /Degree exceeds length/);
});

test('Polynomial Regression Extrapolation NaN gap semantics', () => {
    const data = new Float64Array([
        Number.NaN, 1.0, 4.0, 9.0, 16.0, 25.0,
        Number.NaN, 64.0, 81.0, 100.0, 121.0, 144.0,
    ]);
    const result = wasm.polynomial_regression_extrapolation_js(data, 3, 1, 2);

    assert(result.slice(0, 3).every(v => isNaN(v)));
    assert(Math.abs(result[3] - 16.0) <= 1e-9);
    assert(Math.abs(result[4] - 25.0) <= 1e-9);
    assert(Math.abs(result[5] - 36.0) <= 1e-9);
    assert(result.slice(6, 9).every(v => isNaN(v)));
    assert(Math.abs(result[9] - 121.0) <= 1e-9);
    assert(Math.abs(result[10] - 144.0) <= 1e-9);
    assert(Math.abs(result[11] - 169.0) <= 1e-9);
});

test('Polynomial Regression Extrapolation fast API matches safe API', () => {
    const data = cubicData(24);
    const len = data.length;
    const ptr = wasm.polynomial_regression_extrapolation_alloc(len);

    try {
        const memory = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        memory.set(data);
        wasm.polynomial_regression_extrapolation_into(ptr, ptr, len, 6, 2, 3);
        const result = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        const safe = wasm.polynomial_regression_extrapolation_js(data, 6, 2, 3);
        assertArrayClose(new Float64Array(result), safe, 1e-12, 'fast API mismatch');
    } finally {
        wasm.polynomial_regression_extrapolation_free(ptr, len);
    }
});

test('Polynomial Regression Extrapolation batch matches single', () => {
    const data = quadraticData(22);
    const result = wasm.polynomial_regression_extrapolation_batch(data, {
        length_range: [4, 5, 1],
        extrapolate_range: [1, 2, 1],
        degree_range: [1, 2, 1],
    });

    assert.strictEqual(result.rows, 8);
    assert.strictEqual(result.cols, data.length);
    assert.strictEqual(result.values.length, result.rows * result.cols);

    for (let row = 0; row < result.rows; row++) {
        const combo = result.combos[row];
        const single = wasm.polynomial_regression_extrapolation_js(
            data,
            combo.length,
            combo.extrapolate,
            combo.degree,
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
