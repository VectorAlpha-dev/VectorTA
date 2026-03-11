import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

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

test('linear_regression_intensity_js output contract', () => {
    const source = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.linear_regression_intensity_js(source, 12, 90.0, 90);

    assert.strictEqual(result.length, source.length);
    assert(Array.from(result).some(v => !Number.isNaN(v)));
});

test('linear_regression_intensity_into_host pointer path matches safe API', () => {
    const source = new Float64Array(testData.close.slice(0, 192));
    const safe = wasm.linear_regression_intensity_js(source, 12, 90.0, 24);
    const outPtr = wasm.linear_regression_intensity_alloc(source.length);

    try {
        wasm.linear_regression_intensity_into_host(source, outPtr, 12, 90.0, 24);
        const out = wasm.read_f64_array(outPtr, source.length);
        assertArrayClose(out, safe, 1e-10, 'pointer-path mismatch');
    } finally {
        wasm.linear_regression_intensity_free(outPtr, source.length);
    }
});

test('linear_regression_intensity_batch_js single parameter set matches safe API', () => {
    const source = new Float64Array(testData.close.slice(0, 192));
    const single = wasm.linear_regression_intensity_js(source, 12, 90.0, 24);
    const batch = wasm.linear_regression_intensity_batch_js(source, {
        lookback_period_range: [12, 12, 0],
        range_tolerance_range: [90.0, 90.0, 0.0],
        linreg_length_range: [24, 24, 0],
    });

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, source.length);
    assert.strictEqual(batch.values.length, source.length);
    assert.strictEqual(batch.combos[0].lookback_period, 12);
    assertArrayClose(batch.values, single, 1e-10, 'batch mismatch');
});

test('linear_regression_intensity invalid tolerance throws', () => {
    const source = new Float64Array(testData.close.slice(0, 64));
    assert.throws(
        () => wasm.linear_regression_intensity_js(source, 12, 120.0, 24),
        /Invalid range tolerance/
    );
});
