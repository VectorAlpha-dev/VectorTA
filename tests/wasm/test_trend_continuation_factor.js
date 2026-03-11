import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose, isNaN } from './test_utils.js';

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

test('trend_continuation_factor_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.trend_continuation_factor_js(close, 35);

    assert(result.plus_tcf);
    assert(result.minus_tcf);
    assert.strictEqual(result.plus_tcf.length, close.length);
    assert.strictEqual(result.minus_tcf.length, close.length);
    assert(Array.from(result.plus_tcf.slice(0, 35)).every(v => isNaN(v)));
    assert(Array.from(result.minus_tcf.slice(0, 35)).every(v => isNaN(v)));
    assert(Array.from(result.plus_tcf.slice(35)).some(v => !isNaN(v)));
    assert(Array.from(result.minus_tcf.slice(35)).some(v => !isNaN(v)));
});

test('trend_continuation_factor_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 192));
    const safe = wasm.trend_continuation_factor_js(close, 20);
    const outPtr = wasm.trend_continuation_factor_alloc(close.length);

    try {
        wasm.trend_continuation_factor_into_host(close, outPtr, 20);
        const out = wasm.read_f64_array(outPtr, 2 * close.length);
        const plus = out.slice(0, close.length);
        const minus = out.slice(close.length, 2 * close.length);

        assertArrayClose(plus, safe.plus_tcf, 1e-10, 'plus_tcf mismatch');
        assertArrayClose(minus, safe.minus_tcf, 1e-10, 'minus_tcf mismatch');
    } finally {
        wasm.trend_continuation_factor_free(outPtr, close.length);
    }
});

test('trend_continuation_factor_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 192));
    const batch = wasm.trend_continuation_factor_batch_js(close, {
        length_range: [12, 12, 0],
    });
    const single = wasm.trend_continuation_factor_js(close, 12);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.plus_tcf.length, close.length);
    assert.strictEqual(batch.minus_tcf.length, close.length);
    assert.strictEqual(batch.combos[0].length, 12);
    assertArrayClose(batch.plus_tcf, single.plus_tcf, 1e-10, 'batch plus_tcf mismatch');
    assertArrayClose(batch.minus_tcf, single.minus_tcf, 1e-10, 'batch minus_tcf mismatch');
});
