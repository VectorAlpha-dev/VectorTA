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

test('ehlers_detrending_filter_js output contract', () => {
    const data = new Float64Array(testData.close.slice(0, 320));
    const result = wasm.ehlers_detrending_filter_js(data, 10);

    assert.strictEqual(result.edf.length, data.length);
    assert.strictEqual(result.signal.length, data.length);
    assert(Array.from(result.edf.slice(0, 9)).every(v => isNaN(v)));
    assert(Array.from(result.signal.slice(0, 9)).every(v => isNaN(v)));
    assert(Array.from(result.edf.slice(9)).some(v => !isNaN(v)));
    assert(Array.from(result.signal.slice(9)).some(v => !isNaN(v)));
});

test('ehlers_detrending_filter_into_host pointer path matches safe API', () => {
    const data = new Float64Array(testData.close.slice(0, 320));
    const safe = wasm.ehlers_detrending_filter_js(data, 12);
    const outPtr = wasm.ehlers_detrending_filter_alloc(data.length);

    try {
        wasm.ehlers_detrending_filter_into_host(data, outPtr, 12);
        const flat = wasm.read_f64_array(outPtr, data.length * 2);
        const edf = flat.slice(0, data.length);
        const signal = flat.slice(data.length);
        assertArrayClose(edf, safe.edf, 1e-10, 'edf pointer-path mismatch');
        assertArrayClose(signal, safe.signal, 1e-10, 'signal pointer-path mismatch');
    } finally {
        wasm.ehlers_detrending_filter_free(outPtr, data.length);
    }
});

test('ehlers_detrending_filter_batch_js single parameter set matches safe API', () => {
    const data = new Float64Array(testData.close.slice(0, 320));
    const batch = wasm.ehlers_detrending_filter_batch_js(data, {
        length_range: [11, 11, 0],
    });
    const single = wasm.ehlers_detrending_filter_js(data, 11);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, data.length);
    assert.strictEqual(batch.edf.length, data.length);
    assert.strictEqual(batch.signal.length, data.length);
    assert.strictEqual(batch.combos[0].length, 11);
    assertArrayClose(batch.edf, single.edf, 1e-10, 'edf batch mismatch');
    assertArrayClose(batch.signal, single.signal, 1e-10, 'signal batch mismatch');
});
