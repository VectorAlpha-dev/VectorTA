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

test('pretty_good_oscillator_js output contract', () => {
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const source = new Float64Array(Array.from(close, (_, i) => (high[i] + low[i]) * 0.5));

    const result = wasm.pretty_good_oscillator_js(high, low, close, source, 5);

    assert.strictEqual(result.length, close.length);
    assert(Array.from(result.slice(0, 4)).every(v => isNaN(v)));
    assert(Array.from(result.slice(4)).some(v => !isNaN(v)));
});

test('pretty_good_oscillator_into pointer path matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 192));
    const low = new Float64Array(testData.low.slice(0, 192));
    const close = new Float64Array(testData.close.slice(0, 192));
    const source = new Float64Array(testData.close.slice(0, 192));
    const safe = wasm.pretty_good_oscillator_js(high, low, close, source, 14);
    const outPtr = wasm.pretty_good_oscillator_alloc(close.length);

    try {
        wasm.pretty_good_oscillator_into_host(high, low, close, source, outPtr, 14);
        const out = wasm.read_f64_array(outPtr, close.length);
        assertArrayClose(out, safe, 1e-10, 'pointer-path mismatch');
    } finally {
        wasm.pretty_good_oscillator_free(outPtr, close.length);
    }
});

test('pretty_good_oscillator_batch_js single parameter set matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 192));
    const low = new Float64Array(testData.low.slice(0, 192));
    const close = new Float64Array(testData.close.slice(0, 192));
    const source = new Float64Array(testData.close.slice(0, 192));
    const batch = wasm.pretty_good_oscillator_batch_js(high, low, close, source, {
        length_range: [14, 14, 0],
    });
    const single = wasm.pretty_good_oscillator_js(high, low, close, source, 14);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.values.length, close.length);
    assert.strictEqual(batch.combos[0].length, 14);
    assertArrayClose(batch.values, single, 1e-10, 'batch mismatch');
});
