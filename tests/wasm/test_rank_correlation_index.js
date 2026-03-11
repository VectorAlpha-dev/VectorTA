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

test('rank_correlation_index_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 256));

    const result = wasm.rank_correlation_index_js(close, 12);

    assert.strictEqual(result.length, close.length);
    assert(Array.from(result.slice(0, 11)).every(v => isNaN(v)));
    assert(Array.from(result.slice(11)).some(v => !isNaN(v)));
});

test('rank_correlation_index_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 192));
    const safe = wasm.rank_correlation_index_js(close, 12);
    const outPtr = wasm.rank_correlation_index_alloc(close.length);

    try {
        wasm.rank_correlation_index_into_host(close, outPtr, 12);
        const out = wasm.read_f64_array(outPtr, close.length);
        assertArrayClose(out, safe, 1e-10, 'pointer-path mismatch');
    } finally {
        wasm.rank_correlation_index_free(outPtr, close.length);
    }
});

test('rank_correlation_index_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 192));
    const batch = wasm.rank_correlation_index_batch_js(close, {
        length_range: [12, 12, 0],
    });
    const single = wasm.rank_correlation_index_js(close, 12);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.values.length, close.length);
    assert.strictEqual(batch.combos[0].length, 12);
    assertArrayClose(batch.values, single, 1e-10, 'batch mismatch');
});
