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

test('stochastic_connors_rsi_js output contract', () => {
    const source = new Float64Array(testData.close.slice(0, 256));

    const result = wasm.stochastic_connors_rsi_js(source, 3, 3, 3, 3, 2, 100);

    assert(result.k);
    assert(result.d);
    assert.strictEqual(result.k.length, source.length);
    assert.strictEqual(result.d.length, source.length);
    assert(Array.from(result.k.slice(0, 104)).every(v => isNaN(v)));
    assert(Array.from(result.d.slice(0, 106)).every(v => isNaN(v)));
    assert(Array.from(result.k.slice(104)).some(v => !isNaN(v)));
    assert(Array.from(result.d.slice(106)).some(v => !isNaN(v)));
});

test('stochastic_connors_rsi_into_host pointer path matches safe API', () => {
    const source = new Float64Array(testData.close.slice(0, 220));
    const safe = wasm.stochastic_connors_rsi_js(source, 5, 4, 3, 4, 3, 30);
    const outPtr = wasm.stochastic_connors_rsi_alloc(source.length);

    try {
        wasm.stochastic_connors_rsi_into_host(source, outPtr, 5, 4, 3, 4, 3, 30);
        const out = wasm.read_f64_array(outPtr, 2 * source.length);
        const k = out.slice(0, source.length);
        const d = out.slice(source.length, 2 * source.length);

        assertArrayClose(k, safe.k, 1e-10, 'k mismatch');
        assertArrayClose(d, safe.d, 1e-10, 'd mismatch');
    } finally {
        wasm.stochastic_connors_rsi_free(outPtr, source.length);
    }
});

test('stochastic_connors_rsi_batch_js single parameter set matches safe API', () => {
    const source = new Float64Array(testData.close.slice(0, 220));
    const batch = wasm.stochastic_connors_rsi_batch_js(source, {
        stoch_length_range: [3, 3, 0],
        smooth_k_range: [3, 3, 0],
        smooth_d_range: [3, 3, 0],
        rsi_length_range: [3, 3, 0],
        updown_length_range: [2, 2, 0],
        roc_length_range: [20, 20, 0],
    });
    const single = wasm.stochastic_connors_rsi_js(source, 3, 3, 3, 3, 2, 20);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, source.length);
    assert.strictEqual(batch.k.length, source.length);
    assert.strictEqual(batch.d.length, source.length);
    assert.strictEqual(batch.combos[0].stoch_length, 3);
    assert.strictEqual(batch.combos[0].smooth_k, 3);
    assert.strictEqual(batch.combos[0].smooth_d, 3);
    assert.strictEqual(batch.combos[0].rsi_length, 3);
    assert.strictEqual(batch.combos[0].updown_length, 2);
    assert.strictEqual(batch.combos[0].roc_length, 20);
    assertArrayClose(batch.k, single.k, 1e-10, 'batch k mismatch');
    assertArrayClose(batch.d, single.d, 1e-10, 'batch d mismatch');
});
