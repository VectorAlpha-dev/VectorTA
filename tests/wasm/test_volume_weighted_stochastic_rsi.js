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

test('volume_weighted_stochastic_rsi_js output contract', () => {
    const source = new Float64Array(testData.close.slice(0, 256));
    const volume = new Float64Array(testData.volume.slice(0, 256));

    const result = wasm.volume_weighted_stochastic_rsi_js(source, volume, 14, 14, 3, 3, 'WSMA');

    assert(result.k);
    assert(result.d);
    assert.strictEqual(result.k.length, source.length);
    assert.strictEqual(result.d.length, source.length);
    assert(Array.from(result.k.slice(0, 29)).every(v => isNaN(v)));
    assert(Array.from(result.d.slice(0, 31)).every(v => isNaN(v)));
    assert(Array.from(result.k.slice(29)).some(v => !isNaN(v)));
    assert(Array.from(result.d.slice(31)).some(v => !isNaN(v)));
});

test('volume_weighted_stochastic_rsi_into_host pointer path matches safe API', () => {
    const source = new Float64Array(testData.close.slice(0, 192));
    const volume = new Float64Array(testData.volume.slice(0, 192));
    const safe = wasm.volume_weighted_stochastic_rsi_js(source, volume, 12, 10, 4, 3, 'EMA');
    const outPtr = wasm.volume_weighted_stochastic_rsi_alloc(source.length);

    try {
        wasm.volume_weighted_stochastic_rsi_into_host(
            source,
            volume,
            outPtr,
            12,
            10,
            4,
            3,
            'EMA'
        );
        const out = wasm.read_f64_array(outPtr, 2 * source.length);
        const k = out.slice(0, source.length);
        const d = out.slice(source.length, 2 * source.length);

        assertArrayClose(k, safe.k, 1e-10, 'k mismatch');
        assertArrayClose(d, safe.d, 1e-10, 'd mismatch');
    } finally {
        wasm.volume_weighted_stochastic_rsi_free(outPtr, source.length);
    }
});

test('volume_weighted_stochastic_rsi_batch_js single parameter set matches safe API', () => {
    const source = new Float64Array(testData.close.slice(0, 192));
    const volume = new Float64Array(testData.volume.slice(0, 192));
    const batch = wasm.volume_weighted_stochastic_rsi_batch_js(source, volume, {
        rsi_length_range: [14, 14, 0],
        stoch_length_range: [14, 14, 0],
        k_length_range: [3, 3, 0],
        d_length_range: [3, 3, 0],
        ma_type: 'VWMA',
    });
    const single = wasm.volume_weighted_stochastic_rsi_js(source, volume, 14, 14, 3, 3, 'VWMA');

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, source.length);
    assert.strictEqual(batch.k.length, source.length);
    assert.strictEqual(batch.d.length, source.length);
    assert.strictEqual(batch.combos[0].rsi_length, 14);
    assert.strictEqual(batch.combos[0].stoch_length, 14);
    assert.strictEqual(batch.combos[0].k_length, 3);
    assert.strictEqual(batch.combos[0].d_length, 3);
    assert.strictEqual(batch.combos[0].ma_type, 'VWMA');
    assertArrayClose(batch.k, single.k, 1e-10, 'batch k mismatch');
    assertArrayClose(batch.d, single.d, 1e-10, 'batch d mismatch');
});
