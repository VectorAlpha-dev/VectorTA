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

test('market_meanness_index_js output contract', () => {
    const open = new Float64Array(testData.open.slice(0, 720));
    const close = new Float64Array(testData.close.slice(0, 720));

    const result = wasm.market_meanness_index_js(open, close, 300, 'Price');

    assert.strictEqual(result.mmi.length, close.length);
    assert.strictEqual(result.mmi_smoothed.length, close.length);
    assert(Array.from(result.mmi.slice(0, 299)).every(v => isNaN(v)));
    assert(Array.from(result.mmi_smoothed.slice(0, 598)).every(v => isNaN(v)));
    assert(Array.from(result.mmi.slice(299)).some(v => !isNaN(v)));
    assert(Array.from(result.mmi_smoothed.slice(598)).some(v => !isNaN(v)));
});

test('market_meanness_index_into_host pointer path matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 320));
    const close = new Float64Array(testData.close.slice(0, 320));
    const safe = wasm.market_meanness_index_js(open, close, 40, 'Change');
    const outPtr = wasm.market_meanness_index_alloc(close.length);

    try {
        wasm.market_meanness_index_into_host(open, close, outPtr, 40, 'Change');
        const flat = wasm.read_f64_array(outPtr, close.length * 2);
        const mmi = flat.slice(0, close.length);
        const smoothed = flat.slice(close.length);
        assertArrayClose(mmi, safe.mmi, 1e-10, 'mmi pointer-path mismatch');
        assertArrayClose(smoothed, safe.mmi_smoothed, 1e-10, 'mmi_smoothed pointer-path mismatch');
    } finally {
        wasm.market_meanness_index_free(outPtr, close.length);
    }
});

test('market_meanness_index_batch_js single parameter set matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 320));
    const close = new Float64Array(testData.close.slice(0, 320));
    const batch = wasm.market_meanness_index_batch_js(open, close, {
        length_range: [40, 40, 0],
        source_mode: 'Price',
    });
    const single = wasm.market_meanness_index_js(open, close, 40, 'Price');

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.mmi.length, close.length);
    assert.strictEqual(batch.mmi_smoothed.length, close.length);
    assert.strictEqual(batch.combos[0].length, 40);
    assert.strictEqual(batch.combos[0].source_mode, 'Price');
    assertArrayClose(batch.mmi, single.mmi, 1e-10, 'mmi batch mismatch');
    assertArrayClose(batch.mmi_smoothed, single.mmi_smoothed, 1e-10, 'mmi_smoothed batch mismatch');
});
