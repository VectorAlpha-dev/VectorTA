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

test('insync_index_js output contract', () => {
    const high = new Float64Array(testData.high.slice(0, 320));
    const low = new Float64Array(testData.low.slice(0, 320));
    const close = new Float64Array(testData.close.slice(0, 320));
    const volume = new Float64Array(testData.volume.slice(0, 320));

    const result = wasm.insync_index_js(
        high, low, close, volume,
        10000, 14, 12, 26, 20, 20, 2.0, 14, 18, 10, 14, 14, 3, 1, 10
    );

    assert.strictEqual(result.length, close.length);
    assert(!isNaN(result[0]));
    assert(Array.from(result.slice(25)).some(v => !isNaN(v)));
});

test('insync_index_into_host pointer path matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 192));
    const low = new Float64Array(testData.low.slice(0, 192));
    const close = new Float64Array(testData.close.slice(0, 192));
    const volume = new Float64Array(testData.volume.slice(0, 192));
    const safe = wasm.insync_index_js(
        high, low, close, volume,
        10000, 14, 12, 26, 20, 20, 2.0, 14, 18, 10, 14, 14, 3, 1, 10
    );
    const outPtr = wasm.insync_index_alloc(close.length);

    try {
        wasm.insync_index_into_host(
            high, low, close, volume, outPtr,
            10000, 14, 12, 26, 20, 20, 2.0, 14, 18, 10, 14, 14, 3, 1, 10
        );
        const flat = wasm.read_f64_array(outPtr, close.length);
        assertArrayClose(flat, safe, 1e-10, 'into_host mismatch');
    } finally {
        wasm.insync_index_free(outPtr, close.length);
    }
});

test('insync_index_batch_js single parameter set matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 192));
    const low = new Float64Array(testData.low.slice(0, 192));
    const close = new Float64Array(testData.close.slice(0, 192));
    const volume = new Float64Array(testData.volume.slice(0, 192));
    const batch = wasm.insync_index_batch_js(high, low, close, volume, {});
    const single = wasm.insync_index_js(
        high, low, close, volume,
        10000, 14, 12, 26, 20, 20, 2.0, 14, 18, 10, 14, 14, 3, 1, 10
    );

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.values.length, close.length);
    assertArrayClose(batch.values, single, 1e-10, 'batch mismatch');
});
