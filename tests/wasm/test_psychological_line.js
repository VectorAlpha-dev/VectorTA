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

test('psychological_line_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 256));

    const result = wasm.psychological_line_js(close, 20);

    assert.strictEqual(result.length, close.length);
    assert(Array.from(result.slice(0, 20)).every(v => isNaN(v)));
    assert(Array.from(result.slice(20)).some(v => !isNaN(v)));
});

test('psychological_line_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 192));
    const safe = wasm.psychological_line_js(close, 20);
    const outPtr = wasm.psychological_line_alloc(close.length);

    try {
        wasm.psychological_line_into_host(close, outPtr, 20);
        const out = wasm.read_f64_array(outPtr, close.length);
        assertArrayClose(out, safe, 1e-10, 'pointer-path mismatch');
    } finally {
        wasm.psychological_line_free(outPtr, close.length);
    }
});

test('psychological_line_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 192));
    const batch = wasm.psychological_line_batch_js(close, {
        length_range: [20, 20, 0],
    });
    const single = wasm.psychological_line_js(close, 20);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.values.length, close.length);
    assert.strictEqual(batch.combos[0].length, 20);
    assertArrayClose(batch.values, single, 1e-10, 'batch mismatch');
});
