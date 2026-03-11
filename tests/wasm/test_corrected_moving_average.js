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

test('corrected_moving_average_js output contract', () => {
    const source = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.corrected_moving_average_js(source, 35);

    assert.strictEqual(result.length, source.length);
    assert(Array.from(result.slice(0, 34)).every(v => isNaN(v)));
    assert(Array.from(result.slice(34)).some(v => !isNaN(v)));
});

test('corrected_moving_average_into_host pointer path matches safe API', () => {
    const source = new Float64Array(testData.close.slice(0, 220));
    const safe = wasm.corrected_moving_average_js(source, 20);
    const outPtr = wasm.corrected_moving_average_alloc(source.length);

    try {
        wasm.corrected_moving_average_into_host(source, outPtr, 20);
        const out = wasm.read_f64_array(outPtr, source.length);
        assertArrayClose(out, safe, 1e-10, 'pointer path mismatch');
    } finally {
        wasm.corrected_moving_average_free(outPtr, source.length);
    }
});

test('corrected_moving_average_batch_js single parameter set matches safe API', () => {
    const source = new Float64Array(testData.close.slice(0, 220));
    const batch = wasm.corrected_moving_average_batch_js(source, {
        period_range: [35, 35, 0],
    });
    const single = wasm.corrected_moving_average_js(source, 35);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, source.length);
    assert.strictEqual(batch.periods.length, 1);
    assert.strictEqual(batch.periods[0], 35);
    assert.strictEqual(batch.values.length, source.length);
    assertArrayClose(batch.values, single, 1e-10, 'batch mismatch');
});
