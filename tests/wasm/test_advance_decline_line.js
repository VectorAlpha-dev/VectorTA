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
    try {
        const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
        const importPath = process.platform === 'win32'
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-bindgen target/wasm32-unknown-unknown/release/vector_ta.wasm --out-dir pkg --target nodejs" first');
        throw error;
    }

    testData = loadTestData();
});

test('advance_decline_line_js computes exact cumulative sums', () => {
    const data = new Float64Array([1, 2, 3, 4]);
    const result = wasm.advance_decline_line_js(data);
    assert.deepStrictEqual(Array.from(result), [1, 3, 6, 10]);
});

test('advance_decline_line_js rejects all-NaN input', () => {
    const data = new Float64Array([NaN, NaN, NaN]);
    assert.throws(() => {
        wasm.advance_decline_line_js(data);
    }, /All values are NaN/);
});

test('advance_decline_line_into pointer path matches safe API', () => {
    const data = new Float64Array(testData.close.slice(0, 256));
    const safe = wasm.advance_decline_line_js(data);

    const dataPtr = wasm.copy_f64_array(data);
    const outPtr = wasm.advance_decline_line_alloc(data.length);

    try {
        wasm.advance_decline_line_into(dataPtr, outPtr, data.length);
        const values = wasm.read_f64_array(outPtr, data.length);
        assertArrayClose(values, safe, 1e-10, 'pointer-path mismatch');
    } finally {
        wasm.advance_decline_line_free(outPtr, data.length);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('advance_decline_line_js resets after NaN', () => {
    const data = new Float64Array([1, 2, NaN, 3, 4]);
    const result = wasm.advance_decline_line_js(data);
    assert.strictEqual(result[0], 1);
    assert.strictEqual(result[1], 3);
    assert(isNaN(result[2]));
    assert.strictEqual(result[3], 3);
    assert.strictEqual(result[4], 7);
});

test('advance_decline_line_batch_js single row matches safe API', () => {
    const data = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.advance_decline_line_batch_js(data, {});
    const single = wasm.advance_decline_line_js(data);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, data.length);
    assert.strictEqual(batch.values.length, data.length);
    assert.strictEqual(batch.combos.length, 1);
    assertArrayClose(batch.values, single, 1e-10, 'batch mismatch');
});

test('advance_decline_line_batch_into matches safe API', () => {
    const data = new Float64Array(testData.close.slice(0, 160));
    const safe = wasm.advance_decline_line_js(data);
    const dataPtr = wasm.copy_f64_array(data);
    const outPtr = wasm.advance_decline_line_alloc(data.length);

    try {
        const rows = wasm.advance_decline_line_batch_into(dataPtr, outPtr, data.length);
        assert.strictEqual(rows, 1);
        const values = wasm.read_f64_array(outPtr, data.length);
        assertArrayClose(values, safe, 1e-10, 'batch-into mismatch');
    } finally {
        wasm.advance_decline_line_free(outPtr, data.length);
        wasm.deallocate_f64_array(dataPtr);
    }
});
