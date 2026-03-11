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

test('gopalakrishnan_range_index_js output contract', () => {
    const high = new Float64Array(testData.high.slice(0, 512));
    const low = new Float64Array(testData.low.slice(0, 512));
    const result = wasm.gopalakrishnan_range_index_js(high, low, 5);

    assert.strictEqual(result.length, high.length);
    const firstValid = result.findIndex(v => !isNaN(v));
    assert(firstValid >= 4, `unexpected first valid index: ${firstValid}`);
    assert(result.some(v => !isNaN(v)), 'output should contain valid values');

    const tailStart = Math.min(firstValid + 32, result.length);
    for (let i = tailStart; i < result.length; i++) {
        assert(!isNaN(result[i]), `unexpected NaN at ${i}`);
    }
});

test('gopalakrishnan_range_index_js rejects invalid parameters', () => {
    const high = new Float64Array(testData.high.slice(0, 128));
    const low = new Float64Array(testData.low.slice(0, 128));

    assert.throws(() => {
        wasm.gopalakrishnan_range_index_js(high, low, 1);
    }, /Invalid length/);

    assert.throws(() => {
        wasm.gopalakrishnan_range_index_js(high.subarray(0, 100), low, 5);
    }, /Inconsistent slice lengths|length mismatch/);
});

test('gopalakrishnan_range_index_into pointer path matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const safe = wasm.gopalakrishnan_range_index_js(high, low, 7);

    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const outPtr = wasm.gopalakrishnan_range_index_alloc(high.length);

    try {
        wasm.gopalakrishnan_range_index_into(highPtr, lowPtr, outPtr, high.length, 7);
        const values = wasm.read_f64_array(outPtr, high.length);
        assertArrayClose(values, safe, 1e-10, 'pointer-path mismatch');
    } finally {
        wasm.gopalakrishnan_range_index_free(outPtr, high.length);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
    }
});

test('gopalakrishnan_range_index_batch_js single parameter set matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const batch = wasm.gopalakrishnan_range_index_batch_js(high, low, {
        length_range: [5, 5, 0],
    });
    const single = wasm.gopalakrishnan_range_index_js(high, low, 5);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, high.length);
    assert.deepStrictEqual(batch.lengths, [5]);
    assertArrayClose(batch.values, single, 1e-10, 'batch mismatch');
});

test('gopalakrishnan_range_index_batch_js metadata matches requested grid', () => {
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const batch = wasm.gopalakrishnan_range_index_batch_js(high, low, {
        length_range: [5, 9, 2],
    });

    assert.strictEqual(batch.rows, 3);
    assert.strictEqual(batch.cols, high.length);
    assert.strictEqual(batch.values.length, 3 * high.length);
    assert.deepStrictEqual(batch.lengths, [5, 7, 9]);

    const single = wasm.gopalakrishnan_range_index_js(high, low, 5);
    assertArrayClose(batch.values.slice(0, high.length), single, 1e-10, 'first-row mismatch');
});

test('gopalakrishnan_range_index_batch_js rejects invalid config', () => {
    const high = new Float64Array(testData.high.slice(0, 64));
    const low = new Float64Array(testData.low.slice(0, 64));

    assert.throws(() => {
        wasm.gopalakrishnan_range_index_batch_js(high, low, {
            length_range: [1, 5, 1],
        });
    }, /Invalid length/);
});
