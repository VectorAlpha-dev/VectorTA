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

test('vertical_horizontal_filter_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 512));
    const result = wasm.vertical_horizontal_filter_js(close, 28);

    assert.strictEqual(result.length, close.length);
    const firstValid = result.findIndex(v => !isNaN(v));
    assert(firstValid >= 28, `unexpected first valid index: ${firstValid}`);
    assert(result.some(v => !isNaN(v)), 'output should contain valid values');

    const tailStart = Math.min(firstValid + 32, result.length);
    for (let i = tailStart; i < result.length; i++) {
        assert(!isNaN(result[i]), `unexpected NaN at ${i}`);
    }
});

test('vertical_horizontal_filter_js rejects invalid parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 128));

    assert.throws(() => {
        wasm.vertical_horizontal_filter_js(close, 0);
    }, /Invalid length/);
});

test('vertical_horizontal_filter_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const safe = wasm.vertical_horizontal_filter_js(close, 21);

    const dataPtr = wasm.copy_f64_array(close);
    const outPtr = wasm.vertical_horizontal_filter_alloc(close.length);

    try {
        wasm.vertical_horizontal_filter_into(dataPtr, outPtr, close.length, 21);
        const values = wasm.read_f64_array(outPtr, close.length);
        assertArrayClose(values, safe, 1e-10, 'pointer-path mismatch');
    } finally {
        wasm.vertical_horizontal_filter_free(outPtr, close.length);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('vertical_horizontal_filter_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const batch = wasm.vertical_horizontal_filter_batch_js(close, {
        length_range: [28, 28, 0],
    });
    const single = wasm.vertical_horizontal_filter_js(close, 28);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.lengths, [28]);
    assertArrayClose(batch.values, single, 1e-10, 'batch mismatch');
});

test('vertical_horizontal_filter_batch_js metadata matches requested grid', () => {
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.vertical_horizontal_filter_batch_js(close, {
        length_range: [28, 32, 2],
    });

    assert.strictEqual(batch.rows, 3);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.values.length, 3 * close.length);
    assert.deepStrictEqual(batch.lengths, [28, 30, 32]);

    const single = wasm.vertical_horizontal_filter_js(close, 28);
    assertArrayClose(batch.values.slice(0, close.length), single, 1e-10, 'first-row mismatch');
});

test('vertical_horizontal_filter_batch_js rejects invalid config', () => {
    const close = new Float64Array(testData.close.slice(0, 64));

    assert.throws(() => {
        wasm.vertical_horizontal_filter_batch_js(close, {
            length_range: [0, 5, 1],
        });
    }, /Invalid length/);
});
