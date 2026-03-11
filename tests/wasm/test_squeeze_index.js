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

test('squeeze_index_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 512));
    const result = wasm.squeeze_index_js(close, 50.0, 20);

    assert.strictEqual(result.length, close.length);
    const firstValid = result.findIndex(v => !isNaN(v));
    assert(firstValid >= 19, `unexpected first valid index: ${firstValid}`);
    assert(result.some(v => !isNaN(v)), 'output should contain valid values');

    const tailStart = Math.min(firstValid + 32, result.length);
    for (let i = tailStart; i < result.length; i++) {
        assert(!isNaN(result[i]), `unexpected NaN at ${i}`);
    }
});

test('squeeze_index_js rejects invalid parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 128));

    assert.throws(() => {
        wasm.squeeze_index_js(close, 1.0, 20);
    }, /Invalid conv/);

    assert.throws(() => {
        wasm.squeeze_index_js(close, 50.0, 0);
    }, /Invalid length/);
});

test('squeeze_index_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const safe = wasm.squeeze_index_js(close, 50.0, 20);

    const dataPtr = wasm.copy_f64_array(close);
    const outPtr = wasm.squeeze_index_alloc(close.length);

    try {
        wasm.squeeze_index_into(dataPtr, outPtr, close.length, 50.0, 20);
        const values = wasm.read_f64_array(outPtr, close.length);
        assertArrayClose(values, safe, 1e-10, 'pointer-path mismatch');
    } finally {
        wasm.squeeze_index_free(outPtr, close.length);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('squeeze_index_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const batch = wasm.squeeze_index_batch_js(close, {
        conv_range: [50.0, 50.0, 0.0],
        length_range: [20, 20, 0],
    });
    const single = wasm.squeeze_index_js(close, 50.0, 20);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.convs, [50.0]);
    assert.deepStrictEqual(batch.lengths, [20]);
    assertArrayClose(batch.values, single, 1e-10, 'batch mismatch');
});

test('squeeze_index_batch_js metadata and batch_into match requested grid', () => {
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.squeeze_index_batch_js(close, {
        conv_range: [50.0, 60.0, 10.0],
        length_range: [20, 24, 4],
    });

    assert.strictEqual(batch.rows, 4);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.values.length, 4 * close.length);
    assert.deepStrictEqual(batch.convs, [50.0, 50.0, 60.0, 60.0]);
    assert.deepStrictEqual(batch.lengths, [20, 24, 20, 24]);

    const dataPtr = wasm.copy_f64_array(close);
    const outPtr = wasm.squeeze_index_alloc(batch.values.length);
    try {
        const rows = wasm.squeeze_index_batch_into(
            dataPtr,
            outPtr,
            close.length,
            50.0,
            60.0,
            10.0,
            20,
            24,
            4
        );
        assert.strictEqual(rows, 4);
        const values = wasm.read_f64_array(outPtr, batch.values.length);
        assertArrayClose(values, batch.values, 1e-10, 'batch_into mismatch');
    } finally {
        wasm.squeeze_index_free(outPtr, batch.values.length);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('squeeze_index_batch_js rejects invalid config', () => {
    const close = new Float64Array(testData.close.slice(0, 64));

    assert.throws(() => {
        wasm.squeeze_index_batch_js(close, {
            conv_range: [1.0, 1.0, 0.0],
            length_range: [20, 20, 0],
        });
    }, /Invalid conv/);

    assert.throws(() => {
        wasm.squeeze_index_batch_js(close, {
            conv_range: [50.0, 50.0, 0.0],
            length_range: [0, 5, 1],
        });
    }, /Invalid length/);
});
