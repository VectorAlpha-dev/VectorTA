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

test('atr_percentile_js output contract', () => {
    const high = new Float64Array(testData.high.slice(0, 512));
    const low = new Float64Array(testData.low.slice(0, 512));
    const close = new Float64Array(testData.close.slice(0, 512));
    const result = wasm.atr_percentile_js(high, low, close, 10, 50);

    assert.strictEqual(result.length, close.length);
    const firstValid = result.findIndex(v => !isNaN(v));
    assert(firstValid >= 59, `unexpected first valid index: ${firstValid}`);
    assert(result.some(v => !isNaN(v)), 'output should contain valid values');

    for (const value of result.slice(firstValid + 16, firstValid + 80)) {
        if (!isNaN(value)) {
            assert(value >= 0 && value <= 100, `unexpected percentile: ${value}`);
        }
    }
});

test('atr_percentile_js rejects invalid parameters', () => {
    const high = new Float64Array(testData.high.slice(0, 128));
    const low = new Float64Array(testData.low.slice(0, 128));
    const close = new Float64Array(testData.close.slice(0, 128));

    assert.throws(() => {
        wasm.atr_percentile_js(high, low, close, 0, 10);
    }, /Invalid ATR length/);

    assert.throws(() => {
        wasm.atr_percentile_js(high, low, close, 10, 0);
    }, /Invalid percentile length/);

    assert.throws(() => {
        wasm.atr_percentile_js(high.subarray(0, 100), low, close, 10, 20);
    }, /Inconsistent slice lengths|length mismatch/);
});

test('atr_percentile_into pointer path matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const safe = wasm.atr_percentile_js(high, low, close, 10, 20);

    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const outPtr = wasm.atr_percentile_alloc(close.length);

    try {
        wasm.atr_percentile_into(highPtr, lowPtr, closePtr, outPtr, close.length, 10, 20);
        const values = wasm.read_f64_array(outPtr, close.length);
        assertArrayClose(values, safe, 1e-10, 'pointer-path mismatch');
    } finally {
        wasm.atr_percentile_free(outPtr, close.length);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('atr_percentile_batch_js single parameter set matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const batch = wasm.atr_percentile_batch_js(high, low, close, {
        atr_length_range: [10, 10, 0],
        percentile_length_range: [20, 20, 0],
    });
    const single = wasm.atr_percentile_js(high, low, close, 10, 20);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.atr_lengths, [10]);
    assert.deepStrictEqual(batch.percentile_lengths, [20]);
    assertArrayClose(batch.values, single, 1e-10, 'batch mismatch');
});

test('atr_percentile_batch_js metadata matches requested grid', () => {
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.atr_percentile_batch_js(high, low, close, {
        atr_length_range: [10, 14, 2],
        percentile_length_range: [20, 24, 2],
    });

    assert.strictEqual(batch.rows, 9);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.values.length, 9 * close.length);
    assert.deepStrictEqual(batch.atr_lengths, [10, 10, 10, 12, 12, 12, 14, 14, 14]);
    assert.deepStrictEqual(batch.percentile_lengths, [20, 22, 24, 20, 22, 24, 20, 22, 24]);

    const single = wasm.atr_percentile_js(high, low, close, 10, 20);
    assertArrayClose(batch.values.slice(0, close.length), single, 1e-10, 'first-row mismatch');
});

test('atr_percentile_batch_js rejects invalid config', () => {
    const high = new Float64Array(testData.high.slice(0, 64));
    const low = new Float64Array(testData.low.slice(0, 64));
    const close = new Float64Array(testData.close.slice(0, 64));

    assert.throws(() => {
        wasm.atr_percentile_batch_js(high, low, close, {
            atr_length_range: [0, 10, 1],
        });
    }, /Invalid ATR length/);
});
