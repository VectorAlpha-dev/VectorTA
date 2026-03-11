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

test('bull_power_vs_bear_power_js output contract', () => {
    const open = new Float64Array(testData.open.slice(0, 512));
    const high = new Float64Array(testData.high.slice(0, 512));
    const low = new Float64Array(testData.low.slice(0, 512));
    const close = new Float64Array(testData.close.slice(0, 512));
    const result = wasm.bull_power_vs_bear_power_js(open, high, low, close, 5);

    assert.strictEqual(result.length, close.length);
    const firstValid = result.findIndex(v => !isNaN(v));
    assert(firstValid >= 4, `unexpected first valid index: ${firstValid}`);
    assert(result.some(v => !isNaN(v)), 'output should contain valid values');

    const tailStart = Math.min(firstValid + 32, result.length);
    for (let i = tailStart; i < result.length; i++) {
        assert(!isNaN(result[i]), `unexpected NaN at ${i}`);
    }
});

test('bull_power_vs_bear_power_js rejects invalid parameters', () => {
    const open = new Float64Array(testData.open.slice(0, 128));
    const high = new Float64Array(testData.high.slice(0, 128));
    const low = new Float64Array(testData.low.slice(0, 128));
    const close = new Float64Array(testData.close.slice(0, 128));

    assert.throws(() => {
        wasm.bull_power_vs_bear_power_js(open, high, low, close, 0);
    }, /Invalid period/);

    assert.throws(() => {
        wasm.bull_power_vs_bear_power_js(open.subarray(0, 100), high, low, close, 5);
    }, /Inconsistent slice lengths|length mismatch/);
});

test('bull_power_vs_bear_power_into pointer path matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 256));
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const safe = wasm.bull_power_vs_bear_power_js(open, high, low, close, 7);

    const openPtr = wasm.copy_f64_array(open);
    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const outPtr = wasm.bull_power_vs_bear_power_alloc(close.length);

    try {
        wasm.bull_power_vs_bear_power_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            close.length,
            7,
        );
        const values = wasm.read_f64_array(outPtr, close.length);
        assertArrayClose(values, safe, 1e-10, 'pointer-path mismatch');
    } finally {
        wasm.bull_power_vs_bear_power_free(outPtr, close.length);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('bull_power_vs_bear_power_batch_js single parameter set matches safe API', () => {
    const open = new Float64Array(testData.open.slice(0, 256));
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const batch = wasm.bull_power_vs_bear_power_batch_js(open, high, low, close, {
        period_range: [5, 5, 0],
    });
    const single = wasm.bull_power_vs_bear_power_js(open, high, low, close, 5);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.periods, [5]);
    assertArrayClose(batch.values, single, 1e-10, 'batch mismatch');
});

test('bull_power_vs_bear_power_batch_js metadata matches requested grid', () => {
    const open = new Float64Array(testData.open.slice(0, 200));
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.bull_power_vs_bear_power_batch_js(open, high, low, close, {
        period_range: [5, 9, 2],
    });

    assert.strictEqual(batch.rows, 3);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.values.length, 3 * close.length);
    assert.deepStrictEqual(batch.periods, [5, 7, 9]);

    const single = wasm.bull_power_vs_bear_power_js(open, high, low, close, 5);
    assertArrayClose(batch.values.slice(0, close.length), single, 1e-10, 'first-row mismatch');
});

test('bull_power_vs_bear_power_batch_js rejects invalid config', () => {
    const open = new Float64Array(testData.open.slice(0, 64));
    const high = new Float64Array(testData.high.slice(0, 64));
    const low = new Float64Array(testData.low.slice(0, 64));
    const close = new Float64Array(testData.close.slice(0, 64));

    assert.throws(() => {
        wasm.bull_power_vs_bear_power_batch_js(open, high, low, close, {
            period_range: [0, 5, 1],
        });
    }, /Invalid period/);
});
