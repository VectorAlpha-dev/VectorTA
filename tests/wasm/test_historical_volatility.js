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
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }

    testData = loadTestData();
});

test('historical_volatility_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 512));
    const result = wasm.historical_volatility_js(close, 20, 250.0);

    assert.strictEqual(result.length, close.length);
    const firstValid = result.findIndex(v => !isNaN(v));
    assert(firstValid >= 20, `unexpected first valid index: ${firstValid}`);
    assert(result.some(v => !isNaN(v)), 'output should contain valid values');

    const tailStart = Math.min(firstValid + 32, result.length);
    for (let i = tailStart; i < result.length; i++) {
        assert(!isNaN(result[i]), `unexpected NaN at ${i}`);
    }
});

test('historical_volatility_js rejects invalid parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 128));

    assert.throws(() => {
        wasm.historical_volatility_js(close, 0, 250.0);
    }, /Invalid lookback/);

    assert.throws(() => {
        wasm.historical_volatility_js(close, 20, 0.0);
    }, /Invalid annualization_days/);
});

test('historical_volatility_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const safe = wasm.historical_volatility_js(close, 20, 252.0);

    const inPtr = wasm.copy_f64_array(close);
    const outPtr = wasm.historical_volatility_alloc(close.length);

    try {
        wasm.historical_volatility_into(inPtr, outPtr, close.length, 20, 252.0);
        const values = wasm.read_f64_array(outPtr, close.length);
        assertArrayClose(values, safe, 1e-10, 'pointer-path mismatch');
    } finally {
        wasm.historical_volatility_free(outPtr, close.length);
        wasm.deallocate_f64_array(inPtr);
    }
});

test('historical_volatility_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const batch = wasm.historical_volatility_batch_js(close, {
        lookback_range: [20, 20, 0],
        annualization_days_range: [250.0, 250.0, 0.0],
    });
    const single = wasm.historical_volatility_js(close, 20, 250.0);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.lookbacks, [20]);
    assert.deepStrictEqual(batch.annualization_days, [250.0]);
    assertArrayClose(batch.values, single, 1e-10, 'batch mismatch');
});

test('historical_volatility_batch_js metadata matches requested grid', () => {
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.historical_volatility_batch_js(close, {
        lookback_range: [10, 14, 2],
        annualization_days_range: [250.0, 252.0, 2.0],
    });

    assert.strictEqual(batch.rows, 6);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.values.length, 6 * close.length);
    assert.deepStrictEqual(batch.lookbacks, [10, 10, 12, 12, 14, 14]);
    assert.deepStrictEqual(batch.annualization_days, [250.0, 252.0, 250.0, 252.0, 250.0, 252.0]);

    const single = wasm.historical_volatility_js(close, 10, 250.0);
    assertArrayClose(batch.values.slice(0, close.length), single, 1e-10, 'first-row mismatch');
});

test('historical_volatility_batch_js rejects invalid config', () => {
    const close = new Float64Array(testData.close.slice(0, 64));

    assert.throws(() => {
        wasm.historical_volatility_batch_js(close, {
            lookback_range: [0, 10, 1],
            annualization_days_range: [250.0, 250.0, 0.0],
        });
    }, /Invalid lookback/);
});
