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

test('garman_klass_volatility_js output contract', () => {
    const n = 512;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    const result = wasm.garman_klass_volatility_js(open, high, low, close, 14);

    assert.strictEqual(result.length, n);
    const firstValid = result.findIndex(v => !isNaN(v));
    assert(firstValid >= 13, `unexpected first valid index: ${firstValid}`);
    assert(result.some(v => !isNaN(v)), 'output should contain valid values');

    const tailStart = Math.min(firstValid + 32, result.length);
    for (let i = tailStart; i < result.length; i++) {
        assert(!isNaN(result[i]), `unexpected NaN at ${i}`);
    }
});

test('garman_klass_volatility_js rejects length mismatch', () => {
    const n = 128;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n - 1));

    assert.throws(() => {
        wasm.garman_klass_volatility_js(open, high, low, close, 14);
    }, /Inconsistent slice lengths/);
});

test('garman_klass_volatility_js rejects invalid parameters', () => {
    const n = 128;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    assert.throws(() => {
        wasm.garman_klass_volatility_js(open, high, low, close, 0);
    }, /Invalid lookback/);
});

test('garman_klass_volatility_into pointer path matches safe API', (t) => {
    const n = 256;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    const safe = wasm.garman_klass_volatility_js(open, high, low, close, 20);

    if (typeof wasm.copy_f64_array !== 'function') {
        t.skip('wasm-bindgen wrapper does not expose a raw-memory input helper in this build mode');
        return;
    }

    const openPtr = wasm.copy_f64_array(open);
    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const outPtr = wasm.garman_klass_volatility_alloc(n);

    try {
        wasm.garman_klass_volatility_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            n,
            20,
        );

        const values = wasm.read_f64_array(outPtr, n);

        assertArrayClose(values, safe, 1e-10, 'zero-copy mismatch');
    } finally {
        wasm.garman_klass_volatility_free(outPtr, n);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('garman_klass_volatility_batch_js single parameter set matches safe API', () => {
    const n = 256;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    const batch = wasm.garman_klass_volatility_batch_js(open, high, low, close, {
        lookback_range: [20, 20, 0],
    });

    const single = wasm.garman_klass_volatility_js(open, high, low, close, 20);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, n);
    assert.strictEqual(batch.combos.length, 1);
    assert.strictEqual(batch.values.length, n);
    assert.strictEqual(batch.combos[0].lookback, 20);

    assertArrayClose(batch.values, single, 1e-10, 'batch mismatch');
});

test('garman_klass_volatility_batch_js metadata matches requested lookbacks', () => {
    const n = 200;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    const batch = wasm.garman_klass_volatility_batch_js(open, high, low, close, {
        lookback_range: [10, 14, 2],
    });

    assert.strictEqual(batch.rows, 3);
    assert.strictEqual(batch.cols, n);
    assert.strictEqual(batch.values.length, 3 * n);
    assert.deepStrictEqual(
        batch.combos.map(combo => combo.lookback),
        [10, 12, 14],
    );

    const single = wasm.garman_klass_volatility_js(open, high, low, close, 10);
    assertArrayClose(batch.values.slice(0, n), single, 1e-10, 'first-row mismatch');
});

test('garman_klass_volatility_batch_js rejects invalid config', () => {
    const n = 64;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    assert.throws(() => {
        wasm.garman_klass_volatility_batch_js(open, high, low, close, {
            lookback_range: [10, 12],
        });
    }, /Invalid config/);
});
