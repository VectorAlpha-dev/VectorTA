import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

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

test('trend_follower_js output contract', () => {
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const volume = new Float64Array(testData.volume.slice(0, 256));

    const result = wasm.trend_follower_js(high, low, close, volume, 'ema', 20, 20, 1.0, true, 5);

    assert.strictEqual(result.length, close.length);
    assert(Array.from(result).some(v => !Number.isNaN(v)));
});

test('trend_follower_into_host pointer path matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 192));
    const low = new Float64Array(testData.low.slice(0, 192));
    const close = new Float64Array(testData.close.slice(0, 192));
    const volume = new Float64Array(testData.volume.slice(0, 192));
    const safe = wasm.trend_follower_js(high, low, close, volume, 'ema', 20, 20, 1.0, true, 5);
    const outPtr = wasm.trend_follower_alloc(close.length);

    try {
        wasm.trend_follower_into_host(high, low, close, volume, outPtr, 'ema', 20, 20, 1.0, true, 5);
        const out = wasm.read_f64_array(outPtr, close.length);
        assertArrayClose(out, safe, 1e-10, 'pointer-path mismatch');
    } finally {
        wasm.trend_follower_free(outPtr, close.length);
    }
});

test('trend_follower_batch_js single parameter set matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 192));
    const low = new Float64Array(testData.low.slice(0, 192));
    const close = new Float64Array(testData.close.slice(0, 192));
    const volume = new Float64Array(testData.volume.slice(0, 192));
    const single = wasm.trend_follower_js(high, low, close, volume, 'ema', 20, 20, 1.0, true, 5);
    const batch = wasm.trend_follower_batch_js(high, low, close, volume, {
        trend_period_range: [20, 20, 0],
        ma_period_range: [20, 20, 0],
        channel_rate_percent_range: [1.0, 1.0, 0.0],
        linear_regression_period_range: [5, 5, 0],
        matype: 'ema',
        use_linear_regression: true,
    });

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.values.length, close.length);
    assert.strictEqual(batch.combos[0].matype, 'ema');
    assertArrayClose(batch.values, single, 1e-10, 'batch mismatch');
});

test('trend_follower vwma output changes with volume', () => {
    const high = new Float64Array(testData.high.slice(0, 160));
    const low = new Float64Array(testData.low.slice(0, 160));
    const close = new Float64Array(testData.close.slice(0, 160));
    const volume = new Float64Array(testData.volume.slice(0, 160));
    const reversed = new Float64Array(Array.from(volume).reverse());

    const a = wasm.trend_follower_js(high, low, close, volume, 'vwma', 20, 12, 1.0, false, 5);
    const b = wasm.trend_follower_js(high, low, close, reversed, 'vwma', 20, 12, 1.0, false, 5);

    assert(Array.from(a).some((v, i) => !Number.isNaN(v) && !Number.isNaN(b[i]) && Math.abs(v - b[i]) > 1e-9));
});

test('trend_follower invalid matype throws', () => {
    const high = new Float64Array(testData.high.slice(0, 64));
    const low = new Float64Array(testData.low.slice(0, 64));
    const close = new Float64Array(testData.close.slice(0, 64));
    const volume = new Float64Array(testData.volume.slice(0, 64));

    assert.throws(
        () => wasm.trend_follower_js(high, low, close, volume, 'hma', 20, 20, 1.0, true, 5),
        /Invalid MA type/
    );
});
