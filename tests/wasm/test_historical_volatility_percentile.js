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

test('historical_volatility_percentile_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 256));

    const result = wasm.historical_volatility_percentile_js(close, 5, 10);

    assert(result.hvp);
    assert(result.hvp_sma);
    assert.strictEqual(result.hvp.length, close.length);
    assert.strictEqual(result.hvp_sma.length, close.length);
    assert(Array.from(result.hvp.slice(0, 13)).every(v => isNaN(v)));
    assert(Array.from(result.hvp_sma.slice(0, 17)).every(v => isNaN(v)));
    assert(Array.from(result.hvp.slice(13)).some(v => !isNaN(v)));
    assert(Array.from(result.hvp_sma.slice(17)).some(v => !isNaN(v)));
});

test('historical_volatility_percentile_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 192));
    const safe = wasm.historical_volatility_percentile_js(close, 5, 10);
    const outPtr = wasm.historical_volatility_percentile_alloc(close.length);

    try {
        wasm.historical_volatility_percentile_into_host(close, outPtr, 5, 10);
        const out = wasm.read_f64_array(outPtr, 2 * close.length);
        const hvp = out.slice(0, close.length);
        const hvpSma = out.slice(close.length, 2 * close.length);

        assertArrayClose(hvp, safe.hvp, 1e-10, 'hvp mismatch');
        assertArrayClose(hvpSma, safe.hvp_sma, 1e-10, 'hvp_sma mismatch');
    } finally {
        wasm.historical_volatility_percentile_free(outPtr, close.length);
    }
});

test('historical_volatility_percentile_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 192));
    const batch = wasm.historical_volatility_percentile_batch_js(close, {
        length_range: [5, 5, 0],
        annual_length_range: [10, 10, 0],
    });
    const single = wasm.historical_volatility_percentile_js(close, 5, 10);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.hvp.length, close.length);
    assert.strictEqual(batch.hvp_sma.length, close.length);
    assert.strictEqual(batch.combos[0].length, 5);
    assert.strictEqual(batch.combos[0].annual_length, 10);
    assertArrayClose(batch.hvp, single.hvp, 1e-10, 'batch hvp mismatch');
    assertArrayClose(batch.hvp_sma, single.hvp_sma, 1e-10, 'batch hvp_sma mismatch');
});
