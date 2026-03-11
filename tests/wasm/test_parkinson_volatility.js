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

test('parkinson_volatility_js output contract', () => {
    const n = 256;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));

    const result = wasm.parkinson_volatility_js(high, low, 8);

    assert(result.volatility);
    assert(result.variance);
    assert.strictEqual(result.volatility.length, n);
    assert.strictEqual(result.variance.length, n);

    const firstValid = Array.from(result.volatility).findIndex(v => !isNaN(v));
    assert(firstValid >= 7);
    for (let i = Math.min(firstValid + 32, n); i < n; i++) {
        assert(!isNaN(result.volatility[i]));
        assert(!isNaN(result.variance[i]));
    }
});

test('parkinson_volatility_into pointer path matches safe API', () => {
    const n = 192;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const safe = wasm.parkinson_volatility_js(high, low, 10);

    const outPtr = wasm.parkinson_volatility_alloc(n);

    try {
        wasm.parkinson_volatility_into_host(high, low, outPtr, 10);

        const out = wasm.read_f64_array(outPtr, 2 * n);
        const volatility = out.slice(0, n);
        const variance = out.slice(n, 2 * n);

        assertArrayClose(volatility, safe.volatility, 1e-10, 'volatility mismatch');
        assertArrayClose(variance, safe.variance, 1e-10, 'variance mismatch');
    } finally {
        wasm.parkinson_volatility_free(outPtr, n);
    }
});

test('parkinson_volatility_batch_js single parameter set matches safe API', () => {
    const n = 192;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));

    const batch = wasm.parkinson_volatility_batch_js(high, low, {
        period_range: [10, 10, 0],
    });
    const single = wasm.parkinson_volatility_js(high, low, 10);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, n);
    assert.strictEqual(batch.volatility.length, n);
    assert.strictEqual(batch.variance.length, n);
    assert.strictEqual(batch.combos[0].period, 10);
    assertArrayClose(batch.volatility, single.volatility, 1e-10, 'batch volatility mismatch');
    assertArrayClose(batch.variance, single.variance, 1e-10, 'batch variance mismatch');
});
