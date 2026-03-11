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

test('volatility_ratio_adaptive_rsx_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 256));

    const result = wasm.volatility_ratio_adaptive_rsx_js(close, 6, 0.5);

    assert(result.line);
    assert(result.signal);
    assert.strictEqual(result.line.length, close.length);
    assert.strictEqual(result.signal.length, close.length);
    assert(Array.from(result.line.slice(0, 10)).every(v => isNaN(v)));
    assert(Array.from(result.signal.slice(0, 11)).every(v => isNaN(v)));
    assert(Array.from(result.line.slice(10)).some(v => !isNaN(v)));
    assert(Array.from(result.signal.slice(11)).some(v => !isNaN(v)));
});

test('volatility_ratio_adaptive_rsx_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 192));
    const safe = wasm.volatility_ratio_adaptive_rsx_js(close, 6, 0.5);
    const outPtr = wasm.volatility_ratio_adaptive_rsx_alloc(close.length);

    try {
        wasm.volatility_ratio_adaptive_rsx_into_host(close, outPtr, 6, 0.5);
        const out = wasm.read_f64_array(outPtr, 2 * close.length);
        const line = out.slice(0, close.length);
        const signal = out.slice(close.length, 2 * close.length);

        assertArrayClose(line, safe.line, 1e-10, 'line mismatch');
        assertArrayClose(signal, safe.signal, 1e-10, 'signal mismatch');
    } finally {
        wasm.volatility_ratio_adaptive_rsx_free(outPtr, close.length);
    }
});

test('volatility_ratio_adaptive_rsx_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 192));
    const batch = wasm.volatility_ratio_adaptive_rsx_batch_js(close, {
        period_range: [6, 6, 0],
        speed_range: [0.5, 0.5, 0.0],
    });
    const single = wasm.volatility_ratio_adaptive_rsx_js(close, 6, 0.5);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.line.length, close.length);
    assert.strictEqual(batch.signal.length, close.length);
    assert.strictEqual(batch.combos[0].period, 6);
    assert.strictEqual(batch.combos[0].speed, 0.5);
    assertArrayClose(batch.line, single.line, 1e-10, 'batch line mismatch');
    assertArrayClose(batch.signal, single.signal, 1e-10, 'batch signal mismatch');
});
