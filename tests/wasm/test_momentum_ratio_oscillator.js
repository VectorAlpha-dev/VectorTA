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

test('momentum_ratio_oscillator_js output contract', () => {
    const source = new Float64Array(testData.close.slice(0, 320));

    const result = wasm.momentum_ratio_oscillator_js(source, 50);

    assert.strictEqual(result.line.length, source.length);
    assert.strictEqual(result.signal.length, source.length);
    assert(Array.from(result.line.slice(0, 1)).every(v => isNaN(v)));
    assert(Array.from(result.signal.slice(0, 2)).every(v => isNaN(v)));
    assert(Array.from(result.line.slice(1)).some(v => !isNaN(v)));
    assert(Array.from(result.signal.slice(2)).some(v => !isNaN(v)));
});

test('momentum_ratio_oscillator_into_host pointer path matches safe API', () => {
    const source = new Float64Array(testData.close.slice(0, 240));
    const safe = wasm.momentum_ratio_oscillator_js(source, 35);
    const outPtr = wasm.momentum_ratio_oscillator_alloc(source.length);

    try {
        wasm.momentum_ratio_oscillator_into_host(source, outPtr, 35);
        const out = wasm.read_f64_array(outPtr, 2 * source.length);
        const line = out.slice(0, source.length);
        const signal = out.slice(source.length, 2 * source.length);
        assertArrayClose(line, safe.line, 1e-10, 'line mismatch');
        assertArrayClose(signal, safe.signal, 1e-10, 'signal mismatch');
    } finally {
        wasm.momentum_ratio_oscillator_free(outPtr, source.length);
    }
});

test('momentum_ratio_oscillator_batch_js single parameter set matches safe API', () => {
    const source = new Float64Array(testData.close.slice(0, 240));
    const batch = wasm.momentum_ratio_oscillator_batch_js(source, {
        period_range: [35, 35, 0],
    });
    const single = wasm.momentum_ratio_oscillator_js(source, 35);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, source.length);
    assert.strictEqual(batch.line.length, source.length);
    assert.strictEqual(batch.signal.length, source.length);
    assert.strictEqual(batch.combos[0].period, 35);
    assertArrayClose(batch.line, single.line, 1e-10, 'batch line mismatch');
    assertArrayClose(batch.signal, single.signal, 1e-10, 'batch signal mismatch');
});
