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

test('leavitt_convolution_acceleration_js output contract', () => {
    const data = new Float64Array(testData.close.slice(0, 720));
    const result = wasm.leavitt_convolution_acceleration_js(data, 21, 34, true);

    assert.strictEqual(result.conv_acceleration.length, data.length);
    assert.strictEqual(result.signal.length, data.length);
    assert(Array.from(result.conv_acceleration.slice(0, 53)).every(v => isNaN(v)));
    assert(Array.from(result.signal.slice(0, 53)).every(v => isNaN(v)));
    assert(Array.from(result.conv_acceleration.slice(53)).some(v => !isNaN(v)));
    assert(Array.from(result.signal.slice(53)).some(v => !isNaN(v)));
});

test('leavitt_convolution_acceleration_into_host pointer path matches safe API', () => {
    const data = new Float64Array(testData.close.slice(0, 320));
    const safe = wasm.leavitt_convolution_acceleration_js(data, 21, 34, false);
    const outPtr = wasm.leavitt_convolution_acceleration_alloc(data.length);

    try {
        wasm.leavitt_convolution_acceleration_into_host(data, outPtr, 21, 34, false);
        const flat = wasm.read_f64_array(outPtr, data.length * 2);
        const conv = flat.slice(0, data.length);
        const signal = flat.slice(data.length);
        assertArrayClose(conv, safe.conv_acceleration, 1e-10, 'conv_acceleration pointer-path mismatch');
        assertArrayClose(signal, safe.signal, 1e-10, 'signal pointer-path mismatch');
    } finally {
        wasm.leavitt_convolution_acceleration_free(outPtr, data.length);
    }
});

test('leavitt_convolution_acceleration_batch_js single parameter set matches safe API', () => {
    const data = new Float64Array(testData.close.slice(0, 320));
    const batch = wasm.leavitt_convolution_acceleration_batch_js(data, {
        length_range: [21, 21, 0],
        norm_length_range: [34, 34, 0],
        use_norm_hyperbolic: true,
    });
    const single = wasm.leavitt_convolution_acceleration_js(data, 21, 34, true);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, data.length);
    assert.strictEqual(batch.conv_acceleration.length, data.length);
    assert.strictEqual(batch.signal.length, data.length);
    assert.strictEqual(batch.combos[0].length, 21);
    assert.strictEqual(batch.combos[0].norm_length, 34);
    assert.strictEqual(batch.combos[0].use_norm_hyperbolic, true);
    assertArrayClose(batch.conv_acceleration, single.conv_acceleration, 1e-10, 'conv_acceleration batch mismatch');
    assertArrayClose(batch.signal, single.signal, 1e-10, 'signal batch mismatch');
});
