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

test('on_balance_volume_oscillator_js output contract', () => {
    const source = new Float64Array(testData.close.slice(0, 256));
    const volume = new Float64Array(testData.volume.slice(0, 256));

    const result = wasm.on_balance_volume_oscillator_js(source, volume, 20, 9);

    assert(result.line);
    assert(result.signal);
    assert.strictEqual(result.line.length, source.length);
    assert.strictEqual(result.signal.length, source.length);
    assert(Array.from(result.line.slice(0, 19)).every(v => isNaN(v)));
    assert(Array.from(result.signal.slice(0, 19)).every(v => isNaN(v)));
    assert(Array.from(result.line.slice(19)).some(v => !isNaN(v)));
    assert(Array.from(result.signal.slice(19)).some(v => !isNaN(v)));
});

test('on_balance_volume_oscillator_into_host pointer path matches safe API', () => {
    const source = new Float64Array(testData.close.slice(0, 192));
    const volume = new Float64Array(testData.volume.slice(0, 192));
    const safe = wasm.on_balance_volume_oscillator_js(source, volume, 20, 9);
    const linePtr = wasm.on_balance_volume_oscillator_alloc(source.length);
    const signalPtr = wasm.on_balance_volume_oscillator_alloc(source.length);

    try {
        wasm.on_balance_volume_oscillator_into_host(source, volume, linePtr, signalPtr, 20, 9);
        const line = wasm.read_f64_array(linePtr, source.length);
        const signal = wasm.read_f64_array(signalPtr, source.length);
        assertArrayClose(line, safe.line, 1e-12, 'line pointer mismatch');
        assertArrayClose(signal, safe.signal, 1e-12, 'signal pointer mismatch');
    } finally {
        wasm.on_balance_volume_oscillator_free(linePtr, source.length);
        wasm.on_balance_volume_oscillator_free(signalPtr, source.length);
    }
});

test('on_balance_volume_oscillator_batch_js single parameter set matches safe API', () => {
    const source = new Float64Array(testData.close.slice(0, 192));
    const volume = new Float64Array(testData.volume.slice(0, 192));
    const batch = wasm.on_balance_volume_oscillator_batch_js(source, volume, {
        obv_length_range: [20, 20, 0],
        ema_length_range: [9, 9, 0],
    });
    const single = wasm.on_balance_volume_oscillator_js(source, volume, 20, 9);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, source.length);
    assert.strictEqual(batch.obv_lengths.length, 1);
    assert.strictEqual(batch.ema_lengths.length, 1);
    assert.strictEqual(batch.obv_lengths[0], 20);
    assert.strictEqual(batch.ema_lengths[0], 9);
    assertArrayClose(batch.line, single.line, 1e-12, 'batch line mismatch');
    assertArrayClose(batch.signal, single.signal, 1e-12, 'batch signal mismatch');
});

test('on_balance_volume_oscillator invalid length throws', () => {
    const source = new Float64Array(testData.close.slice(0, 64));
    const volume = new Float64Array(testData.volume.slice(0, 64));
    assert.throws(() => {
        wasm.on_balance_volume_oscillator_js(source, volume, 0, 9);
    }, /Invalid OBV length/);
});
