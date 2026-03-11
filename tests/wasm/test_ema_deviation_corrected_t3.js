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

test('ema_deviation_corrected_t3_js output contract', () => {
    const source = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.ema_deviation_corrected_t3_js(source, 10, 0.7, 0);

    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, source.length);
    assert.strictEqual(result.values.length, source.length * 2);

    const corrected = result.values.slice(0, source.length);
    const t3 = result.values.slice(source.length);
    assert(Array.from(corrected).some(v => !Number.isNaN(v)));
    assert(Array.from(t3).some(v => !Number.isNaN(v)));
});

test('ema_deviation_corrected_t3 constant input keeps corrected equal to t3', () => {
    const source = new Float64Array(64);
    source.fill(100.0);
    const result = wasm.ema_deviation_corrected_t3_js(source, 10, 0.7, 0);
    const corrected = result.values.slice(0, source.length);
    const t3 = result.values.slice(source.length);
    assertArrayClose(corrected, t3, 1e-12, 'constant input mismatch');
});

test('ema_deviation_corrected_t3_into_host pointer path matches safe API', () => {
    const source = new Float64Array(testData.close.slice(0, 220));
    const safe = wasm.ema_deviation_corrected_t3_js(source, 10, 0.7, 0);
    const correctedPtr = wasm.ema_deviation_corrected_t3_alloc(source.length);
    const t3Ptr = wasm.ema_deviation_corrected_t3_alloc(source.length);

    try {
        wasm.ema_deviation_corrected_t3_into_host(source, correctedPtr, t3Ptr, 10, 0.7, 0);
        const corrected = wasm.read_f64_array(correctedPtr, source.length);
        const t3 = wasm.read_f64_array(t3Ptr, source.length);
        assertArrayClose(corrected, safe.values.slice(0, source.length), 1e-12, 'corrected pointer mismatch');
        assertArrayClose(t3, safe.values.slice(source.length), 1e-12, 't3 pointer mismatch');
    } finally {
        wasm.ema_deviation_corrected_t3_free(correctedPtr, source.length);
        wasm.ema_deviation_corrected_t3_free(t3Ptr, source.length);
    }
});

test('ema_deviation_corrected_t3_batch_js single parameter set matches safe API', () => {
    const source = new Float64Array(testData.close.slice(0, 220));
    const safe = wasm.ema_deviation_corrected_t3_js(source, 10, 0.7, 0);
    const batch = wasm.ema_deviation_corrected_t3_batch_js(source, {
        period_range: [10, 10, 0],
        hot_range: [0.7, 0.7, 0.0],
        t3_mode_range: [0, 0, 0],
    });

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, source.length);
    assert.strictEqual(batch.periods.length, 1);
    assert.strictEqual(batch.hots.length, 1);
    assert.strictEqual(batch.t3_modes.length, 1);
    assert.strictEqual(batch.periods[0], 10);
    assert.strictEqual(batch.t3_modes[0], 0);
    assert.strictEqual(batch.hots[0], 0.7);
    assertArrayClose(batch.corrected, safe.values.slice(0, source.length), 1e-12, 'corrected batch mismatch');
    assertArrayClose(batch.t3, safe.values.slice(source.length), 1e-12, 't3 batch mismatch');
});

test('ema_deviation_corrected_t3 invalid mode throws', () => {
    const source = new Float64Array(testData.close.slice(0, 64));
    assert.throws(() => {
        wasm.ema_deviation_corrected_t3_js(source, 10, 0.7, 2);
    }, /Invalid T3 mode/);
});
