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

test('supertrend_oscillator output contract', () => {
    const high = new Float64Array(testData.high.slice(0, 320));
    const low = new Float64Array(testData.low.slice(0, 320));
    const source = new Float64Array(testData.close.slice(0, 320));

    const result = wasm.supertrend_oscillator(high, low, source, 10, 2.0, 72);

    assert.strictEqual(result.oscillator.length, source.length);
    assert.strictEqual(result.signal.length, source.length);
    assert.strictEqual(result.histogram.length, source.length);
    assert(Array.from(result.oscillator.slice(0, 9)).every(v => isNaN(v)));
    assert(Array.from(result.signal.slice(0, 9)).every(v => isNaN(v)));
    assert(Array.from(result.histogram.slice(0, 9)).every(v => isNaN(v)));
    assert(Array.from(result.oscillator.slice(9)).some(v => !isNaN(v)));
    assert(Array.from(result.signal.slice(9)).some(v => !isNaN(v)));
    assert(Array.from(result.histogram.slice(9)).some(v => !isNaN(v)));
});

test('supertrend_oscillator_into_host pointer path matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 240));
    const low = new Float64Array(testData.low.slice(0, 240));
    const source = new Float64Array(testData.close.slice(0, 240));
    const safe = wasm.supertrend_oscillator(high, low, source, 12, 2.5, 18);
    const outPtr = wasm.supertrend_oscillator_alloc(source.length);

    try {
        wasm.supertrend_oscillator_into_host(high, low, source, outPtr, 12, 2.5, 18);
        const out = wasm.read_f64_array(outPtr, 3 * source.length);
        const oscillator = out.slice(0, source.length);
        const signal = out.slice(source.length, 2 * source.length);
        const histogram = out.slice(2 * source.length, 3 * source.length);
        assertArrayClose(oscillator, safe.oscillator, 1e-10, 'oscillator mismatch');
        assertArrayClose(signal, safe.signal, 1e-10, 'signal mismatch');
        assertArrayClose(histogram, safe.histogram, 1e-10, 'histogram mismatch');
    } finally {
        wasm.supertrend_oscillator_free(outPtr, source.length);
    }
});

test('supertrend_oscillator_batch single parameter set matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 240));
    const low = new Float64Array(testData.low.slice(0, 240));
    const source = new Float64Array(testData.close.slice(0, 240));
    const batch = wasm.supertrend_oscillator_batch(high, low, source, {
        length_range: [12, 12, 0],
        mult_range: [2.5, 2.5, 0.0],
        smooth_range: [18, 18, 0],
    });
    const single = wasm.supertrend_oscillator(high, low, source, 12, 2.5, 18);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, source.length);
    assert.strictEqual(batch.combos[0].length, 12);
    assert.strictEqual(batch.combos[0].mult, 2.5);
    assert.strictEqual(batch.combos[0].smooth, 18);
    assertArrayClose(batch.oscillator, single.oscillator, 1e-10, 'batch oscillator mismatch');
    assertArrayClose(batch.signal, single.signal, 1e-10, 'batch signal mismatch');
    assertArrayClose(batch.histogram, single.histogram, 1e-10, 'batch histogram mismatch');
});

test('supertrend_oscillator_batch metadata', () => {
    const high = new Float64Array(testData.high.slice(0, 180));
    const low = new Float64Array(testData.low.slice(0, 180));
    const source = new Float64Array(testData.close.slice(0, 180));
    const batch = wasm.supertrend_oscillator_batch(high, low, source, {
        length_range: [10, 12, 2],
        mult_range: [1.5, 2.5, 0.5],
        smooth_range: [18, 18, 0],
    });

    assert.strictEqual(batch.rows, 6);
    assert.strictEqual(batch.cols, source.length);
    assert.strictEqual(batch.combos.length, 6);
    assert.deepStrictEqual(
        batch.combos.map(combo => combo.length),
        [10, 10, 10, 12, 12, 12],
    );
    assert.deepStrictEqual(
        batch.combos.map(combo => combo.smooth),
        [18, 18, 18, 18, 18, 18],
    );
});

test('supertrend_oscillator_batch_into pointer path matches safe batch API', () => {
    const high = new Float64Array(testData.high.slice(0, 180));
    const low = new Float64Array(testData.low.slice(0, 180));
    const source = new Float64Array(testData.close.slice(0, 180));
    const batch = wasm.supertrend_oscillator_batch(high, low, source, {
        length_range: [10, 12, 2],
        mult_range: [1.5, 2.5, 0.5],
        smooth_range: [18, 18, 0],
    });

    const rows = batch.rows;
    const total = rows * source.length;
    const highPtr = wasm.supertrend_oscillator_alloc(source.length);
    const lowPtr = wasm.supertrend_oscillator_alloc(source.length);
    const sourcePtr = wasm.supertrend_oscillator_alloc(source.length);
    const oscillatorPtr = wasm.supertrend_oscillator_alloc(total);
    const signalPtr = wasm.supertrend_oscillator_alloc(total);
    const histogramPtr = wasm.supertrend_oscillator_alloc(total);

    try {
        const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        if (!memory || !memory.buffer) {
            console.warn('Skipping supertrend_oscillator_batch_into test: wasm memory accessor unavailable');
            return;
        }
        const highMem = new Float64Array(memory.buffer, highPtr, source.length);
        const lowMem = new Float64Array(memory.buffer, lowPtr, source.length);
        const sourceMem = new Float64Array(memory.buffer, sourcePtr, source.length);
        highMem.set(high);
        lowMem.set(low);
        sourceMem.set(source);

        const returnedRows = wasm.supertrend_oscillator_batch_into(
            highPtr,
            lowPtr,
            sourcePtr,
            oscillatorPtr,
            signalPtr,
            histogramPtr,
            source.length,
            10,
            12,
            2,
            1.5,
            2.5,
            0.5,
            18,
            18,
            0,
        );

        assert.strictEqual(returnedRows, rows);
        const oscillator = wasm.read_f64_array(oscillatorPtr, total);
        const signal = wasm.read_f64_array(signalPtr, total);
        const histogram = wasm.read_f64_array(histogramPtr, total);
        assertArrayClose(oscillator, batch.oscillator, 1e-10, 'batch_into oscillator mismatch');
        assertArrayClose(signal, batch.signal, 1e-10, 'batch_into signal mismatch');
        assertArrayClose(histogram, batch.histogram, 1e-10, 'batch_into histogram mismatch');
    } finally {
        wasm.supertrend_oscillator_free(highPtr, source.length);
        wasm.supertrend_oscillator_free(lowPtr, source.length);
        wasm.supertrend_oscillator_free(sourcePtr, source.length);
        wasm.supertrend_oscillator_free(oscillatorPtr, total);
        wasm.supertrend_oscillator_free(signalPtr, total);
        wasm.supertrend_oscillator_free(histogramPtr, total);
    }
});
