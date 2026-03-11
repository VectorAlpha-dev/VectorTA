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

test('keltner_channel_width_oscillator_js output contract', () => {
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const source = new Float64Array(Array.from(close, (_, i) => (high[i] + low[i] + close[i]) / 3));

    const result = wasm.keltner_channel_width_oscillator_js(
        high,
        low,
        close,
        source,
        20,
        2.0,
        true,
        'Average True Range',
        10,
    );

    assert.strictEqual(result.kbw.length, close.length);
    assert.strictEqual(result.kbw_sma.length, close.length);
    assert(Array.from(result.kbw.slice(0, 19)).every(v => isNaN(v)));
    assert(Array.from(result.kbw_sma.slice(0, 38)).every(v => isNaN(v)));
    assert(Array.from(result.kbw.slice(19)).some(v => !isNaN(v)));
    assert(Array.from(result.kbw_sma.slice(38)).some(v => !isNaN(v)));
});

test('keltner_channel_width_oscillator_into pointer path matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 192));
    const low = new Float64Array(testData.low.slice(0, 192));
    const close = new Float64Array(testData.close.slice(0, 192));
    const source = new Float64Array(testData.close.slice(0, 192));
    const safe = wasm.keltner_channel_width_oscillator_js(
        high,
        low,
        close,
        source,
        20,
        2.0,
        true,
        'Average True Range',
        10,
    );
    const outPtr = wasm.keltner_channel_width_oscillator_alloc(close.length);

    try {
        wasm.keltner_channel_width_oscillator_into_host(
            high,
            low,
            close,
            source,
            outPtr,
            20,
            2.0,
            true,
            'Average True Range',
            10,
        );
        const flat = wasm.read_f64_array(outPtr, close.length * 2);
        const kbw = flat.slice(0, close.length);
        const kbwSma = flat.slice(close.length);
        assertArrayClose(kbw, safe.kbw, 1e-10, 'kbw pointer-path mismatch');
        assertArrayClose(kbwSma, safe.kbw_sma, 1e-10, 'kbw_sma pointer-path mismatch');
    } finally {
        wasm.keltner_channel_width_oscillator_free(outPtr, close.length);
    }
});

test('keltner_channel_width_oscillator_batch_js single parameter set matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 192));
    const low = new Float64Array(testData.low.slice(0, 192));
    const close = new Float64Array(testData.close.slice(0, 192));
    const source = new Float64Array(testData.close.slice(0, 192));
    const batch = wasm.keltner_channel_width_oscillator_batch_js(high, low, close, source, {
        length_range: [20, 20, 0],
        multiplier_range: [2.0, 2.0, 0.0],
        atr_length_range: [10, 10, 0],
        use_exponential: true,
        bands_style: 'Average True Range',
    });
    const single = wasm.keltner_channel_width_oscillator_js(
        high,
        low,
        close,
        source,
        20,
        2.0,
        true,
        'Average True Range',
        10,
    );

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.kbw.length, close.length);
    assert.strictEqual(batch.kbw_sma.length, close.length);
    assert.strictEqual(batch.combos[0].length, 20);
    assert.strictEqual(batch.combos[0].multiplier, 2.0);
    assert.strictEqual(batch.combos[0].atr_length, 10);
    assert.strictEqual(batch.combos[0].use_exponential, true);
    assert.strictEqual(batch.combos[0].bands_style, 'Average True Range');
    assertArrayClose(batch.kbw, single.kbw, 1e-10, 'kbw batch mismatch');
    assertArrayClose(batch.kbw_sma, single.kbw_sma, 1e-10, 'kbw_sma batch mismatch');
});
