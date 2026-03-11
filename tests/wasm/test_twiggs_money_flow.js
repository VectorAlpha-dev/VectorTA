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

test('twiggs_money_flow_js output contract', () => {
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const volume = new Float64Array(testData.volume.slice(0, 256));

    const result = wasm.twiggs_money_flow_js(high, low, close, volume, 5, 4, 'WMA');

    assert(result.tmf);
    assert(result.smoothed);
    assert.strictEqual(result.tmf.length, close.length);
    assert.strictEqual(result.smoothed.length, close.length);
    assert(Array.from(result.tmf.slice(0, 5)).every(v => isNaN(v)));
    assert(Array.from(result.smoothed.slice(0, 9)).every(v => isNaN(v)));
    assert(Array.from(result.tmf.slice(5)).some(v => !isNaN(v)));
    assert(Array.from(result.smoothed.slice(9)).some(v => !isNaN(v)));
});

test('twiggs_money_flow_into pointer path matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 192));
    const low = new Float64Array(testData.low.slice(0, 192));
    const close = new Float64Array(testData.close.slice(0, 192));
    const volume = new Float64Array(testData.volume.slice(0, 192));
    const safe = wasm.twiggs_money_flow_js(high, low, close, volume, 5, 4, 'EMA');
    const outPtr = wasm.twiggs_money_flow_alloc(close.length);

    try {
        wasm.twiggs_money_flow_into_host(high, low, close, volume, outPtr, 5, 4, 'EMA');
        const out = wasm.read_f64_array(outPtr, 2 * close.length);
        const tmf = out.slice(0, close.length);
        const smoothed = out.slice(close.length, 2 * close.length);

        assertArrayClose(tmf, safe.tmf, 1e-10, 'tmf mismatch');
        assertArrayClose(smoothed, safe.smoothed, 1e-10, 'smoothed mismatch');
    } finally {
        wasm.twiggs_money_flow_free(outPtr, close.length);
    }
});

test('twiggs_money_flow_batch_js single parameter set matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 192));
    const low = new Float64Array(testData.low.slice(0, 192));
    const close = new Float64Array(testData.close.slice(0, 192));
    const volume = new Float64Array(testData.volume.slice(0, 192));
    const batch = wasm.twiggs_money_flow_batch_js(high, low, close, volume, {
        length_range: [5, 5, 0],
        smoothing_length_range: [4, 4, 0],
        ma_type: 'VWMA',
    });
    const single = wasm.twiggs_money_flow_js(high, low, close, volume, 5, 4, 'VWMA');

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.tmf.length, close.length);
    assert.strictEqual(batch.smoothed.length, close.length);
    assert.strictEqual(batch.combos[0].length, 5);
    assert.strictEqual(batch.combos[0].smoothing_length, 4);
    assert.strictEqual(batch.combos[0].ma_type, 'VWMA');
    assertArrayClose(batch.tmf, single.tmf, 1e-10, 'batch tmf mismatch');
    assertArrayClose(batch.smoothed, single.smoothed, 1e-10, 'batch smoothed mismatch');
});
