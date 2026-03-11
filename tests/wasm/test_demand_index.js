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
    const importPath =
        process.platform === 'win32'
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('demand_index_js output contract', () => {
    const high = new Float64Array(testData.high.slice(0, 512));
    const low = new Float64Array(testData.low.slice(0, 512));
    const close = new Float64Array(testData.close.slice(0, 512));
    const volume = new Float64Array(testData.volume.slice(0, 512));
    const result = wasm.demand_index_js(high, low, close, volume, 19, 19, 19, 'ema');

    assert.strictEqual(result.demand_index.length, close.length);
    assert.strictEqual(result.signal.length, close.length);
    assert(result.demand_index.some(v => !isNaN(v)));
    assert(result.signal.some(v => !isNaN(v)));

    for (const value of result.demand_index) {
        if (!isNaN(value)) {
            assert(value >= -100 && value <= 100, `unexpected DI value ${value}`);
        }
    }
});

test('demand_index_js rejects invalid parameters', () => {
    const high = new Float64Array(testData.high.slice(0, 128));
    const low = new Float64Array(testData.low.slice(0, 128));
    const close = new Float64Array(testData.close.slice(0, 128));
    const volume = new Float64Array(testData.volume.slice(0, 128));

    assert.throws(() => {
        wasm.demand_index_js(high, low, close, volume, 0, 19, 19, 'ema');
    }, /Invalid len_bs/);

    assert.throws(() => {
        wasm.demand_index_js(high, low, close, volume, 19, 19, 19, 'bad_ma');
    }, /Invalid ma_type/);
});

test('demand_index_into pointer path matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const volume = new Float64Array(testData.volume.slice(0, 256));
    const safe = wasm.demand_index_js(high, low, close, volume, 19, 19, 19, 'ema');

    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const volumePtr = wasm.copy_f64_array(volume);
    const diPtr = wasm.demand_index_alloc(close.length);
    const signalPtr = wasm.demand_index_alloc(close.length);

    try {
        wasm.demand_index_into(
            highPtr,
            lowPtr,
            closePtr,
            volumePtr,
            diPtr,
            signalPtr,
            close.length,
            19,
            19,
            19,
            'ema',
        );
        const diValues = wasm.read_f64_array(diPtr, close.length);
        const signalValues = wasm.read_f64_array(signalPtr, close.length);
        assertArrayClose(diValues, safe.demand_index, 1e-10, 'DI pointer-path mismatch');
        assertArrayClose(signalValues, safe.signal, 1e-10, 'signal pointer-path mismatch');
    } finally {
        wasm.demand_index_free(diPtr, close.length);
        wasm.demand_index_free(signalPtr, close.length);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
        wasm.deallocate_f64_array(volumePtr);
    }
});

test('demand_index_batch_js single parameter set matches safe API', () => {
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    const volume = new Float64Array(testData.volume.slice(0, 256));
    const batch = wasm.demand_index_batch_js(high, low, close, volume, {
        len_bs_range: [19, 19, 0],
        len_bs_ma_range: [19, 19, 0],
        len_di_ma_range: [19, 19, 0],
        ma_type: 'ema',
    });
    const safe = wasm.demand_index_js(high, low, close, volume, 19, 19, 19, 'ema');

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assertArrayClose(batch.demand_index.slice(0, close.length), safe.demand_index, 1e-10, 'batch DI mismatch');
    assertArrayClose(batch.signal.slice(0, close.length), safe.signal, 1e-10, 'batch signal mismatch');
});

test('demand_index_batch_js metadata matches requested grid', () => {
    const high = new Float64Array(testData.high.slice(0, 192));
    const low = new Float64Array(testData.low.slice(0, 192));
    const close = new Float64Array(testData.close.slice(0, 192));
    const volume = new Float64Array(testData.volume.slice(0, 192));
    const batch = wasm.demand_index_batch_js(high, low, close, volume, {
        len_bs_range: [10, 12, 2],
        len_bs_ma_range: [8, 10, 2],
        len_di_ma_range: [5, 6, 1],
        ma_type: 'ema',
    });

    assert.strictEqual(batch.rows, 8);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.demand_index.length, 8 * close.length);
    assert.strictEqual(batch.signal.length, 8 * close.length);
    assert.deepStrictEqual(batch.len_bs, [10, 10, 10, 10, 12, 12, 12, 12]);
    assert.deepStrictEqual(batch.len_bs_ma, [8, 8, 10, 10, 8, 8, 10, 10]);
    assert.deepStrictEqual(batch.len_di_ma, [5, 6, 5, 6, 5, 6, 5, 6]);
});

test('demand_index_batch_js rejects invalid config', () => {
    const high = new Float64Array(testData.high.slice(0, 128));
    const low = new Float64Array(testData.low.slice(0, 128));
    const close = new Float64Array(testData.close.slice(0, 128));
    const volume = new Float64Array(testData.volume.slice(0, 128));

    assert.throws(() => {
        wasm.demand_index_batch_js(high, low, close, volume, {
            len_bs_range: [0, 10, 1],
            len_bs_ma_range: [19, 19, 0],
            len_di_ma_range: [19, 19, 0],
        });
    }, /Invalid len_bs/);
});
