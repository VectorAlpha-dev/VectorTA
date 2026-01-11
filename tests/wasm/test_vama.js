/**
 * WASM binding tests for VAMA (Volatility Adjusted Moving Average).
 *
 * Notes:
 * - This indicator uses only a price series (no volume).
 * - The config for the batch API uses `base_period_range` and `vol_period_range`.
 */
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertClose, isNaN } from './test_utils.js';
import { compareWithRust } from './rust-comparison.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

const DEFAULT_PARAMS = {
    base_period: 113,
    vol_period: 51,
    smoothing: true,
    smooth_type: 3,
    smooth_period: 5,
};

test.before(async () => {
    try {
        const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
        const importPath = process.platform === 'win32'
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }

    testData = loadTestData();
});

test('VAMA partial params', () => {
    const close = new Float64Array(testData.close);
    const result = wasm.vama_js(
        close,
        DEFAULT_PARAMS.base_period,
        DEFAULT_PARAMS.vol_period,
        DEFAULT_PARAMS.smoothing,
        DEFAULT_PARAMS.smooth_type,
        DEFAULT_PARAMS.smooth_period
    );
    assert.strictEqual(result.length, close.length);
});

test('VAMA accuracy vs Rust', async () => {
    const close = new Float64Array(testData.close);
    const result = wasm.vama_js(
        close,
        DEFAULT_PARAMS.base_period,
        DEFAULT_PARAMS.vol_period,
        DEFAULT_PARAMS.smoothing,
        DEFAULT_PARAMS.smooth_type,
        DEFAULT_PARAMS.smooth_period
    );
    assert.strictEqual(result.length, close.length);

    await compareWithRust('vama', result, 'close', DEFAULT_PARAMS, 1e-10);
});

test('VAMA empty input', () => {
    const empty = new Float64Array([]);
    assert.throws(() => {
        wasm.vama_js(
            empty,
            DEFAULT_PARAMS.base_period,
            DEFAULT_PARAMS.vol_period,
            DEFAULT_PARAMS.smoothing,
            DEFAULT_PARAMS.smooth_type,
            DEFAULT_PARAMS.smooth_period
        );
    }, /empty/i);
});

test('VAMA all NaN input', () => {
    const allNaN = new Float64Array(128).fill(NaN);
    assert.throws(() => {
        wasm.vama_js(
            allNaN,
            DEFAULT_PARAMS.base_period,
            DEFAULT_PARAMS.vol_period,
            DEFAULT_PARAMS.smoothing,
            DEFAULT_PARAMS.smooth_type,
            DEFAULT_PARAMS.smooth_period
        );
    }, /All values are NaN/i);
});

test('VAMA invalid base period', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    assert.throws(() => {
        wasm.vama_js(data, 0, 5, false, 3, 5);
    }, /Invalid period/i);
});

test('VAMA invalid smooth type', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    assert.throws(() => {
        wasm.vama_js(data, 5, 5, true, 4, 3);
    }, /Invalid smooth type/i);
});

test('VAMA warmup NaNs', () => {
    const close = new Float64Array(testData.close.slice(0, 1000));
    const base = 21;
    const vol = 13;
    const smoothing = false;

    const result = wasm.vama_js(close, base, vol, smoothing, 3, 5);
    assert.strictEqual(result.length, close.length);

    const warmup = Math.max(base, vol) - 1;
    for (let i = 0; i < warmup; i++) {
        assert(isNaN(result[i]), `Expected NaN at warmup index ${i}`);
    }
    for (let i = warmup; i < Math.min(warmup + 50, result.length); i++) {
        assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
    }
});

test('VAMA zero-copy API', () => {
    const data = new Float64Array(testData.close.slice(0, 256));
    const len = data.length;

    const inPtr = wasm.vama_alloc(len);
    const outPtr = wasm.vama_alloc(len);
    assert(inPtr !== 0, 'Failed to allocate input buffer');
    assert(outPtr !== 0, 'Failed to allocate output buffer');

    try {
        const memory = wasm.__wasm.memory.buffer;
        new Float64Array(memory, inPtr, len).set(data);

        wasm.vama_into(
            inPtr,
            outPtr,
            len,
            DEFAULT_PARAMS.base_period,
            DEFAULT_PARAMS.vol_period,
            DEFAULT_PARAMS.smoothing,
            DEFAULT_PARAMS.smooth_type,
            DEFAULT_PARAMS.smooth_period
        );

        const memory2 = wasm.__wasm.memory.buffer;
        const outView = new Float64Array(memory2, outPtr, len);

        const regular = wasm.vama_js(
            data,
            DEFAULT_PARAMS.base_period,
            DEFAULT_PARAMS.vol_period,
            DEFAULT_PARAMS.smoothing,
            DEFAULT_PARAMS.smooth_type,
            DEFAULT_PARAMS.smooth_period
        );
        for (let i = 0; i < len; i++) {
            if (isNaN(regular[i]) && isNaN(outView[i])) continue;
            assertClose(outView[i], regular[i], 1e-10, `vama_into mismatch at ${i}`);
        }
    } finally {
        wasm.vama_free(inPtr, len);
        wasm.vama_free(outPtr, len);
    }
});

test('VAMA batch single parameter set', () => {
    const close = new Float64Array(testData.close.slice(0, 512));
    const config = {
        base_period_range: [21, 21, 0],
        vol_period_range: [13, 13, 0],
    };

    const result = wasm.vama_batch(close, config);
    assert(result.values);
    assert(result.combos);
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, close.length);
    assert.strictEqual(result.combos.length, 1);

    
    const single = wasm.vama_js(close, 21, 13, false, 3, 5);
    const row = result.values.slice(0, close.length);
    for (let i = 0; i < close.length; i++) {
        if (isNaN(single[i]) && isNaN(row[i])) continue;
        assertClose(row[i], single[i], 1e-10, `batch vs single mismatch at ${i}`);
    }
});

test('VAMA batch multiple parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const config = {
        base_period_range: [20, 30, 5], 
        vol_period_range: [10, 14, 2],  
    };

    const result = wasm.vama_batch(close, config);
    const expectedRows = 3 * 3;
    assert.strictEqual(result.rows, expectedRows);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, expectedRows * close.length);
    assert.strictEqual(result.combos.length, expectedRows);
});

test.after(() => {
    console.log('VAMA WASM tests completed');
});

