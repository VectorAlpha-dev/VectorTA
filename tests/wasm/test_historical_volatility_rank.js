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

test('historical_volatility_rank_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.historical_volatility_rank_js(close, 10, 20, 365.0, 1.0);

    assert(result.hvr);
    assert(result.hv);
    assert.strictEqual(result.hvr.length, close.length);
    assert.strictEqual(result.hv.length, close.length);

    const hvFirst = result.hv.findIndex(v => !isNaN(v));
    const hvrFirst = result.hvr.findIndex(v => !isNaN(v));
    assert.strictEqual(hvFirst, 10);
    assert.strictEqual(hvrFirst, 29);
});

test('historical_volatility_rank_js rejects invalid parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 128));

    assert.throws(() => {
        wasm.historical_volatility_rank_js(close, 0, 20, 365.0, 1.0);
    }, /Invalid hv_length/);

    assert.throws(() => {
        wasm.historical_volatility_rank_js(close, 10, 0, 365.0, 1.0);
    }, /Invalid rank_length/);

    assert.throws(() => {
        wasm.historical_volatility_rank_js(close, 10, 20, 0.0, 1.0);
    }, /Invalid annualization_days/);
});

test('historical_volatility_rank_into pointer path matches safe API', (t) => {
    const close = new Float64Array(testData.close.slice(0, 200));
    const safe = wasm.historical_volatility_rank_js(close, 10, 20, 365.0, 1.0);
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    if (!memory || !memory.buffer) {
        t.skip('raw wasm memory is not exposed by this package build');
        return;
    }

    const inPtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.historical_volatility_rank_alloc(close.length);

    try {
        new Float64Array(memory.buffer, inPtr, close.length).set(close);
        wasm.historical_volatility_rank_into(
            inPtr,
            outPtr,
            close.length,
            10,
            20,
            365.0,
            1.0,
        );

        const out = new Float64Array(memory.buffer, outPtr, 2 * close.length);
        const hvr = Array.from(out.subarray(0, close.length));
        const hv = Array.from(out.subarray(close.length, 2 * close.length));

        assertArrayClose(hvr, safe.hvr, 1e-10, 'pointer hvr mismatch');
        assertArrayClose(hv, safe.hv, 1e-10, 'pointer hv mismatch');
    } finally {
        wasm.historical_volatility_rank_free(outPtr, close.length);
        wasm.deallocate_f64_array(inPtr);
    }
});

test('historical_volatility_rank_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.historical_volatility_rank_batch_js(close, {
        hv_length_range: [10, 10, 0],
        rank_length_range: [20, 20, 0],
        annualization_days_range: [365.0, 365.0, 0.0],
        bar_days_range: [1.0, 1.0, 0.0],
    });
    const single = wasm.historical_volatility_rank_js(close, 10, 20, 365.0, 1.0);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.hvr.length, close.length);
    assert.strictEqual(batch.hv.length, close.length);
    assert.strictEqual(batch.combos[0].hv_length, 10);
    assert.strictEqual(batch.combos[0].rank_length, 20);

    assertArrayClose(batch.hvr, single.hvr, 1e-10, 'batch hvr mismatch');
    assertArrayClose(batch.hv, single.hv, 1e-10, 'batch hv mismatch');
});

test('historical_volatility_rank_batch_js metadata matches requested ranges', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const batch = wasm.historical_volatility_rank_batch_js(close, {
        hv_length_range: [10, 12, 2],
        rank_length_range: [20, 24, 4],
        annualization_days_range: [252.0, 252.0, 0.0],
        bar_days_range: [1.0, 1.0, 0.0],
    });

    assert.strictEqual(batch.rows, 4);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(
        batch.combos.map(combo => combo.hv_length),
        [10, 10, 12, 12],
    );
    assert.deepStrictEqual(
        batch.combos.map(combo => combo.rank_length),
        [20, 24, 20, 24],
    );
});
