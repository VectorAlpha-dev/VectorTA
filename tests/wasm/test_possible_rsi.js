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

test('possible_rsi_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 384));
    const result = wasm.possible_rsi_js(
        close, 32, 'regular', 100, 'gaussian_fisher', 15, 15, 20, 0.2, 0.2,
        'zeroline_crossover', false, 15
    );

    assert.strictEqual(result.value.length, close.length);
    assert.strictEqual(result.buy_level.length, close.length);
    assert.strictEqual(result.sell_level.length, close.length);
    assert.strictEqual(result.middle_level.length, close.length);
    assert.strictEqual(result.state.length, close.length);
    assert(result.value.some(v => !isNaN(v)));
    assert(!isNaN(result.value[result.value.length - 1]));
});

test('possible_rsi_js rejects invalid params', () => {
    const close = new Float64Array(testData.close.slice(0, 96));

    assert.throws(() => {
        wasm.possible_rsi_js(
            close, 0, 'regular', 100, 'gaussian_fisher', 15, 15, 20, 0.2, 0.2,
            'zeroline_crossover', false, 15
        );
    }, /Invalid period/);

    assert.throws(() => {
        wasm.possible_rsi_js(
            close, 32, 'bad_mode', 100, 'gaussian_fisher', 15, 15, 20, 0.2, 0.2,
            'zeroline_crossover', false, 15
        );
    }, /Invalid RSI mode/);
});

test('possible_rsi_into pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 260));
    const safe = wasm.possible_rsi_js(
        close, 28, 'cutler', 90, 'softmax', 12, 11, 18, 0.2, 0.2,
        'levels_crossover', true, 13
    );
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const dataPtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.possible_rsi_alloc(close.length);

    try {
        new Float64Array(memory.buffer, dataPtr, close.length).set(close);
        wasm.possible_rsi_into(
            dataPtr, outPtr, close.length, 28, 'cutler', 90, 'softmax', 12, 11, 18,
            0.2, 0.2, 'levels_crossover', true, 13
        );
        const flat = new Float64Array(memory.buffer, outPtr, 7 * close.length);
        const value = Array.from(flat.slice(0, close.length));
        const buy = Array.from(flat.slice(close.length, 2 * close.length));
        const sell = Array.from(flat.slice(2 * close.length, 3 * close.length));
        const middle = Array.from(flat.slice(3 * close.length, 4 * close.length));
        const state = Array.from(flat.slice(4 * close.length, 5 * close.length));
        assertArrayClose(value, safe.value, 1e-12, 'pointer value mismatch');
        assertArrayClose(buy, safe.buy_level, 1e-12, 'pointer buy mismatch');
        assertArrayClose(sell, safe.sell_level, 1e-12, 'pointer sell mismatch');
        assertArrayClose(middle, safe.middle_level, 1e-12, 'pointer middle mismatch');
        assertArrayClose(state, safe.state, 1e-12, 'pointer state mismatch');
    } finally {
        wasm.possible_rsi_free(outPtr, close.length);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('possible_rsi_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 240));
    const batch = wasm.possible_rsi_batch_js(close, {
        period_range: [32, 32, 0],
        rsi_mode: 'regular',
        norm_period_range: [100, 100, 0],
        normalization_mode: 'gaussian_fisher',
        normalization_length_range: [15, 15, 0],
        nonlag_period_range: [15, 15, 0],
        dynamic_zone_period_range: [20, 20, 0],
        buy_probability_range: [0.2, 0.2, 0.0],
        sell_probability_range: [0.2, 0.2, 0.0],
        signal_type: 'zeroline_crossover',
        run_highpass: false,
        highpass_period: 15,
    });
    const single = wasm.possible_rsi_js(
        close, 32, 'regular', 100, 'gaussian_fisher', 15, 15, 20, 0.2, 0.2,
        'zeroline_crossover', false, 15
    );

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].period, 32);
    assert.strictEqual(batch.combos[0].rsi_mode, 'regular');
    assertArrayClose(batch.value, single.value, 1e-12, 'batch value mismatch');
    assertArrayClose(batch.state, single.state, 1e-12, 'batch state mismatch');
});

test('possible_rsi_batch_into metadata matches requested ranges', () => {
    const close = new Float64Array(testData.close.slice(0, 220));
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 9;
    const total = rows * close.length;
    const dataPtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.possible_rsi_alloc(total);

    try {
        new Float64Array(memory.buffer, dataPtr, close.length).set(close);
        const actualRows = wasm.possible_rsi_batch_into(
            dataPtr, outPtr, close.length,
            30, 32, 1, 'regular',
            100, 100, 0,
            'gaussian_fisher',
            15, 15, 0,
            14, 16, 1,
            20, 20, 0,
            0.2, 0.2, 0.0,
            0.2, 0.2, 0.0,
            'zeroline_crossover', false, 15
        );
        assert.strictEqual(actualRows, rows);

        const flat = new Float64Array(memory.buffer, outPtr, 7 * total);
        const value = Array.from(flat.slice(0, total));
        const state = Array.from(flat.slice(4 * total, 5 * total));
        const jsBatch = wasm.possible_rsi_batch_js(close, {
            period_range: [30, 32, 1],
            rsi_mode: 'regular',
            norm_period_range: [100, 100, 0],
            normalization_mode: 'gaussian_fisher',
            normalization_length_range: [15, 15, 0],
            nonlag_period_range: [14, 16, 1],
            dynamic_zone_period_range: [20, 20, 0],
            buy_probability_range: [0.2, 0.2, 0.0],
            sell_probability_range: [0.2, 0.2, 0.0],
            signal_type: 'zeroline_crossover',
            run_highpass: false,
            highpass_period: 15,
        });
        assertArrayClose(value, jsBatch.value, 1e-12, 'batch_into value mismatch');
        assertArrayClose(state, jsBatch.state, 1e-12, 'batch_into state mismatch');
    } finally {
        wasm.possible_rsi_free(outPtr, total);
        wasm.deallocate_f64_array(dataPtr);
    }
});
