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

test('logarithmic_moving_average output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 320));
    const volume = new Float64Array(0);
    const result = wasm.logarithmic_moving_average(
        close,
        volume,
        100,
        2.5,
        'ema',
        10,
        1.2,
        0.5,
        -0.5,
    );

    assert.strictEqual(result.lma.length, close.length);
    assert.strictEqual(result.signal.length, close.length);
    assert.strictEqual(result.position.length, close.length);
    assert.strictEqual(result.momentum_confirmed.length, close.length);
    assert(Array.from(result.lma.slice(0, 99)).every(v => isNaN(v)));
    assert(Array.from(result.signal.slice(120)).some(v => !isNaN(v)));
});

test('logarithmic_moving_average rejects vwma without volume', () => {
    const close = new Float64Array(testData.close.slice(0, 180));
    const volume = new Float64Array(0);

    assert.throws(() => {
        wasm.logarithmic_moving_average(
            close,
            volume,
            100,
            2.5,
            'vwma',
            10,
            1.2,
            0.5,
            -0.5,
        );
    }, /requires volume/i);
});

test('logarithmic_moving_average_into_host pointer path matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 240));
    const volume = new Float64Array(testData.volume.slice(0, 240));
    const safe = wasm.logarithmic_moving_average(
        close,
        volume,
        100,
        2.5,
        'vwma',
        10,
        1.2,
        0.5,
        -0.5,
    );
    const outPtr = wasm.logarithmic_moving_average_alloc(close.length);

    try {
        wasm.logarithmic_moving_average_into_host(
            close,
            volume,
            outPtr,
            100,
            2.5,
            'vwma',
            10,
            1.2,
            0.5,
            -0.5,
        );
        const out = wasm.read_f64_array(outPtr, 4 * close.length);
        const lma = out.slice(0, close.length);
        const signal = out.slice(close.length, 2 * close.length);
        const position = out.slice(2 * close.length, 3 * close.length);
        const momentumConfirmed = out.slice(3 * close.length, 4 * close.length);
        assertArrayClose(lma, safe.lma, 1e-10, 'lma mismatch');
        assertArrayClose(signal, safe.signal, 1e-10, 'signal mismatch');
        assertArrayClose(position, safe.position, 1e-10, 'position mismatch');
        assertArrayClose(momentumConfirmed, safe.momentum_confirmed, 1e-10, 'momentum mismatch');
    } finally {
        wasm.logarithmic_moving_average_free(outPtr, close.length);
    }
});

test('logarithmic_moving_average_batch single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 220));
    const volume = new Float64Array(testData.volume.slice(0, 220));
    const batch = wasm.logarithmic_moving_average_batch(close, volume, {
        period_range: [100, 100, 0],
        steepness_range: [2.5, 2.5, 0.0],
        smooth_range: [10, 10, 0],
        momentum_weight_range: [1.2, 1.2, 0.0],
        long_threshold_range: [0.5, 0.5, 0.0],
        short_threshold_range: [-0.5, -0.5, 0.0],
        ma_type: 'vwma',
    });
    const single = wasm.logarithmic_moving_average(
        close,
        volume,
        100,
        2.5,
        'vwma',
        10,
        1.2,
        0.5,
        -0.5,
    );

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos[0].ma_type, 'vwma');
    assertArrayClose(batch.signal, single.signal, 1e-10, 'batch signal mismatch');
});

test('logarithmic_moving_average_batch metadata', () => {
    const close = new Float64Array(testData.close.slice(0, 180));
    const batch = wasm.logarithmic_moving_average_batch(close, new Float64Array(0), {
        period_range: [90, 100, 10],
        steepness_range: [2.0, 2.5, 0.5],
        smooth_range: [8, 8, 0],
        momentum_weight_range: [1.2, 1.2, 0.0],
        long_threshold_range: [0.5, 0.5, 0.0],
        short_threshold_range: [-0.5, -0.5, 0.0],
        ma_type: 'ema',
    });

    assert.strictEqual(batch.rows, 4);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.combos.length, 4);
    assert.deepStrictEqual(batch.combos.map(combo => combo.period), [90, 90, 100, 100]);
});

test('logarithmic_moving_average_batch_into pointer path matches safe batch API', () => {
    const close = new Float64Array(testData.close.slice(0, 180));
    const volume = new Float64Array(testData.volume.slice(0, 180));
    const batch = wasm.logarithmic_moving_average_batch(close, volume, {
        period_range: [90, 100, 10],
        steepness_range: [2.0, 2.5, 0.5],
        smooth_range: [8, 8, 0],
        momentum_weight_range: [1.2, 1.2, 0.0],
        long_threshold_range: [0.5, 0.5, 0.0],
        short_threshold_range: [-0.5, -0.5, 0.0],
        ma_type: 'vwma',
    });

    const rows = batch.rows;
    const total = rows * close.length;
    const dataPtr = wasm.logarithmic_moving_average_alloc(close.length);
    const volumePtr = wasm.logarithmic_moving_average_alloc(close.length);
    const lmaPtr = wasm.logarithmic_moving_average_alloc(total);
    const signalPtr = wasm.logarithmic_moving_average_alloc(total);
    const positionPtr = wasm.logarithmic_moving_average_alloc(total);
    const momentumPtr = wasm.logarithmic_moving_average_alloc(total);

    try {
        const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        if (!memory || !memory.buffer) {
            console.warn('Skipping logarithmic_moving_average_batch_into test: wasm memory accessor unavailable');
            return;
        }
        new Float64Array(memory.buffer, dataPtr, close.length).set(close);
        new Float64Array(memory.buffer, volumePtr, close.length).set(volume);

        const returnedRows = wasm.logarithmic_moving_average_batch_into(
            dataPtr,
            volumePtr,
            close.length,
            lmaPtr,
            signalPtr,
            positionPtr,
            momentumPtr,
            close.length,
            90,
            100,
            10,
            2.0,
            2.5,
            0.5,
            8,
            8,
            0,
            1.2,
            1.2,
            0.0,
            0.5,
            0.5,
            0.0,
            -0.5,
            -0.5,
            0.0,
            'vwma',
        );

        assert.strictEqual(returnedRows, rows);
        assertArrayClose(wasm.read_f64_array(lmaPtr, total), batch.lma, 1e-10, 'batch_into lma mismatch');
        assertArrayClose(wasm.read_f64_array(signalPtr, total), batch.signal, 1e-10, 'batch_into signal mismatch');
        assertArrayClose(wasm.read_f64_array(positionPtr, total), batch.position, 1e-10, 'batch_into position mismatch');
        assertArrayClose(
            wasm.read_f64_array(momentumPtr, total),
            batch.momentum_confirmed,
            1e-10,
            'batch_into momentum mismatch',
        );
    } finally {
        wasm.logarithmic_moving_average_free(dataPtr, close.length);
        wasm.logarithmic_moving_average_free(volumePtr, close.length);
        wasm.logarithmic_moving_average_free(lmaPtr, total);
        wasm.logarithmic_moving_average_free(signalPtr, total);
        wasm.logarithmic_moving_average_free(positionPtr, total);
        wasm.logarithmic_moving_average_free(momentumPtr, total);
    }
});
