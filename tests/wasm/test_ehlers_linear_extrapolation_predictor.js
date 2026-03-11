import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function sampleData() {
    return new Float64Array([
        100.0, 100.5, 101.0, 100.7, 100.2, 99.8, 99.5, 99.7, 100.0,
        100.4, 100.8, 101.0, 100.9, 100.7, 100.4, 100.1, 99.9, NaN,
        100.0, 100.3, 100.6, 100.8, 100.7, 100.5, 100.2, 100.0, 99.8,
        99.9, 100.1, 100.4, 100.7, 100.9, 101.0, 100.8, 100.5,
    ]);
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
});

test('ehlers_linear_extrapolation_predictor_js output contract on constant series', () => {
    const data = new Float64Array(24).fill(100.0);
    const out = wasm.ehlers_linear_extrapolation_predictor_js(
        data, 8, 4, 0.7, 3, 'predict_filter_crosses'
    );

    assert.strictEqual(out.prediction.length, data.length);
    assert(out.prediction.slice(0, 15).every(isNaN));
    assert(out.filter.slice(0, 15).every(isNaN));
    assert(out.state.slice(0, 15).every(isNaN));
    assert(out.go_long.slice(0, 15).every(isNaN));
    assert(out.go_short.slice(0, 15).every(isNaN));
    assert(out.prediction.slice(15).every(v => Math.abs(v) <= 1e-12));
    assert(out.filter.slice(15).every(v => Math.abs(v) <= 1e-12));
    assert(out.state.slice(15).every(v => Math.abs(v) <= 1e-12));
    assert(out.go_long.slice(15).every(v => Math.abs(v) <= 1e-12));
    assert(out.go_short.slice(15).every(v => Math.abs(v) <= 1e-12));
});

test('ehlers_linear_extrapolation_predictor_js rejects invalid parameters', () => {
    const data = new Float64Array(Array.from({ length: 32 }, (_, i) => 100 + i));

    assert.throws(
        () => wasm.ehlers_linear_extrapolation_predictor_js(
            data, 8, 4, 0.7, 11, 'predict_filter_crosses'
        ),
        /Invalid bars_forward/
    );
    assert.throws(
        () => wasm.ehlers_linear_extrapolation_predictor_js(
            data, 8, 4, 0.7, 3, 'bad_mode'
        ),
        /Invalid signal_mode/
    );
});

test('ehlers_linear_extrapolation_predictor_into pointer path matches safe API', () => {
    const data = new Float64Array(Array.from({ length: 96 }, (_, i) => 90 + i * 0.2));
    const safe = wasm.ehlers_linear_extrapolation_predictor_js(
        data, 8, 4, 0.7, 3, 'filter_middle_crosses'
    );

    const dataPtr = wasm.copy_f64_array(data);
    const predictionPtr = wasm.ehlers_linear_extrapolation_predictor_alloc(data.length);
    const filterPtr = wasm.ehlers_linear_extrapolation_predictor_alloc(data.length);
    const statePtr = wasm.ehlers_linear_extrapolation_predictor_alloc(data.length);
    const longPtr = wasm.ehlers_linear_extrapolation_predictor_alloc(data.length);
    const shortPtr = wasm.ehlers_linear_extrapolation_predictor_alloc(data.length);

    try {
        wasm.ehlers_linear_extrapolation_predictor_into(
            dataPtr,
            predictionPtr,
            filterPtr,
            statePtr,
            longPtr,
            shortPtr,
            data.length,
            8,
            4,
            0.7,
            3,
            'filter_middle_crosses'
        );
        assertArrayClose(
            wasm.read_f64_array(predictionPtr, data.length),
            safe.prediction,
            1e-12,
            'prediction mismatch'
        );
        assertArrayClose(
            wasm.read_f64_array(filterPtr, data.length),
            safe.filter,
            1e-12,
            'filter mismatch'
        );
        assertArrayClose(
            wasm.read_f64_array(statePtr, data.length),
            safe.state,
            1e-12,
            'state mismatch'
        );
        assertArrayClose(
            wasm.read_f64_array(longPtr, data.length),
            safe.go_long,
            1e-12,
            'go_long mismatch'
        );
        assertArrayClose(
            wasm.read_f64_array(shortPtr, data.length),
            safe.go_short,
            1e-12,
            'go_short mismatch'
        );
    } finally {
        wasm.ehlers_linear_extrapolation_predictor_free(predictionPtr, data.length);
        wasm.ehlers_linear_extrapolation_predictor_free(filterPtr, data.length);
        wasm.ehlers_linear_extrapolation_predictor_free(statePtr, data.length);
        wasm.ehlers_linear_extrapolation_predictor_free(longPtr, data.length);
        wasm.ehlers_linear_extrapolation_predictor_free(shortPtr, data.length);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('ehlers_linear_extrapolation_predictor_js reset after NaN', () => {
    const out = wasm.ehlers_linear_extrapolation_predictor_js(
        sampleData(), 8, 4, 0.7, 3, 'predict_middle_crosses'
    );

    assert(isNaN(out.prediction[17]));
    assert(isNaN(out.prediction[32]));
    assert(Number.isFinite(out.prediction.at(-1)));
});

test('ehlers_linear_extrapolation_predictor_batch_js single parameter set matches safe API', () => {
    const data = new Float64Array(Array.from({ length: 96 }, (_, i) => 90 + i * 0.2));
    const batch = wasm.ehlers_linear_extrapolation_predictor_batch_js(data, {
        high_pass_length_range: [8, 8, 0],
        low_pass_length_range: [4, 4, 0],
        gain_range: [0.7, 0.7, 0.0],
        bars_forward_range: [3, 3, 0],
        signal_mode: 'filter_middle_crosses',
    });
    const single = wasm.ehlers_linear_extrapolation_predictor_js(
        data, 8, 4, 0.7, 3, 'filter_middle_crosses'
    );

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, data.length);
    assert.deepStrictEqual(batch.high_pass_lengths, [8]);
    assert.deepStrictEqual(batch.low_pass_lengths, [4]);
    assert.deepStrictEqual(batch.gains, [0.7]);
    assert.deepStrictEqual(batch.bars_forwards, [3]);
    assert.deepStrictEqual(batch.signal_modes, ['filter_middle_crosses']);
    assertArrayClose(batch.prediction, single.prediction, 1e-12, 'batch prediction mismatch');
    assertArrayClose(batch.filter, single.filter, 1e-12, 'batch filter mismatch');
    assertArrayClose(batch.state, single.state, 1e-12, 'batch state mismatch');
    assertArrayClose(batch.go_long, single.go_long, 1e-12, 'batch go_long mismatch');
    assertArrayClose(batch.go_short, single.go_short, 1e-12, 'batch go_short mismatch');
});

test('ehlers_linear_extrapolation_predictor_batch_js metadata and batch_into', () => {
    const data = new Float64Array(Array.from({ length: 96 }, (_, i) => 95 + i * 0.25));
    const batch = wasm.ehlers_linear_extrapolation_predictor_batch_js(data, {
        high_pass_length_range: [8, 10, 2],
        low_pass_length_range: [4, 6, 2],
        gain_range: [0.7, 0.7, 0.0],
        bars_forward_range: [3, 5, 2],
        signal_mode: 'predict_filter_crosses',
    });

    assert.strictEqual(batch.rows, 8);
    assert.strictEqual(batch.cols, data.length);
    assert.deepStrictEqual(batch.high_pass_lengths, [8, 8, 8, 8, 10, 10, 10, 10]);
    assert.deepStrictEqual(batch.low_pass_lengths, [4, 4, 6, 6, 4, 4, 6, 6]);
    assert.deepStrictEqual(batch.gains, [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]);
    assert.deepStrictEqual(batch.bars_forwards, [3, 5, 3, 5, 3, 5, 3, 5]);
    assert.deepStrictEqual(batch.signal_modes, Array(8).fill('predict_filter_crosses'));

    const dataPtr = wasm.copy_f64_array(data);
    const total = batch.rows * data.length;
    const predictionPtr = wasm.ehlers_linear_extrapolation_predictor_alloc(total);
    const filterPtr = wasm.ehlers_linear_extrapolation_predictor_alloc(total);
    const statePtr = wasm.ehlers_linear_extrapolation_predictor_alloc(total);
    const longPtr = wasm.ehlers_linear_extrapolation_predictor_alloc(total);
    const shortPtr = wasm.ehlers_linear_extrapolation_predictor_alloc(total);

    try {
        const rows = wasm.ehlers_linear_extrapolation_predictor_batch_into(
            dataPtr,
            predictionPtr,
            filterPtr,
            statePtr,
            longPtr,
            shortPtr,
            data.length,
            8, 10, 2,
            4, 6, 2,
            0.7, 0.7, 0.0,
            3, 5, 2,
            'predict_filter_crosses'
        );
        assert.strictEqual(rows, 8);
        assertArrayClose(
            wasm.read_f64_array(predictionPtr, total),
            batch.prediction,
            1e-12,
            'batch_into prediction mismatch'
        );
        assertArrayClose(
            wasm.read_f64_array(filterPtr, total),
            batch.filter,
            1e-12,
            'batch_into filter mismatch'
        );
        assertArrayClose(
            wasm.read_f64_array(statePtr, total),
            batch.state,
            1e-12,
            'batch_into state mismatch'
        );
        assertArrayClose(
            wasm.read_f64_array(longPtr, total),
            batch.go_long,
            1e-12,
            'batch_into go_long mismatch'
        );
        assertArrayClose(
            wasm.read_f64_array(shortPtr, total),
            batch.go_short,
            1e-12,
            'batch_into go_short mismatch'
        );
    } finally {
        wasm.ehlers_linear_extrapolation_predictor_free(predictionPtr, total);
        wasm.ehlers_linear_extrapolation_predictor_free(filterPtr, total);
        wasm.ehlers_linear_extrapolation_predictor_free(statePtr, total);
        wasm.ehlers_linear_extrapolation_predictor_free(longPtr, total);
        wasm.ehlers_linear_extrapolation_predictor_free(shortPtr, total);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('ehlers_linear_extrapolation_predictor_batch_js rejects invalid config', () => {
    const data = new Float64Array(Array.from({ length: 32 }, (_, i) => 100 + i));
    assert.throws(
        () => wasm.ehlers_linear_extrapolation_predictor_batch_js(data, {
            high_pass_length_range: [8, 8, 0],
            low_pass_length_range: [4, 4, 0],
            gain_range: [0.7, 0.7, 0.0],
            bars_forward_range: [11, 11, 0],
            signal_mode: 'predict_filter_crosses',
        }),
        /Invalid bars_forward/
    );
});
