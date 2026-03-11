import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function sampleData(length, slotsPerDay) {
    const data = new Float64Array(length);
    for (let i = 0; i < length; i += 1) {
        const slot = i % slotsPerDay;
        const day = Math.floor(i / slotsPerDay);
        data[i] =
            1000.0 +
            day * 5.0 +
            Math.sin(slot * 0.11) * 30.0 +
            Math.cos(slot * 0.03) * 12.0 +
            (slot / slotsPerDay) * 25.0;
    }
    return data;
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath =
        process.platform === 'win32'
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
    wasm = await import(importPath);
});

test('half_causal_estimator_js output contract', () => {
    const slotsPerDay = 60;
    const data = sampleData(slotsPerDay * 4, slotsPerDay);
    const out = wasm.half_causal_estimator_js(data, {
        slots_per_day: slotsPerDay,
        enable_expected_value: true,
    });

    assert.strictEqual(out.estimate.length, data.length);
    assert.strictEqual(out.expected_value.length, data.length);
    assert(Number.isFinite(out.estimate.at(-1)));
    assert(Number.isFinite(out.expected_value.at(-1)));
});

test('half_causal_estimator_js rejects invalid parameters', () => {
    const data = sampleData(128, 32);
    assert.throws(
        () => wasm.half_causal_estimator_js(data, { slots_per_day: 1 }),
        /Invalid slots_per_day/,
    );
});

test('half_causal_estimator_into pointer path matches safe API', () => {
    const slotsPerDay = 48;
    const data = sampleData(slotsPerDay * 4, slotsPerDay);
    const safe = wasm.half_causal_estimator_js(data, {
        slots_per_day: slotsPerDay,
        enable_expected_value: true,
    });

    const inPtr = wasm.copy_f64_array(data);
    const estimatePtr = wasm.half_causal_estimator_alloc(data.length);
    const expectedPtr = wasm.half_causal_estimator_alloc(data.length);

    try {
        wasm.half_causal_estimator_into(inPtr, estimatePtr, expectedPtr, data.length, {
            slots_per_day: slotsPerDay,
            enable_expected_value: true,
        });
        assertArrayClose(
            wasm.read_f64_array(estimatePtr, data.length),
            safe.estimate,
            1e-12,
            'estimate mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(expectedPtr, data.length),
            safe.expected_value,
            1e-12,
            'expected_value mismatch',
        );
    } finally {
        wasm.half_causal_estimator_free(estimatePtr, data.length);
        wasm.half_causal_estimator_free(expectedPtr, data.length);
        wasm.deallocate_f64_array(inPtr);
    }
});

test('half_causal_estimator_batch_js single parameter set matches safe API', () => {
    const slotsPerDay = 60;
    const data = sampleData(slotsPerDay * 4, slotsPerDay);
    const batch = wasm.half_causal_estimator_batch_js(data, {
        slots_per_day: slotsPerDay,
        enable_expected_value: true,
    });
    const safe = wasm.half_causal_estimator_js(data, {
        slots_per_day: slotsPerDay,
        enable_expected_value: true,
    });

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, data.length);
    assertArrayClose(batch.estimate, safe.estimate, 1e-12, 'estimate batch mismatch');
    assertArrayClose(
        batch.expected_value,
        safe.expected_value,
        1e-12,
        'expected batch mismatch',
    );
});

test('half_causal_estimator_batch_into matches batch_js', () => {
    const slotsPerDay = 48;
    const data = sampleData(slotsPerDay * 4, slotsPerDay);
    const batch = wasm.half_causal_estimator_batch_js(data, {
        slots_per_day: slotsPerDay,
        data_period_range: [4, 5, 1],
        extra_smoothing_range: [0, 1, 1],
        enable_expected_value: true,
    });

    const inPtr = wasm.copy_f64_array(data);
    const total = batch.rows * data.length;
    const estimatePtr = wasm.half_causal_estimator_alloc(total);
    const expectedPtr = wasm.half_causal_estimator_alloc(total);

    try {
        const rows = wasm.half_causal_estimator_batch_into(
            inPtr,
            estimatePtr,
            expectedPtr,
            data.length,
            {
                slots_per_day: slotsPerDay,
                data_period_range: [4, 5, 1],
                extra_smoothing_range: [0, 1, 1],
                enable_expected_value: true,
            },
        );
        assert.strictEqual(rows, batch.rows);
        assertArrayClose(
            wasm.read_f64_array(estimatePtr, total),
            batch.estimate,
            1e-12,
            'batch estimate mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(expectedPtr, total),
            batch.expected_value,
            1e-12,
            'batch expected mismatch',
        );
    } finally {
        wasm.half_causal_estimator_free(estimatePtr, total);
        wasm.half_causal_estimator_free(expectedPtr, total);
        wasm.deallocate_f64_array(inPtr);
    }
});
