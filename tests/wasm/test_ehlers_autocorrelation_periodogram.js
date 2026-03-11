import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function cycleData(length = 256, period = 20.0) {
    return new Float64Array(Array.from({ length }, (_, i) => {
        const phase = 2.0 * Math.PI * i / period;
        return Math.sin(phase) + 0.15 * Math.cos(phase * 0.5);
    }));
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
});

test('eacp_js output contract on cycle', () => {
    const data = cycleData();
    const out = wasm.ehlers_autocorrelation_periodogram_js(data, 8, 48, 3, true);

    assert.strictEqual(out.dominant_cycle.length, data.length);
    assert.strictEqual(out.normalized_power.length, data.length);
    assert(out.dominant_cycle.slice(0, 50).every(isNaN));
    assert(Number.isFinite(out.dominant_cycle.at(-1)));
    assert(Number.isFinite(out.normalized_power.at(-1)));
    assert(out.dominant_cycle.at(-1) > 15.0 && out.dominant_cycle.at(-1) < 25.0);
    assert(out.normalized_power.at(-1) >= 0.0 && out.normalized_power.at(-1) <= 1.0);
});

test('eacp_js rejects invalid parameters', () => {
    const data = cycleData(64, 16.0);
    assert.throws(
        () => wasm.ehlers_autocorrelation_periodogram_js(data, 2, 32, 3, true),
        /Invalid min_period/
    );
    assert.throws(
        () => wasm.ehlers_autocorrelation_periodogram_js(data, 16, 8, 3, true),
        /Invalid period order/
    );
});

test('eacp_into pointer path matches safe API', () => {
    const data = cycleData(160, 22.0);
    const safe = wasm.ehlers_autocorrelation_periodogram_js(data, 8, 48, 3, true);
    const dataPtr = wasm.copy_f64_array(data);
    const domPtr = wasm.ehlers_autocorrelation_periodogram_alloc(data.length);
    const pwrPtr = wasm.ehlers_autocorrelation_periodogram_alloc(data.length);

    try {
        wasm.ehlers_autocorrelation_periodogram_into(dataPtr, domPtr, pwrPtr, data.length, 8, 48, 3, true);
        assertArrayClose(
            wasm.read_f64_array(domPtr, data.length),
            safe.dominant_cycle,
            1e-9,
            'dominant cycle mismatch'
        );
        assertArrayClose(
            wasm.read_f64_array(pwrPtr, data.length),
            safe.normalized_power,
            1e-9,
            'normalized power mismatch'
        );
    } finally {
        wasm.ehlers_autocorrelation_periodogram_free(domPtr, data.length);
        wasm.ehlers_autocorrelation_periodogram_free(pwrPtr, data.length);
        wasm.deallocate_f64_array(dataPtr);
    }
});

test('eacp_js reset after NaN', () => {
    const data = cycleData(192, 18.0);
    data[90] = NaN;
    const out = wasm.ehlers_autocorrelation_periodogram_js(data, 8, 32, 3, true);

    assert(isNaN(out.dominant_cycle[90]));
    assert(isNaN(out.dominant_cycle[120]));
    assert(Number.isFinite(out.dominant_cycle.at(-1)));
});

test('eacp_batch_js single parameter set matches safe API', () => {
    const data = cycleData(160, 22.0);
    const batch = wasm.ehlers_autocorrelation_periodogram_batch_js(data, {
        min_period_range: [8, 8, 0],
        max_period_range: [48, 48, 0],
        avg_length_range: [3, 3, 0],
        enhance: true,
    });
    const single = wasm.ehlers_autocorrelation_periodogram_js(data, 8, 48, 3, true);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, data.length);
    assert.deepStrictEqual(batch.min_periods, [8]);
    assert.deepStrictEqual(batch.max_periods, [48]);
    assert.deepStrictEqual(batch.avg_lengths, [3]);
    assert.deepStrictEqual(batch.enhance_flags, [true]);
    assertArrayClose(batch.dominant_cycle, single.dominant_cycle, 1e-9, 'batch dominant mismatch');
    assertArrayClose(batch.normalized_power, single.normalized_power, 1e-9, 'batch power mismatch');
});

test('eacp_batch_js metadata and batch_into', () => {
    const data = cycleData(160, 21.0);
    const batch = wasm.ehlers_autocorrelation_periodogram_batch_js(data, {
        min_period_range: [8, 10, 2],
        max_period_range: [24, 28, 4],
        avg_length_range: [3, 3, 0],
        enhance: false,
    });

    assert.strictEqual(batch.rows, 4);
    assert.strictEqual(batch.cols, data.length);
    assert.deepStrictEqual(batch.min_periods, [8, 8, 10, 10]);
    assert.deepStrictEqual(batch.max_periods, [24, 28, 24, 28]);
    assert.deepStrictEqual(batch.avg_lengths, [3, 3, 3, 3]);
    assert.deepStrictEqual(batch.enhance_flags, [false, false, false, false]);

    const dataPtr = wasm.copy_f64_array(data);
    const total = batch.rows * data.length;
    const domPtr = wasm.ehlers_autocorrelation_periodogram_alloc(total);
    const pwrPtr = wasm.ehlers_autocorrelation_periodogram_alloc(total);

    try {
        const rows = wasm.ehlers_autocorrelation_periodogram_batch_into(
            dataPtr, domPtr, pwrPtr, data.length,
            8, 10, 2,
            24, 28, 4,
            3, 3, 0,
            false
        );
        assert.strictEqual(rows, 4);
        assertArrayClose(
            wasm.read_f64_array(domPtr, total),
            batch.dominant_cycle,
            1e-9,
            'batch_into dominant mismatch'
        );
        assertArrayClose(
            wasm.read_f64_array(pwrPtr, total),
            batch.normalized_power,
            1e-9,
            'batch_into power mismatch'
        );
    } finally {
        wasm.ehlers_autocorrelation_periodogram_free(domPtr, total);
        wasm.ehlers_autocorrelation_periodogram_free(pwrPtr, total);
        wasm.deallocate_f64_array(dataPtr);
    }
});
