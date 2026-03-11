import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function sampleOhlcv(length = 256) {
    const high = new Float64Array(length);
    const low = new Float64Array(length);
    const close = new Float64Array(length);
    const volume = new Float64Array(length);
    for (let i = 0; i < length; i++) {
        const open = 100 + i * 0.03 + Math.sin(i * 0.08);
        close[i] = open + Math.cos(i * 0.11) * 0.9;
        high[i] = Math.max(open, close[i]) + 0.6 + Math.abs(Math.sin(i * 0.03)) * 0.25;
        low[i] = Math.min(open, close[i]) - 0.6 - Math.abs(Math.cos(i * 0.05)) * 0.2;
        volume[i] = 1000 + i * 4 + Math.sin(i * 0.09) * 180;
    }
    return { high, low, close, volume };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    const imported = await import(importPath);
    wasm = imported.default ?? imported;
});

test('volume_energy_reservoirs_js output contract', () => {
    const { high, low, close, volume } = sampleOhlcv();
    const out = wasm.volume_energy_reservoirs_js(high, low, close, volume, 20, 1.5);

    assert.strictEqual(out.momentum.length, close.length);
    assert.strictEqual(out.reservoir.length, close.length);
    assert.strictEqual(out.squeeze_active.length, close.length);
    assert.strictEqual(out.squeeze_start.length, close.length);
    assert.strictEqual(out.range_high.length, close.length);
    assert.strictEqual(out.range_low.length, close.length);
    for (const value of out.reservoir) {
        if (Number.isFinite(value)) {
            assert(value >= -1e-12 && value <= 10 + 1e-12);
        }
    }
});

test('volume_energy_reservoirs_js rejects invalid parameters', () => {
    const { high, low, close, volume } = sampleOhlcv(32);
    assert.throws(
        () => wasm.volume_energy_reservoirs_js(high, low, close, volume, 4, 1.5),
        /Invalid length/,
    );
    assert.throws(
        () => wasm.volume_energy_reservoirs_js(high, low, close, volume, 20, Number.NaN),
        /Invalid sensitivity/,
    );
});

test('volume_energy_reservoirs_into pointer path matches safe API', () => {
    const { high, low, close, volume } = sampleOhlcv(128);
    const safe = wasm.volume_energy_reservoirs_js(high, low, close, volume, 18, 1.7);

    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const volumePtr = wasm.copy_f64_array(volume);
    const momentumPtr = wasm.volume_energy_reservoirs_alloc(close.length);
    const reservoirPtr = wasm.volume_energy_reservoirs_alloc(close.length);
    const squeezeActivePtr = wasm.volume_energy_reservoirs_alloc(close.length);
    const squeezeStartPtr = wasm.volume_energy_reservoirs_alloc(close.length);
    const rangeHighPtr = wasm.volume_energy_reservoirs_alloc(close.length);
    const rangeLowPtr = wasm.volume_energy_reservoirs_alloc(close.length);

    try {
        wasm.volume_energy_reservoirs_into(
            highPtr,
            lowPtr,
            closePtr,
            volumePtr,
            momentumPtr,
            reservoirPtr,
            squeezeActivePtr,
            squeezeStartPtr,
            rangeHighPtr,
            rangeLowPtr,
            close.length,
            18,
            1.7,
        );
        assertArrayClose(wasm.read_f64_array(momentumPtr, close.length), safe.momentum, 1e-12, 'momentum mismatch');
        assertArrayClose(wasm.read_f64_array(reservoirPtr, close.length), safe.reservoir, 1e-12, 'reservoir mismatch');
        assertArrayClose(wasm.read_f64_array(squeezeActivePtr, close.length), safe.squeeze_active, 1e-12, 'squeeze_active mismatch');
        assertArrayClose(wasm.read_f64_array(squeezeStartPtr, close.length), safe.squeeze_start, 1e-12, 'squeeze_start mismatch');
        assertArrayClose(wasm.read_f64_array(rangeHighPtr, close.length), safe.range_high, 1e-12, 'range_high mismatch');
        assertArrayClose(wasm.read_f64_array(rangeLowPtr, close.length), safe.range_low, 1e-12, 'range_low mismatch');
    } finally {
        wasm.volume_energy_reservoirs_free(momentumPtr, close.length);
        wasm.volume_energy_reservoirs_free(reservoirPtr, close.length);
        wasm.volume_energy_reservoirs_free(squeezeActivePtr, close.length);
        wasm.volume_energy_reservoirs_free(squeezeStartPtr, close.length);
        wasm.volume_energy_reservoirs_free(rangeHighPtr, close.length);
        wasm.volume_energy_reservoirs_free(rangeLowPtr, close.length);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
        wasm.deallocate_f64_array(volumePtr);
    }
});

test('volume_energy_reservoirs_js resets after NaN', () => {
    const { high, low, close, volume } = sampleOhlcv(220);
    high[110] = Number.NaN;
    low[110] = Number.NaN;
    close[110] = Number.NaN;
    volume[110] = Number.NaN;
    const out = wasm.volume_energy_reservoirs_js(high, low, close, volume, 18, 1.7);

    assert(isNaN(out.momentum[110]));
    assert(isNaN(out.reservoir[110]));
    assert(isNaN(out.squeeze_active[110]));
});

test('volume_energy_reservoirs_batch_js rejects invalid config', () => {
    const { high, low, close, volume } = sampleOhlcv(16);
    assert.throws(
        () => wasm.volume_energy_reservoirs_batch_js(high, low, close, volume, { length_range: 'bad' }),
        /Invalid config/,
    );
});

test('volume_energy_reservoirs_batch_js single parameter set matches safe API', () => {
    const { high, low, close, volume } = sampleOhlcv(128);
    const batch = wasm.volume_energy_reservoirs_batch_js(high, low, close, volume, {
        length_range: [18, 18, 0],
        sensitivity_range: [1.7, 1.7, 0.0],
    });
    const single = wasm.volume_energy_reservoirs_js(high, low, close, volume, 18, 1.7);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.lengths, [18]);
    assert.deepStrictEqual(batch.sensitivities, [1.7]);
    assertArrayClose(batch.momentum, single.momentum, 1e-12, 'batch momentum mismatch');
    assertArrayClose(batch.reservoir, single.reservoir, 1e-12, 'batch reservoir mismatch');
});

test('volume_energy_reservoirs_batch_js metadata and batch_into', () => {
    const { high, low, close, volume } = sampleOhlcv(96);
    const batch = wasm.volume_energy_reservoirs_batch_js(high, low, close, volume, {
        length_range: [16, 20, 4],
        sensitivity_range: [1.5, 2.0, 0.5],
    });

    assert.strictEqual(batch.rows, 4);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.lengths, [16, 16, 20, 20]);
    assert.deepStrictEqual(batch.sensitivities, [1.5, 2.0, 1.5, 2.0]);

    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const volumePtr = wasm.copy_f64_array(volume);
    const total = batch.rows * close.length;
    const momentumPtr = wasm.volume_energy_reservoirs_alloc(total);
    const reservoirPtr = wasm.volume_energy_reservoirs_alloc(total);
    const squeezeActivePtr = wasm.volume_energy_reservoirs_alloc(total);
    const squeezeStartPtr = wasm.volume_energy_reservoirs_alloc(total);
    const rangeHighPtr = wasm.volume_energy_reservoirs_alloc(total);
    const rangeLowPtr = wasm.volume_energy_reservoirs_alloc(total);

    try {
        const rows = wasm.volume_energy_reservoirs_batch_into(
            highPtr,
            lowPtr,
            closePtr,
            volumePtr,
            momentumPtr,
            reservoirPtr,
            squeezeActivePtr,
            squeezeStartPtr,
            rangeHighPtr,
            rangeLowPtr,
            close.length,
            16, 20, 4,
            1.5, 2.0, 0.5,
        );
        assert.strictEqual(rows, 4);
        assertArrayClose(wasm.read_f64_array(momentumPtr, total), batch.momentum, 1e-12, 'batch_into momentum mismatch');
        assertArrayClose(wasm.read_f64_array(reservoirPtr, total), batch.reservoir, 1e-12, 'batch_into reservoir mismatch');
    } finally {
        wasm.volume_energy_reservoirs_free(momentumPtr, total);
        wasm.volume_energy_reservoirs_free(reservoirPtr, total);
        wasm.volume_energy_reservoirs_free(squeezeActivePtr, total);
        wasm.volume_energy_reservoirs_free(squeezeStartPtr, total);
        wasm.volume_energy_reservoirs_free(rangeHighPtr, total);
        wasm.volume_energy_reservoirs_free(rangeLowPtr, total);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
        wasm.deallocate_f64_array(volumePtr);
    }
});
