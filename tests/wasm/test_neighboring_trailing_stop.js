import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function sampleOhlc(length = 256) {
    const high = new Float64Array(length);
    const low = new Float64Array(length);
    const close = new Float64Array(length);
    for (let i = 0; i < length; i++) {
        const open = 100 + i * 0.04 + Math.sin(i * 0.07);
        close[i] = open + Math.cos(i * 0.11) * 0.85;
        high[i] = Math.max(open, close[i]) + 0.55 + Math.abs(Math.sin(i * 0.03)) * 0.2;
        low[i] = Math.min(open, close[i]) - 0.55 - Math.abs(Math.cos(i * 0.05)) * 0.2;
    }
    return { high, low, close };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    const imported = await import(importPath);
    wasm = imported.default ?? imported;
});

test('neighboring_trailing_stop_js output contract', () => {
    const { high, low, close } = sampleOhlc();
    const out = wasm.neighboring_trailing_stop_js(high, low, close, 200, 50, 90.0, 5);

    assert.strictEqual(out.trailing_stop.length, close.length);
    assert.strictEqual(out.bullish_band.length, close.length);
    assert.strictEqual(out.bearish_band.length, close.length);
    assert.strictEqual(out.direction.length, close.length);
    assert.strictEqual(out.discovery_bull.length, close.length);
    assert.strictEqual(out.discovery_bear.length, close.length);
});

test('neighboring_trailing_stop_js rejects invalid parameters', () => {
    const { high, low, close } = sampleOhlc(64);
    assert.throws(
        () => wasm.neighboring_trailing_stop_js(high, low, close, 50, 50, 90.0, 5),
        /Invalid buffer_size/,
    );
    assert.throws(
        () => wasm.neighboring_trailing_stop_js(high, low, close, 200, 2, 90.0, 5),
        /Invalid k/,
    );
    assert.throws(
        () => wasm.neighboring_trailing_stop_js(high, low, close, 200, 50, Number.NaN, 5),
        /Invalid percentile/,
    );
});

test('neighboring_trailing_stop_into pointer path matches safe API', () => {
    const { high, low, close } = sampleOhlc(128);
    const safe = wasm.neighboring_trailing_stop_js(high, low, close, 180, 30, 87.5, 4);

    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const trailingStopPtr = wasm.neighboring_trailing_stop_alloc(close.length);
    const bullishBandPtr = wasm.neighboring_trailing_stop_alloc(close.length);
    const bearishBandPtr = wasm.neighboring_trailing_stop_alloc(close.length);
    const directionPtr = wasm.neighboring_trailing_stop_alloc(close.length);
    const discoveryBullPtr = wasm.neighboring_trailing_stop_alloc(close.length);
    const discoveryBearPtr = wasm.neighboring_trailing_stop_alloc(close.length);

    try {
        wasm.neighboring_trailing_stop_into(
            highPtr,
            lowPtr,
            closePtr,
            trailingStopPtr,
            bullishBandPtr,
            bearishBandPtr,
            directionPtr,
            discoveryBullPtr,
            discoveryBearPtr,
            close.length,
            180,
            30,
            87.5,
            4,
        );
        assertArrayClose(
            wasm.read_f64_array(trailingStopPtr, close.length),
            safe.trailing_stop,
            1e-12,
            'trailing_stop mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(bullishBandPtr, close.length),
            safe.bullish_band,
            1e-12,
            'bullish_band mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(bearishBandPtr, close.length),
            safe.bearish_band,
            1e-12,
            'bearish_band mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(directionPtr, close.length),
            safe.direction,
            1e-12,
            'direction mismatch',
        );
    } finally {
        wasm.neighboring_trailing_stop_free(trailingStopPtr, close.length);
        wasm.neighboring_trailing_stop_free(bullishBandPtr, close.length);
        wasm.neighboring_trailing_stop_free(bearishBandPtr, close.length);
        wasm.neighboring_trailing_stop_free(directionPtr, close.length);
        wasm.neighboring_trailing_stop_free(discoveryBullPtr, close.length);
        wasm.neighboring_trailing_stop_free(discoveryBearPtr, close.length);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('neighboring_trailing_stop_js resets after NaN', () => {
    const { high, low, close } = sampleOhlc(220);
    high[110] = Number.NaN;
    low[110] = Number.NaN;
    close[110] = Number.NaN;
    const out = wasm.neighboring_trailing_stop_js(high, low, close, 180, 25, 92.0, 4);

    assert(isNaN(out.trailing_stop[110]));
    assert(isNaN(out.bullish_band[110]));
    assert(isNaN(out.direction[110]));
});

test('neighboring_trailing_stop_batch_js rejects invalid config', () => {
    const { high, low, close } = sampleOhlc(32);
    assert.throws(
        () => wasm.neighboring_trailing_stop_batch_js(high, low, close, { buffer_size_range: 'bad' }),
        /Invalid config/,
    );
});

test('neighboring_trailing_stop_batch_js single parameter set matches safe API', () => {
    const { high, low, close } = sampleOhlc(128);
    const batch = wasm.neighboring_trailing_stop_batch_js(high, low, close, {
        buffer_size_range: [180, 180, 0],
        k_range: [30, 30, 0],
        percentile_range: [87.5, 87.5, 0.0],
        smooth_range: [4, 4, 0],
    });
    const single = wasm.neighboring_trailing_stop_js(high, low, close, 180, 30, 87.5, 4);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.buffer_sizes, [180]);
    assert.deepStrictEqual(batch.ks, [30]);
    assert.deepStrictEqual(batch.percentiles, [87.5]);
    assert.deepStrictEqual(batch.smoothings, [4]);
    assertArrayClose(batch.trailing_stop, single.trailing_stop, 1e-12, 'batch trailing_stop mismatch');
    assertArrayClose(batch.bullish_band, single.bullish_band, 1e-12, 'batch bullish_band mismatch');
});

test('neighboring_trailing_stop_batch_js metadata and batch_into', () => {
    const { high, low, close } = sampleOhlc(96);
    const batch = wasm.neighboring_trailing_stop_batch_js(high, low, close, {
        buffer_size_range: [150, 150, 0],
        k_range: [20, 24, 4],
        percentile_range: [85.0, 90.0, 5.0],
        smooth_range: [3, 3, 0],
    });

    assert.strictEqual(batch.rows, 4);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.buffer_sizes, [150, 150, 150, 150]);
    assert.deepStrictEqual(batch.ks, [20, 20, 24, 24]);
    assert.deepStrictEqual(batch.percentiles, [85.0, 90.0, 85.0, 90.0]);
    assert.deepStrictEqual(batch.smoothings, [3, 3, 3, 3]);

    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const total = batch.rows * close.length;
    const trailingStopPtr = wasm.neighboring_trailing_stop_alloc(total);
    const bullishBandPtr = wasm.neighboring_trailing_stop_alloc(total);
    const bearishBandPtr = wasm.neighboring_trailing_stop_alloc(total);
    const directionPtr = wasm.neighboring_trailing_stop_alloc(total);
    const discoveryBullPtr = wasm.neighboring_trailing_stop_alloc(total);
    const discoveryBearPtr = wasm.neighboring_trailing_stop_alloc(total);

    try {
        const rows = wasm.neighboring_trailing_stop_batch_into(
            highPtr,
            lowPtr,
            closePtr,
            trailingStopPtr,
            bullishBandPtr,
            bearishBandPtr,
            directionPtr,
            discoveryBullPtr,
            discoveryBearPtr,
            close.length,
            150, 150, 0,
            20, 24, 4,
            85.0, 90.0, 5.0,
            3, 3, 0,
        );
        assert.strictEqual(rows, 4);
        assertArrayClose(
            wasm.read_f64_array(trailingStopPtr, total),
            batch.trailing_stop,
            1e-12,
            'batch_into trailing_stop mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(bullishBandPtr, total),
            batch.bullish_band,
            1e-12,
            'batch_into bullish_band mismatch',
        );
    } finally {
        wasm.neighboring_trailing_stop_free(trailingStopPtr, total);
        wasm.neighboring_trailing_stop_free(bullishBandPtr, total);
        wasm.neighboring_trailing_stop_free(bearishBandPtr, total);
        wasm.neighboring_trailing_stop_free(directionPtr, total);
        wasm.neighboring_trailing_stop_free(discoveryBullPtr, total);
        wasm.neighboring_trailing_stop_free(discoveryBearPtr, total);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});
