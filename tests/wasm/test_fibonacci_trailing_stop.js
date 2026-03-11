import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function sampleOhlc(length = 256) {
    const open = new Float64Array(
        Array.from({ length }, (_, i) => 100 + i * 0.03 + Math.sin(i * 0.09)),
    );
    const close = new Float64Array(
        Array.from(open, (o, i) => o + Math.cos(i * 0.13) * 0.8),
    );
    const high = new Float64Array(
        Array.from({ length }, (_, i) => Math.max(open[i], close[i]) + 0.5 + Math.abs(Math.sin(i * 0.05)) * 0.2),
    );
    const low = new Float64Array(
        Array.from({ length }, (_, i) => Math.min(open[i], close[i]) - 0.5 - Math.abs(Math.cos(i * 0.07)) * 0.2),
    );
    return { open, high, low, close };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    const imported = await import(importPath);
    wasm = imported.default ?? imported;
});

test('fibonacci_trailing_stop_js output contract', () => {
    const { high, low, close } = sampleOhlc();
    const out = wasm.fibonacci_trailing_stop_js(high, low, close, 20, 1, -0.382, 'close');

    assert.strictEqual(out.trailing_stop.length, close.length);
    assert.strictEqual(out.long_stop.length, close.length);
    assert.strictEqual(out.short_stop.length, close.length);
    assert.strictEqual(out.direction.length, close.length);
    assert(Number.isFinite(out.trailing_stop[0]));
    for (let i = 0; i < out.direction.length; i++) {
        const value = out.direction[i];
        if (Number.isFinite(value)) {
            assert(value === -1 || value === 0 || value === 1);
        }
    }
});

test('fibonacci_trailing_stop_js rejects invalid parameters', () => {
    const { high, low, close } = sampleOhlc(32);
    assert.throws(
        () => wasm.fibonacci_trailing_stop_js(high, low, close, 0, 1, -0.382, 'close'),
        /Invalid left_bars/,
    );
    assert.throws(
        () => wasm.fibonacci_trailing_stop_js(high, low, close, 20, 1, -0.382, 'bad'),
        /Invalid trigger/,
    );
});

test('fibonacci_trailing_stop_into pointer path matches safe API', () => {
    const { high, low, close } = sampleOhlc(160);
    const safe = wasm.fibonacci_trailing_stop_js(high, low, close, 12, 2, -0.236, 'wick');

    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const trailingStopPtr = wasm.fibonacci_trailing_stop_alloc(close.length);
    const longStopPtr = wasm.fibonacci_trailing_stop_alloc(close.length);
    const shortStopPtr = wasm.fibonacci_trailing_stop_alloc(close.length);
    const directionPtr = wasm.fibonacci_trailing_stop_alloc(close.length);

    try {
        wasm.fibonacci_trailing_stop_into(
            highPtr,
            lowPtr,
            closePtr,
            trailingStopPtr,
            longStopPtr,
            shortStopPtr,
            directionPtr,
            close.length,
            12,
            2,
            -0.236,
            'wick',
        );
        assertArrayClose(
            wasm.read_f64_array(trailingStopPtr, close.length),
            safe.trailing_stop,
            1e-12,
            'trailing stop mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(longStopPtr, close.length),
            safe.long_stop,
            1e-12,
            'long stop mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(shortStopPtr, close.length),
            safe.short_stop,
            1e-12,
            'short stop mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(directionPtr, close.length),
            safe.direction,
            1e-12,
            'direction mismatch',
        );
    } finally {
        wasm.fibonacci_trailing_stop_free(trailingStopPtr, close.length);
        wasm.fibonacci_trailing_stop_free(longStopPtr, close.length);
        wasm.fibonacci_trailing_stop_free(shortStopPtr, close.length);
        wasm.fibonacci_trailing_stop_free(directionPtr, close.length);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('fibonacci_trailing_stop_js resets after NaN', () => {
    const { high, low, close } = sampleOhlc(220);
    high[110] = Number.NaN;
    low[110] = Number.NaN;
    close[110] = Number.NaN;
    const out = wasm.fibonacci_trailing_stop_js(high, low, close, 12, 2, -0.236, 'close');

    assert(isNaN(out.trailing_stop[110]));
    assert(isNaN(out.long_stop[110]));
    assert(isNaN(out.short_stop[110]));
    assert(isNaN(out.direction[110]));
});

test('fibonacci_trailing_stop_batch_js rejects invalid config', () => {
    const { high, low, close } = sampleOhlc(32);
    assert.throws(
        () => wasm.fibonacci_trailing_stop_batch_js(high, low, close, { left_bars_range: 'bad' }),
        /Invalid config/,
    );
});

test('fibonacci_trailing_stop_batch_js single parameter set matches safe API', () => {
    const { high, low, close } = sampleOhlc(160);
    const batch = wasm.fibonacci_trailing_stop_batch_js(high, low, close, {
        left_bars_range: [12, 12, 0],
        right_bars_range: [2, 2, 0],
        level_range: [-0.236, -0.236, 0],
        trigger: 'wick',
    });
    const single = wasm.fibonacci_trailing_stop_js(high, low, close, 12, 2, -0.236, 'wick');

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.left_bars, [12]);
    assert.deepStrictEqual(batch.right_bars, [2]);
    assertArrayClose(batch.levels, [-0.236], 1e-12, 'level metadata mismatch');
    assertArrayClose(batch.trailing_stop, single.trailing_stop, 1e-12, 'batch trailing stop mismatch');
    assertArrayClose(batch.long_stop, single.long_stop, 1e-12, 'batch long stop mismatch');
    assertArrayClose(batch.short_stop, single.short_stop, 1e-12, 'batch short stop mismatch');
    assertArrayClose(batch.direction, single.direction, 1e-12, 'batch direction mismatch');
});

test('fibonacci_trailing_stop_batch_js metadata and batch_into', () => {
    const { high, low, close } = sampleOhlc(120);
    const batch = wasm.fibonacci_trailing_stop_batch_js(high, low, close, {
        left_bars_range: [10, 12, 2],
        right_bars_range: [1, 2, 1],
        level_range: [-0.382, -0.236, 0.146],
        trigger: 'close',
    });

    assert.strictEqual(batch.rows, 8);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.left_bars, [10, 10, 10, 10, 12, 12, 12, 12]);
    assert.deepStrictEqual(batch.right_bars, [1, 1, 2, 2, 1, 1, 2, 2]);
    assertArrayClose(
        batch.levels,
        [-0.382, -0.236, -0.382, -0.236, -0.382, -0.236, -0.382, -0.236],
        1e-12,
        'level grid mismatch',
    );

    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const total = batch.rows * close.length;
    const trailingStopPtr = wasm.fibonacci_trailing_stop_alloc(total);
    const longStopPtr = wasm.fibonacci_trailing_stop_alloc(total);
    const shortStopPtr = wasm.fibonacci_trailing_stop_alloc(total);
    const directionPtr = wasm.fibonacci_trailing_stop_alloc(total);

    try {
        const rows = wasm.fibonacci_trailing_stop_batch_into(
            highPtr,
            lowPtr,
            closePtr,
            trailingStopPtr,
            longStopPtr,
            shortStopPtr,
            directionPtr,
            close.length,
            10,
            12,
            2,
            1,
            2,
            1,
            -0.382,
            -0.236,
            0.146,
            'close',
        );
        assert.strictEqual(rows, 8);
        assertArrayClose(
            wasm.read_f64_array(trailingStopPtr, total),
            batch.trailing_stop,
            1e-12,
            'batch_into trailing stop mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(longStopPtr, total),
            batch.long_stop,
            1e-12,
            'batch_into long stop mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(shortStopPtr, total),
            batch.short_stop,
            1e-12,
            'batch_into short stop mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(directionPtr, total),
            batch.direction,
            1e-12,
            'batch_into direction mismatch',
        );
    } finally {
        wasm.fibonacci_trailing_stop_free(trailingStopPtr, total);
        wasm.fibonacci_trailing_stop_free(longStopPtr, total);
        wasm.fibonacci_trailing_stop_free(shortStopPtr, total);
        wasm.fibonacci_trailing_stop_free(directionPtr, total);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});
