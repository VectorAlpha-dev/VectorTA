import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function sampleOhlc(length = 256) {
    const open = new Float64Array(length);
    const high = new Float64Array(length);
    const low = new Float64Array(length);
    const close = new Float64Array(length);
    for (let i = 0; i < length; i++) {
        open[i] = 100 + i * 0.05 + Math.sin(i * 0.09) * 1.3;
        close[i] = open[i] + Math.cos(i * 0.07) * 1.1;
        high[i] = Math.max(open[i], close[i]) + 0.65 + Math.abs(Math.sin(i * 0.03)) * 0.25;
        low[i] = Math.min(open[i], close[i]) - 0.65 - Math.abs(Math.cos(i * 0.05)) * 0.25;
    }
    return { open, high, low, close };
}

function outputKeys() {
    return [
        'fast_hema',
        'slow_hema',
        'trend_direction',
        'bar_state',
        'bullish_crossover',
        'bearish_crossunder',
        'box_offset',
        'bull_box_top',
        'bull_box_bottom',
        'bear_box_top',
        'bear_box_bottom',
        'bullish_test',
        'bearish_test',
        'bullish_test_level',
        'bearish_test_level',
    ];
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath =
        process.platform === 'win32'
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
    const imported = await import(importPath);
    wasm = imported.default ?? imported;
});

test('hema_trend_levels_js output contract', () => {
    const { open, high, low, close } = sampleOhlc();
    const out = wasm.hema_trend_levels_js(open, high, low, close, 20, 40);

    for (const key of outputKeys()) {
        assert.strictEqual(out[key].length, close.length);
    }

    assert(out.fast_hema.some(v => !isNaN(v)));
    assert(out.slow_hema.some(v => !isNaN(v)));

    for (const key of ['trend_direction', 'bar_state']) {
        for (const value of out[key].filter(v => !isNaN(v))) {
            assert(value === -1 || value === 0 || value === 1);
        }
    }

    for (const key of ['bullish_crossover', 'bearish_crossunder', 'bullish_test', 'bearish_test']) {
        for (const value of out[key].filter(v => !isNaN(v))) {
            assert(value === 0 || value === 1);
        }
    }
});

test('hema_trend_levels_js rejects invalid inputs', () => {
    const { open, high, low, close } = sampleOhlc(32);
    assert.throws(
        () => wasm.hema_trend_levels_js(open.subarray(0, 16), high, low, close, 20, 40),
        /Inconsistent slice lengths/,
    );
    assert.throws(() => wasm.hema_trend_levels_js(open, high, low, close, 0, 40), /Invalid fast_length/);
});

test('hema_trend_levels_into pointer path matches safe API', () => {
    const { open, high, low, close } = sampleOhlc(128);
    const safe = wasm.hema_trend_levels_js(open, high, low, close, 20, 40);

    const openPtr = wasm.copy_f64_array(open);
    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);

    const ptrs = {};
    for (const key of outputKeys()) {
        ptrs[key] = wasm.hema_trend_levels_alloc(close.length);
    }

    try {
        wasm.hema_trend_levels_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            ptrs.fast_hema,
            ptrs.slow_hema,
            ptrs.trend_direction,
            ptrs.bar_state,
            ptrs.bullish_crossover,
            ptrs.bearish_crossunder,
            ptrs.box_offset,
            ptrs.bull_box_top,
            ptrs.bull_box_bottom,
            ptrs.bear_box_top,
            ptrs.bear_box_bottom,
            ptrs.bullish_test,
            ptrs.bearish_test,
            ptrs.bullish_test_level,
            ptrs.bearish_test_level,
            close.length,
            20,
            40,
        );

        for (const key of outputKeys()) {
            assertArrayClose(
                wasm.read_f64_array(ptrs[key], close.length),
                safe[key],
                1e-12,
                key + ' mismatch',
            );
        }
    } finally {
        for (const key of outputKeys()) {
            wasm.hema_trend_levels_free(ptrs[key], close.length);
        }
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('hema_trend_levels_js resets after NaN', () => {
    const { open, high, low, close } = sampleOhlc(220);
    open[90] = Number.NaN;
    high[90] = Number.NaN;
    low[90] = Number.NaN;
    close[90] = Number.NaN;

    const out = wasm.hema_trend_levels_js(open, high, low, close, 20, 40);
    for (const key of outputKeys()) {
        assert(isNaN(out[key][90]), key + ' expected NaN at reset point');
    }
});

test('HemaTrendLevelsContext update_into matches safe API', () => {
    const { open, high, low, close } = sampleOhlc(128);
    const safe = wasm.hema_trend_levels_js(open, high, low, close, 20, 40);

    const ctx = new wasm.HemaTrendLevelsContext(20, 40);
    assert.strictEqual(ctx.get_warmup_period(), 0);

    const openPtr = wasm.copy_f64_array(open);
    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const ptrs = {};
    for (const key of outputKeys()) {
        ptrs[key] = wasm.hema_trend_levels_alloc(close.length);
    }

    try {
        ctx.update_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            ptrs.fast_hema,
            ptrs.slow_hema,
            ptrs.trend_direction,
            ptrs.bar_state,
            ptrs.bullish_crossover,
            ptrs.bearish_crossunder,
            ptrs.box_offset,
            ptrs.bull_box_top,
            ptrs.bull_box_bottom,
            ptrs.bear_box_top,
            ptrs.bear_box_bottom,
            ptrs.bullish_test,
            ptrs.bearish_test,
            ptrs.bullish_test_level,
            ptrs.bearish_test_level,
            close.length,
        );

        for (const key of outputKeys()) {
            assertArrayClose(
                wasm.read_f64_array(ptrs[key], close.length),
                safe[key],
                1e-12,
                key + ' context mismatch',
            );
        }
    } finally {
        for (const key of outputKeys()) {
            wasm.hema_trend_levels_free(ptrs[key], close.length);
        }
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('hema_trend_levels_batch_js metadata and values', () => {
    const { open, high, low, close } = sampleOhlc(128);
    const batch = wasm.hema_trend_levels_batch_js(open, high, low, close, {
        fast_length_range: [20, 22, 2],
        slow_length_range: [40, 40, 0],
    });

    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.fast_lengths, [20, 22]);
    assert.deepStrictEqual(batch.slow_lengths, [40, 40]);

    const first = wasm.hema_trend_levels_js(open, high, low, close, 20, 40);
    const second = wasm.hema_trend_levels_js(open, high, low, close, 22, 40);

    const total = close.length;
    for (const key of outputKeys()) {
        assertArrayClose(batch[key].slice(0, total), first[key], 1e-12, key + ' batch row0 mismatch');
        assertArrayClose(batch[key].slice(total, total * 2), second[key], 1e-12, key + ' batch row1 mismatch');
    }
});

test('hema_trend_levels_batch_into matches batch_js', () => {
    const { open, high, low, close } = sampleOhlc(96);
    const rows = 2;
    const total = rows * close.length;
    const batch = wasm.hema_trend_levels_batch_js(open, high, low, close, {
        fast_length_range: [20, 22, 2],
        slow_length_range: [40, 40, 0],
    });

    const openPtr = wasm.copy_f64_array(open);
    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const ptrs = {};
    for (const key of outputKeys()) {
        ptrs[key] = wasm.hema_trend_levels_alloc(total);
    }

    try {
        const processed = wasm.hema_trend_levels_batch_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            ptrs.fast_hema,
            ptrs.slow_hema,
            ptrs.trend_direction,
            ptrs.bar_state,
            ptrs.bullish_crossover,
            ptrs.bearish_crossunder,
            ptrs.box_offset,
            ptrs.bull_box_top,
            ptrs.bull_box_bottom,
            ptrs.bear_box_top,
            ptrs.bear_box_bottom,
            ptrs.bullish_test,
            ptrs.bearish_test,
            ptrs.bullish_test_level,
            ptrs.bearish_test_level,
            close.length,
            20,
            22,
            2,
            40,
            40,
            0,
        );
        assert.strictEqual(processed, 2);

        for (const key of outputKeys()) {
            assertArrayClose(
                wasm.read_f64_array(ptrs[key], total),
                batch[key],
                1e-12,
                key + ' batch_into mismatch',
            );
        }
    } finally {
        for (const key of outputKeys()) {
            wasm.hema_trend_levels_free(ptrs[key], total);
        }
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});
