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
        'basis',
        'trend',
        'upper_0618',
        'upper_1000',
        'upper_1618',
        'upper_2618',
        'lower_0618',
        'lower_1000',
        'lower_1618',
        'lower_2618',
        'tp_long_band',
        'tp_short_band',
        'long_entry',
        'short_entry',
        'rejection_long',
        'rejection_short',
        'long_bounce',
        'short_bounce',
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

test('fibonacci_entry_bands_js output contract', () => {
    const { open, high, low, close } = sampleOhlc();
    const out = wasm.fibonacci_entry_bands_js(open, high, low, close, 'hlc3', 21, 14, true, 'low');

    for (const key of outputKeys()) {
        assert.strictEqual(out[key].length, close.length);
    }

    assert(out.basis.some(v => !isNaN(v)));
    assert(out.tp_long_band.some(v => !isNaN(v)));
    assert(out.tp_short_band.some(v => !isNaN(v)));

    for (const value of out.trend.filter(v => !isNaN(v))) {
        assert(value === -1 || value === 0 || value === 1);
    }

    for (const key of [
        'long_entry',
        'short_entry',
        'rejection_long',
        'rejection_short',
        'long_bounce',
        'short_bounce',
    ]) {
        for (const value of out[key].filter(v => !isNaN(v))) {
            assert(value === 0 || value === 1);
        }
    }
});

test('fibonacci_entry_bands_js rejects invalid inputs', () => {
    const { open, high, low, close } = sampleOhlc(32);
    assert.throws(
        () => wasm.fibonacci_entry_bands_js(open.subarray(0, 16), high, low, close, 'hlc3', 21, 14, true, 'low'),
        /Inconsistent slice lengths/,
    );
    assert.throws(
        () => wasm.fibonacci_entry_bands_js(open, high, low, close, 'bad', 21, 14, true, 'low'),
        /Invalid source/,
    );
    assert.throws(
        () => wasm.fibonacci_entry_bands_js(open, high, low, close, 'hlc3', 0, 14, true, 'low'),
        /Invalid length/,
    );
    assert.throws(
        () => wasm.fibonacci_entry_bands_js(open, high, low, close, 'hlc3', 21, 14, true, 'bad'),
        /Invalid TP aggressiveness/,
    );
});

test('fibonacci_entry_bands_into pointer path matches safe API', () => {
    const { open, high, low, close } = sampleOhlc(128);
    const safe = wasm.fibonacci_entry_bands_js(open, high, low, close, 'hlc3', 16, 11, true, 'high');

    const openPtr = wasm.copy_f64_array(open);
    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const ptrs = {};
    for (const key of outputKeys()) {
        ptrs[key] = wasm.fibonacci_entry_bands_alloc(close.length);
    }

    try {
        wasm.fibonacci_entry_bands_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            ptrs.basis,
            ptrs.trend,
            ptrs.upper_0618,
            ptrs.upper_1000,
            ptrs.upper_1618,
            ptrs.upper_2618,
            ptrs.lower_0618,
            ptrs.lower_1000,
            ptrs.lower_1618,
            ptrs.lower_2618,
            ptrs.tp_long_band,
            ptrs.tp_short_band,
            ptrs.long_entry,
            ptrs.short_entry,
            ptrs.rejection_long,
            ptrs.rejection_short,
            ptrs.long_bounce,
            ptrs.short_bounce,
            close.length,
            'hlc3',
            16,
            11,
            true,
            'high',
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
            wasm.fibonacci_entry_bands_free(ptrs[key], close.length);
        }
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('fibonacci_entry_bands_js resets after NaN', () => {
    const { open, high, low, close } = sampleOhlc(220);
    open[90] = Number.NaN;
    high[90] = Number.NaN;
    low[90] = Number.NaN;
    close[90] = Number.NaN;

    const out = wasm.fibonacci_entry_bands_js(open, high, low, close, 'hlc3', 16, 11, true, 'high');
    for (const key of outputKeys()) {
        assert(isNaN(out[key][90]), key + ' expected NaN at reset point');
    }
});

test('fibonacci_entry_bands_batch_js single parameter set matches safe API', () => {
    const { open, high, low, close } = sampleOhlc(160);
    const batch = wasm.fibonacci_entry_bands_batch_js(open, high, low, close, {
        length_range: [17, 17, 0],
        atr_length_range: [10, 10, 0],
        source: 'hlc3',
        use_atr: true,
        tp_aggressiveness: 'medium',
    });
    const single = wasm.fibonacci_entry_bands_js(open, high, low, close, 'hlc3', 17, 10, true, 'medium');

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.lengths, [17]);
    assert.deepStrictEqual(batch.atr_lengths, [10]);
    assert.deepStrictEqual(batch.sources, ['hlc3']);
    assert.deepStrictEqual(batch.use_atr_flags, [true]);
    assert.deepStrictEqual(batch.tp_aggressiveness_values, ['medium']);

    for (const key of outputKeys()) {
        assertArrayClose(batch[key], single[key], 1e-12, key + ' batch mismatch');
    }
});

test('fibonacci_entry_bands_batch_js metadata and batch_into', () => {
    const { open, high, low, close } = sampleOhlc(120);
    const batch = wasm.fibonacci_entry_bands_batch_js(open, high, low, close, {
        length_range: [16, 18, 2],
        atr_length_range: [10, 12, 2],
        source: 'hl2',
        use_atr: false,
        tp_aggressiveness: 'low',
    });

    assert.strictEqual(batch.rows, 4);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.lengths, [16, 16, 18, 18]);
    assert.deepStrictEqual(batch.atr_lengths, [10, 12, 10, 12]);
    assert.deepStrictEqual(batch.sources, ['hl2', 'hl2', 'hl2', 'hl2']);
    assert.deepStrictEqual(batch.use_atr_flags, [false, false, false, false]);
    assert.deepStrictEqual(batch.tp_aggressiveness_values, ['low', 'low', 'low', 'low']);

    const openPtr = wasm.copy_f64_array(open);
    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const total = batch.rows * close.length;
    const ptrs = {};
    for (const key of outputKeys()) {
        ptrs[key] = wasm.fibonacci_entry_bands_alloc(total);
    }

    try {
        const rows = wasm.fibonacci_entry_bands_batch_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            ptrs.basis,
            ptrs.trend,
            ptrs.upper_0618,
            ptrs.upper_1000,
            ptrs.upper_1618,
            ptrs.upper_2618,
            ptrs.lower_0618,
            ptrs.lower_1000,
            ptrs.lower_1618,
            ptrs.lower_2618,
            ptrs.tp_long_band,
            ptrs.tp_short_band,
            ptrs.long_entry,
            ptrs.short_entry,
            ptrs.rejection_long,
            ptrs.rejection_short,
            ptrs.long_bounce,
            ptrs.short_bounce,
            close.length,
            16,
            18,
            2,
            10,
            12,
            2,
            'hl2',
            false,
            'low',
        );
        assert.strictEqual(rows, 4);

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
            wasm.fibonacci_entry_bands_free(ptrs[key], total);
        }
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('fibonacci_entry_bands_batch_js rejects invalid config', () => {
    const { open, high, low, close } = sampleOhlc(32);
    assert.throws(
        () => wasm.fibonacci_entry_bands_batch_js(open, high, low, close, { length_range: 'bad' }),
        /Invalid config/,
    );
});
