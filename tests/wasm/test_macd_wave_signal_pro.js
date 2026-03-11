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
        open[i] = 100 + i * 0.08 + Math.sin(i * 0.05) * 0.7;
        close[i] = open[i] + Math.cos(i * 0.09) * 0.9;
        high[i] = Math.max(open[i], close[i]) + 0.55 + Math.abs(Math.sin(i * 0.03)) * 0.2;
        low[i] = Math.min(open[i], close[i]) - 0.55 - Math.abs(Math.cos(i * 0.05)) * 0.2;
    }
    return { open, high, low, close };
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

test('macd_wave_signal_pro_js output contract', () => {
    const { open, high, low, close } = sampleOhlc();
    const out = wasm.macd_wave_signal_pro_js(open, high, low, close);

    assert.strictEqual(out.diff.length, close.length);
    assert.strictEqual(out.dea.length, close.length);
    assert.strictEqual(out.macd_histogram.length, close.length);
    assert.strictEqual(out.line_convergence.length, close.length);
    assert.strictEqual(out.buy_signal.length, close.length);
    assert.strictEqual(out.sell_signal.length, close.length);
    assert(out.diff.some(v => !isNaN(v)));
    assert(out.line_convergence.some(v => !isNaN(v)));
});

test('macd_wave_signal_pro_js rejects invalid inputs', () => {
    const { open, high, low, close } = sampleOhlc(32);
    assert.throws(() => wasm.macd_wave_signal_pro_js(open, high, low, close), /Not enough valid data/);
    assert.throws(
        () => wasm.macd_wave_signal_pro_js(open.subarray(0, 16), high, low, close),
        /Inconsistent slice lengths/,
    );
});

test('macd_wave_signal_pro_into pointer path matches safe API', () => {
    const { open, high, low, close } = sampleOhlc(128);
    const safe = wasm.macd_wave_signal_pro_js(open, high, low, close);

    const openPtr = wasm.copy_f64_array(open);
    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const diffPtr = wasm.macd_wave_signal_pro_alloc(close.length);
    const deaPtr = wasm.macd_wave_signal_pro_alloc(close.length);
    const macdPtr = wasm.macd_wave_signal_pro_alloc(close.length);
    const linePtr = wasm.macd_wave_signal_pro_alloc(close.length);
    const buyPtr = wasm.macd_wave_signal_pro_alloc(close.length);
    const sellPtr = wasm.macd_wave_signal_pro_alloc(close.length);

    try {
        wasm.macd_wave_signal_pro_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            diffPtr,
            deaPtr,
            macdPtr,
            linePtr,
            buyPtr,
            sellPtr,
            close.length,
        );
        assertArrayClose(wasm.read_f64_array(diffPtr, close.length), safe.diff, 1e-12, 'diff mismatch');
        assertArrayClose(wasm.read_f64_array(deaPtr, close.length), safe.dea, 1e-12, 'dea mismatch');
        assertArrayClose(
            wasm.read_f64_array(macdPtr, close.length),
            safe.macd_histogram,
            1e-12,
            'macd_histogram mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(linePtr, close.length),
            safe.line_convergence,
            1e-12,
            'line_convergence mismatch',
        );
    } finally {
        wasm.macd_wave_signal_pro_free(diffPtr, close.length);
        wasm.macd_wave_signal_pro_free(deaPtr, close.length);
        wasm.macd_wave_signal_pro_free(macdPtr, close.length);
        wasm.macd_wave_signal_pro_free(linePtr, close.length);
        wasm.macd_wave_signal_pro_free(buyPtr, close.length);
        wasm.macd_wave_signal_pro_free(sellPtr, close.length);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('macd_wave_signal_pro_js resets after NaN', () => {
    const { open, high, low, close } = sampleOhlc(220);
    open[90] = Number.NaN;
    high[90] = Number.NaN;
    low[90] = Number.NaN;
    close[90] = Number.NaN;

    const out = wasm.macd_wave_signal_pro_js(open, high, low, close);
    assert(isNaN(out.diff[90]));
    assert(isNaN(out.dea[90]));
    assert(isNaN(out.line_convergence[90]));
});

test('MacdWaveSignalProContext update_into matches safe API', () => {
    const { open, high, low, close } = sampleOhlc(128);
    const safe = wasm.macd_wave_signal_pro_js(open, high, low, close);

    const ctx = new wasm.MacdWaveSignalProContext();
    assert.strictEqual(ctx.get_warmup_period(), 39);

    const openPtr = wasm.copy_f64_array(open);
    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const diffPtr = wasm.macd_wave_signal_pro_alloc(close.length);
    const deaPtr = wasm.macd_wave_signal_pro_alloc(close.length);
    const macdPtr = wasm.macd_wave_signal_pro_alloc(close.length);
    const linePtr = wasm.macd_wave_signal_pro_alloc(close.length);
    const buyPtr = wasm.macd_wave_signal_pro_alloc(close.length);
    const sellPtr = wasm.macd_wave_signal_pro_alloc(close.length);

    try {
        ctx.update_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            diffPtr,
            deaPtr,
            macdPtr,
            linePtr,
            buyPtr,
            sellPtr,
            close.length,
        );
        assertArrayClose(wasm.read_f64_array(diffPtr, close.length), safe.diff, 1e-12, 'context diff mismatch');
        assertArrayClose(wasm.read_f64_array(deaPtr, close.length), safe.dea, 1e-12, 'context dea mismatch');
        assertArrayClose(
            wasm.read_f64_array(macdPtr, close.length),
            safe.macd_histogram,
            1e-12,
            'context macd mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(linePtr, close.length),
            safe.line_convergence,
            1e-12,
            'context line mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(buyPtr, close.length),
            safe.buy_signal,
            1e-12,
            'context buy mismatch',
        );
        assertArrayClose(
            wasm.read_f64_array(sellPtr, close.length),
            safe.sell_signal,
            1e-12,
            'context sell mismatch',
        );
    } finally {
        wasm.macd_wave_signal_pro_free(diffPtr, close.length);
        wasm.macd_wave_signal_pro_free(deaPtr, close.length);
        wasm.macd_wave_signal_pro_free(macdPtr, close.length);
        wasm.macd_wave_signal_pro_free(linePtr, close.length);
        wasm.macd_wave_signal_pro_free(buyPtr, close.length);
        wasm.macd_wave_signal_pro_free(sellPtr, close.length);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('macd_wave_signal_pro_batch_js single row matches safe API', () => {
    const { open, high, low, close } = sampleOhlc(128);
    const batch = wasm.macd_wave_signal_pro_batch_js(open, high, low, close, {});
    const single = wasm.macd_wave_signal_pro_js(open, high, low, close);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assertArrayClose(batch.diff, single.diff, 1e-12, 'batch diff mismatch');
    assertArrayClose(batch.dea, single.dea, 1e-12, 'batch dea mismatch');
    assertArrayClose(batch.macd_histogram, single.macd_histogram, 1e-12, 'batch macd mismatch');
    assertArrayClose(
        batch.line_convergence,
        single.line_convergence,
        1e-12,
        'batch line_convergence mismatch',
    );
});

test('macd_wave_signal_pro_batch_js invalid config and batch_into', () => {
    const { open, high, low, close } = sampleOhlc(96);
    assert.throws(() => wasm.macd_wave_signal_pro_batch_js(open, high, low, close, 'bad'), /Invalid config/);

    const batch = wasm.macd_wave_signal_pro_batch_js(open, high, low, close, {});
    const total = batch.rows * close.length;
    const openPtr = wasm.copy_f64_array(open);
    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const diffPtr = wasm.macd_wave_signal_pro_alloc(total);
    const deaPtr = wasm.macd_wave_signal_pro_alloc(total);
    const macdPtr = wasm.macd_wave_signal_pro_alloc(total);
    const linePtr = wasm.macd_wave_signal_pro_alloc(total);
    const buyPtr = wasm.macd_wave_signal_pro_alloc(total);
    const sellPtr = wasm.macd_wave_signal_pro_alloc(total);

    try {
        const rows = wasm.macd_wave_signal_pro_batch_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            diffPtr,
            deaPtr,
            macdPtr,
            linePtr,
            buyPtr,
            sellPtr,
            close.length,
        );
        assert.strictEqual(rows, 1);
        assertArrayClose(wasm.read_f64_array(diffPtr, total), batch.diff, 1e-12, 'batch_into diff mismatch');
        assertArrayClose(
            wasm.read_f64_array(linePtr, total),
            batch.line_convergence,
            1e-12,
            'batch_into line mismatch',
        );
    } finally {
        wasm.macd_wave_signal_pro_free(diffPtr, total);
        wasm.macd_wave_signal_pro_free(deaPtr, total);
        wasm.macd_wave_signal_pro_free(macdPtr, total);
        wasm.macd_wave_signal_pro_free(linePtr, total);
        wasm.macd_wave_signal_pro_free(buyPtr, total);
        wasm.macd_wave_signal_pro_free(sellPtr, total);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});
