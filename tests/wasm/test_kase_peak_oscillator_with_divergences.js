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
    let price = 100.0;
    for (let i = 0; i < length; i++) {
        const wave = Math.sin(i * 0.17) * 1.8 + Math.cos(i * 0.07) * 0.9;
        price += wave * 0.35 + 0.12;
        close[i] = price;
        high[i] = price + 0.8 + (i % 5) * 0.05;
        low[i] = price - 0.7 - (i % 7) * 0.04;
    }
    return { high, low, close };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
});

test('kase_peak_oscillator_with_divergences_js output contract', () => {
    const { high, low, close } = sampleOhlc();
    const out = wasm.kase_peak_oscillator_with_divergences_js(
        high, low, close, 2.0, 8, 65, 40.0, true, 5, 5, 60, 5, true, false, true, false
    );
    assert.strictEqual(out.oscillator.length, close.length);
    assert(out.oscillator.slice(0, 66).every(isNaN));
    assertArrayClose(out.oscillator, out.histogram, 1e-12, 'histogram mismatch');
});

test('kase_peak_oscillator_with_divergences_js rejects invalid parameters', () => {
    const { high, low, close } = sampleOhlc(80);
    assert.throws(
        () => wasm.kase_peak_oscillator_with_divergences_js(
            high, low, close, 2.0, 10, 10, 40.0, true, 5, 5, 60, 5, true, false, true, false
        ),
        /Invalid cycle order/
    );
});

test('kase_peak_oscillator_with_divergences_into pointer path matches safe API', () => {
    const { high, low, close } = sampleOhlc(192);
    const safe = wasm.kase_peak_oscillator_with_divergences_js(
        high, low, close, 2.0, 8, 65, 40.0, true, 5, 5, 60, 5, true, false, true, false
    );

    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const total = close.length;
    const ptrs = Array.from({ length: 11 }, () => wasm.kase_peak_oscillator_with_divergences_alloc(total));

    try {
        wasm.kase_peak_oscillator_with_divergences_into(
            highPtr, lowPtr, closePtr,
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5],
            ptrs[6], ptrs[7], ptrs[8], ptrs[9], ptrs[10],
            total, 2.0, 8, 65, 40.0, true, 5, 5, 60, 5, true, false, true, false
        );
        assertArrayClose(wasm.read_f64_array(ptrs[0], total), safe.oscillator, 1e-12, 'osc mismatch');
        assertArrayClose(wasm.read_f64_array(ptrs[4], total), safe.market_extreme, 1e-12, 'market mismatch');
    } finally {
        for (const ptr of ptrs) {
            wasm.kase_peak_oscillator_with_divergences_free(ptr, total);
        }
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('kase_peak_oscillator_with_divergences_batch_js metadata and parity', () => {
    const { high, low, close } = sampleOhlc(180);
    const batch = wasm.kase_peak_oscillator_with_divergences_batch_js(high, low, close, {
        deviations_range: [2.0, 2.0, 0.0],
        short_cycle_range: [8, 8, 0],
        long_cycle_range: [65, 65, 0],
        sensitivity_range: [40.0, 40.0, 0.0],
    });
    const single = wasm.kase_peak_oscillator_with_divergences_js(
        high, low, close, 2.0, 8, 65, 40.0, true, 5, 5, 60, 5, true, false, true, false
    );
    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.short_cycles, [8]);
    assert.deepStrictEqual(batch.long_cycles, [65]);
    assertArrayClose(batch.oscillator, single.oscillator, 1e-12, 'batch osc mismatch');
});

test('kase_peak_oscillator_with_divergences_batch_into matches batch_js', () => {
    const { high, low, close } = sampleOhlc(180);
    const batch = wasm.kase_peak_oscillator_with_divergences_batch_js(high, low, close, {
        deviations_range: [2.0, 2.0, 0.0],
        short_cycle_range: [8, 8, 0],
        long_cycle_range: [65, 65, 0],
        sensitivity_range: [40.0, 40.0, 0.0],
    });

    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const total = close.length;
    const ptrs = Array.from({ length: 11 }, () => wasm.kase_peak_oscillator_with_divergences_alloc(total));

    try {
        const rows = wasm.kase_peak_oscillator_with_divergences_batch_into(
            highPtr, lowPtr, closePtr,
            ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5],
            ptrs[6], ptrs[7], ptrs[8], ptrs[9], ptrs[10],
            total,
            2.0, 2.0, 0.0,
            8, 8, 0,
            65, 65, 0,
            40.0, 40.0, 0.0,
            true, 5, 5, 60, 5, true, false, true, false
        );
        assert.strictEqual(rows, 1);
        assertArrayClose(
            wasm.read_f64_array(ptrs[0], total * rows),
            batch.oscillator,
            1e-12,
            'batch_into osc mismatch'
        );
        assertArrayClose(
            wasm.read_f64_array(ptrs[4], total * rows),
            batch.market_extreme,
            1e-12,
            'batch_into market mismatch'
        );
    } finally {
        for (const ptr of ptrs) {
            wasm.kase_peak_oscillator_with_divergences_free(ptr, total);
        }
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});
