import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function sampleOhlc(length) {
    const open = new Float64Array(length);
    const high = new Float64Array(length);
    const low = new Float64Array(length);
    const close = new Float64Array(length);
    open.fill(NaN);
    high.fill(NaN);
    low.fill(NaN);
    close.fill(NaN);
    let prev = 100.0;
    for (let i = 2; i < length; i += 1) {
        const x = i;
        const wave = Math.sin(x * 0.11) * 2.4 + Math.cos(x * 0.037) * 1.3;
        const o = prev + wave * 0.35;
        const c = o + Math.sin(x * 0.19) * 1.1 - Math.cos(x * 0.07) * 0.4;
        const h = Math.max(o, c) + 0.6 + Math.abs(Math.sin(x * 0.03)) * 0.25;
        const l = Math.min(o, c) - 0.6 - Math.abs(Math.cos(x * 0.02)) * 0.25;
        open[i] = o;
        high[i] = h;
        low[i] = l;
        close[i] = c;
        prev = c;
    }
    return { open, high, low, close };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
});

test('grover_llorens_cycle_oscillator_js output contract', () => {
    const { open, high, low, close } = sampleOhlc(512);
    const out = wasm.grover_llorens_cycle_oscillator_js(
        open, high, low, close, 100, 10.0, 'close', true, 20
    );

    assert.strictEqual(out.values.length, close.length);
    const firstValid = out.values.findIndex(v => !isNaN(v));
    assert(firstValid >= 100);
    const tailStart = Math.min(firstValid + 32, out.values.length);
    for (let i = tailStart; i < out.values.length; i += 1) {
        assert(!isNaN(out.values[i]), `unexpected NaN at ${i}`);
    }
});

test('grover_llorens_cycle_oscillator_js rejects invalid parameters', () => {
    const { open, high, low, close } = sampleOhlc(64);

    assert.throws(
        () => wasm.grover_llorens_cycle_oscillator_js(
            open, high, low, close, 0, 10.0, 'close', true, 20
        ),
        /Invalid length/
    );
    assert.throws(
        () => wasm.grover_llorens_cycle_oscillator_js(
            open, high, low, close, 10, 10.0, 'bad', true, 20
        ),
        /Invalid source/
    );
});

test('grover_llorens_cycle_oscillator_into pointer path matches safe API', () => {
    const { open, high, low, close } = sampleOhlc(192);
    const safe = wasm.grover_llorens_cycle_oscillator_js(
        open, high, low, close, 50, 8.0, 'hlc3', true, 14
    );

    const openPtr = wasm.copy_f64_array(open);
    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const valuesPtr = wasm.grover_llorens_cycle_oscillator_alloc(close.length);

    try {
        wasm.grover_llorens_cycle_oscillator_into(
            openPtr, highPtr, lowPtr, closePtr, valuesPtr, close.length, 50, 8.0, 'hlc3', true, 14
        );
        assertArrayClose(
            wasm.read_f64_array(valuesPtr, close.length),
            safe.values,
            1e-12,
            'pointer-path mismatch',
        );
    } finally {
        wasm.grover_llorens_cycle_oscillator_free(valuesPtr, close.length);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('grover_llorens_cycle_oscillator_js reset after NaN', () => {
    const { open, high, low, close } = sampleOhlc(256);
    open[120] = NaN;
    high[120] = NaN;
    low[120] = NaN;
    close[120] = NaN;

    const out = wasm.grover_llorens_cycle_oscillator_js(
        open, high, low, close, 60, 8.0, 'hlc3', true, 14
    );

    assert(isNaN(out.values[120]));
    assert(isNaN(out.values[121]));
    assert(isNaN(out.values[140]));
    assert(Number.isFinite(out.values.at(-1)));
});

test('grover_llorens_cycle_oscillator_batch_js single parameter set matches safe API', () => {
    const { open, high, low, close } = sampleOhlc(192);
    const batch = wasm.grover_llorens_cycle_oscillator_batch_js(open, high, low, close, {
        length_range: [50, 50, 0],
        mult_range: [8.0, 8.0, 0.0],
        source: 'close',
        smooth: true,
        rsi_period_range: [14, 14, 0],
    });
    const single = wasm.grover_llorens_cycle_oscillator_js(
        open, high, low, close, 50, 8.0, 'close', true, 14
    );

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.lengths, [50]);
    assert.deepStrictEqual(batch.mults, [8.0]);
    assert.deepStrictEqual(batch.sources, ['close']);
    assert.deepStrictEqual(batch.smooth_flags, [true]);
    assert.deepStrictEqual(batch.rsi_periods, [14]);
    assertArrayClose(batch.values, single.values, 1e-12, 'batch mismatch');
});

test('grover_llorens_cycle_oscillator_batch_js metadata and batch_into', () => {
    const { open, high, low, close } = sampleOhlc(160);
    const batch = wasm.grover_llorens_cycle_oscillator_batch_js(open, high, low, close, {
        length_range: [40, 60, 20],
        mult_range: [8.0, 10.0, 2.0],
        source: 'hlc3',
        smooth: false,
        rsi_period_range: [14, 16, 2],
    });

    assert.strictEqual(batch.rows, 8);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.lengths, [40, 40, 40, 40, 60, 60, 60, 60]);
    assert.deepStrictEqual(batch.mults, [8.0, 8.0, 10.0, 10.0, 8.0, 8.0, 10.0, 10.0]);
    assert.deepStrictEqual(batch.sources, Array(8).fill('hlc3'));
    assert.deepStrictEqual(batch.smooth_flags, Array(8).fill(false));
    assert.deepStrictEqual(batch.rsi_periods, [14, 16, 14, 16, 14, 16, 14, 16]);

    const openPtr = wasm.copy_f64_array(open);
    const highPtr = wasm.copy_f64_array(high);
    const lowPtr = wasm.copy_f64_array(low);
    const closePtr = wasm.copy_f64_array(close);
    const total = batch.rows * close.length;
    const valuesPtr = wasm.grover_llorens_cycle_oscillator_alloc(total);

    try {
        const rows = wasm.grover_llorens_cycle_oscillator_batch_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            valuesPtr,
            close.length,
            40,
            60,
            20,
            8.0,
            10.0,
            2.0,
            'hlc3',
            false,
            14,
            16,
            2,
        );
        assert.strictEqual(rows, 8);
        assertArrayClose(
            wasm.read_f64_array(valuesPtr, total),
            batch.values,
            1e-12,
            'batch_into mismatch',
        );
    } finally {
        wasm.grover_llorens_cycle_oscillator_free(valuesPtr, total);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('grover_llorens_cycle_oscillator_batch_js rejects invalid config', () => {
    const { open, high, low, close } = sampleOhlc(64);
    assert.throws(
        () => wasm.grover_llorens_cycle_oscillator_batch_js(open, high, low, close, {
            length_range: [0, 10, 1],
            mult_range: [8.0, 8.0, 0.0],
            source: 'close',
            smooth: true,
            rsi_period_range: [14, 14, 0],
        }),
        /Invalid length/
    );
});
