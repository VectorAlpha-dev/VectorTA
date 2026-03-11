import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function oscillatingData(size) {
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        const x = i;
        const c = 100 + Math.sin(x * 0.19) * 12 + Math.cos(x * 0.043) * 5 + 0.03 * x;
        close[i] = c;
        high[i] = c + 1.4 + (i % 5) * 0.07;
        low[i] = c - 1.3 - (i % 3) * 0.05;
    }
    return { high, low, close };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath =
        process.platform === 'win32' ? 'file:///' + wasmPath.replace(/\\/g, '/') : wasmPath;
    wasm = await import(importPath);
});

test('Vdubus generator outputs hist and force', () => {
    const { high, low, close } = oscillatingData(320);
    const result = wasm.vdubus_divergence_wave_pattern_generator_js(
        high,
        low,
        close,
        9,
        24,
        21,
        34,
        5,
        3,
        0.15,
        true,
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        true,
    );
    assert(result.hist.some(Number.isFinite));
    assert(result.opposing_force.some(Number.isFinite));
});

test('Vdubus generator invalid periods', () => {
    const { high, low, close } = oscillatingData(64);
    assert.throws(() => {
        wasm.vdubus_divergence_wave_pattern_generator_js(
            high,
            low,
            close,
            9,
            24,
            21,
            0,
            5,
            3,
            0.15,
            true,
            true,
            true,
            true,
            false,
            false,
            false,
            false,
            false,
            true,
        );
    }, /invalid periods/i);
});

test('Vdubus generator NaN gap restart', () => {
    const { high, low, close } = oscillatingData(260);
    high[140] = Number.NaN;
    low[140] = Number.NaN;
    close[140] = Number.NaN;
    const result = wasm.vdubus_divergence_wave_pattern_generator_js(
        high,
        low,
        close,
        9,
        24,
        21,
        34,
        5,
        3,
        0.15,
        true,
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        true,
    );
    const restartEnd = Math.min(result.hist.length, 140 + 34 + 5 - 1);
    for (let i = 140; i < restartEnd; i++) {
        assert(isNaN(result.hist[i]));
    }
});

test('Vdubus generator fast API matches safe API hist', () => {
    const { high, low, close } = oscillatingData(200);
    const len = close.length;
    const ptrs = Array.from({ length: 15 }, () =>
        wasm.vdubus_divergence_wave_pattern_generator_alloc(len),
    );
    try {
        new Float64Array(wasm.__wasm.memory.buffer, ptrs[0], len).set(high);
        new Float64Array(wasm.__wasm.memory.buffer, ptrs[1], len).set(low);
        new Float64Array(wasm.__wasm.memory.buffer, ptrs[2], len).set(close);
        wasm.vdubus_divergence_wave_pattern_generator_into(
            ptrs[0],
            ptrs[1],
            ptrs[2],
            ptrs[3],
            ptrs[4],
            ptrs[5],
            ptrs[6],
            ptrs[7],
            ptrs[8],
            ptrs[9],
            ptrs[10],
            ptrs[11],
            ptrs[12],
            ptrs[13],
            ptrs[14],
            len,
            9,
            24,
            21,
            34,
            5,
            3,
            0.15,
            true,
            true,
            true,
            true,
            false,
            false,
            false,
            false,
            false,
            true,
        );
        const safe = wasm.vdubus_divergence_wave_pattern_generator_js(
            high,
            low,
            close,
            9,
            24,
            21,
            34,
            5,
            3,
            0.15,
            true,
            true,
            true,
            true,
            false,
            false,
            false,
            false,
            false,
            true,
        );
        assertArrayClose(new Float64Array(wasm.__wasm.memory.buffer, ptrs[14], len), safe.hist, 1e-12);
    } finally {
        for (const ptr of ptrs) {
            wasm.vdubus_divergence_wave_pattern_generator_free(ptr, len);
        }
    }
});

test('Vdubus generator batch matches single hist', () => {
    const { high, low, close } = oscillatingData(220);
    const batch = wasm.vdubus_divergence_wave_pattern_generator_batch(high, low, close, {
        fast_depth_range: [7, 9, 2],
        slow_depth_range: [20, 24, 4],
        fast_length_range: [15, 17, 2],
        slow_length_range: [28, 30, 2],
        signal_length_range: [4, 5, 1],
        lookback_range: [2, 3, 1],
        err_tol_range: [0.10, 0.15, 0.05],
        show_standard: true,
        show_climax: true,
        show_rounded: true,
        show_predator: true,
        show_gartley: false,
        show_bat: false,
        show_butterfly: false,
        show_crab: false,
        show_deep: false,
        show_hs: true,
    });
    assert.strictEqual(batch.rows, 128);
    assert.strictEqual(batch.cols, close.length);
    const single = wasm.vdubus_divergence_wave_pattern_generator_js(
        high,
        low,
        close,
        7,
        20,
        15,
        28,
        4,
        2,
        0.10,
        true,
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        true,
    );
    assertArrayClose(batch.hist.slice(0, close.length), single.hist, 1e-12);
});
