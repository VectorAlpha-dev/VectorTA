import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function sampleHlc() {
    const high = [];
    const low = [];
    const close = [];

    for (let i = 0; i < 60; i++) {
        close.push(100 + i * 0.65 + ((i % 5) - 2) * 0.08);
    }
    for (let i = 60; i < 100; i++) {
        const x = i - 60;
        close.push(139 + x * 0.05 - (i % 4) * 0.12);
    }
    for (let i = 100; i < 120; i++) {
        const x = i - 100;
        close.push(140 + x * 0.55);
    }
    for (let i = 120; i < 140; i++) {
        const x = i - 120;
        close.push(150 - x * 2.4);
    }
    for (let i = 140; i < 160; i++) {
        const x = i - 140;
        close.push(102 + x * 1.8);
    }

    for (let i = 0; i < close.length; i++) {
        const wiggle = (i % 3) * 0.15;
        high.push(close[i] + 1.6 + wiggle);
        low.push(close[i] - 1.5 - wiggle * 0.8);
    }

    return {
        high: new Float64Array(high),
        low: new Float64Array(low),
        close: new Float64Array(close),
    };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath =
        process.platform === 'win32' ? 'file:///' + wasmPath.replace(/\\/g, '/') : wasmPath;
    wasm = await import(importPath);
});

test('Exponential Trend outputs are present', () => {
    const data = sampleHlc();
    const result = wasm.exponential_trend_js(data.high, data.low, data.close, 0.00003, 4.0, 1.0);
    assert(result.uptrend_base.some(Number.isFinite));
    assert(result.downtrend_base.some(Number.isFinite));
});

test('Exponential Trend stream matches batch', () => {
    const data = sampleHlc();
    const batch = wasm.exponential_trend_js(data.high, data.low, data.close, 0.00003, 4.0, 1.0);
    const stream = new wasm.ExponentialTrendStreamWasm(0.00003, 4.0, 1.0);

    const uptrendBase = new Array(data.close.length).fill(NaN);
    for (let i = 0; i < data.close.length; i++) {
        const current = stream.update(data.high[i], data.low[i], data.close[i]);
        const decoded = current === null ? null : current;
        if (decoded) {
            uptrendBase[i] = decoded[0];
        }
    }

    assertArrayClose(uptrendBase, batch.uptrend_base, 0.0, 0.0, 'stream mismatch');
});

test('Exponential Trend fast API matches safe API', () => {
    const data = sampleHlc();
    const len = data.close.length;
    const inputPtrs = [
        wasm.exponential_trend_alloc(len),
        wasm.exponential_trend_alloc(len),
        wasm.exponential_trend_alloc(len),
    ];
    const outputPtrs = [
        wasm.exponential_trend_alloc(len),
        wasm.exponential_trend_alloc(len),
        wasm.exponential_trend_alloc(len),
        wasm.exponential_trend_alloc(len),
        wasm.exponential_trend_alloc(len),
        wasm.exponential_trend_alloc(len),
    ];

    try {
        new Float64Array(wasm.__wasm.memory.buffer, inputPtrs[0], len).set(data.high);
        new Float64Array(wasm.__wasm.memory.buffer, inputPtrs[1], len).set(data.low);
        new Float64Array(wasm.__wasm.memory.buffer, inputPtrs[2], len).set(data.close);

        wasm.exponential_trend_into(
            inputPtrs[0],
            inputPtrs[1],
            inputPtrs[2],
            outputPtrs[0],
            outputPtrs[1],
            outputPtrs[2],
            outputPtrs[3],
            outputPtrs[4],
            outputPtrs[5],
            len,
            0.00003,
            4.0,
            1.0,
        );

        const safe = wasm.exponential_trend_js(data.high, data.low, data.close, 0.00003, 4.0, 1.0);
        assertArrayClose(
            new Float64Array(wasm.__wasm.memory.buffer, outputPtrs[0], len),
            safe.uptrend_base,
            0.0,
            0.0,
            'fast API mismatch',
        );
    } finally {
        for (const ptr of [...inputPtrs, ...outputPtrs]) {
            wasm.exponential_trend_free(ptr, len);
        }
    }
});

test('Exponential Trend batch returns rows', () => {
    const data = sampleHlc();
    const result = wasm.exponential_trend_batch(data.high, data.low, data.close, {
        exp_rate_range: [0.00003, 0.00003, 0.0],
        initial_distance_range: [4.0, 4.1, 0.1],
        width_multiplier_range: [1.0, 1.0, 0.0],
    });
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, data.close.length);
});
