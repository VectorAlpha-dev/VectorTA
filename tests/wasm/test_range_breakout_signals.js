import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function sampleOhlcv() {
    const open = [];
    const high = [];
    const low = [];
    const close = [];
    const volume = [];

    for (let i = 0; i < 24; i++) {
        const base = 100 + i * 0.35;
        const o = base - 1.4;
        const c = base + (i % 2 === 0 ? 1.7 : -1.6);
        open.push(o);
        close.push(c);
        high.push(Math.max(o, c) + 1.1);
        low.push(Math.min(o, c) - 1.0);
        volume.push(900 + i * 8);
    }

    for (let i = 24; i < 36; i++) {
        const base = 108.5 + (i - 24) * 0.03;
        open.push(base - 0.03);
        close.push(base + 0.03);
        high.push(base + 0.18);
        low.push(base - 0.18);
        volume.push(1600 + i * 6);
    }

    open.push(109.2);
    high.push(112.4);
    low.push(109.0);
    close.push(112.1);
    volume.push(2200);

    for (let i = 37; i < 60; i++) {
        const base = 111.8 - (i - 37) * 0.08;
        const o = base + 0.8;
        const c = base - 0.9;
        open.push(o);
        close.push(c);
        high.push(Math.max(o, c) + 0.9);
        low.push(Math.min(o, c) - 0.8);
        volume.push(1100 + i * 5);
    }

    for (let i = 60; i < 72; i++) {
        const base = 104.7 - (i - 60) * 0.02;
        open.push(base + 0.02);
        close.push(base - 0.02);
        high.push(base + 0.16);
        low.push(base - 0.16);
        volume.push(1750 + i * 5);
    }

    open.push(103.9);
    high.push(104.0);
    low.push(100.6);
    close.push(100.9);
    volume.push(2400);

    for (let i = 73; i < 96; i++) {
        const base = 101.4 + (i - 73) * 0.06;
        const o = base - 0.3;
        const c = base + 0.25;
        open.push(o);
        close.push(c);
        high.push(Math.max(o, c) + 0.45);
        low.push(Math.min(o, c) - 0.45);
        volume.push(1200 + i * 3);
    }

    return {
        open: new Float64Array(open),
        high: new Float64Array(high),
        low: new Float64Array(low),
        close: new Float64Array(close),
        volume: new Float64Array(volume),
    };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath =
        process.platform === 'win32' ? 'file:///' + wasmPath.replace(/\\/g, '/') : wasmPath;
    wasm = await import(importPath);
});

test('Range Breakout Signals outputs are present', () => {
    const data = sampleOhlcv();
    const result = wasm.range_breakout_signals_js(
        data.open,
        data.high,
        data.low,
        data.close,
        data.volume,
        20,
        5,
    );
    assert(result.range_top.some(Number.isFinite));
    assert(result.range_bottom.some(Number.isFinite));
});

test('Range Breakout Signals stream matches batch', async () => {
    const data = sampleOhlcv();
    const batch = wasm.range_breakout_signals_js(
        data.open,
        data.high,
        data.low,
        data.close,
        data.volume,
        18,
        4,
    );
    const stream = new wasm.RangeBreakoutSignalsStreamWasm(18, 4);

    const rangeTop = new Array(data.close.length).fill(NaN);
    for (let i = 0; i < data.close.length; i++) {
        const current = stream.update(
            data.open[i],
            data.high[i],
            data.low[i],
            data.close[i],
            data.volume[i],
        );
        const decoded = current === null ? null : current;
        if (decoded) {
            rangeTop[i] = decoded[0];
        }
    }

    assertArrayClose(rangeTop, batch.range_top, 0.0, 0.0, 'stream mismatch');
});

test('Range Breakout Signals fast API matches safe API', () => {
    const data = sampleOhlcv();
    const len = data.close.length;
    const inputPtrs = [
        wasm.range_breakout_signals_alloc(len),
        wasm.range_breakout_signals_alloc(len),
        wasm.range_breakout_signals_alloc(len),
        wasm.range_breakout_signals_alloc(len),
        wasm.range_breakout_signals_alloc(len),
    ];
    const outputPtrs = [
        wasm.range_breakout_signals_alloc(len),
        wasm.range_breakout_signals_alloc(len),
        wasm.range_breakout_signals_alloc(len),
        wasm.range_breakout_signals_alloc(len),
        wasm.range_breakout_signals_alloc(len),
        wasm.range_breakout_signals_alloc(len),
    ];

    try {
        new Float64Array(wasm.__wasm.memory.buffer, inputPtrs[0], len).set(data.open);
        new Float64Array(wasm.__wasm.memory.buffer, inputPtrs[1], len).set(data.high);
        new Float64Array(wasm.__wasm.memory.buffer, inputPtrs[2], len).set(data.low);
        new Float64Array(wasm.__wasm.memory.buffer, inputPtrs[3], len).set(data.close);
        new Float64Array(wasm.__wasm.memory.buffer, inputPtrs[4], len).set(data.volume);

        wasm.range_breakout_signals_into(
            inputPtrs[0],
            inputPtrs[1],
            inputPtrs[2],
            inputPtrs[3],
            inputPtrs[4],
            outputPtrs[0],
            outputPtrs[1],
            outputPtrs[2],
            outputPtrs[3],
            outputPtrs[4],
            outputPtrs[5],
            len,
            20,
            5,
        );

        const safe = wasm.range_breakout_signals_js(
            data.open,
            data.high,
            data.low,
            data.close,
            data.volume,
            20,
            5,
        );
        assertArrayClose(
            new Float64Array(wasm.__wasm.memory.buffer, outputPtrs[0], len),
            safe.range_top,
            0.0,
            0.0,
            'fast API mismatch',
        );
    } finally {
        for (const ptr of [...inputPtrs, ...outputPtrs]) {
            wasm.range_breakout_signals_free(ptr, len);
        }
    }
});

test('Range Breakout Signals batch returns rows', () => {
    const data = sampleOhlcv();
    const result = wasm.range_breakout_signals_batch(
        data.open,
        data.high,
        data.low,
        data.close,
        data.volume,
        {
            range_length_range: [20, 21, 1],
            confirmation_length_range: [5, 5, 0],
        },
    );
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, data.close.length);
});
