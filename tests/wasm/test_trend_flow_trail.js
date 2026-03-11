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
    for (let i = 0; i < 180; i++) {
        const base = i < 60 ? 100 + i * 0.35 : i < 120 ? 121 - (i - 60) * 0.22 : 108 + (i - 120) * 0.42;
        const wiggle = ((i % 7) - 3) * 0.18;
        const c = base + wiggle;
        const o = c - ((i % 5) - 2) * 0.11;
        open.push(o);
        close.push(c);
        high.push(Math.max(c, o) + 1.2 + (i % 3) * 0.07);
        low.push(Math.min(c, o) - 1.1 - (i % 4) * 0.05);
        volume.push(1000 + (i % 9) * 37 + i * 4);
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
    const importPath = process.platform === 'win32' ? 'file:///' + wasmPath.replace(/\\/g, '/') : wasmPath;
    wasm = await import(importPath);
});

test('Trend Flow Trail outputs are present', () => {
    const data = sampleOhlcv();
    const result = wasm.trend_flow_trail_js(data.open, data.high, data.low, data.close, data.volume, 33, 3.3, 14);
    assert(result.alpha_trail.some(Number.isFinite));
    assert(result.mfi.some(Number.isFinite));
});

test('Trend Flow Trail stream matches batch', () => {
    const data = sampleOhlcv();
    const batch = wasm.trend_flow_trail_js(data.open, data.high, data.low, data.close, data.volume, 33, 3.3, 14);
    const stream = new wasm.TrendFlowTrailStreamWasm(33, 3.3, 14);
    const alphaTrail = new Array(data.close.length).fill(NaN);
    for (let i = 0; i < data.close.length; i++) {
        const current = stream.update(data.open[i], data.high[i], data.low[i], data.close[i], data.volume[i]);
        if (current) {
            alphaTrail[i] = current[0];
        }
    }
    assertArrayClose(alphaTrail, batch.alpha_trail, 0.0, 0.0, 'stream mismatch');
});

test('Trend Flow Trail batch returns rows', () => {
    const data = sampleOhlcv();
    const result = wasm.trend_flow_trail_batch(data.open, data.high, data.low, data.close, data.volume, {
        alpha_length_range: [33, 33, 0],
        alpha_multiplier_range: [3.3, 3.4, 0.1],
        mfi_length_range: [14, 14, 0],
    });
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, data.close.length);
});
