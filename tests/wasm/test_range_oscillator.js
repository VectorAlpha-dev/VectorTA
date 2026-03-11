import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, loadTestData } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function manualRangeOscillator(high, low, close, length, mult) {
    const n = close.length;
    const out = {
        oscillator: new Float64Array(n),
        ma: new Float64Array(n),
        upper_band: new Float64Array(n),
        lower_band: new Float64Array(n),
        range_width: new Float64Array(n),
        in_range: new Float64Array(n),
        trend: new Float64Array(n),
        break_up: new Float64Array(n),
        break_down: new Float64Array(n),
    };
    for (const key of Object.keys(out)) out[key].fill(NaN);

    let atr200 = null;
    let atr2000 = null;
    let count200 = 0;
    let count2000 = 0;
    let sum200 = 0.0;
    let sum2000 = 0.0;
    let prevClose200 = NaN;
    let prevClose2000 = NaN;
    let closes = [];
    let trend = 0.0;

    for (let i = 0; i < n; i++) {
        const h = high[i];
        const l = low[i];
        const c = close[i];
        if (!Number.isFinite(h) || !Number.isFinite(l) || !Number.isFinite(c)) {
            atr200 = null;
            atr2000 = null;
            count200 = 0;
            count2000 = 0;
            sum200 = 0.0;
            sum2000 = 0.0;
            prevClose200 = NaN;
            prevClose2000 = NaN;
            closes = [];
            trend = 0.0;
            continue;
        }

        const tr200 = Number.isFinite(prevClose200)
            ? Math.max(h - l, Math.abs(h - prevClose200), Math.abs(l - prevClose200))
            : h - l;
        prevClose200 = c;
        if (atr200 === null) {
            count200 += 1;
            sum200 += tr200;
            if (count200 === 200) atr200 = sum200 / 200.0;
        } else {
            atr200 = ((atr200 * 199.0) + tr200) / 200.0;
        }

        const tr2000 = Number.isFinite(prevClose2000)
            ? Math.max(h - l, Math.abs(h - prevClose2000), Math.abs(l - prevClose2000))
            : h - l;
        prevClose2000 = c;
        if (atr2000 === null) {
            count2000 += 1;
            sum2000 += tr2000;
            if (count2000 === 2000) atr2000 = sum2000 / 2000.0;
        } else {
            atr2000 = ((atr2000 * 1999.0) + tr2000) / 2000.0;
        }

        closes.push(c);
        if (closes.length > length + 1) closes.shift();

        const atrRaw = atr2000 ?? atr200;
        if (atrRaw === null || closes.length < length + 1) continue;

        let sumWeighted = 0.0;
        let sumWeights = 0.0;
        for (let j = 0; j < length; j++) {
            const curr = closes[closes.length - 1 - j];
            const prev = closes[closes.length - 2 - j];
            if (Math.abs(prev) <= 1e-12) continue;
            const w = Math.abs(curr - prev) / prev;
            sumWeighted += curr * w;
            sumWeights += w;
        }
        if (Math.abs(sumWeights) <= 1e-12) continue;

        const ma = sumWeighted / sumWeights;
        const width = atrRaw * mult;
        let maxDist = 0.0;
        for (let j = 0; j < length; j++) {
            maxDist = Math.max(maxDist, Math.abs(closes[closes.length - 1 - j] - ma));
        }

        if (c > ma) trend = 1.0;
        else if (c < ma) trend = -1.0;

        const upper = ma + width;
        const lower = ma - width;
        const osc = Math.abs(width) <= 1e-12 ? NaN : 100.0 * (c - ma) / width;

        out.oscillator[i] = osc;
        out.ma[i] = ma;
        out.upper_band[i] = upper;
        out.lower_band[i] = lower;
        out.range_width[i] = width;
        out.in_range[i] = maxDist <= width ? 1.0 : 0.0;
        out.trend[i] = trend;
        out.break_up[i] = c > upper ? 1.0 : 0.0;
        out.break_down[i] = c < lower ? 1.0 : 0.0;
    }

    return out;
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    const mod = await import(importPath);
    wasm = mod.default ?? mod;
    testData = loadTestData();
});

test('range_oscillator_js matches manual reference', () => {
    const high = new Float64Array(testData.high.slice(0, 320));
    const low = new Float64Array(testData.low.slice(0, 320));
    const close = new Float64Array(testData.close.slice(0, 320));
    const result = wasm.range_oscillator_js(high, low, close, 50, 2.0);
    const expected = manualRangeOscillator(high, low, close, 50, 2.0);

    for (const key of Object.keys(expected)) {
        assertArrayClose(result[key], expected[key], 1e-12, `${key} mismatch`);
    }
});

test('range_oscillator_batch first row matches single', () => {
    const high = new Float64Array(testData.high.slice(0, 320));
    const low = new Float64Array(testData.low.slice(0, 320));
    const close = new Float64Array(testData.close.slice(0, 320));
    const batch = wasm.range_oscillator_batch(high, low, close, {
        length_range: [50, 52, 2],
        mult_range: [2.0, 2.5, 0.5],
    });
    const single = wasm.range_oscillator_js(high, low, close, 50, 2.0);

    assert.strictEqual(batch.rows, 4);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.lengths, [50, 50, 52, 52]);
    assert.deepStrictEqual(batch.mults, [2.0, 2.5, 2.0, 2.5]);

    for (const key of Object.keys(single)) {
        assertArrayClose(batch[key].slice(0, close.length), single[key], 1e-12, `${key} batch mismatch`);
    }
});

test('range_oscillator_js rejects invalid params', () => {
    const high = new Float64Array(testData.high.slice(0, 128));
    const low = new Float64Array(testData.low.slice(0, 128));
    const close = new Float64Array(testData.close.slice(0, 128));

    assert.throws(() => {
        wasm.range_oscillator_js(high, low, close, 0, 2.0);
    }, /invalid length/i);
    assert.throws(() => {
        wasm.range_oscillator_js(high, low, close, 50, 0.0);
    }, /invalid mult/i);
});
