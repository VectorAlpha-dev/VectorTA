import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, loadTestData } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function makeRma(period) {
    return { period, count: 0, sum: 0.0, value: null };
}

function updateRma(state, input) {
    if (state.value !== null) {
        state.value = ((state.value * (state.period - 1.0)) + input) / state.period;
        return state.value;
    }
    state.count += 1;
    state.sum += input;
    if (state.count === state.period) {
        state.value = state.sum / state.period;
        return state.value;
    }
    return null;
}

function wma(window, period, value) {
    window.push(value);
    if (window.length > period) window.shift();
    if (window.length < period) return null;
    let num = 0.0;
    let den = 0.0;
    for (let i = 0; i < period; i++) {
        const weight = i + 1;
        num += window[i] * weight;
        den += weight;
    }
    return num / den;
}

function hma(state, period, value) {
    const half = Math.max(1, Math.floor(period / 2));
    const sqrtLen = Math.max(1, Math.floor(Math.sqrt(period)));
    const h = wma(state.half, half, value);
    const f = wma(state.full, period, value);
    if (h === null || f === null) return null;
    return wma(state.sqrt, sqrtLen, 2.0 * h - f);
}

function manualVolumeWeightedRelativeStrengthIndex(source, volume, rsiLength, rangeLength, maLength, maType) {
    const n = source.length;
    const out = {
        rsi: new Float64Array(n),
        consolidation_strength: new Float64Array(n),
        rsi_ma: new Float64Array(n),
        bearish_tp: new Float64Array(n),
        bullish_tp: new Float64Array(n),
    };
    for (const key of Object.keys(out)) out[key].fill(NaN);

    let prevSource = null;
    let prevRsi = null;
    let prevDir = null;
    let validRsiCount = 0;
    const upRma = makeRma(rsiLength);
    const downRma = makeRma(rsiLength);
    const volRma = makeRma(rsiLength);

    const smaWindow = [];
    let smaSum = 0.0;
    const vwmaPv = [];
    const vwmaVol = [];
    let vwmaPvSum = 0.0;
    let vwmaVolSum = 0.0;
    let emaState = null;
    const maRma = makeRma(maLength);

    const xWindow = [];
    let xSum = 0.0;
    const shortState = { half: [], full: [], sqrt: [] };
    const longState = { half: [], full: [], sqrt: [] };

    for (let i = 0; i < n; i++) {
        const s = source[i];
        const v = volume[i];
        if (!Number.isFinite(s) || !Number.isFinite(v)) {
            prevSource = null;
            prevRsi = null;
            prevDir = null;
            validRsiCount = 0;
            upRma.count = downRma.count = volRma.count = 0;
            upRma.sum = downRma.sum = volRma.sum = 0.0;
            upRma.value = downRma.value = volRma.value = null;
            smaWindow.length = 0;
            smaSum = 0.0;
            vwmaPv.length = 0;
            vwmaVol.length = 0;
            vwmaPvSum = 0.0;
            vwmaVolSum = 0.0;
            emaState = null;
            maRma.count = 0;
            maRma.sum = 0.0;
            maRma.value = null;
            xWindow.length = 0;
            xSum = 0.0;
            shortState.half.length = shortState.full.length = shortState.sqrt.length = 0;
            longState.half.length = longState.full.length = longState.sqrt.length = 0;
            continue;
        }
        if (prevSource === null) {
            prevSource = s;
            continue;
        }

        const delta = s - prevSource;
        prevSource = s;
        const upNum = updateRma(upRma, Math.max(delta, 0.0) * v);
        const downNum = updateRma(downRma, Math.max(-delta, 0.0) * v);
        const volAvg = updateRma(volRma, v);
        if (upNum === null || downNum === null || volAvg === null || Math.abs(volAvg) <= 1e-12) continue;

        const up = upNum / volAvg;
        const down = downNum / volAvg;
        const rsi = Math.abs(down) <= 1e-12 ? 100.0 : Math.abs(up) <= 1e-12 ? 0.0 : 100.0 - (100.0 / (1.0 + up / down));
        out.rsi[i] = rsi;

        const type = maType.toLowerCase();
        if (type === 'ema') {
            const alpha = 2.0 / (maLength + 1.0);
            emaState = emaState === null ? rsi : alpha * rsi + (1.0 - alpha) * emaState;
            out.rsi_ma[i] = emaState;
        } else if (type === 'sma') {
            smaWindow.push(rsi);
            smaSum += rsi;
            if (smaWindow.length > maLength) smaSum -= smaWindow.shift();
            if (smaWindow.length === maLength) out.rsi_ma[i] = smaSum / maLength;
        } else if (type === 'wma') {
            const value = wma(smaWindow, maLength, rsi);
            if (value !== null) out.rsi_ma[i] = value;
        } else if (type === 'hma') {
            const value = hma({ half: vwmaPv, full: vwmaVol, sqrt: smaWindow }, maLength, rsi);
            if (value !== null) out.rsi_ma[i] = value;
        } else if (type === 'smma (rma)' || type === 'smma' || type === 'rma') {
            const value = updateRma(maRma, rsi);
            if (value !== null) out.rsi_ma[i] = value;
        } else if (type === 'vwma') {
            vwmaPv.push(rsi * v);
            vwmaVol.push(v);
            vwmaPvSum += rsi * v;
            vwmaVolSum += v;
            if (vwmaPv.length > maLength) {
                vwmaPvSum -= vwmaPv.shift();
                vwmaVolSum -= vwmaVol.shift();
            }
            if (vwmaPv.length === maLength && Math.abs(vwmaVolSum) > 1e-12) {
                out.rsi_ma[i] = vwmaPvSum / vwmaVolSum;
            }
        }

        if (prevRsi !== null && prevRsi >= 80.0 && rsi < 80.0) out.bearish_tp[i] = 95.0;
        if (prevRsi !== null && prevRsi <= 20.0 && rsi > 20.0) out.bullish_tp[i] = 5.0;

        validRsiCount += 1;
        const dir = rsi > 50.0 ? 1 : -1;
        const transition = prevDir !== null && prevDir + dir === 0 ? 1.0 : 0.0;
        prevDir = dir;
        prevRsi = rsi;

        const x = transition / Math.min(validRsiCount, rangeLength);
        xWindow.push(x);
        xSum += x;
        if (xWindow.length > rangeLength) xSum -= xWindow.shift();
        if (xWindow.length === rangeLength) {
            const p = xSum / rangeLength;
            const shortValue = hma(shortState, Math.max(1, Math.floor(rangeLength / 2)), p);
            if (shortValue !== null) {
                const f = shortValue * (Math.max(1, Math.floor(rangeLength / 2)) * (5.0 / 3.0));
                const longValue = hma(longState, rangeLength * 2, f);
                if (longValue !== null) out.consolidation_strength[i] = Math.max(longValue, 0.0);
            }
        }
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

test('volume_weighted_relative_strength_index_js matches manual reference', () => {
    const source = new Float64Array(testData.close.slice(0, 320));
    const volume = new Float64Array(testData.volume.slice(0, 320));
    const result = wasm.volume_weighted_relative_strength_index_js(source, volume, 14, 10, 14, 'EMA');
    const expected = manualVolumeWeightedRelativeStrengthIndex(source, volume, 14, 10, 14, 'EMA');

    for (const key of Object.keys(expected)) {
        assertArrayClose(result[key], expected[key], 1e-12, `${key} mismatch`);
    }
});

test('volume_weighted_relative_strength_index_batch first row matches single', () => {
    const source = new Float64Array(testData.close.slice(0, 256));
    const volume = new Float64Array(testData.volume.slice(0, 256));
    const batch = wasm.volume_weighted_relative_strength_index_batch(source, volume, {
        rsi_length_range: [14, 16, 2],
        range_length_range: [10, 10, 0],
        ma_length_range: [14, 14, 0],
        ma_type: 'EMA',
    });
    const single = wasm.volume_weighted_relative_strength_index_js(source, volume, 14, 10, 14, 'EMA');

    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, source.length);
    assert.deepStrictEqual(batch.rsi_lengths, [14, 16]);
    assert.deepStrictEqual(batch.range_lengths, [10, 10]);
    assert.deepStrictEqual(batch.ma_lengths, [14, 14]);
    assert.deepStrictEqual(batch.ma_types, ['EMA', 'EMA']);

    for (const key of Object.keys(single)) {
        assertArrayClose(batch[key].slice(0, source.length), single[key], 1e-12, `${key} batch mismatch`);
    }
});

test('volume_weighted_relative_strength_index_js rejects invalid params and exports exist', () => {
    assert.strictEqual(typeof wasm.volume_weighted_relative_strength_index_js, 'function');
    assert.strictEqual(typeof wasm.volume_weighted_relative_strength_index_batch, 'function');

    const source = new Float64Array(testData.close.slice(0, 128));
    const volume = new Float64Array(testData.volume.slice(0, 128));
    assert.throws(() => {
        wasm.volume_weighted_relative_strength_index_js(source, volume, 0, 10, 14, 'EMA');
    }, /invalid rsi_length/i);
    assert.throws(() => {
        wasm.volume_weighted_relative_strength_index_js(source, volume, 14, 0, 14, 'EMA');
    }, /invalid range_length/i);
    assert.throws(() => {
        wasm.volume_weighted_relative_strength_index_js(source, volume, 14, 10, 14, 'BAD');
    }, /invalid ma_type/i);
});
