import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, loadTestData } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function manualAdaptiveBoundsRsi(data, rsiLength, alpha) {
    const n = data.length;
    const out = {
        rsi: new Float64Array(n),
        lower_bound: new Float64Array(n),
        lower_mid: new Float64Array(n),
        mid: new Float64Array(n),
        upper_mid: new Float64Array(n),
        upper_bound: new Float64Array(n),
        regime: new Float64Array(n),
        regime_flip: new Float64Array(n),
        lower_signal: new Float64Array(n),
        upper_signal: new Float64Array(n),
    };
    for (const key of Object.keys(out)) {
        out[key].fill(NaN);
    }

    let prevPrice = NaN;
    let seedCount = 0;
    let sumGain = 0.0;
    let sumLoss = 0.0;
    let avgGain = 0.0;
    let avgLoss = 0.0;
    let seeded = false;
    const invP = 1.0 / rsiLength;
    const beta = 1.0 - invP;

    let c1 = 20.0;
    let c2 = 40.0;
    let c3 = 50.0;
    let c4 = 60.0;
    let c5 = 80.0;
    let prevRsi = NaN;
    let prevC1 = NaN;
    let prevC5 = NaN;
    let prevRegime = NaN;
    let canShowLower = true;
    let canShowUpper = true;

    for (let i = 0; i < n; i++) {
        const value = data[i];
        if (!Number.isFinite(value)) {
            prevPrice = NaN;
            seedCount = 0;
            sumGain = 0.0;
            sumLoss = 0.0;
            avgGain = 0.0;
            avgLoss = 0.0;
            seeded = false;
            prevRsi = NaN;
            prevC1 = NaN;
            prevC5 = NaN;
            prevRegime = NaN;
            continue;
        }

        if (!Number.isFinite(prevPrice)) {
            prevPrice = value;
            continue;
        }

        const delta = value - prevPrice;
        prevPrice = value;
        const gain = Math.max(delta, 0.0);
        const loss = Math.max(-delta, 0.0);

        if (!seeded) {
            sumGain += gain;
            sumLoss += loss;
            seedCount += 1;
            if (seedCount !== rsiLength) {
                continue;
            }
            seeded = true;
            avgGain = sumGain * invP;
            avgLoss = sumLoss * invP;
        } else {
            avgGain = avgGain * beta + invP * gain;
            avgLoss = avgLoss * beta + invP * loss;
        }

        const denom = avgGain + avgLoss;
        const rsi = denom === 0.0 ? 50.0 : 100.0 * avgGain / denom;

        const distances = [
            Math.abs(rsi - c1),
            Math.abs(rsi - c2),
            Math.abs(rsi - c3),
            Math.abs(rsi - c4),
            Math.abs(rsi - c5),
        ];
        const winner = distances.indexOf(Math.min(...distances));
        if (winner === 0) c1 += (rsi - c1) * alpha;
        else if (winner === 1) c2 += (rsi - c2) * alpha;
        else if (winner === 2) c3 += (rsi - c3) * alpha;
        else if (winner === 3) c4 += (rsi - c4) * alpha;
        else c5 += (rsi - c5) * alpha;

        const regime = rsi <= c1 ? -2 : rsi <= c2 ? -1 : rsi <= c3 ? 0 : rsi <= c4 ? 1 : 2;
        const crossedMid = Number.isFinite(prevRsi) &&
            ((prevRsi <= 50.0 && rsi > 50.0) || (prevRsi >= 50.0 && rsi < 50.0));
        if (crossedMid) {
            canShowLower = true;
            canShowUpper = true;
        }

        const lowerSignal = Number.isFinite(prevRsi) && Number.isFinite(prevC1) &&
            prevRsi >= prevC1 && rsi < c1 && canShowLower;
        const upperSignal = Number.isFinite(prevRsi) && Number.isFinite(prevC5) &&
            prevRsi <= prevC5 && rsi > c5 && canShowUpper;
        if (lowerSignal) canShowLower = false;
        if (upperSignal) canShowUpper = false;

        const regimeFlip = Number.isFinite(prevRegime) && prevRegime === 0 && regime !== 0;

        out.rsi[i] = rsi;
        out.lower_bound[i] = c1;
        out.lower_mid[i] = c2;
        out.mid[i] = c3;
        out.upper_mid[i] = c4;
        out.upper_bound[i] = c5;
        out.regime[i] = regime;
        out.regime_flip[i] = regimeFlip ? 1.0 : 0.0;
        out.lower_signal[i] = lowerSignal ? 1.0 : 0.0;
        out.upper_signal[i] = upperSignal ? 1.0 : 0.0;

        prevRsi = rsi;
        prevC1 = c1;
        prevC5 = c5;
        prevRegime = regime;
    }

    return out;
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('adaptive_bounds_rsi_js matches manual reference', () => {
    const close = new Float64Array(testData.close.slice(0, 180));
    const result = wasm.adaptive_bounds_rsi_js(close, 14, 0.1);
    const expected = manualAdaptiveBoundsRsi(close, 14, 0.1);

    for (const key of Object.keys(expected)) {
        assertArrayClose(result[key], expected[key], 1e-12, `${key} mismatch`);
    }
});

test('adaptive_bounds_rsi_batch first row matches single', () => {
    const close = new Float64Array(testData.close.slice(0, 128));
    const batch = wasm.adaptive_bounds_rsi_batch(close, {
        rsi_length_range: [14, 16, 2],
        alpha_range: [0.1, 0.2, 0.1],
    });
    const single = wasm.adaptive_bounds_rsi_js(close, 14, 0.1);

    assert.strictEqual(batch.rows, 4);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.rsi_lengths, [14, 14, 16, 16]);
    assert.deepStrictEqual(batch.alphas, [0.1, 0.2, 0.1, 0.2]);

    for (const key of [
        'rsi',
        'lower_bound',
        'lower_mid',
        'mid',
        'upper_mid',
        'upper_bound',
        'regime',
        'regime_flip',
        'lower_signal',
        'upper_signal',
    ]) {
        assertArrayClose(
            batch[key].slice(0, close.length),
            single[key],
            1e-12,
            `batch ${key} mismatch`,
        );
    }
});

test('adaptive_bounds_rsi_js rejects invalid params', () => {
    const close = new Float64Array(testData.close.slice(0, 64));
    assert.throws(() => {
        wasm.adaptive_bounds_rsi_js(close, 0, 0.1);
    }, /invalid rsi_length/i);
    assert.throws(() => {
        wasm.adaptive_bounds_rsi_js(close, 14, 0.0);
    }, /invalid alpha/i);
});
