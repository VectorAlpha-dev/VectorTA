import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, loadTestData } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function manualRangeFilteredTrendSignals(
    high,
    low,
    close,
    kalmanAlpha = 0.01,
    kalmanBeta = 0.1,
    kalmanPeriod = 77,
    dev = 1.2,
    supertrendFactor = 0.7,
    supertrendAtrPeriod = 7,
) {
    const n = close.length;
    const out = {
        kalman: new Float64Array(n),
        supertrend: new Float64Array(n),
        upper_band: new Float64Array(n),
        lower_band: new Float64Array(n),
        trend: new Float64Array(n),
        kalman_trend: new Float64Array(n),
        state: new Float64Array(n),
        market_trending: new Float64Array(n),
        market_ranging: new Float64Array(n),
        short_term_bullish: new Float64Array(n),
        short_term_bearish: new Float64Array(n),
        long_term_bullish: new Float64Array(n),
        long_term_bearish: new Float64Array(n),
    };
    for (const key of Object.keys(out)) out[key].fill(NaN);

    const wbuf = [];
    let wsum = 0.0;
    let wweighted = 0.0;
    const wdiv = (200 * 201) / 2.0;
    let atrSum = 0.0;
    let atrCount = 0;
    let atrVal = null;
    let atrPrevClose = NaN;
    let kalmanValue = NaN;
    let kalmanCov = 1.0;
    let prevClose = NaN;
    let prevLowerBand = NaN;
    let prevUpperBand = NaN;
    let prevSupertrend = NaN;
    let prevKalman = NaN;
    let prevAtrReady = false;
    let trendState = 0.0;
    let prevTrend = null;
    let prevKtrend = null;
    let prevState = null;

    for (let i = 0; i < n; i++) {
        const h = high[i];
        const l = low[i];
        const c = close[i];

        const gain = kalmanCov / (kalmanCov + kalmanAlpha * kalmanPeriod);
        if (Number.isNaN(kalmanValue) && Number.isFinite(prevClose)) {
            kalmanValue = prevClose;
        }
        let kalman = NaN;
        if (Number.isFinite(kalmanValue)) {
            kalmanValue = kalmanValue + gain * (c - kalmanValue);
            kalman = kalmanValue;
        }
        kalmanCov = (1.0 - gain) * kalmanCov + kalmanBeta / kalmanPeriod;
        prevClose = c;

        const tr = Number.isFinite(atrPrevClose)
            ? Math.max(h - l, Math.abs(h - atrPrevClose), Math.abs(l - atrPrevClose))
            : h - l;
        atrPrevClose = c;
        if (atrVal === null) {
            atrCount += 1;
            atrSum += tr;
            if (atrCount === supertrendAtrPeriod) atrVal = atrSum / supertrendAtrPeriod;
        } else {
            atrVal = ((atrVal * (supertrendAtrPeriod - 1)) + tr) / supertrendAtrPeriod;
        }

        const width = h - l;
        let vola = null;
        if (wbuf.length < 200) {
            wbuf.push(width);
            wsum += width;
            wweighted += wbuf.length * width;
            if (wbuf.length === 200) vola = wweighted / wdiv;
        } else {
            const oldest = wbuf.shift();
            const oldSum = wsum;
            wbuf.push(width);
            wweighted = wweighted - oldSum + 200.0 * width;
            wsum = oldSum - oldest + width;
            vola = wweighted / wdiv;
        }

        let supertrend = null;
        let direction = null;
        if (Number.isFinite(kalman) && atrVal !== null) {
            let upper = kalman + supertrendFactor * atrVal;
            let lower = kalman - supertrendFactor * atrVal;
            const pl = Number.isNaN(prevLowerBand) ? lower : prevLowerBand;
            const pu = Number.isNaN(prevUpperBand) ? upper : prevUpperBand;
            const pk = Number.isNaN(prevKalman) ? kalman : prevKalman;
            if (!(lower > pl || pk < pl)) lower = pl;
            if (!(upper < pu || pk > pu)) upper = pu;
            if (!prevAtrReady) direction = 1;
            else if (prevSupertrend === pu) direction = kalman > upper ? -1 : 1;
            else direction = kalman < lower ? 1 : -1;
            supertrend = direction === -1 ? lower : upper;
            prevLowerBand = lower;
            prevUpperBand = upper;
            prevSupertrend = supertrend;
            prevKalman = kalman;
            prevAtrReady = true;
        }

        if (vola === null || supertrend === null) continue;

        const upperBand = kalman + vola * dev;
        const lowerBand = kalman - vola * dev;
        if (c > upperBand) trendState = 1.0;
        else if (c < lowerBand) trendState = -1.0;

        const ktrend = direction < 0 ? 1.0 : -1.0;
        const state = trendState * ktrend;
        out.kalman[i] = kalman;
        out.supertrend[i] = supertrend;
        out.upper_band[i] = upperBand;
        out.lower_band[i] = lowerBand;
        out.trend[i] = trendState;
        out.kalman_trend[i] = ktrend;
        out.state[i] = state;
        out.market_trending[i] = prevState !== null && state > 0 && prevState <= 0 ? 1.0 : 0.0;
        out.market_ranging[i] = prevState !== null && state < 0 && prevState >= 0 ? 1.0 : 0.0;
        out.short_term_bullish[i] = prevTrend !== null && trendState > 0 && prevTrend <= 0 ? 1.0 : 0.0;
        out.short_term_bearish[i] = prevTrend !== null && trendState < 0 && prevTrend >= 0 ? 1.0 : 0.0;
        out.long_term_bullish[i] = prevKtrend !== null && ktrend > 0 && prevKtrend <= 0 ? 1.0 : 0.0;
        out.long_term_bearish[i] = prevKtrend !== null && ktrend < 0 && prevKtrend >= 0 ? 1.0 : 0.0;
        prevTrend = trendState;
        prevKtrend = ktrend;
        prevState = state;
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

test('range_filtered_trend_signals_js matches manual reference', () => {
    const high = new Float64Array(testData.high.slice(0, 360));
    const low = new Float64Array(testData.low.slice(0, 360));
    const close = new Float64Array(testData.close.slice(0, 360));
    const result = wasm.range_filtered_trend_signals_js(high, low, close, 0.01, 0.1, 77, 1.2, 0.7, 7);
    const expected = manualRangeFilteredTrendSignals(high, low, close);
    for (const key of Object.keys(expected)) {
        assertArrayClose(result[key], expected[key], 1e-12, `${key} mismatch`);
    }
});

test('range_filtered_trend_signals_batch first row matches single', () => {
    const high = new Float64Array(testData.high.slice(0, 360));
    const low = new Float64Array(testData.low.slice(0, 360));
    const close = new Float64Array(testData.close.slice(0, 360));
    const batch = wasm.range_filtered_trend_signals_batch(high, low, close, {
        kalman_alpha_range: [0.01, 0.02, 0.01],
        kalman_beta_range: [0.1, 0.1, 0.0],
        kalman_period_range: [77, 77, 0],
        dev_range: [1.2, 1.2, 0.0],
        supertrend_factor_range: [0.7, 0.7, 0.0],
        supertrend_atr_period_range: [7, 7, 0],
    });
    const single = wasm.range_filtered_trend_signals_js(high, low, close, 0.01, 0.1, 77, 1.2, 0.7, 7);

    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.kalman_alphas, [0.01, 0.02]);
    for (const key of Object.keys(single)) {
        assertArrayClose(batch[key].slice(0, close.length), single[key], 1e-12, `${key} batch mismatch`);
    }
});

test('range_filtered_trend_signals_js rejects invalid params', () => {
    const high = new Float64Array(testData.high.slice(0, 256));
    const low = new Float64Array(testData.low.slice(0, 256));
    const close = new Float64Array(testData.close.slice(0, 256));
    assert.throws(() => {
        wasm.range_filtered_trend_signals_js(high, low, close, 0.0, 0.1, 77, 1.2, 0.7, 7);
    }, /invalid kalman_alpha/i);
    assert.throws(() => {
        wasm.range_filtered_trend_signals_js(high, low, close, 0.01, 0.1, 77, -1.0, 0.7, 7);
    }, /invalid dev/i);
});
