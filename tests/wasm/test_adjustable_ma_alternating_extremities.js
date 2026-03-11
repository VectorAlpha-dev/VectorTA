import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, loadTestData } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function buildWeights(length, alpha, beta) {
    const weights = new Float64Array(length);
    let sum = 0.0;
    for (let i = 0; i < length; i++) {
        const x = i / (length - 1);
        const weight = Math.sin((2.0 * Math.PI) * (x ** alpha)) * (1.0 - (x ** beta));
        weights[i] = weight;
        sum += weight;
    }
    for (let i = 0; i < length; i++) {
        weights[i] /= sum;
    }
    return weights;
}

function weightedFilter(source, first, length, weights) {
    const out = new Float64Array(source.length);
    out.fill(NaN);
    const start = first + length - 1;
    for (let i = start; i < source.length; i++) {
        let acc = 0.0;
        for (let j = 0; j < length; j++) {
            acc += source[i - j] * weights[j];
        }
        out[i] = acc;
    }
    return out;
}

function pineCross(prevA, prevB, currA, currB) {
    if (
        !Number.isFinite(prevA) ||
        !Number.isFinite(prevB) ||
        !Number.isFinite(currA) ||
        !Number.isFinite(currB)
    ) {
        return false;
    }
    return (currA > currB && prevA <= prevB) || (currA < currB && prevA >= prevB);
}

function manualAmaAe(high, low, close, length, mult, alpha, beta) {
    const n = close.length;
    const out = {
        ma: new Float64Array(n),
        upper: new Float64Array(n),
        lower: new Float64Array(n),
        extremity: new Float64Array(n),
        state: new Float64Array(n),
        changed: new Float64Array(n),
        smoothed_open: new Float64Array(n),
        smoothed_high: new Float64Array(n),
        smoothed_low: new Float64Array(n),
        smoothed_close: new Float64Array(n),
    };
    for (const values of Object.values(out)) {
        values.fill(NaN);
    }

    let first = -1;
    for (let i = 0; i < n; i++) {
        if (Number.isFinite(high[i]) && Number.isFinite(low[i]) && Number.isFinite(close[i])) {
            first = i;
            break;
        }
    }
    if (first === -1) {
        return out;
    }

    const weights = buildWeights(length, alpha, beta);
    const maWarm = first + length - 1;
    const bandWarm = first + (2 * length) - 2;

    out.ma = weightedFilter(close, first, length, weights);
    out.smoothed_high = weightedFilter(high, first, length, weights);
    out.smoothed_low = weightedFilter(low, first, length, weights);
    out.smoothed_close = out.ma.slice();

    for (let i = maWarm + 2; i < n; i++) {
        out.smoothed_open[i] = 0.5 * (out.smoothed_close[i - 1] + out.smoothed_close[i - 2]);
    }

    let rolling = 0.0;
    for (let i = maWarm; i <= bandWarm; i++) {
        rolling += Math.abs(close[i] - out.ma[i]);
    }
    let dev = (rolling / length) * mult;
    out.upper[bandWarm] = out.ma[bandWarm] + dev;
    out.lower[bandWarm] = out.ma[bandWarm] - dev;

    for (let i = bandWarm + 1; i < n; i++) {
        rolling += Math.abs(close[i] - out.ma[i]);
        rolling -= Math.abs(close[i - length] - out.ma[i - length]);
        dev = (rolling / length) * mult;
        out.upper[i] = out.ma[i] + dev;
        out.lower[i] = out.ma[i] - dev;
    }

    out.state[bandWarm] = 0.0;
    out.changed[bandWarm] = 0.0;
    out.extremity[bandWarm] = out.lower[bandWarm];

    for (let i = bandWarm + 1; i < n; i++) {
        const prevState = out.state[i - 1];
        const crossHigh = pineCross(high[i - 1], out.upper[i - 1], high[i], out.upper[i]);
        const crossLow = pineCross(low[i - 1], out.lower[i - 1], low[i], out.lower[i]);
        const nextState = crossHigh ? 1.0 : (crossLow ? 0.0 : prevState);
        out.state[i] = nextState;
        out.changed[i] = Math.abs(nextState - prevState) > 0.0 ? 1.0 : 0.0;
        out.extremity[i] = nextState >= 0.5 ? out.upper[i] : out.lower[i];
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

test('adjustable_ma_alternating_extremities_js matches manual reference', () => {
    const n = 160;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const params = { length: 12, mult: 2.4, alpha: 1.3, beta: 0.7 };

    const result = wasm.adjustable_ma_alternating_extremities_js(
        high,
        low,
        close,
        params.length,
        params.mult,
        params.alpha,
        params.beta,
    );
    const expected = manualAmaAe(
        high,
        low,
        close,
        params.length,
        params.mult,
        params.alpha,
        params.beta,
    );

    for (const key of Object.keys(expected)) {
        assertArrayClose(result[key], expected[key], 1e-11, `${key} mismatch`);
    }
});

test('adjustable_ma_alternating_extremities_batch first row matches single', () => {
    const n = 128;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    const batch = wasm.adjustable_ma_alternating_extremities_batch(high, low, close, {
        length_range: [10, 12, 2],
        mult_range: [2.0, 2.5, 0.5],
        alpha_range: [1.0, 1.0, 0.0],
        beta_range: [0.5, 0.5, 0.0],
    });
    const single = wasm.adjustable_ma_alternating_extremities_js(high, low, close, 10, 2.0, 1.0, 0.5);

    assert.strictEqual(batch.rows, 4);
    assert.strictEqual(batch.cols, n);
    assert.deepStrictEqual(batch.lengths, [10, 10, 12, 12]);
    assert.deepStrictEqual(batch.mults, [2.0, 2.5, 2.0, 2.5]);
    assert.deepStrictEqual(batch.alphas, [1.0, 1.0, 1.0, 1.0]);
    assert.deepStrictEqual(batch.betas, [0.5, 0.5, 0.5, 0.5]);

    for (const key of Object.keys(single)) {
        assertArrayClose(batch[key].slice(0, n), single[key], 1e-12, `${key} batch mismatch`);
    }
});

test('adjustable_ma_alternating_extremities_js rejects invalid params', () => {
    const n = 64;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    assert.throws(() => {
        wasm.adjustable_ma_alternating_extremities_js(high, low, close, 1, 2.0, 1.0, 0.5);
    }, /invalid length/i);
    assert.throws(() => {
        wasm.adjustable_ma_alternating_extremities_js(high, low, close, 10, 0.5, 1.0, 0.5);
    }, /invalid mult/i);
    assert.throws(() => {
        wasm.adjustable_ma_alternating_extremities_js(high, low, close, 10, 2.0, -1.0, 0.5);
    }, /invalid alpha/i);
    assert.throws(() => {
        wasm.adjustable_ma_alternating_extremities_js(high, low, close, 10, 2.0, 1.0, -0.1);
    }, /invalid beta/i);
});
