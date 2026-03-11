import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, loadTestData } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function manualQqeWeightedOscillator(data, length, factor, smooth, weight) {
    const n = data.length;
    const rsi = new Float64Array(n);
    const trailingStop = new Float64Array(n);
    rsi.fill(NaN);
    trailingStop.fill(NaN);

    let first = -1;
    let validCount = 0;
    for (let i = 0; i < n; i++) {
        if (Number.isFinite(data[i])) {
            if (first === -1) first = i;
            validCount += 1;
        }
    }
    if (first === -1 || validCount < length + 1) {
        return { rsi, trailing_stop: trailingStop };
    }

    const alpha = 2.0 / (smooth + 1.0);
    let prevSrc = data[first];
    let prevRsi = NaN;
    let prevTs = NaN;
    let numCount = 0;
    let denCount = 0;
    let diffCount = 0;
    let numSum = 0.0;
    let denSum = 0.0;
    let diffSum = 0.0;
    let numState = NaN;
    let denState = NaN;
    let diffState = NaN;
    let emaState = NaN;

    for (let i = first + 1; i < n; i++) {
        const current = data[i];
        if (!Number.isFinite(current)) {
            prevSrc = NaN;
            continue;
        }
        if (!Number.isFinite(prevSrc)) {
            prevSrc = current;
            continue;
        }

        const delta = current - prevSrc;
        const w = Number.isFinite(prevRsi) && Number.isFinite(prevTs) && delta * (prevRsi - prevTs) > 0.0 ? weight : 1.0;
        const weightedDelta = delta * w;
        const absWeightedDelta = Math.abs(weightedDelta);

        numCount += 1;
        if (numCount < length) {
            numSum += weightedDelta;
        } else if (numCount === length) {
            numSum += weightedDelta;
            numState = numSum / length;
        } else {
            numState = ((numState * (length - 1.0)) + weightedDelta) / length;
        }

        denCount += 1;
        if (denCount < length) {
            denSum += absWeightedDelta;
        } else if (denCount === length) {
            denSum += absWeightedDelta;
            denState = denSum / length;
        } else {
            denState = ((denState * (length - 1.0)) + absWeightedDelta) / length;
        }

        if (Number.isFinite(numState) && Number.isFinite(denState) && denState !== 0.0) {
            const ratio = numState / denState;
            emaState = Number.isFinite(emaState) ? alpha * ratio + (1.0 - alpha) * emaState : ratio;
            const currRsi = 50.0 * emaState + 50.0;
            rsi[i] = currRsi;

            let diff = NaN;
            if (Number.isFinite(prevRsi)) {
                const diffInput = Math.abs(currRsi - prevRsi);
                diffCount += 1;
                if (diffCount < length) {
                    diffSum += diffInput;
                } else if (diffCount === length) {
                    diffSum += diffInput;
                    diffState = diffSum / length;
                    diff = diffState;
                } else {
                    diffState = ((diffState * (length - 1.0)) + diffInput) / length;
                    diff = diffState;
                }
            }

            let currTs = currRsi;
            if (Number.isFinite(diff)) {
                const crossover = Number.isFinite(prevTs) && currRsi > prevTs && prevRsi <= prevTs;
                const crossunder = Number.isFinite(prevTs) && currRsi < prevTs && prevRsi >= prevTs;
                if (crossover) {
                    currTs = currRsi - diff * factor;
                } else if (crossunder) {
                    currTs = currRsi + diff * factor;
                } else if (Number.isFinite(prevTs)) {
                    currTs = currRsi > prevTs
                        ? Math.max(currRsi - diff * factor, prevTs)
                        : Math.min(currRsi + diff * factor, prevTs);
                }
            }

            trailingStop[i] = currTs;
            prevRsi = currRsi;
            prevTs = currTs;
        }

        prevSrc = current;
    }

    return { rsi, trailing_stop: trailingStop };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('qqe_weighted_oscillator_js matches manual reference', () => {
    const close = new Float64Array(testData.close.slice(0, 160));
    const result = wasm.qqe_weighted_oscillator_js(close, 14, 4.236, 5, 2.4);
    const expected = manualQqeWeightedOscillator(close, 14, 4.236, 5, 2.4);

    assertArrayClose(result.rsi, expected.rsi, 1e-12, 'rsi mismatch');
    assertArrayClose(result.trailing_stop, expected.trailing_stop, 1e-12, 'trailing stop mismatch');
});

test('qqe_weighted_oscillator_batch first row matches single', () => {
    const close = new Float64Array(testData.close.slice(0, 128));
    const batch = wasm.qqe_weighted_oscillator_batch(close, {
        length_range: [14, 16, 2],
        factor_range: [4.236, 4.736, 0.5],
        smooth_range: [5, 5, 0],
        weight_range: [2.0, 2.0, 0.0],
    });
    const single = wasm.qqe_weighted_oscillator_js(close, 14, 4.236, 5, 2.0);

    assert.strictEqual(batch.rows, 4);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.lengths, [14, 14, 16, 16]);
    assert.deepStrictEqual(batch.factors, [4.236, 4.736, 4.236, 4.736]);
    assert.deepStrictEqual(batch.smooths, [5, 5, 5, 5]);
    assert.deepStrictEqual(batch.weights, [2.0, 2.0, 2.0, 2.0]);

    assertArrayClose(batch.rsi.slice(0, close.length), single.rsi, 1e-12, 'batch rsi mismatch');
    assertArrayClose(
        batch.trailing_stop.slice(0, close.length),
        single.trailing_stop,
        1e-12,
        'batch trailing stop mismatch',
    );
});

test('qqe_weighted_oscillator_js rejects invalid params', () => {
    const close = new Float64Array(testData.close.slice(0, 64));
    assert.throws(() => {
        wasm.qqe_weighted_oscillator_js(close, 0, 4.236, 5, 2.0);
    }, /invalid length/i);
    assert.throws(() => {
        wasm.qqe_weighted_oscillator_js(close, 14, -1.0, 5, 2.0);
    }, /invalid factor/i);
    assert.throws(() => {
        wasm.qqe_weighted_oscillator_js(close, 14, 4.236, 0, 2.0);
    }, /invalid smooth/i);
});
