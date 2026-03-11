import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, loadTestData } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function manualForwardBackwardExponentialOscillator(data, length, smooth) {
    const n = data.length;
    const forwardBackward = new Float64Array(n);
    const backward = new Float64Array(n);
    const histogram = new Float64Array(n);
    forwardBackward.fill(NaN);
    backward.fill(NaN);
    histogram.fill(NaN);

    const alpha = 2.0 / (smooth + 1.0);
    let ema1State = NaN;
    let ema2State = NaN;
    let prevEma2 = NaN;
    const ema1Window = [];
    const diffs = [];

    for (let i = 0; i < n; i++) {
        const value = data[i];
        if (!Number.isFinite(value)) {
            ema1State = NaN;
            ema2State = NaN;
            prevEma2 = NaN;
            ema1Window.length = 0;
            diffs.length = 0;
            continue;
        }

        ema1State = Number.isFinite(ema1State) ? alpha * value + (1.0 - alpha) * ema1State : value;
        ema1Window.push(ema1State);
        if (ema1Window.length > length) {
            ema1Window.shift();
        }

        if (ema1Window.length === length) {
            let ema2 = ema1Window[ema1Window.length - 1];
            let prev = ema2;
            let num = 0.0;
            let den = 0.0;
            for (let j = ema1Window.length - 2; j >= 0; j--) {
                ema2 += alpha * (ema1Window[j] - ema2);
                const dt = prev - ema2;
                num += dt;
                den += Math.abs(dt);
                prev = ema2;
            }
            if (den !== 0.0) {
                forwardBackward[i] = num / den * 50.0 + 50.0;
            }
        }

        ema2State = Number.isFinite(ema2State) ? alpha * ema1State + (1.0 - alpha) * ema2State : ema1State;
        if (Number.isFinite(prevEma2)) {
            diffs.push(ema2State - prevEma2);
            if (diffs.length > length) {
                diffs.shift();
            }
            if (diffs.length === length) {
                let num = 0.0;
                let den = 0.0;
                for (const diff of diffs) {
                    num += diff;
                    den += Math.abs(diff);
                }
                if (den !== 0.0) {
                    backward[i] = num / den * 50.0 + 50.0;
                    if (Number.isFinite(forwardBackward[i])) {
                        histogram[i] = (forwardBackward[i] - backward[i]) / 4.0 + 50.0;
                    }
                }
            }
        }
        prevEma2 = ema2State;
    }

    return {
        forward_backward: forwardBackward,
        backward,
        histogram,
    };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('forward_backward_exponential_oscillator_js matches manual reference', () => {
    const close = new Float64Array(testData.close.slice(0, 180));
    const result = wasm.forward_backward_exponential_oscillator_js(close, 20, 10);
    const expected = manualForwardBackwardExponentialOscillator(close, 20, 10);

    assertArrayClose(result.forward_backward, expected.forward_backward, 1e-12, 'forward_backward mismatch');
    assertArrayClose(result.backward, expected.backward, 1e-12, 'backward mismatch');
    assertArrayClose(result.histogram, expected.histogram, 1e-12, 'histogram mismatch');
});

test('forward_backward_exponential_oscillator_batch first row matches single', () => {
    const close = new Float64Array(testData.close.slice(0, 128));
    const batch = wasm.forward_backward_exponential_oscillator_batch(close, {
        length_range: [20, 22, 2],
        smooth_range: [10, 12, 2],
    });
    const single = wasm.forward_backward_exponential_oscillator_js(close, 20, 10);

    assert.strictEqual(batch.rows, 4);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(batch.lengths, [20, 20, 22, 22]);
    assert.deepStrictEqual(batch.smooths, [10, 12, 10, 12]);

    assertArrayClose(
        batch.forward_backward.slice(0, close.length),
        single.forward_backward,
        1e-12,
        'batch forward_backward mismatch',
    );
    assertArrayClose(
        batch.backward.slice(0, close.length),
        single.backward,
        1e-12,
        'batch backward mismatch',
    );
    assertArrayClose(
        batch.histogram.slice(0, close.length),
        single.histogram,
        1e-12,
        'batch histogram mismatch',
    );
});

test('forward_backward_exponential_oscillator_js rejects invalid params', () => {
    const close = new Float64Array(testData.close.slice(0, 64));
    assert.throws(() => {
        wasm.forward_backward_exponential_oscillator_js(close, 0, 10);
    }, /invalid length/i);
    assert.throws(() => {
        wasm.forward_backward_exponential_oscillator_js(close, 20, 0);
    }, /invalid smooth/i);
});
