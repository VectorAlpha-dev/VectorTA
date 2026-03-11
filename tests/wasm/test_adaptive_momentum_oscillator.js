import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let data;

function getMemory(mod) {
    if (mod.memory && mod.memory.buffer) {
        return mod.memory;
    }
    if (typeof mod.__wbindgen_memory === 'function') {
        const memory = mod.__wbindgen_memory();
        if (memory && memory.buffer) {
            return memory;
        }
    }
    if (mod.__wasm && mod.__wasm.memory && mod.__wasm.memory.buffer) {
        return mod.__wasm.memory;
    }
    return null;
}

function naiveAdaptiveMomentumOscillator(values, length, smoothingLength) {
    const amo = new Array(values.length).fill(NaN);
    const ama = new Array(values.length).fill(NaN);
    const raw = new Array(values.length).fill(NaN);

    for (let idx = 0; idx < values.length; idx++) {
        const value = values[idx];
        if (!Number.isFinite(value) || idx < length) {
            continue;
        }
        let bestAbs = -1.0;
        let bestDelta = NaN;
        let valid = true;
        for (let lag = 1; lag <= length; lag++) {
            const past = values[idx - lag];
            if (!Number.isFinite(past)) {
                valid = false;
                break;
            }
            const delta = value - past;
            const absDelta = Math.abs(delta);
            if (absDelta >= bestAbs) {
                bestAbs = absDelta;
                bestDelta = delta;
            }
        }
        if (valid) {
            raw[idx] = bestDelta;
        }
    }

    const pf = smoothingLength;
    const xSum = pf * (pf + 1) * 0.5;
    const x2Sum = pf * (pf + 1) * (2 * pf + 1) / 6;
    const denom = pf * x2Sum - xSum * xSum;
    for (let idx = smoothingLength - 1; idx < values.length; idx++) {
        const window = raw.slice(idx + 1 - smoothingLength, idx + 1);
        if (!window.every(Number.isFinite)) {
            continue;
        }
        let ySum = 0.0;
        let xySum = 0.0;
        for (let i = 0; i < window.length; i++) {
            ySum += window[i];
            xySum += (i + 1) * window[i];
        }
        const b = (pf * xySum - xSum * ySum) / denom;
        const a = (ySum - b * xSum) / pf;
        amo[idx] = a + b * pf;
    }

    let amaState = 0.0;
    let prev = NaN;
    let havePrev = false;
    const changeRing = new Array(length).fill(0.0);
    let head = 0;
    let count = 0;
    let changeSum = 0.0;
    for (let idx = 0; idx < values.length; idx++) {
        const current = amo[idx];
        const change = havePrev && Number.isFinite(current) && Number.isFinite(prev)
            ? Math.abs(current - prev)
            : 0.0;

        if (count < length) {
            changeRing[head] = change;
            changeSum += change;
            count += 1;
        } else {
            const old = changeRing[head];
            changeRing[head] = change;
            changeSum += change - old;
        }
        head = (head + 1) % length;

        if (Number.isFinite(current)) {
            if (changeSum > 0.0) {
                const efficiencyRatio = Math.abs(current) / changeSum;
                const delta = efficiencyRatio * (current - amaState);
                if (Number.isFinite(delta)) {
                    amaState += delta;
                }
            }
            ama[idx] = amaState;
        }

        prev = current;
        havePrev = true;
    }

    return { amo, ama };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);

    const testData = loadTestData();
    data = testData.close;
});

test('Adaptive Momentum Oscillator basic output shape', () => {
    const result = wasm.adaptive_momentum_oscillator_js(data, 14, 9);

    assert.strictEqual(result.amo.length, data.length);
    assert.strictEqual(result.ama.length, data.length);
    assert(result.amo.slice(32).some((value) => !isNaN(value)));
    assert(result.ama.slice(32).some((value) => !isNaN(value)));
});

test('Adaptive Momentum Oscillator matches naive reference', () => {
    const sample = [10.0, 10.4, 10.2, 10.9, 11.3, 11.0, 11.8, 11.5, 12.0, 11.7, 12.4, 12.1];
    const actual = wasm.adaptive_momentum_oscillator_js(sample, 3, 2);
    const expected = naiveAdaptiveMomentumOscillator(sample, 3, 2);
    assertArrayClose(actual.amo, expected.amo, 1e-12, 1e-12, 'AMO naive parity mismatch');
    assertArrayClose(actual.ama, expected.ama, 1e-9, 1e-9, 'AMA naive parity mismatch');
});

test('Adaptive Momentum Oscillator batch matches single', () => {
    const single = wasm.adaptive_momentum_oscillator_js(data, 14, 9);
    const batch = wasm.adaptive_momentum_oscillator_batch(data, {
        length_range: [14, 14, 0],
        smoothing_length_range: [9, 9, 0],
    });

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, data.length);
    assertArrayClose(batch.amo, single.amo, 1e-12, 1e-12, 'AMO batch mismatch');
    assertArrayClose(batch.ama, single.ama, 1e-12, 1e-12, 'AMA batch mismatch');
});

test('Adaptive Momentum Oscillator zero-copy and stream paths work', async () => {
    const len = data.length;
    const inPtr = wasm.adaptive_momentum_oscillator_alloc(len);
    const amoPtr = wasm.adaptive_momentum_oscillator_alloc(len);
    const amaPtr = wasm.adaptive_momentum_oscillator_alloc(len);

    try {
        const memory = getMemory(wasm);
        assert(memory, 'WASM memory accessor unavailable');
        new Float64Array(memory.buffer, inPtr, len).set(data);

        wasm.adaptive_momentum_oscillator_into(inPtr, amoPtr, amaPtr, len, 14, 9);
        const expected = wasm.adaptive_momentum_oscillator_js(data, 14, 9);
        assertArrayClose(Array.from(new Float64Array(memory.buffer, amoPtr, len)), expected.amo, 1e-12, 1e-12, 'AMO zero-copy mismatch');
        assertArrayClose(Array.from(new Float64Array(memory.buffer, amaPtr, len)), expected.ama, 1e-12, 1e-12, 'AMA zero-copy mismatch');

        const rows = wasm.adaptive_momentum_oscillator_batch_into(
            inPtr,
            amoPtr,
            amaPtr,
            len,
            14,
            14,
            0,
            9,
            9,
            0
        );
        assert.strictEqual(rows, 1);
        assertArrayClose(Array.from(new Float64Array(memory.buffer, amoPtr, rows * len)), expected.amo, 1e-12, 1e-12, 'AMO batch-into mismatch');
        assertArrayClose(Array.from(new Float64Array(memory.buffer, amaPtr, rows * len)), expected.ama, 1e-12, 1e-12, 'AMA batch-into mismatch');

        const stream = new wasm.AdaptiveMomentumOscillatorStreamWasm(14, 9);
        const amo = new Array(len).fill(NaN);
        const ama = new Array(len).fill(NaN);
        for (let i = 0; i < len; i++) {
            const result = await stream.update(data[i]);
            if (result !== null) {
                amo[i] = result.amo;
                ama[i] = result.ama;
            }
        }
        assertArrayClose(amo, expected.amo, 1e-12, 1e-12, 'AMO stream mismatch');
        assertArrayClose(ama, expected.ama, 1e-12, 1e-12, 'AMA stream mismatch');
        stream.reset();
        assert.strictEqual(await stream.update(NaN), null);
    } finally {
        wasm.adaptive_momentum_oscillator_free(inPtr, len);
        wasm.adaptive_momentum_oscillator_free(amoPtr, len);
        wasm.adaptive_momentum_oscillator_free(amaPtr, len);
    }
});

test('Adaptive Momentum Oscillator errors', () => {
    assert.throws(() => wasm.adaptive_momentum_oscillator_js([], 14, 9), /empty|Empty/i);
    assert.throws(() => wasm.adaptive_momentum_oscillator_js(new Float64Array(16).fill(NaN), 3, 2), /all values are NaN/i);
    assert.throws(() => wasm.adaptive_momentum_oscillator_js(new Float64Array([1, 2, 3]), 0, 2), /invalid length|Invalid length/i);
    assert.throws(() => wasm.adaptive_momentum_oscillator_js(new Float64Array([1, 2, 3]), 2, 0), /invalid smoothing_length|Invalid smoothing_length/i);
    assert.throws(
        () => wasm.adaptive_momentum_oscillator_js(new Float64Array([1, 2, 3]), 2, 2),
        /not enough valid data|Not enough valid data/i
    );
});
