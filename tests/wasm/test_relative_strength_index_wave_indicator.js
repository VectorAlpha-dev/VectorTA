import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function manualReference(source, high, low, rsiLength = 14, length1 = 2, length2 = 5, length3 = 9, length4 = 13) {
    const makeRsiState = (period) => ({
        period,
        invP: 1 / period,
        beta: 1 - 1 / period,
        initialized: false,
        prev: NaN,
        deltasSeen: 0,
        sumGain: 0,
        sumLoss: 0,
        avgGain: 0,
        avgLoss: 0,
        ready: false,
    });

    const updateRsi = (state, value) => {
        if (!Number.isFinite(value)) {
            Object.assign(state, makeRsiState(state.period));
            return NaN;
        }
        if (!state.initialized) {
            state.initialized = true;
            state.prev = value;
            return NaN;
        }
        const delta = value - state.prev;
        state.prev = value;
        const gain = Math.max(delta, 0);
        const loss = Math.max(-delta, 0);
        if (!state.ready) {
            state.sumGain += gain;
            state.sumLoss += loss;
            state.deltasSeen += 1;
            if (state.deltasSeen < state.period) {
                return NaN;
            }
            state.avgGain = state.sumGain * state.invP;
            state.avgLoss = state.sumLoss * state.invP;
            state.ready = true;
        } else {
            state.avgGain = state.avgGain * state.beta + state.invP * gain;
            state.avgLoss = state.avgLoss * state.beta + state.invP * loss;
        }
        const denom = state.avgGain + state.avgLoss;
        return denom === 0 ? 50 : 100 * state.avgGain / denom;
    };

    const makeWmaState = (len) => ({
        len,
        denom: (len * (len + 1)) / 2,
        buf: new Float64Array(len),
        pos: 0,
        count: 0,
        sum: 0,
        weightedSum: 0,
    });

    const updateWma = (state, value) => {
        if (!Number.isFinite(value)) {
            state.pos = 0;
            state.count = 0;
            state.sum = 0;
            state.weightedSum = 0;
            return NaN;
        }
        if (state.count < state.len) {
            state.buf[state.count] = value;
            state.count += 1;
            state.sum += value;
            state.weightedSum += state.count * value;
            return state.count === state.len ? state.weightedSum / state.denom : NaN;
        }
        const oldSum = state.sum;
        const old = state.buf[state.pos];
        state.buf[state.pos] = value;
        state.pos = (state.pos + 1) % state.len;
        state.weightedSum = state.weightedSum + state.len * value - oldSum;
        state.sum = oldSum + value - old;
        return state.weightedSum / state.denom;
    };

    const rsiSource = makeRsiState(rsiLength);
    const rsiHigh = makeRsiState(rsiLength);
    const rsiLow = makeRsiState(rsiLength);
    const wma1 = makeWmaState(length1);
    const wma2 = makeWmaState(length2);
    const wma3 = makeWmaState(length3);
    const wma4 = makeWmaState(length4);
    const out1 = new Float64Array(source.length);
    const out2 = new Float64Array(source.length);
    const out3 = new Float64Array(source.length);
    const out4 = new Float64Array(source.length);
    const state = new Float64Array(source.length);
    out1.fill(NaN);
    out2.fill(NaN);
    out3.fill(NaN);
    out4.fill(NaN);
    state.fill(NaN);
    let prevSlo = null;

    for (let i = 0; i < source.length; i++) {
        const s = source[i];
        const h = high[i];
        const l = low[i];
        if (!Number.isFinite(s) || !Number.isFinite(h) || !Number.isFinite(l)) {
            Object.assign(rsiSource, makeRsiState(rsiLength));
            Object.assign(rsiHigh, makeRsiState(rsiLength));
            Object.assign(rsiLow, makeRsiState(rsiLength));
            Object.assign(wma1, makeWmaState(length1));
            Object.assign(wma2, makeWmaState(length2));
            Object.assign(wma3, makeWmaState(length3));
            Object.assign(wma4, makeWmaState(length4));
            prevSlo = null;
            continue;
        }
        const srcRsi = updateRsi(rsiSource, s);
        const highRsi = updateRsi(rsiHigh, h);
        const lowRsi = updateRsi(rsiLow, l);
        if (!Number.isFinite(srcRsi) || !Number.isFinite(highRsi) || !Number.isFinite(lowRsi)) {
            continue;
        }
        const hlcRsi = (highRsi + lowRsi + 2 * srcRsi) * 0.25;
        out1[i] = updateWma(wma1, hlcRsi);
        out2[i] = updateWma(wma2, hlcRsi);
        out3[i] = updateWma(wma3, hlcRsi);
        out4[i] = updateWma(wma4, hlcRsi);
        if (Number.isFinite(out1[i]) && Number.isFinite(out2[i])) {
            const slo = out1[i] - out2[i];
            const prev = prevSlo === null ? 0 : prevSlo;
            if (slo > 0) {
                state[i] = slo > prev ? 2 : 1;
            } else if (slo < 0) {
                state[i] = slo < prev ? -2 : -1;
            } else {
                state[i] = 0;
            }
            prevSlo = slo;
        }
    }

    return { rsi_ma1: out1, rsi_ma2: out2, rsi_ma3: out3, rsi_ma4: out4, state };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('relative_strength_index_wave_indicator_js matches manual reference', () => {
    const high = new Float64Array(testData.high.slice(0, 180));
    const low = new Float64Array(testData.low.slice(0, 180));
    const close = new Float64Array(testData.close.slice(0, 180));
    const source = new Float64Array(high.length);
    for (let i = 0; i < source.length; i++) {
        source[i] = (high[i] + low[i] + 2 * close[i]) * 0.25;
    }
    const result = wasm.relative_strength_index_wave_indicator_js(source, high, low, 14, 2, 5, 9, 13);
    const expected = manualReference(source, high, low, 14, 2, 5, 9, 13);
    assertArrayClose(result.rsi_ma1, expected.rsi_ma1, 5e-12, 'rsi_ma1 mismatch');
    assertArrayClose(result.rsi_ma2, expected.rsi_ma2, 5e-12, 'rsi_ma2 mismatch');
    assertArrayClose(result.rsi_ma3, expected.rsi_ma3, 5e-12, 'rsi_ma3 mismatch');
    assertArrayClose(result.rsi_ma4, expected.rsi_ma4, 5e-12, 'rsi_ma4 mismatch');
    assertArrayClose(result.state, expected.state, 5e-12, 'state mismatch');
});

test('relative_strength_index_wave_indicator_batch_js first row matches single', () => {
    const high = new Float64Array(testData.high.slice(0, 128));
    const low = new Float64Array(testData.low.slice(0, 128));
    const close = new Float64Array(testData.close.slice(0, 128));
    const source = new Float64Array(high.length);
    for (let i = 0; i < source.length; i++) {
        source[i] = (high[i] + low[i] + 2 * close[i]) * 0.25;
    }
    const batch = wasm.relative_strength_index_wave_indicator_batch_js(source, high, low, {
        rsi_length_range: [14, 15, 1],
        length1_range: [2, 2, 0],
        length2_range: [5, 5, 0],
        length3_range: [9, 9, 0],
        length4_range: [13, 13, 0],
    });
    const single = wasm.relative_strength_index_wave_indicator_js(source, high, low, 14, 2, 5, 9, 13);
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, source.length);
    assert.deepStrictEqual(Array.from(batch.rsi_lengths), [14, 15]);
    assertArrayClose(batch.rsi_ma1.slice(0, source.length), single.rsi_ma1, 5e-12, 'batch rsi_ma1 mismatch');
    assertArrayClose(batch.state.slice(0, source.length), single.state, 5e-12, 'batch state mismatch');
});

test('relative_strength_index_wave_indicator_into pointer path matches safe API', () => {
    const n = 144;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const source = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        source[i] = (high[i] + low[i] + 2 * close[i]) * 0.25;
    }
    const safe = wasm.relative_strength_index_wave_indicator_js(source, high, low, 14, 2, 5, 9, 13);
    const sourcePtr = wasm.allocate_f64_array(n);
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const out1Ptr = wasm.relative_strength_index_wave_indicator_alloc(n);
    const out2Ptr = wasm.relative_strength_index_wave_indicator_alloc(n);
    const out3Ptr = wasm.relative_strength_index_wave_indicator_alloc(n);
    const out4Ptr = wasm.relative_strength_index_wave_indicator_alloc(n);
    const outStatePtr = wasm.relative_strength_index_wave_indicator_alloc(n);
    try {
        wasm.write_f64_array(sourcePtr, source);
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        wasm.relative_strength_index_wave_indicator_into(
            sourcePtr, highPtr, lowPtr, out1Ptr, out2Ptr, out3Ptr, out4Ptr, outStatePtr, n, 14, 2, 5, 9, 13,
        );
        assertArrayClose(wasm.read_f64_array(out1Ptr, n), safe.rsi_ma1, 5e-12, 'pointer rsi_ma1 mismatch');
        assertArrayClose(wasm.read_f64_array(out2Ptr, n), safe.rsi_ma2, 5e-12, 'pointer rsi_ma2 mismatch');
        assertArrayClose(wasm.read_f64_array(out3Ptr, n), safe.rsi_ma3, 5e-12, 'pointer rsi_ma3 mismatch');
        assertArrayClose(wasm.read_f64_array(out4Ptr, n), safe.rsi_ma4, 5e-12, 'pointer rsi_ma4 mismatch');
        assertArrayClose(wasm.read_f64_array(outStatePtr, n), safe.state, 5e-12, 'pointer state mismatch');
    } finally {
        wasm.deallocate_f64_array(sourcePtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.relative_strength_index_wave_indicator_free(out1Ptr, n);
        wasm.relative_strength_index_wave_indicator_free(out2Ptr, n);
        wasm.relative_strength_index_wave_indicator_free(out3Ptr, n);
        wasm.relative_strength_index_wave_indicator_free(out4Ptr, n);
        wasm.relative_strength_index_wave_indicator_free(outStatePtr, n);
    }
});

test('relative_strength_index_wave_indicator_batch_into returns row count', () => {
    const n = 128;
    const rows = 2;
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));
    const source = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        source[i] = (high[i] + low[i] + 2 * close[i]) * 0.25;
    }
    const sourcePtr = wasm.allocate_f64_array(n);
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const out1Ptr = wasm.relative_strength_index_wave_indicator_alloc(rows * n);
    const out2Ptr = wasm.relative_strength_index_wave_indicator_alloc(rows * n);
    const out3Ptr = wasm.relative_strength_index_wave_indicator_alloc(rows * n);
    const out4Ptr = wasm.relative_strength_index_wave_indicator_alloc(rows * n);
    const outStatePtr = wasm.relative_strength_index_wave_indicator_alloc(rows * n);
    try {
        wasm.write_f64_array(sourcePtr, source);
        wasm.write_f64_array(highPtr, high);
        wasm.write_f64_array(lowPtr, low);
        const returnedRows = wasm.relative_strength_index_wave_indicator_batch_into(
            sourcePtr, highPtr, lowPtr, out1Ptr, out2Ptr, out3Ptr, out4Ptr, outStatePtr, n,
            14, 15, 1, 2, 2, 0, 5, 5, 0, 9, 9, 0, 13, 13, 0,
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.deallocate_f64_array(sourcePtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.relative_strength_index_wave_indicator_free(out1Ptr, rows * n);
        wasm.relative_strength_index_wave_indicator_free(out2Ptr, rows * n);
        wasm.relative_strength_index_wave_indicator_free(out3Ptr, rows * n);
        wasm.relative_strength_index_wave_indicator_free(out4Ptr, rows * n);
        wasm.relative_strength_index_wave_indicator_free(outStatePtr, rows * n);
    }
});

test('relative_strength_index_wave_indicator_js rejects invalid rsi_length', () => {
    const high = new Float64Array(testData.high.slice(0, 64));
    const low = new Float64Array(testData.low.slice(0, 64));
    const close = new Float64Array(testData.close.slice(0, 64));
    const source = new Float64Array(high.length);
    for (let i = 0; i < source.length; i++) {
        source[i] = (high[i] + low[i] + 2 * close[i]) * 0.25;
    }
    assert.throws(
        () => wasm.relative_strength_index_wave_indicator_js(source, high, low, 0, 2, 5, 9, 13),
        /Invalid rsi_length/,
    );
});
