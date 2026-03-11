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

function naiveVelocity(values, length, smoothLength) {
    const out = new Array(values.length).fill(NaN);
    const firstValid = values.findIndex((value) => Number.isFinite(value));
    if (firstValid === -1) {
        return out;
    }

    const raw = new Array(values.length).fill(NaN);
    const denom = smoothLength * (smoothLength + 1) / 2;

    for (let idx = firstValid; idx < values.length; idx++) {
        const value = values[idx];
        if (!Number.isFinite(value)) {
            continue;
        }
        let acc = 0.0;
        for (let lag = 1; lag <= length; lag++) {
            const hist = idx >= lag && Number.isFinite(values[idx - lag]) ? values[idx - lag] : 0.0;
            acc += (value - hist) / lag;
        }
        raw[idx] = acc / length;
    }

    for (let idx = firstValid + smoothLength - 1; idx < values.length; idx++) {
        let weighted = 0.0;
        let valid = true;
        for (let offset = 0; offset < smoothLength; offset++) {
            const rawValue = raw[idx - smoothLength + 1 + offset];
            if (!Number.isFinite(rawValue)) {
                valid = false;
                break;
            }
            weighted += (offset + 1) * rawValue;
        }
        if (valid) {
            out[idx] = weighted / denom;
        }
    }

    return out;
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);

    const testData = loadTestData();
    data = Float64Array.from(
        testData.high.map((high, idx) => (high + testData.low[idx] + 2.0 * testData.close[idx]) * 0.25)
    );
});

test('Velocity basic output shape', () => {
    const result = wasm.velocity_js(data, 21, 5);

    assert.strictEqual(result.length, data.length);
    for (let i = 0; i < 4; i++) {
        assert(isNaN(result[i]));
    }
    let hasFinite = false;
    for (let i = 32; i < result.length; i++) {
        hasFinite ||= !isNaN(result[i]);
    }
    assert(hasFinite);
});

test('Velocity matches naive reference', () => {
    const sample = [NaN, NaN, 10.0, 11.0, 12.0, 13.0, 12.0, 14.0];
    const result = wasm.velocity_js(sample, 3, 2);
    const expected = naiveVelocity(sample, 3, 2);
    assertArrayClose(result, expected, 1e-12, 1e-12, 'Velocity naive parity mismatch');
});

test('Velocity batch matches single', () => {
    const single = wasm.velocity_js(data, 21, 5);
    const batch = wasm.velocity_batch(data, {
        length_range: [21, 21, 0],
        smooth_length_range: [5, 5, 0],
    });

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, data.length);
    assert.strictEqual(batch.combos.length, 1);
    assertArrayClose(batch.values, single, 1e-12, 1e-12, 'Velocity batch mismatch');
});

test('Velocity zero-copy path works', () => {
    const len = data.length;
    const inPtr = wasm.velocity_alloc(len);
    const outPtr = wasm.velocity_alloc(len);

    try {
        const memory = getMemory(wasm);
        assert(memory, 'WASM memory accessor unavailable');

        new Float64Array(memory.buffer, inPtr, len).set(data);
        wasm.velocity_into(inPtr, outPtr, len, 21, 5);

        const actual = Array.from(new Float64Array(memory.buffer, outPtr, len));
        const expected = wasm.velocity_js(data, 21, 5);
        assertArrayClose(actual, expected, 1e-12, 1e-12, 'Velocity zero-copy mismatch');
    } finally {
        wasm.velocity_free(inPtr, len);
        wasm.velocity_free(outPtr, len);
    }
});

test('Velocity stream matches single and reset works', async () => {
    const single = wasm.velocity_js(data, 21, 5);
    const stream = new wasm.VelocityStreamWasm(21, 5);
    const values = new Array(data.length).fill(NaN);

    for (let i = 0; i < data.length; i++) {
        const result = await stream.update(data[i]);
        if (result !== null) {
            values[i] = result;
        }
    }

    assertArrayClose(values, single, 1e-12, 1e-12, 'Velocity stream mismatch');
    stream.reset();
    assert.strictEqual(await stream.update(NaN), null);
});

test('Velocity errors', () => {
    assert.throws(() => wasm.velocity_js([], 21, 5), /empty|Empty/i);
    assert.throws(() => wasm.velocity_js(new Float64Array(16).fill(NaN), 21, 5), /all values are NaN/i);
    assert.throws(() => wasm.velocity_js(new Float64Array([1, 2, 3]), 1, 5), /invalid length|Invalid length/i);
    assert.throws(() => wasm.velocity_js(new Float64Array([1, 2, 3]), 2, 0), /invalid smoothing length|Invalid smoothing length/i);
    assert.throws(
        () => wasm.velocity_js(new Float64Array([1, 2, 3]), 2, 5),
        /not enough valid data|Not enough valid data/i
    );
});
