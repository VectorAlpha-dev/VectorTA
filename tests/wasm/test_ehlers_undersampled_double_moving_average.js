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

function naiveEudma(values, fastLength, slowLength, sampleLength) {
    const fast = new Array(values.length).fill(NaN);
    const slow = new Array(values.length).fill(NaN);
    const firstValid = values.findIndex((value) => Number.isFinite(value));
    if (firstValid === -1) {
        return { fast, slow };
    }

    const sampled = new Array(values.length).fill(NaN);
    for (let idx = 0; idx < values.length; idx++) {
        if (idx % sampleLength === 0) {
            sampled[idx] = values[idx];
        } else if (idx > 0 && Number.isFinite(sampled[idx - 1])) {
            sampled[idx] = sampled[idx - 1];
        } else {
            sampled[idx] = 0.0;
        }
    }

    const buildWeights = (length) => {
        const weights = new Array(length);
        let norm = 0.0;
        for (let i = 1; i <= length; i++) {
            const weight = 1.0 - Math.cos(2.0 * Math.PI * i / (length + 1.0));
            weights[i - 1] = weight;
            norm += weight;
        }
        return { weights, norm };
    };

    const fastWeights = buildWeights(fastLength);
    const slowWeights = buildWeights(slowLength);

    for (let idx = firstValid; idx < values.length; idx++) {
        let fastAcc = 0.0;
        for (let offset = 0; offset < fastLength; offset++) {
            if (idx >= offset) {
                const value = sampled[idx - offset];
                fastAcc += fastWeights.weights[offset] * (Number.isFinite(value) ? value : 0.0);
            }
        }
        fast[idx] = fastAcc / fastWeights.norm;

        let slowAcc = 0.0;
        for (let offset = 0; offset < slowLength; offset++) {
            if (idx >= offset) {
                const value = sampled[idx - offset];
                slowAcc += slowWeights.weights[offset] * (Number.isFinite(value) ? value : 0.0);
            }
        }
        slow[idx] = slowAcc / slowWeights.norm;
    }

    return { fast, slow };
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

test('EUDMA basic output shape', () => {
    const result = wasm.ehlers_undersampled_double_moving_average(data, 6, 12, 5);
    assert.strictEqual(result.fast.length, data.length);
    assert.strictEqual(result.slow.length, data.length);
    assert(!isNaN(result.fast[0]));
    assert(!isNaN(result.slow[0]));
    let hasFast = false;
    let hasSlow = false;
    for (let i = 16; i < data.length; i++) {
        hasFast ||= !isNaN(result.fast[i]);
        hasSlow ||= !isNaN(result.slow[i]);
    }
    assert(hasFast);
    assert(hasSlow);
});

test('EUDMA matches naive reference', () => {
    const sample = [NaN, 10.0, 11.0, 12.0, 13.0, 12.5, 14.0, 15.0];
    const result = wasm.ehlers_undersampled_double_moving_average(sample, 3, 4, 2);
    const expected = naiveEudma(sample, 3, 4, 2);
    assertArrayClose(result.fast, expected.fast, 1e-12, 1e-12, 'EUDMA fast mismatch');
    assertArrayClose(result.slow, expected.slow, 1e-12, 1e-12, 'EUDMA slow mismatch');
});

test('EUDMA batch matches single', () => {
    const single = wasm.ehlers_undersampled_double_moving_average(data, 6, 12, 5);
    const batch = wasm.ehlers_undersampled_double_moving_average_batch(data, {
        fast_length_range: [6, 6, 0],
        slow_length_range: [12, 12, 0],
        sample_length_range: [5, 5, 0],
    });

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, data.length);
    assert.strictEqual(batch.combos.length, 1);
    assertArrayClose(batch.fast_values, single.fast, 1e-12, 1e-12, 'EUDMA batch fast mismatch');
    assertArrayClose(batch.slow_values, single.slow, 1e-12, 1e-12, 'EUDMA batch slow mismatch');
});

test('EUDMA zero-copy path works', () => {
    const len = data.length;
    const inPtr = wasm.ehlers_undersampled_double_moving_average_alloc(len);
    const fastPtr = wasm.ehlers_undersampled_double_moving_average_alloc(len);
    const slowPtr = wasm.ehlers_undersampled_double_moving_average_alloc(len);

    try {
        const memory = getMemory(wasm);
        assert(memory, 'WASM memory accessor unavailable');

        new Float64Array(memory.buffer, inPtr, len).set(data);
        wasm.ehlers_undersampled_double_moving_average_into(
            inPtr,
            fastPtr,
            slowPtr,
            len,
            6,
            12,
            5
        );

        const actualFast = Array.from(new Float64Array(memory.buffer, fastPtr, len));
        const actualSlow = Array.from(new Float64Array(memory.buffer, slowPtr, len));
        const expected = wasm.ehlers_undersampled_double_moving_average(data, 6, 12, 5);
        assertArrayClose(actualFast, expected.fast, 1e-12, 1e-12, 'EUDMA zero-copy fast mismatch');
        assertArrayClose(actualSlow, expected.slow, 1e-12, 1e-12, 'EUDMA zero-copy slow mismatch');
    } finally {
        wasm.ehlers_undersampled_double_moving_average_free(inPtr, len);
        wasm.ehlers_undersampled_double_moving_average_free(fastPtr, len);
        wasm.ehlers_undersampled_double_moving_average_free(slowPtr, len);
    }
});

test('EUDMA batch zero-copy path works', () => {
    const len = data.length;
    const rows = 4;
    const total = rows * len;
    const inPtr = wasm.ehlers_undersampled_double_moving_average_alloc(len);
    const fastPtr = wasm.ehlers_undersampled_double_moving_average_alloc(total);
    const slowPtr = wasm.ehlers_undersampled_double_moving_average_alloc(total);

    try {
        const memory = getMemory(wasm);
        assert(memory, 'WASM memory accessor unavailable');

        new Float64Array(memory.buffer, inPtr, len).set(data);
        const actualRows = wasm.ehlers_undersampled_double_moving_average_batch_into(
            inPtr,
            fastPtr,
            slowPtr,
            len,
            6,
            8,
            2,
            12,
            14,
            2,
            5,
            5,
            0
        );

        assert.strictEqual(actualRows, rows);
        const fastValues = Array.from(new Float64Array(memory.buffer, fastPtr, total));
        const slowValues = Array.from(new Float64Array(memory.buffer, slowPtr, total));
        const batch = wasm.ehlers_undersampled_double_moving_average_batch(data, {
            fast_length_range: [6, 8, 2],
            slow_length_range: [12, 14, 2],
            sample_length_range: [5, 5, 0],
        });
        assertArrayClose(fastValues, batch.fast_values, 1e-12, 1e-12, 'EUDMA batch into fast mismatch');
        assertArrayClose(slowValues, batch.slow_values, 1e-12, 1e-12, 'EUDMA batch into slow mismatch');
    } finally {
        wasm.ehlers_undersampled_double_moving_average_free(inPtr, len);
        wasm.ehlers_undersampled_double_moving_average_free(fastPtr, total);
        wasm.ehlers_undersampled_double_moving_average_free(slowPtr, total);
    }
});

test('EUDMA stream matches single and reset works', async () => {
    const single = wasm.ehlers_undersampled_double_moving_average(data, 6, 12, 5);
    const stream = new wasm.EhlersUndersampledDoubleMovingAverageStreamWasm(6, 12, 5);
    const fast = new Array(data.length).fill(NaN);
    const slow = new Array(data.length).fill(NaN);

    for (let i = 0; i < data.length; i++) {
        const result = await stream.update(data[i]);
        if (result !== null) {
            fast[i] = result.fast;
            slow[i] = result.slow;
        }
    }

    assertArrayClose(fast, single.fast, 1e-12, 1e-12, 'EUDMA stream fast mismatch');
    assertArrayClose(slow, single.slow, 1e-12, 1e-12, 'EUDMA stream slow mismatch');
    stream.reset();
    assert.strictEqual(await stream.update(NaN), null);
});

test('EUDMA errors', () => {
    assert.throws(
        () => wasm.ehlers_undersampled_double_moving_average([], 6, 12, 5),
        /empty|Empty/i
    );
    assert.throws(
        () => wasm.ehlers_undersampled_double_moving_average(new Float64Array(16).fill(NaN), 6, 12, 5),
        /all values are NaN/i
    );
    assert.throws(
        () => wasm.ehlers_undersampled_double_moving_average(new Float64Array([1, 2, 3]), 0, 12, 5),
        /invalid fast_length|Invalid fast_length/i
    );
    assert.throws(
        () => wasm.ehlers_undersampled_double_moving_average(new Float64Array([1, 2, 3]), 6, 0, 5),
        /invalid slow_length|Invalid slow_length/i
    );
    assert.throws(
        () => wasm.ehlers_undersampled_double_moving_average(new Float64Array([1, 2, 3]), 6, 12, 0),
        /invalid sample_length|Invalid sample_length/i
    );
});
