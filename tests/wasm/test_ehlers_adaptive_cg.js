import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let data;

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);

    const testData = loadTestData();
    data = Float64Array.from(
        testData.high.map((high, idx) => 0.5 * (high + testData.low[idx]))
    );
});

test('Ehlers Adaptive CG basic output shape', () => {
    const result = wasm.ehlers_adaptive_cg_js(data, 0.07);

    assert.strictEqual(result.cg.length, data.length);
    assert.strictEqual(result.trigger.length, data.length);

    for (let i = 0; i < 13; i++) {
        assert(isNaN(result.cg[i]));
    }
    for (let i = 0; i < 14; i++) {
        assert(isNaN(result.trigger[i]));
    }

    let hasCg = false;
    let hasTrigger = false;
    for (let i = 64; i < data.length; i++) {
        hasCg ||= !isNaN(result.cg[i]);
        hasTrigger ||= !isNaN(result.trigger[i]);
    }
    assert(hasCg);
    assert(hasTrigger);
});

test('Ehlers Adaptive CG fast API matches safe API', () => {
    const safe = wasm.ehlers_adaptive_cg_js(data, 0.07);
    const len = data.length;

    const inPtr = wasm.ehlers_adaptive_cg_alloc(len);
    const cgPtr = wasm.ehlers_adaptive_cg_alloc(len);
    const triggerPtr = wasm.ehlers_adaptive_cg_alloc(len);

    try {
        new Float64Array(wasm.__wasm.memory.buffer, inPtr, len).set(data);
        wasm.ehlers_adaptive_cg_into(inPtr, cgPtr, triggerPtr, len, 0.07);

        const cg = Array.from(new Float64Array(wasm.__wasm.memory.buffer, cgPtr, len));
        const trigger = Array.from(new Float64Array(wasm.__wasm.memory.buffer, triggerPtr, len));

        assertArrayClose(cg, safe.cg, 1e-10, 1e-10, 'CG mismatch');
        assertArrayClose(trigger, safe.trigger, 1e-10, 1e-10, 'Trigger mismatch');
    } finally {
        wasm.ehlers_adaptive_cg_free(inPtr, len);
        wasm.ehlers_adaptive_cg_free(cgPtr, len);
        wasm.ehlers_adaptive_cg_free(triggerPtr, len);
    }
});

test('Ehlers Adaptive CG aliasing path works', () => {
    const len = data.length;
    const inPtr = wasm.ehlers_adaptive_cg_alloc(len);
    const triggerPtr = wasm.ehlers_adaptive_cg_alloc(len);

    try {
        new Float64Array(wasm.__wasm.memory.buffer, inPtr, len).set(data);
        wasm.ehlers_adaptive_cg_into(inPtr, inPtr, triggerPtr, len, 0.07);

        const cg = Array.from(new Float64Array(wasm.__wasm.memory.buffer, inPtr, len));
        const trigger = Array.from(new Float64Array(wasm.__wasm.memory.buffer, triggerPtr, len));

        assert.strictEqual(cg.length, len);
        assert.strictEqual(trigger.length, len);
    } finally {
        wasm.ehlers_adaptive_cg_free(inPtr, len);
        wasm.ehlers_adaptive_cg_free(triggerPtr, len);
    }
});

test('Ehlers Adaptive CG batch matches single', () => {
    const single = wasm.ehlers_adaptive_cg_js(data, 0.07);
    const batch = wasm.ehlers_adaptive_cg_batch(data, {
        alpha_range: [0.07, 0.07, 0.0]
    });

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, data.length);
    assert.strictEqual(batch.combos.length, 1);

    assertArrayClose(batch.cg, single.cg, 1e-10, 1e-10, 'Batch CG mismatch');
    assertArrayClose(batch.trigger, single.trigger, 1e-10, 1e-10, 'Batch trigger mismatch');
});

test('Ehlers Adaptive CG stream matches single and reset works', () => {
    const single = wasm.ehlers_adaptive_cg_js(data, 0.07);
    const stream = new wasm.EhlersAdaptiveCgStreamWasm(0.07);
    const cg = new Array(data.length).fill(NaN);
    const trigger = new Array(data.length).fill(NaN);

    for (let i = 0; i < data.length; i++) {
        const result = stream.update(data[i]);
        if (result !== null) {
            cg[i] = result.cg;
            trigger[i] = result.trigger;
        }
    }

    assertArrayClose(cg, single.cg, 1e-10, 1e-10, 'Stream CG mismatch');
    assertArrayClose(trigger, single.trigger, 1e-10, 1e-10, 'Stream trigger mismatch');

    stream.reset();
    assert.strictEqual(stream.update(data[0]), null);
});

test('Ehlers Adaptive CG errors', () => {
    assert.throws(() => wasm.ehlers_adaptive_cg_js([], 0.07), /empty|Empty/i);
    assert.throws(() => wasm.ehlers_adaptive_cg_js(new Float64Array(20).fill(NaN), 0.07), /all values are NaN/i);
    assert.throws(() => wasm.ehlers_adaptive_cg_js(new Float64Array([1, 2, 3, 4]), 0.07), /not enough valid data/i);
    assert.throws(() => wasm.ehlers_adaptive_cg_js(data, 0.0), /invalid alpha|Invalid alpha/i);
});
