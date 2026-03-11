import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function manualReference(data, alpha = 0.07) {
    const cycle = new Float64Array(data.length);
    const trigger = new Float64Array(data.length);
    const smooth = new Float64Array(data.length);
    cycle.fill(NaN);
    trigger.fill(NaN);
    smooth.fill(NaN);

    const coeffCycle = (1 - 0.5 * alpha) * (1 - 0.5 * alpha);
    const coeffPrev1 = 2 * (1 - alpha);
    const coeffPrev2 = (1 - alpha) * (1 - alpha);

    for (let i = 0; i < data.length; i++) {
        if (!Number.isFinite(data[i])) {
            continue;
        }
        if (
            i >= 3
            && Number.isFinite(data[i - 1])
            && Number.isFinite(data[i - 2])
            && Number.isFinite(data[i - 3])
        ) {
            smooth[i] = (data[i] + 2 * data[i - 1] + 2 * data[i - 2] + data[i - 3]) / 6;
        }

        let cycleMain = NaN;
        if (
            i >= 2
            && Number.isFinite(smooth[i])
            && Number.isFinite(smooth[i - 1])
            && Number.isFinite(smooth[i - 2])
            && Number.isFinite(cycle[i - 1])
            && Number.isFinite(cycle[i - 2])
        ) {
            cycleMain =
                coeffCycle * (smooth[i] - 2 * smooth[i - 1] + smooth[i - 2])
                + coeffPrev1 * cycle[i - 1]
                - coeffPrev2 * cycle[i - 2];
        }

        let cycleFallback = NaN;
        if (i >= 2 && Number.isFinite(data[i - 1]) && Number.isFinite(data[i - 2])) {
            cycleFallback = (data[i] - 2 * data[i - 1] + data[i - 2]) / 4;
        }

        cycle[i] = i < 7 ? cycleFallback : cycleMain;
        if (i >= 1 && Number.isFinite(cycle[i - 1])) {
            trigger[i] = cycle[i - 1];
        }
    }

    return { cycle, trigger };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('ehlers_simple_cycle_indicator_js matches manual reference', () => {
    const n = 160;
    const source = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        source[i] = (testData.high[i] + testData.low[i]) * 0.5;
    }
    const result = wasm.ehlers_simple_cycle_indicator_js(source, 0.07);
    const expected = manualReference(source, 0.07);
    assertArrayClose(result.cycle, expected.cycle, 1e-12, 'cycle mismatch');
    assertArrayClose(result.trigger, expected.trigger, 1e-12, 'trigger mismatch');
});

test('ehlers_simple_cycle_indicator_batch_js first row matches single', () => {
    const n = 128;
    const source = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        source[i] = (testData.high[i] + testData.low[i]) * 0.5;
    }
    const batch = wasm.ehlers_simple_cycle_indicator_batch_js(source, {
        alpha_range: [0.07, 0.09, 0.02],
    });
    const single = wasm.ehlers_simple_cycle_indicator_js(source, 0.07);
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, n);
    assertArrayClose(batch.alphas, [0.07, 0.09], 1e-12, 'alpha grid mismatch');
    assertArrayClose(batch.cycle.slice(0, n), single.cycle, 1e-12, 'batch cycle mismatch');
    assertArrayClose(batch.trigger.slice(0, n), single.trigger, 1e-12, 'batch trigger mismatch');
});

test('ehlers_simple_cycle_indicator_into pointer path matches safe API', () => {
    const n = 144;
    const source = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        source[i] = (testData.high[i] + testData.low[i]) * 0.5;
    }
    const safe = wasm.ehlers_simple_cycle_indicator_js(source, 0.07);
    const inPtr = wasm.allocate_f64_array(n);
    const outCyclePtr = wasm.ehlers_simple_cycle_indicator_alloc(n);
    const outTriggerPtr = wasm.ehlers_simple_cycle_indicator_alloc(n);
    try {
        wasm.write_f64_array(inPtr, source);
        wasm.ehlers_simple_cycle_indicator_into(inPtr, outCyclePtr, outTriggerPtr, n, 0.07);
        const gotCycle = wasm.read_f64_array(outCyclePtr, n);
        const gotTrigger = wasm.read_f64_array(outTriggerPtr, n);
        assertArrayClose(gotCycle, safe.cycle, 1e-12, 'pointer cycle mismatch');
        assertArrayClose(gotTrigger, safe.trigger, 1e-12, 'pointer trigger mismatch');
    } finally {
        wasm.deallocate_f64_array(inPtr);
        wasm.ehlers_simple_cycle_indicator_free(outCyclePtr, n);
        wasm.ehlers_simple_cycle_indicator_free(outTriggerPtr, n);
    }
});

test('ehlers_simple_cycle_indicator_batch_into returns row count', () => {
    const n = 128;
    const rows = 2;
    const source = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        source[i] = (testData.high[i] + testData.low[i]) * 0.5;
    }
    const inPtr = wasm.allocate_f64_array(n);
    const outCyclePtr = wasm.ehlers_simple_cycle_indicator_alloc(rows * n);
    const outTriggerPtr = wasm.ehlers_simple_cycle_indicator_alloc(rows * n);
    try {
        wasm.write_f64_array(inPtr, source);
        const returnedRows = wasm.ehlers_simple_cycle_indicator_batch_into(
            inPtr,
            outCyclePtr,
            outTriggerPtr,
            n,
            0.07,
            0.09,
            0.02,
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.deallocate_f64_array(inPtr);
        wasm.ehlers_simple_cycle_indicator_free(outCyclePtr, rows * n);
        wasm.ehlers_simple_cycle_indicator_free(outTriggerPtr, rows * n);
    }
});

test('ehlers_simple_cycle_indicator_js rejects invalid alpha', () => {
    const source = new Float64Array(testData.close.slice(0, 64));
    assert.throws(
        () => wasm.ehlers_simple_cycle_indicator_js(source, 1.5),
        /Invalid alpha/,
    );
});
