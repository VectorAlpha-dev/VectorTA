import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function median3(a, b, c) {
    if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) {
        return NaN;
    }
    return (a + b + c) - Math.min(a, b, c) - Math.max(a, b, c);
}

function manualReference(data, alpha = 0.07) {
    const length = data.length;
    const smooth = new Float64Array(length);
    const cycle = new Float64Array(length);
    const q1 = new Float64Array(length);
    const i1 = new Float64Array(length);
    const dp = new Float64Array(length);
    const ip = new Float64Array(length);
    const p = new Float64Array(length);
    const adaptive = new Float64Array(length);
    const trigger = new Float64Array(length);
    smooth.fill(NaN);
    cycle.fill(NaN);
    q1.fill(NaN);
    dp.fill(NaN);
    ip.fill(NaN);
    p.fill(NaN);
    adaptive.fill(NaN);
    trigger.fill(NaN);

    const cycleCoef = (1 - 0.5 * alpha) * (1 - 0.5 * alpha);
    const cyclePrev1 = 2 * (1 - alpha);
    const cyclePrev2 = (1 - alpha) * (1 - alpha);

    for (let idx = 0; idx < length; idx++) {
        if (!Number.isFinite(data[idx])) {
            continue;
        }

        if (
            idx >= 3
            && Number.isFinite(data[idx - 1])
            && Number.isFinite(data[idx - 2])
            && Number.isFinite(data[idx - 3])
        ) {
            smooth[idx] = (data[idx] + 2 * data[idx - 1] + 2 * data[idx - 2] + data[idx - 3]) / 6;
        }

        let cycleFallback = NaN;
        if (idx >= 2 && Number.isFinite(data[idx - 1]) && Number.isFinite(data[idx - 2])) {
            cycleFallback = (data[idx] - 2 * data[idx - 1] + data[idx - 2]) / 4;
        }

        let cycleMain = NaN;
        if (
            idx >= 2
            && Number.isFinite(smooth[idx])
            && Number.isFinite(smooth[idx - 1])
            && Number.isFinite(smooth[idx - 2])
            && Number.isFinite(cycle[idx - 1])
            && Number.isFinite(cycle[idx - 2])
        ) {
            cycleMain =
                cycleCoef * (smooth[idx] - 2 * smooth[idx - 1] + smooth[idx - 2])
                + cyclePrev1 * cycle[idx - 1]
                - cyclePrev2 * cycle[idx - 2];
        }

        cycle[idx] = idx < 7 ? cycleFallback : cycleMain;

        if (Number.isFinite(cycle[idx])) {
            const c2 = idx >= 2 && Number.isFinite(cycle[idx - 2]) ? cycle[idx - 2] : 0;
            const c4 = idx >= 4 && Number.isFinite(cycle[idx - 4]) ? cycle[idx - 4] : 0;
            const c6 = idx >= 6 && Number.isFinite(cycle[idx - 6]) ? cycle[idx - 6] : 0;
            const ip1 = idx >= 1 && Number.isFinite(ip[idx - 1]) ? ip[idx - 1] : 0;
            q1[idx] = (0.0962 * cycle[idx] + 0.5769 * c2 - 0.5769 * c4 - 0.0962 * c6)
                * (0.5 + 0.08 * ip1);
        }

        i1[idx] = idx >= 3 && Number.isFinite(cycle[idx - 3]) ? cycle[idx - 3] : 0;

        const prevQ1 = idx >= 1 ? q1[idx - 1] : NaN;
        const prevI1 = idx >= 1 ? i1[idx - 1] : 0;
        const dpRaw =
            Number.isFinite(q1[idx]) && Number.isFinite(prevQ1) && q1[idx] !== 0 && prevQ1 !== 0
                ? ((i1[idx] / q1[idx]) - (prevI1 / prevQ1))
                    / (1 + (i1[idx] * prevI1) / (q1[idx] * prevQ1))
                : 0;
        dp[idx] = Math.min(1.1, Math.max(0.1, dpRaw));

        const mdInner = idx >= 4 ? median3(dp[idx - 2], dp[idx - 3], dp[idx - 4]) : NaN;
        const md = idx >= 1 ? median3(dp[idx], dp[idx - 1], mdInner) : NaN;
        const dc = md === 0 ? 15 : (2 * Math.PI / md) + 0.5;
        ip[idx] = 0.33 * dc + 0.67 * (idx >= 1 && Number.isFinite(ip[idx - 1]) ? ip[idx - 1] : 0);
        p[idx] = 0.15 * ip[idx] + 0.85 * (idx >= 1 && Number.isFinite(p[idx - 1]) ? p[idx - 1] : 0);

        let adaptiveMain = NaN;
        if (
            idx >= 2
            && Number.isFinite(smooth[idx])
            && Number.isFinite(smooth[idx - 1])
            && Number.isFinite(smooth[idx - 2])
            && Number.isFinite(adaptive[idx - 1])
            && Number.isFinite(adaptive[idx - 2])
        ) {
            const a1 = 2 / (p[idx] + 1);
            const adaptCoef = (1 - 0.5 * a1) * (1 - 0.5 * a1);
            const oneMinusA1 = 1 - a1;
            adaptiveMain =
                adaptCoef * (smooth[idx] - 2 * smooth[idx - 1] + smooth[idx - 2])
                + 2 * oneMinusA1 * adaptive[idx - 1]
                - oneMinusA1 * oneMinusA1 * adaptive[idx - 2];
        }

        adaptive[idx] = idx < 7
            ? cycleFallback
            : (Number.isFinite(adaptiveMain) ? adaptiveMain : cycleFallback);

        if (idx >= 1 && Number.isFinite(adaptive[idx - 1])) {
            trigger[idx] = adaptive[idx - 1];
        }
    }

    return { cycle: adaptive, trigger };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('ehlers_adaptive_cyber_cycle_js matches manual reference', () => {
    const n = 192;
    const source = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        source[i] = (testData.high[i] + testData.low[i]) * 0.5;
    }
    const result = wasm.ehlers_adaptive_cyber_cycle_js(source, 0.07);
    const expected = manualReference(source, 0.07);
    assertArrayClose(result.cycle, expected.cycle, 1e-12, 'cycle mismatch');
    assertArrayClose(result.trigger, expected.trigger, 1e-12, 'trigger mismatch');
});

test('ehlers_adaptive_cyber_cycle_batch_js first row matches single', () => {
    const n = 144;
    const source = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        source[i] = (testData.high[i] + testData.low[i]) * 0.5;
    }
    const batch = wasm.ehlers_adaptive_cyber_cycle_batch_js(source, {
        alpha_range: [0.07, 0.09, 0.02],
    });
    const single = wasm.ehlers_adaptive_cyber_cycle_js(source, 0.07);
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, n);
    assertArrayClose(batch.alphas, [0.07, 0.09], 1e-12, 'alpha grid mismatch');
    assertArrayClose(batch.cycle.slice(0, n), single.cycle, 1e-12, 'batch cycle mismatch');
    assertArrayClose(batch.trigger.slice(0, n), single.trigger, 1e-12, 'batch trigger mismatch');
});

test('ehlers_adaptive_cyber_cycle_into pointer path matches safe API', () => {
    const n = 160;
    const source = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        source[i] = (testData.high[i] + testData.low[i]) * 0.5;
    }
    const safe = wasm.ehlers_adaptive_cyber_cycle_js(source, 0.07);
    const inPtr = wasm.allocate_f64_array(n);
    const outCyclePtr = wasm.ehlers_adaptive_cyber_cycle_alloc(n);
    const outTriggerPtr = wasm.ehlers_adaptive_cyber_cycle_alloc(n);
    try {
        wasm.write_f64_array(inPtr, source);
        wasm.ehlers_adaptive_cyber_cycle_into(inPtr, outCyclePtr, outTriggerPtr, n, 0.07);
        const gotCycle = wasm.read_f64_array(outCyclePtr, n);
        const gotTrigger = wasm.read_f64_array(outTriggerPtr, n);
        assertArrayClose(gotCycle, safe.cycle, 1e-12, 'pointer cycle mismatch');
        assertArrayClose(gotTrigger, safe.trigger, 1e-12, 'pointer trigger mismatch');
    } finally {
        wasm.deallocate_f64_array(inPtr);
        wasm.ehlers_adaptive_cyber_cycle_free(outCyclePtr, n);
        wasm.ehlers_adaptive_cyber_cycle_free(outTriggerPtr, n);
    }
});

test('ehlers_adaptive_cyber_cycle_batch_into returns row count', () => {
    const n = 128;
    const rows = 2;
    const source = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        source[i] = (testData.high[i] + testData.low[i]) * 0.5;
    }
    const inPtr = wasm.allocate_f64_array(n);
    const outCyclePtr = wasm.ehlers_adaptive_cyber_cycle_alloc(rows * n);
    const outTriggerPtr = wasm.ehlers_adaptive_cyber_cycle_alloc(rows * n);
    try {
        wasm.write_f64_array(inPtr, source);
        const returnedRows = wasm.ehlers_adaptive_cyber_cycle_batch_into(
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
        wasm.ehlers_adaptive_cyber_cycle_free(outCyclePtr, rows * n);
        wasm.ehlers_adaptive_cyber_cycle_free(outTriggerPtr, rows * n);
    }
});

test('ehlers_adaptive_cyber_cycle_js rejects invalid alpha', () => {
    const source = new Float64Array(testData.close.slice(0, 64));
    assert.throws(
        () => wasm.ehlers_adaptive_cyber_cycle_js(source, 1.5),
        /Invalid alpha/,
    );
});
