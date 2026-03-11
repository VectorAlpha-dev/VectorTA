import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function naiveWaveSmoother(data, period, phase) {
    const phaseRad = phase * Math.PI / 180.0;
    const weights = [];
    let weightSum = 0.0;
    for (let i = 0; i <= period; i++) {
        const w = Math.sin(i * Math.PI / period + phaseRad) * Math.cos(i * Math.PI / (2 * period));
        weights.push(w);
        weightSum += w;
    }
    if (!Number.isFinite(weightSum) || Math.abs(weightSum) <= Number.EPSILON) {
        return new Float64Array(data.length).fill(NaN);
    }
    for (let i = 0; i < weights.length; i++) {
        weights[i] /= weightSum;
    }

    const src = new Float64Array(data.length).fill(NaN);
    let prevRaw = NaN;
    let firstNz = NaN;
    for (let i = 0; i < data.length; i++) {
        const value = data[i];
        src[i] = Number.isFinite(value) ? 0.5 * (value + (Number.isFinite(prevRaw) ? prevRaw : 0.0)) : NaN;
        prevRaw = value;
        if (Number.isNaN(firstNz) && Number.isFinite(src[i])) {
            firstNz = src[i];
        }
    }

    const out = new Float64Array(data.length).fill(NaN);
    const first = data.findIndex(v => Number.isFinite(v));
    if (first < 0) {
        return out;
    }
    for (let i = first; i < data.length; i++) {
        let acc = 0.0;
        for (let lag = 0; lag <= period; lag++) {
            const hist = i >= lag ? src[i - lag] : firstNz;
            acc += (Number.isFinite(hist) ? hist : firstNz) * weights[lag];
        }
        out[i] = acc;
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

test('wave_smoother_js output contract', () => {
    const source = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.wave_smoother_js(source, 20, 70.0);
    assert.strictEqual(result.length, source.length);
    assert(Array.from(result).some(v => !Number.isNaN(v)));
});

test('wave_smoother handles long period / early history like naive implementation', () => {
    const source = new Float64Array([10.0, 12.0, 14.0]);
    const result = wasm.wave_smoother_js(source, 20, 70.0);
    const expected = naiveWaveSmoother(source, 20, 70.0);
    assertArrayClose(result, expected, 1e-12, 'naive mismatch');
});

test('wave_smoother_into_host pointer path matches safe API', () => {
    const source = new Float64Array(testData.close.slice(0, 220));
    const safe = wasm.wave_smoother_js(source, 20, 70.0);
    const outPtr = wasm.wave_smoother_alloc(source.length);

    try {
        wasm.wave_smoother_into_host(source, outPtr, 20, 70.0);
        const out = wasm.read_f64_array(outPtr, source.length);
        assertArrayClose(out, safe, 1e-12, 'pointer mismatch');
    } finally {
        wasm.wave_smoother_free(outPtr, source.length);
    }
});

test('wave_smoother_batch_js single parameter set matches safe API', () => {
    const source = new Float64Array(testData.close.slice(0, 220));
    const safe = wasm.wave_smoother_js(source, 20, 70.0);
    const batch = wasm.wave_smoother_batch_js(source, {
        period_range: [20, 20, 0],
        phase_range: [70.0, 70.0, 0.0],
    });

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, source.length);
    assert.strictEqual(batch.periods[0], 20);
    assert.strictEqual(batch.phases[0], 70.0);
    assertArrayClose(batch.values, safe, 1e-12, 'batch mismatch');
});

test('wave_smoother invalid phase throws', () => {
    const source = new Float64Array(testData.close.slice(0, 64));
    assert.throws(() => {
        wasm.wave_smoother_js(source, 20, 120.0);
    }, /Invalid phase/);
});
