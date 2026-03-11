import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

function manualReference(data, domesticCycleLength = 15) {
    const out = new Float64Array(data.length);
    out.fill(NaN);
    if (domesticCycleLength <= 0) {
        return out;
    }
    const angle = (2 * Math.PI) / domesticCycleLength;
    for (let i = 0; i < data.length; i++) {
        if (i + 1 < domesticCycleLength) {
            continue;
        }
        let real = 0;
        let imag = 0;
        let valid = true;
        for (let j = 0; j < domesticCycleLength; j++) {
            const value = data[i - j];
            if (!Number.isFinite(value)) {
                valid = false;
                break;
            }
            const theta = angle * j;
            real += Math.cos(theta) * value;
            imag += Math.sin(theta) * value;
        }
        if (!valid) {
            continue;
        }
        let phase = Math.abs(real) > 0.001
            ? Math.atan(imag / real) * 180 / Math.PI
            : imag > 0 ? 90 : imag < 0 ? -90 : 0;
        if (real < 0) {
            phase += 180;
        }
        phase += 90;
        if (phase < 0) {
            phase += 360;
        }
        if (phase > 360) {
            phase -= 360;
        }
        out[i] = phase;
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

test('l1_ehlers_phasor_js matches manual reference', () => {
    const source = new Float64Array(testData.close.slice(0, 160));
    const result = wasm.l1_ehlers_phasor_js(source, 15);
    const expected = manualReference(source, 15);
    assertArrayClose(result, expected, 5e-11, 'phasor mismatch');
});

test('l1_ehlers_phasor_batch_js first row matches single', () => {
    const source = new Float64Array(testData.close.slice(0, 128));
    const batch = wasm.l1_ehlers_phasor_batch_js(source, {
        domestic_cycle_length_range: [15, 17, 2],
    });
    const single = wasm.l1_ehlers_phasor_js(source, 15);
    assert.strictEqual(batch.rows, 2);
    assert.strictEqual(batch.cols, source.length);
    assert.deepStrictEqual(
        batch.combos.map((combo) => combo.domestic_cycle_length),
        [15, 17],
    );
    assertArrayClose(batch.values.slice(0, source.length), single, 5e-11, 'batch mismatch');
});

test('l1_ehlers_phasor_into pointer path matches safe API', () => {
    const n = 144;
    const source = new Float64Array(testData.close.slice(0, n));
    const safe = wasm.l1_ehlers_phasor_js(source, 15);
    const inPtr = wasm.allocate_f64_array(n);
    const outPtr = wasm.l1_ehlers_phasor_alloc(n);
    try {
        wasm.write_f64_array(inPtr, source);
        wasm.l1_ehlers_phasor_into(inPtr, outPtr, n, 15);
        const got = wasm.read_f64_array(outPtr, n);
        assertArrayClose(got, safe, 5e-11, 'pointer mismatch');
    } finally {
        wasm.deallocate_f64_array(inPtr);
        wasm.l1_ehlers_phasor_free(outPtr, n);
    }
});

test('l1_ehlers_phasor_batch_into returns row count', () => {
    const n = 128;
    const rows = 2;
    const source = new Float64Array(testData.close.slice(0, n));
    const inPtr = wasm.allocate_f64_array(n);
    const outPtr = wasm.l1_ehlers_phasor_alloc(rows * n);
    try {
        wasm.write_f64_array(inPtr, source);
        const returnedRows = wasm.l1_ehlers_phasor_batch_into(
            inPtr,
            outPtr,
            n,
            15,
            17,
            2,
        );
        assert.strictEqual(returnedRows, rows);
    } finally {
        wasm.deallocate_f64_array(inPtr);
        wasm.l1_ehlers_phasor_free(outPtr, rows * n);
    }
});

test('l1_ehlers_phasor_js rejects invalid length', () => {
    const source = new Float64Array(testData.close.slice(0, 64));
    assert.throws(
        () => wasm.l1_ehlers_phasor_js(source, 0),
        /Invalid domestic_cycle_length/,
    );
});
