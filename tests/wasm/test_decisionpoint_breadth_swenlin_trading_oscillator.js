import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;

function sampleBreadth(length) {
    const advancing = new Float64Array(length);
    const declining = new Float64Array(length);
    for (let i = 0; i < length; i += 1) {
        const x = i;
        advancing[i] = 1500.0 + x * 0.8 + Math.sin(x * 0.07) * 120.0 + 40.0;
        declining[i] = 1300.0 + x * 0.5 + Math.cos(x * 0.05) * 95.0 + 30.0;
    }
    return { advancing, declining };
}

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath =
        process.platform === 'win32'
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
    wasm = await import(importPath);
});

test('decisionpoint_breadth_swenlin_trading_oscillator_js output contract', () => {
    const { advancing, declining } = sampleBreadth(256);
    const values = wasm.decisionpoint_breadth_swenlin_trading_oscillator_js(advancing, declining);

    assert.strictEqual(values.length, advancing.length);
    assert.strictEqual(values.findIndex(v => !isNaN(v)), 4);
    assert(Number.isFinite(values.at(-1)));
});

test('decisionpoint_breadth_swenlin_trading_oscillator_js rejects invalid input', () => {
    const { advancing, declining } = sampleBreadth(32);
    assert.throws(
        () =>
            wasm.decisionpoint_breadth_swenlin_trading_oscillator_js(
                advancing.subarray(0, 31),
                declining,
            ),
        /Inconsistent slice lengths/,
    );
});

test('decisionpoint_breadth_swenlin_trading_oscillator_into pointer path matches safe API', () => {
    const { advancing, declining } = sampleBreadth(192);
    const safe = wasm.decisionpoint_breadth_swenlin_trading_oscillator_js(advancing, declining);

    const advPtr = wasm.copy_f64_array(advancing);
    const decPtr = wasm.copy_f64_array(declining);
    const outPtr = wasm.decisionpoint_breadth_swenlin_trading_oscillator_alloc(advancing.length);

    try {
        wasm.decisionpoint_breadth_swenlin_trading_oscillator_into(
            advPtr,
            decPtr,
            outPtr,
            advancing.length,
        );
        assertArrayClose(
            wasm.read_f64_array(outPtr, advancing.length),
            safe,
            1e-12,
            'pointer-path mismatch',
        );
    } finally {
        wasm.decisionpoint_breadth_swenlin_trading_oscillator_free(outPtr, advancing.length);
        wasm.deallocate_f64_array(advPtr);
        wasm.deallocate_f64_array(decPtr);
    }
});

test('decisionpoint_breadth_swenlin_trading_oscillator_js resets after zero-total breadth', () => {
    const { advancing, declining } = sampleBreadth(180);
    advancing[90] = 0.0;
    declining[90] = 0.0;

    const values = wasm.decisionpoint_breadth_swenlin_trading_oscillator_js(advancing, declining);
    assert(isNaN(values[90]));
    assert(Number.isFinite(values.at(-1)));
});

test('decisionpoint_breadth_swenlin_trading_oscillator_batch_js matches safe API', () => {
    const { advancing, declining } = sampleBreadth(192);
    const batch = wasm.decisionpoint_breadth_swenlin_trading_oscillator_batch_js(
        advancing,
        declining,
        {},
    );
    const safe = wasm.decisionpoint_breadth_swenlin_trading_oscillator_js(advancing, declining);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, advancing.length);
    assertArrayClose(batch.values.slice(0, advancing.length), safe, 1e-12, 'batch mismatch');
});

test('decisionpoint_breadth_swenlin_trading_oscillator_batch_into matches batch_js', () => {
    const { advancing, declining } = sampleBreadth(160);
    const batch = wasm.decisionpoint_breadth_swenlin_trading_oscillator_batch_js(
        advancing,
        declining,
        {},
    );

    const advPtr = wasm.copy_f64_array(advancing);
    const decPtr = wasm.copy_f64_array(declining);
    const outPtr = wasm.decisionpoint_breadth_swenlin_trading_oscillator_alloc(advancing.length);

    try {
        const rows = wasm.decisionpoint_breadth_swenlin_trading_oscillator_batch_into(
            advPtr,
            decPtr,
            outPtr,
            advancing.length,
        );
        assert.strictEqual(rows, 1);
        assertArrayClose(
            wasm.read_f64_array(outPtr, advancing.length),
            batch.values,
            1e-12,
            'batch_into mismatch',
        );
    } finally {
        wasm.decisionpoint_breadth_swenlin_trading_oscillator_free(outPtr, advancing.length);
        wasm.deallocate_f64_array(advPtr);
        wasm.deallocate_f64_array(decPtr);
    }
});
