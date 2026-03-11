import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData, assertArrayClose, isNaN } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
    try {
        const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
        const importPath = process.platform === 'win32'
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }

    testData = loadTestData();
});

test('rogers_satchell_volatility_js output contract', () => {
    const n = 512;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    const result = wasm.rogers_satchell_volatility_js(open, high, low, close, 14, 6);

    assert(result.rs, 'Should have rs output');
    assert(result.signal, 'Should have signal output');

    const rs = Array.from(result.rs);
    const signal = Array.from(result.signal);

    assert.strictEqual(rs.length, n);
    assert.strictEqual(signal.length, n);

    const rsFirstValid = rs.findIndex(v => !isNaN(v));
    const signalFirstValid = signal.findIndex(v => !isNaN(v));
    assert(rsFirstValid >= 13, `unexpected rs first valid index: ${rsFirstValid}`);
    assert(signalFirstValid >= 18, `unexpected signal first valid index: ${signalFirstValid}`);

    const tailStart = Math.min(signalFirstValid + 32, signal.length);
    for (let i = tailStart; i < signal.length; i++) {
        assert(!isNaN(rs[i]), `unexpected rs NaN at ${i}`);
        assert(!isNaN(signal[i]), `unexpected signal NaN at ${i}`);
    }
});

test('rogers_satchell_volatility_js rejects length mismatch', () => {
    const n = 128;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n - 1));

    assert.throws(() => {
        wasm.rogers_satchell_volatility_js(open, high, low, close, 14, 6);
    }, /length mismatch/);
});

test('rogers_satchell_volatility_js rejects invalid parameters', () => {
    const n = 128;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    assert.throws(() => {
        wasm.rogers_satchell_volatility_js(open, high, low, close, 0, 6);
    }, /Invalid lookback/);

    assert.throws(() => {
        wasm.rogers_satchell_volatility_js(open, high, low, close, 14, 0);
    }, /Invalid signal length/);
});

test('rogers_satchell_volatility_into pointer path matches safe API', (t) => {
    const n = 256;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    const safe = wasm.rogers_satchell_volatility_js(open, high, low, close, 20, 5);
    const memory = wasm.__wasm?.memory || wasm.memory;
    if (!memory || !memory.buffer) {
        t.skip('raw wasm memory is not exposed by this package build');
        return;
    }

    const openPtr = wasm.allocate_f64_array(n);
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const closePtr = wasm.allocate_f64_array(n);
    const outPtr = wasm.rogers_satchell_volatility_alloc(n);

    try {
        new Float64Array(memory.buffer, openPtr, n).set(open);
        new Float64Array(memory.buffer, highPtr, n).set(high);
        new Float64Array(memory.buffer, lowPtr, n).set(low);
        new Float64Array(memory.buffer, closePtr, n).set(close);

        wasm.rogers_satchell_volatility_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            n,
            20,
            5,
        );

        const outView = new Float64Array(memory.buffer, outPtr, 2 * n);
        const rs = Array.from(outView.subarray(0, n));
        const signal = Array.from(outView.subarray(n, 2 * n));

        assertArrayClose(rs, safe.rs, 1e-10, 'zero-copy rs mismatch');
        assertArrayClose(signal, safe.signal, 1e-10, 'zero-copy signal mismatch');
    } finally {
        wasm.rogers_satchell_volatility_free(outPtr, n);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('rogers_satchell_volatility_batch_js single parameter set matches safe API', () => {
    const n = 256;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    const batch = wasm.rogers_satchell_volatility_batch_js(open, high, low, close, {
        lookback_range: [20, 20, 0],
        signal_length_range: [5, 5, 0],
    });

    const single = wasm.rogers_satchell_volatility_js(open, high, low, close, 20, 5);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, n);
    assert.strictEqual(batch.combos.length, 1);
    assert.strictEqual(batch.rs.length, n);
    assert.strictEqual(batch.signal.length, n);
    assert.strictEqual(batch.combos[0].lookback, 20);
    assert.strictEqual(batch.combos[0].signal_length, 5);

    assertArrayClose(batch.rs, single.rs, 1e-10, 'batch rs mismatch');
    assertArrayClose(batch.signal, single.signal, 1e-10, 'batch signal mismatch');
});

test('rogers_satchell_volatility_batch_js metadata matches requested ranges', () => {
    const n = 200;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    const batch = wasm.rogers_satchell_volatility_batch_js(open, high, low, close, {
        lookback_range: [10, 14, 2],
        signal_length_range: [4, 6, 2],
    });

    assert.strictEqual(batch.rows, 6);
    assert.strictEqual(batch.cols, n);
    assert.strictEqual(batch.rs.length, 6 * n);
    assert.strictEqual(batch.signal.length, 6 * n);
    assert.deepStrictEqual(
        batch.combos.map(combo => combo.lookback),
        [10, 10, 12, 12, 14, 14],
    );
    assert.deepStrictEqual(
        batch.combos.map(combo => combo.signal_length),
        [4, 6, 4, 6, 4, 6],
    );

    const single = wasm.rogers_satchell_volatility_js(open, high, low, close, 10, 4);
    assertArrayClose(batch.rs.slice(0, n), single.rs, 1e-10, 'first-row rs mismatch');
    assertArrayClose(batch.signal.slice(0, n), single.signal, 1e-10, 'first-row signal mismatch');
});

test('rogers_satchell_volatility_batch_js rejects invalid config', () => {
    const n = 64;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    assert.throws(() => {
        wasm.rogers_satchell_volatility_batch_js(open, high, low, close, {
            lookback_range: [10, 12],
            signal_length_range: [4, 6, 2],
        });
    }, /Invalid config/);

    assert.throws(() => {
        wasm.rogers_satchell_volatility_batch_js(open, high, low, close, {
            lookback_range: [10, 12, 2],
            signal_length_range: [4, 6],
        });
    }, /Invalid config/);
});
