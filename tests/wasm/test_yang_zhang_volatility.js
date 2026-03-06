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

test('yang_zhang_volatility_js output contract', () => {
    const n = 512;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    const result = wasm.yang_zhang_volatility_js(open, high, low, close, 14, false, 0.34);

    assert(result.yz, 'Should have yz output');
    assert(result.rs, 'Should have rs output');

    const yz = Array.from(result.yz);
    const rs = Array.from(result.rs);

    assert.strictEqual(yz.length, n);
    assert.strictEqual(rs.length, n);

    const firstValid = yz.findIndex(v => !isNaN(v));
    assert(firstValid >= 14, `unexpected first valid index: ${firstValid}`);
    assert(yz.some(v => !isNaN(v)), 'yz should contain valid values');
    assert(rs.some(v => !isNaN(v)), 'rs should contain valid values');

    const tailStart = Math.min(firstValid + 32, yz.length);
    for (let i = tailStart; i < yz.length; i++) {
        assert(!isNaN(yz[i]), `unexpected yz NaN at ${i}`);
        assert(!isNaN(rs[i]), `unexpected rs NaN at ${i}`);
    }
});

test('yang_zhang_volatility_js rejects length mismatch', () => {
    const n = 128;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n - 1));

    assert.throws(() => {
        wasm.yang_zhang_volatility_js(open, high, low, close, 14, false, 0.34);
    }, /length mismatch/);
});

test('yang_zhang_volatility_js rejects invalid parameters', () => {
    const n = 128;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    assert.throws(() => {
        wasm.yang_zhang_volatility_js(open, high, low, close, 0, false, 0.34);
    }, /Invalid lookback/);

    assert.throws(() => {
        wasm.yang_zhang_volatility_js(open, high, low, close, 14, true, 1.5);
    }, /Invalid k/);
});

test('yang_zhang_volatility_into pointer path matches safe API', () => {
    const n = 256;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    const safe = wasm.yang_zhang_volatility_js(open, high, low, close, 20, true, 0.42);

    const openPtr = wasm.allocate_f64_array(n);
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const closePtr = wasm.allocate_f64_array(n);
    const outPtr = wasm.yang_zhang_volatility_alloc(n);

    try {
        const memory = wasm.__wasm.memory.buffer;
        new Float64Array(memory, openPtr, n).set(open);
        new Float64Array(memory, highPtr, n).set(high);
        new Float64Array(memory, lowPtr, n).set(low);
        new Float64Array(memory, closePtr, n).set(close);

        wasm.yang_zhang_volatility_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            n,
            20,
            true,
            0.42,
        );

        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, 2 * n);
        const yz = Array.from(outView.subarray(0, n));
        const rs = Array.from(outView.subarray(n, 2 * n));

        assertArrayClose(yz, safe.yz, 1e-10, 'zero-copy yz mismatch');
        assertArrayClose(rs, safe.rs, 1e-10, 'zero-copy rs mismatch');
    } finally {
        wasm.yang_zhang_volatility_free(outPtr, n);
        wasm.deallocate_f64_array(openPtr);
        wasm.deallocate_f64_array(highPtr);
        wasm.deallocate_f64_array(lowPtr);
        wasm.deallocate_f64_array(closePtr);
    }
});

test('yang_zhang_volatility_batch_js single parameter set matches safe API', () => {
    const n = 256;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    const batch = wasm.yang_zhang_volatility_batch_js(open, high, low, close, {
        lookback_range: [20, 20, 0],
        k_override: true,
        k_range: [0.42, 0.42, 0.0],
    });

    const single = wasm.yang_zhang_volatility_js(open, high, low, close, 20, true, 0.42);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, n);
    assert.strictEqual(batch.combos.length, 1);
    assert.strictEqual(batch.yz.length, n);
    assert.strictEqual(batch.rs.length, n);
    assert.strictEqual(batch.combos[0].lookback, 20);
    assert.strictEqual(batch.combos[0].k_override, true);
    assert.ok(Math.abs(batch.combos[0].k - 0.42) <= 1e-12);

    assertArrayClose(batch.yz, single.yz, 1e-10, 'batch yz mismatch');
    assertArrayClose(batch.rs, single.rs, 1e-10, 'batch rs mismatch');
});

test('yang_zhang_volatility_batch_js metadata matches requested lookbacks', () => {
    const n = 200;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    const batch = wasm.yang_zhang_volatility_batch_js(open, high, low, close, {
        lookback_range: [10, 14, 2],
        k_override: true,
        k_range: [0.42, 0.42, 0.0],
    });

    assert.strictEqual(batch.rows, 3);
    assert.strictEqual(batch.cols, n);
    assert.strictEqual(batch.yz.length, 3 * n);
    assert.strictEqual(batch.rs.length, 3 * n);
    assert.deepStrictEqual(
        batch.combos.map(combo => combo.lookback),
        [10, 12, 14],
    );
    assert.deepStrictEqual(
        batch.combos.map(combo => combo.k_override),
        [true, true, true],
    );
    for (const combo of batch.combos) {
        assert.ok(Math.abs(combo.k - 0.42) <= 1e-12);
    }

    const single = wasm.yang_zhang_volatility_js(open, high, low, close, 10, true, 0.42);
    assertArrayClose(batch.yz.slice(0, n), single.yz, 1e-10, 'first-row yz mismatch');
    assertArrayClose(batch.rs.slice(0, n), single.rs, 1e-10, 'first-row rs mismatch');
});

test('yang_zhang_volatility_batch_js rejects invalid config', () => {
    const n = 64;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    assert.throws(() => {
        wasm.yang_zhang_volatility_batch_js(open, high, low, close, {
            lookback_range: [10, 12],
            k_override: true,
            k_range: [0.42, 0.42, 0.0],
        });
    }, /Invalid config/);

    assert.throws(() => {
        wasm.yang_zhang_volatility_batch_js(open, high, low, close, {
            lookback_range: [10, 12, 2],
            k_override: true,
            k_range: [0.42, 0.42],
        });
    }, /Invalid config/);
});
