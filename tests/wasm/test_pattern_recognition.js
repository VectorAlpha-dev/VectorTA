import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadTestData } from './test_utils.js';

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

test('pattern_recognition_js output contract', () => {
    const n = 512;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    const out = wasm.pattern_recognition_js(open, high, low, close);

    assert.strictEqual(typeof out.rows, 'number');
    assert.strictEqual(typeof out.cols, 'number');
    assert.strictEqual(out.cols, n);
    assert.strictEqual(out.pattern_ids.length, out.rows);
    assert.strictEqual(out.values.length, out.rows * out.cols);
    assert(out.pattern_ids.includes('cdldoji'));

    const limit = Math.min(out.values.length, 1024);
    for (let i = 0; i < limit; i++) {
        assert(out.values[i] === 0 || out.values[i] === 1);
    }
});

test('pattern_recognition_js rejects length mismatch', () => {
    const n = 128;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n - 1));

    assert.throws(() => {
        wasm.pattern_recognition_js(open, high, low, close);
    }, /length mismatch|open=128/);
});

test('pattern_recognition_into pointer path matches safe API', () => {
    const n = 256;
    const open = new Float64Array(testData.open.slice(0, n));
    const high = new Float64Array(testData.high.slice(0, n));
    const low = new Float64Array(testData.low.slice(0, n));
    const close = new Float64Array(testData.close.slice(0, n));

    const safe = wasm.pattern_recognition_js(open, high, low, close);
    const rows = safe.rows;
    const outLen = rows * n;

    const openPtr = wasm.allocate_f64_array(n);
    const highPtr = wasm.allocate_f64_array(n);
    const lowPtr = wasm.allocate_f64_array(n);
    const closePtr = wasm.allocate_f64_array(n);
    const outPtr = wasm.pattern_recognition_alloc(outLen);

    const openView = new Float64Array(wasm.__wasm.memory.buffer, openPtr, n);
    const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, n);
    const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, n);
    const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, n);
    openView.set(open);
    highView.set(high);
    lowView.set(low);
    closeView.set(close);

    const meta = wasm.pattern_recognition_into(
        openPtr,
        highPtr,
        lowPtr,
        closePtr,
        outPtr,
        n,
        outLen,
    );

    const outView = new Uint8Array(wasm.__wasm.memory.buffer, outPtr, outLen);
    const outCopy = new Uint8Array(outView);

    assert.strictEqual(meta.rows, rows);
    assert.strictEqual(meta.cols, n);
    assert.strictEqual(meta.pattern_ids.length, rows);

    assert.deepStrictEqual(Array.from(outCopy), Array.from(safe.values));

    assert.throws(() => {
        wasm.pattern_recognition_into(
            openPtr,
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            n,
            outLen - 1,
        );
    }, /output buffer too small/);

    wasm.pattern_recognition_free(outPtr, outLen);
});
