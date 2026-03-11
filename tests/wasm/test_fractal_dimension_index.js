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
    const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);
    testData = loadTestData();
});

test('fractal_dimension_index_js output contract', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const result = wasm.fractal_dimension_index_js(close, 30);

    assert.strictEqual(result.length, close.length);
    const first = result.findIndex(v => !isNaN(v));
    assert.strictEqual(first, 29);
});

test('fractal_dimension_index_js rejects invalid parameters', () => {
    const close = new Float64Array(testData.close.slice(0, 128));

    assert.throws(() => {
        wasm.fractal_dimension_index_js(close, 1);
    }, /Invalid length/);
});

test('fractal_dimension_index_into pointer path matches safe API', (t) => {
    const close = new Float64Array(testData.close.slice(0, 200));
    const safe = wasm.fractal_dimension_index_js(close, 30);
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    if (!memory || !memory.buffer) {
        t.skip('raw wasm memory is not exposed by this package build');
        return;
    }

    const inPtr = wasm.allocate_f64_array(close.length);
    const outPtr = wasm.fractal_dimension_index_alloc(close.length);

    try {
        new Float64Array(memory.buffer, inPtr, close.length).set(close);
        wasm.fractal_dimension_index_into(inPtr, outPtr, close.length, 30);
        const out = Array.from(new Float64Array(memory.buffer, outPtr, close.length));
        assertArrayClose(out, safe, 1e-12, 'pointer values mismatch');
    } finally {
        wasm.fractal_dimension_index_free(outPtr, close.length);
        wasm.deallocate_f64_array(inPtr);
    }
});

test('fractal_dimension_index_batch_js single parameter set matches safe API', () => {
    const close = new Float64Array(testData.close.slice(0, 200));
    const batch = wasm.fractal_dimension_index_batch_js(close, {
        length_range: [30, 30, 0],
    });
    const single = wasm.fractal_dimension_index_js(close, 30);

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, close.length);
    assert.strictEqual(batch.values.length, close.length);
    assert.strictEqual(batch.combos[0].length, 30);
    assertArrayClose(batch.values, single, 1e-12, 'batch values mismatch');
});

test('fractal_dimension_index_batch_js metadata matches requested ranges', () => {
    const close = new Float64Array(testData.close.slice(0, 256));
    const batch = wasm.fractal_dimension_index_batch_js(close, {
        length_range: [20, 24, 2],
    });

    assert.strictEqual(batch.rows, 3);
    assert.strictEqual(batch.cols, close.length);
    assert.deepStrictEqual(
        batch.combos.map(combo => combo.length),
        [20, 22, 24],
    );
});
