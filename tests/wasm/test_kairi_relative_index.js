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

test('kairi_relative_index_js output contract', () => {
    const source = new Float64Array(testData.close.slice(0, 256));
    const volume = new Float64Array(testData.volume.slice(0, 256));
    const result = wasm.kairi_relative_index_js(source, volume, 50, 'SMA');

    assert.strictEqual(result.length, source.length);
    assert.strictEqual(result.findIndex(v => !isNaN(v)), 49);
});

test('kairi_relative_index_js rejects invalid parameters', () => {
    const source = new Float64Array(testData.close.slice(0, 64));
    const volume = new Float64Array(testData.volume.slice(0, 64));

    assert.throws(() => {
        wasm.kairi_relative_index_js(source, volume, 1, 'SMA');
    }, /Invalid length/);

    assert.throws(() => {
        wasm.kairi_relative_index_js(source, volume, 10, 'BAD');
    }, /Invalid moving average type/);
});

test('kairi_relative_index_into pointer path matches safe API', () => {
    const source = new Float64Array(testData.close.slice(0, 200));
    const volume = new Float64Array(testData.volume.slice(0, 200));
    const safe = wasm.kairi_relative_index_js(source, volume, 20, 'EMA');
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const sourcePtr = wasm.allocate_f64_array(source.length);
    const volumePtr = wasm.allocate_f64_array(volume.length);
    const outPtr = wasm.kairi_relative_index_alloc(source.length);

    try {
        new Float64Array(memory.buffer, sourcePtr, source.length).set(source);
        new Float64Array(memory.buffer, volumePtr, volume.length).set(volume);
        wasm.kairi_relative_index_into(sourcePtr, volumePtr, outPtr, source.length, 20, 'EMA');
        const actual = Array.from(new Float64Array(memory.buffer, outPtr, source.length));
        assertArrayClose(actual, safe, 1e-12, 'pointer mismatch');
    } finally {
        wasm.kairi_relative_index_free(outPtr, source.length);
        wasm.deallocate_f64_array(sourcePtr);
        wasm.deallocate_f64_array(volumePtr);
    }
});

test('kairi_relative_index_batch_js single parameter set matches safe API', () => {
    const source = new Float64Array(testData.close.slice(0, 200));
    const volume = new Float64Array(testData.volume.slice(0, 200));
    const batch = wasm.kairi_relative_index_batch_js(source, volume, {
        length_range: [20, 20, 0],
        ma_type: 'HMA',
    });
    const single = wasm.kairi_relative_index_js(source, volume, 20, 'HMA');

    assert.strictEqual(batch.rows, 1);
    assert.strictEqual(batch.cols, source.length);
    assert.strictEqual(batch.combos[0].length, 20);
    assertArrayClose(batch.values, single, 1e-12, 'batch mismatch');
});

test('kairi_relative_index_batch_into metadata matches requested ranges', () => {
    const source = new Float64Array(testData.close.slice(0, 128));
    const volume = new Float64Array(testData.volume.slice(0, 128));
    const memory = wasm.wasm_memory
        ? wasm.wasm_memory()
        : (wasm.__wasm?.memory || (wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory));
    assert(memory && memory.buffer, 'raw wasm memory is not exposed by this package build');

    const rows = 3;
    const total = rows * source.length;
    const sourcePtr = wasm.allocate_f64_array(source.length);
    const volumePtr = wasm.allocate_f64_array(volume.length);
    const outPtr = wasm.kairi_relative_index_alloc(total);

    try {
        new Float64Array(memory.buffer, sourcePtr, source.length).set(source);
        new Float64Array(memory.buffer, volumePtr, volume.length).set(volume);
        const actualRows = wasm.kairi_relative_index_batch_into(
            sourcePtr,
            volumePtr,
            outPtr,
            source.length,
            10,
            14,
            2,
            'VWMA',
        );
        assert.strictEqual(actualRows, rows);

        const flat = Array.from(new Float64Array(memory.buffer, outPtr, total));
        const jsBatch = wasm.kairi_relative_index_batch_js(source, volume, {
            length_range: [10, 14, 2],
            ma_type: 'VWMA',
        });
        assertArrayClose(flat, jsBatch.values, 1e-12, 'batch_into mismatch');
    } finally {
        wasm.kairi_relative_index_free(outPtr, total);
        wasm.deallocate_f64_array(sourcePtr);
        wasm.deallocate_f64_array(volumePtr);
    }
});
