/**
 * WASM binding tests for VPT (Volume Price Trend).
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { 
    loadTestData, 
    assertArrayClose, 
    assertClose,
    isNaN,
    assertAllNaN,
    assertNoNaN,
    EXPECTED_OUTPUTS 
} from './test_utils.js';
import { compareWithRust } from './rust-comparison.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
    
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
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

test('VPT basic candles', () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.vpt_js(close, volume);
    assert.strictEqual(result.length, close.length);
});

test('VPT basic slices', () => {
    
    const price = new Float64Array([1.0, 1.1, 1.05, 1.2, 1.3]);
    const volume = new Float64Array([1000.0, 1100.0, 1200.0, 1300.0, 1400.0]);
    
    const result = wasm.vpt_js(price, volume);
    assert.strictEqual(result.length, price.length);
    
    
    assert(isNaN(result[0]), 'First value should be NaN');
    assert(isNaN(result[1]), 'Second value should be NaN');
    
    assert(!isNaN(result[2]), 'Third value should not be NaN');
});

test('VPT accuracy from CSV', async () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.vpt_js(close, volume);
    
    const expected_last_five = [
        -18292.323972247592,
        -18292.510374716476,
        -18292.803266539282,
        -18292.62919783763,
        -18296.152568643138,
    ];
    
    assert(result.length >= 5);
    const last5 = result.slice(-5);
    
    
    assertArrayClose(
        last5,
        expected_last_five,
        1e-9,
        "VPT last 5 values mismatch"
    );
    
    
    await compareWithRust('vpt', result);
});

test('VPT not enough data', () => {
    
    const price = new Float64Array([100.0]);
    const volume = new Float64Array([500.0]);
    
    assert.throws(
        () => wasm.vpt_js(price, volume),
        /Not enough valid data/,
        'Should throw with insufficient data'
    );
});

test('VPT empty data', () => {
    
    const price = new Float64Array([]);
    const volume = new Float64Array([]);
    
    assert.throws(
        () => wasm.vpt_js(price, volume),
        /Empty data/,
        'Should throw with empty data'
    );
});

test('VPT all NaN', () => {
    
    const price = new Float64Array([NaN, NaN, NaN]);
    const volume = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(
        () => wasm.vpt_js(price, volume),
        /All values are NaN/,
        'Should throw with all NaN values'
    );
});

test('VPT mismatched lengths', () => {
    
    const price = new Float64Array([1.0, 2.0, 3.0]);
    const volume = new Float64Array([100.0, 200.0]); 
    
    assert.throws(
        () => wasm.vpt_js(price, volume),
        /Empty data/,
        'Should throw with mismatched lengths'
    );
});

test('VPT in-place operation', () => {
    
    const price = new Float64Array([1.0, 1.1, 1.05, 1.2, 1.3]);
    const volume = new Float64Array([1000.0, 1100.0, 1200.0, 1300.0, 1400.0]);
    const len = price.length;
    
    
    const out_ptr = wasm.vpt_alloc(len);
    const output = new Float64Array(wasm.__wasm.memory.buffer, out_ptr, len);
    
    
    output.set(price);
    
    
    
    const vol_ptr = wasm.vpt_alloc(len);
    const volumeWasm = new Float64Array(wasm.__wasm.memory.buffer, vol_ptr, len);
    volumeWasm.set(volume);
    
    wasm.vpt_into(out_ptr, vol_ptr, out_ptr, len);
    
    
    wasm.vpt_free(vol_ptr, len);
    
    
    assert(isNaN(output[0]), 'First value should be NaN');
    assert(isNaN(output[1]), 'Second value should be NaN');
    assert(!isNaN(output[2]), 'Should have computed values after in-place operation');
    
    
    wasm.vpt_free(out_ptr, len);
});

test('VPT batch operations', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    
    const config = {};
    
    const result = wasm.vpt_batch(close, volume, config);
    
    assert(result.values, 'Should have values array');
    assert.strictEqual(result.rows, 1, 'Should have single row (no parameters)');
    assert.strictEqual(result.cols, close.length, 'Columns should match data length');
    
    
    const single = wasm.vpt_js(close, volume);
    assertArrayClose(
        result.values,
        single,
        1e-10,
        "Batch vs single VPT mismatch"
    );
});

test('VPT NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    
    close[10] = NaN;
    volume[20] = NaN;
    
    const result = wasm.vpt_js(close, volume);
    assert.strictEqual(result.length, close.length);
    
    
    assert(isNaN(result[0]), 'First value should be NaN');
    
    
    assert(isNaN(result[11]), 'Value after NaN price should be NaN');
    assert(isNaN(result[21]), 'Value after NaN volume should be NaN');
});
