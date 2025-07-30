/**
 * WASM binding tests for WCLPRICE indicator.
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
    assertNoNaN
} from './test_utils.js';
import { compareWithRust } from './rust-comparison.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
    // Load WASM module
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('WCLPRICE slices', () => {
    // Test WCLPRICE with simple slice data - mirrors check_wclprice_slices
    const high = new Float64Array([59230.0, 59220.0, 59077.0, 59160.0, 58717.0]);
    const low = new Float64Array([59222.0, 59211.0, 59077.0, 59143.0, 58708.0]);
    const close = new Float64Array([59225.0, 59210.0, 59080.0, 59150.0, 58710.0]);
    
    const result = wasm.wclprice_js(high, low, close);
    const expected = [59225.5, 59212.75, 59078.5, 59150.75, 58711.25];
    
    assertArrayClose(result, expected, 1e-2, "WCLPRICE values mismatch");
});

test('WCLPRICE candles', () => {
    // Test WCLPRICE with full candle data - mirrors check_wclprice_candles
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.wclprice_js(high, low, close);
    assert.strictEqual(result.length, close.length);
    
    // Check some values are reasonable (should be between low and high)
    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i])) {
            assert(low[i] <= result[i] && result[i] <= high[i], 
                `WCLPRICE value ${result[i]} at index ${i} is outside range [${low[i]}, ${high[i]}]`);
        }
    }
});

test('WCLPRICE empty data', () => {
    // Test WCLPRICE fails with empty data - mirrors check_wclprice_empty_data
    const high = new Float64Array([]);
    const low = new Float64Array([]);
    const close = new Float64Array([]);
    
    assert.throws(() => {
        wasm.wclprice_js(high, low, close);
    }, /Empty data/);
});

test('WCLPRICE all NaN', () => {
    // Test WCLPRICE fails with all NaN values - mirrors check_wclprice_all_nan
    const high = new Float64Array([NaN, NaN]);
    const low = new Float64Array([NaN, NaN]);
    const close = new Float64Array([NaN, NaN]);
    
    assert.throws(() => {
        wasm.wclprice_js(high, low, close);
    }, /All values are NaN/);
});

test('WCLPRICE partial NaN', () => {
    // Test WCLPRICE handles partial NaN values - mirrors check_wclprice_partial_nan
    const high = new Float64Array([NaN, 59000.0]);
    const low = new Float64Array([NaN, 58950.0]);
    const close = new Float64Array([NaN, 58975.0]);
    
    const result = wasm.wclprice_js(high, low, close);
    
    // First value should be NaN
    assert(isNaN(result[0]));
    // Second value should be calculated
    const expected = (59000.0 + 58950.0 + 2.0 * 58975.0) / 4.0;
    assertClose(result[1], expected, 1e-8, "WCLPRICE calculation incorrect");
});

test('WCLPRICE formula', () => {
    // Test WCLPRICE formula (high + low + 2*close) / 4
    const high = new Float64Array([100.0]);
    const low = new Float64Array([90.0]);
    const close = new Float64Array([95.0]);
    
    const result = wasm.wclprice_js(high, low, close);
    const expected = (100.0 + 90.0 + 2.0 * 95.0) / 4.0; // = 95.0
    
    assertClose(result[0], expected, 1e-10, "WCLPRICE formula incorrect");
});

test('WCLPRICE mismatched lengths', () => {
    // Test WCLPRICE handles mismatched input lengths
    const high = new Float64Array([100.0, 101.0, 102.0]);
    const low = new Float64Array([90.0, 91.0]); // Shorter
    const close = new Float64Array([95.0, 96.0, 97.0]);
    
    // Should process up to the shortest length
    const result = wasm.wclprice_js(high, low, close);
    assert.strictEqual(result.length, 2); // min(3, 2, 3) = 2
});

test('WCLPRICE fast API (no aliasing)', () => {
    // Test fast API without aliasing
    const high = new Float64Array([100.0, 101.0, 102.0]);
    const low = new Float64Array([90.0, 91.0, 92.0]);
    const close = new Float64Array([95.0, 96.0, 97.0]);
    const output = new Float64Array(3);
    
    const highPtr = wasm.wclprice_alloc(3);
    const lowPtr = wasm.wclprice_alloc(3);
    const closePtr = wasm.wclprice_alloc(3);
    const outPtr = wasm.wclprice_alloc(3);
    
    const memory = new Float64Array(wasm.memory.buffer);
    memory.set(high, highPtr / 8);
    memory.set(low, lowPtr / 8);
    memory.set(close, closePtr / 8);
    
    wasm.wclprice_into(highPtr, lowPtr, closePtr, outPtr, 3);
    
    const result = new Float64Array(wasm.memory.buffer, outPtr, 3);
    
    // Check results
    for (let i = 0; i < 3; i++) {
        const expected = (high[i] + low[i] + 2.0 * close[i]) / 4.0;
        assertClose(result[i], expected, 1e-10, `Fast API value mismatch at index ${i}`);
    }
    
    // Clean up
    wasm.wclprice_free(highPtr, 3);
    wasm.wclprice_free(lowPtr, 3);
    wasm.wclprice_free(closePtr, 3);
    wasm.wclprice_free(outPtr, 3);
});

test('WCLPRICE fast API (with aliasing)', () => {
    // Test fast API with aliasing (output = high)
    const data = new Float64Array([100.0, 101.0, 102.0]);
    const low = new Float64Array([90.0, 91.0, 92.0]);
    const close = new Float64Array([95.0, 96.0, 97.0]);
    
    const dataPtr = wasm.wclprice_alloc(3);
    const lowPtr = wasm.wclprice_alloc(3);
    const closePtr = wasm.wclprice_alloc(3);
    
    const memory = new Float64Array(wasm.memory.buffer);
    memory.set(data, dataPtr / 8);
    memory.set(low, lowPtr / 8);
    memory.set(close, closePtr / 8);
    
    // Use dataPtr as both high input and output (aliasing)
    wasm.wclprice_into(dataPtr, lowPtr, closePtr, dataPtr, 3);
    
    const result = new Float64Array(wasm.memory.buffer, dataPtr, 3);
    
    // Check results
    for (let i = 0; i < 3; i++) {
        const expected = (data[i] + low[i] + 2.0 * close[i]) / 4.0;
        assertClose(result[i], expected, 1e-10, `Fast API aliasing value mismatch at index ${i}`);
    }
    
    // Clean up
    wasm.wclprice_free(dataPtr, 3);
    wasm.wclprice_free(lowPtr, 3);
    wasm.wclprice_free(closePtr, 3);
});

test('WCLPRICE batch', () => {
    // Test batch API
    const high = new Float64Array([100.0, 101.0, 102.0]);
    const low = new Float64Array([90.0, 91.0, 92.0]);
    const close = new Float64Array([95.0, 96.0, 97.0]);
    
    // WCLPRICE has no parameters, so config is empty
    const config = {};
    
    const result = wasm.wclprice_batch(high, low, close, config);
    
    assert.strictEqual(result.rows, 1); // No parameters, so only 1 row
    assert.strictEqual(result.cols, 3);
    assert.strictEqual(result.values.length, 3);
    
    // Check values match single calculation
    const singleResult = wasm.wclprice_js(high, low, close);
    assertArrayClose(result.values, singleResult, 1e-10, "Batch values mismatch");
});

test('WCLPRICE with test data', async () => {
    // Test with real data
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result = wasm.wclprice_js(high, low, close);
    
    // Since WCLPRICE has no parameters, we pass null for params
    await compareWithRust('wclprice', result, 'hlc', null);
});