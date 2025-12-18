/**
 * WASM binding tests for OBV indicator.
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

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
    const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
    const importPath = process.platform === 'win32'
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    wasm = await import(importPath);

    testData = loadTestData();
});

test('OBV accuracy', () => {
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.obv_js(close, volume);
    assert.strictEqual(result.length, close.length);
    
    // Expected values from Rust tests (last 5 values)
    const expectedLastFive = [
        -329661.6180239202,
        -329767.87639284023,
        -329889.94421654026,
        -329801.35075036023,
        -330218.2007503602,
    ];
    
    // Check last 5 values
    const last5 = result.slice(-5);
    assertArrayClose(last5, expectedLastFive, 1e-6, "OBV last 5 values mismatch");
});

test('OBV empty data', () => {
    const emptyClose = new Float64Array([]);
    const emptyVolume = new Float64Array([]);
    
    assert.throws(() => {
        wasm.obv_js(emptyClose, emptyVolume);
    }, /empty/i);
});

test('OBV mismatched lengths', () => {
    const close = new Float64Array([1.0, 2.0, 3.0]);
    const volume = new Float64Array([100.0, 200.0]);
    
    assert.throws(() => {
        wasm.obv_js(close, volume);
    }, /Data length mismatch/);
});

test('OBV all NaN', () => {
    const close = new Float64Array([NaN, NaN]);
    const volume = new Float64Array([NaN, NaN]);
    
    assert.throws(() => {
        wasm.obv_js(close, volume);
    }, /All values are NaN/);
});

test('OBV batch', () => {
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.obv_batch(close, volume);
    
    assert(result.values, 'Batch result should have values');
    assert.strictEqual(result.rows, 1, 'OBV batch should have 1 row (no parameters)');
    assert.strictEqual(result.cols, close.length, 'Columns should match input length');
    assert.strictEqual(result.values.length, close.length, 'Values array should match input length');
    
    // Should match single calculation
    const singleResult = wasm.obv_js(close, volume);
    assertArrayClose(result.values, singleResult, 1e-10, "Batch should match single calculation");
});

test('OBV fast API (in-place)', () => {
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const len = close.length;
    
    // Allocate buffers for inputs and output
    const closePtr = wasm.obv_alloc(len);
    const volumePtr = wasm.obv_alloc(len);
    const outPtr = wasm.obv_alloc(len);
    
    try {
        // Create views into WASM memory for inputs
        let closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        let volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        
        // Copy data into WASM memory
        closeView.set(close);
        volumeView.set(volume);
        
        // Get expected result first (before calling obv_into)
        const expected = wasm.obv_js(close, volume);
        
        // Use fast API with raw pointers
        wasm.obv_into(closePtr, volumePtr, outPtr, len);
        
        // IMPORTANT: Recreate view after the call as memory may have grown
        // This invalidates any existing TypedArrays
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        // Compare results
        assertArrayClose(result, expected, 1e-10, "Fast API should match safe API");
    } finally {
        // Clean up
        wasm.obv_free(closePtr, len);
        wasm.obv_free(volumePtr, len);
        wasm.obv_free(outPtr, len);
    }
});

test('OBV fast API with aliasing', () => {
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset
    const volume = new Float64Array(testData.volume.slice(0, 100));
    const len = close.length;
    
    // Allocate buffers for close and volume
    const closePtr = wasm.obv_alloc(len);
    const volumePtr = wasm.obv_alloc(len);
    
    try {
        // Create views into WASM memory
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        
        // Copy data into WASM memory
        closeView.set(close);
        volumeView.set(volume);
        
        // Test aliasing - use closePtr as output (overwriting close data)
        wasm.obv_into(closePtr, volumePtr, closePtr, len);
        
        // Read result from close buffer (which now contains OBV output)
        const result = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        
        // Should produce correct result despite aliasing
        const expected = wasm.obv_js(close, volume);
        assertArrayClose(result, expected, 1e-10, "Fast API should handle aliasing correctly");
    } finally {
        // Clean up
        wasm.obv_free(closePtr, len);
        wasm.obv_free(volumePtr, len);
    }
});

test('OBV NaN handling', () => {
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.obv_js(close, volume);
    assert.strictEqual(result.length, close.length);
    
    // Find first valid data point
    let firstValid = null;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i]) && !isNaN(volume[i])) {
            firstValid = i;
            break;
        }
    }
    
    if (firstValid !== null) {
        // First valid value should be 0.0 (OBV starts at 0)
        assertClose(result[firstValid], 0.0, 1e-10, `OBV should start at 0.0, got ${result[firstValid]}`);
        
        // All values before first valid should be NaN
        if (firstValid > 0) {
            for (let i = 0; i < firstValid; i++) {
                assert(isNaN(result[i]), `Expected NaN at index ${i} before first valid data`);
            }
        }
    }
});

test.after(() => {
    console.log('OBV WASM tests completed');
});
