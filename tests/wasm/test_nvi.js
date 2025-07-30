/**
 * WASM binding tests for NVI indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
const test = require('node:test');
const assert = require('node:assert');
const path = require('path');
const { 
    loadTestData, 
    assertArrayClose, 
    assertClose,
    isNaN,
    assertAllNaN,
    assertNoNaN,
    EXPECTED_OUTPUTS 
} = require('./test_utils');

let wasm;
let testData;

test.before(async () => {
    // Load WASM module
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        wasm = await import(wasmPath);
        await wasm.default();
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('NVI accuracy - safe API', () => {
    const close = testData.close;
    const volume = testData.volume;
    
    // Run NVI
    const result = wasm.nvi_js(close, volume);
    
    // Check last 5 values match expected from EXPECTED_OUTPUTS
    const expected = EXPECTED_OUTPUTS.nvi.last_5_values;
    const last5 = result.slice(-5);
    
    assertArrayClose(last5, expected, 1e-5, 'NVI accuracy test failed');
});

test('NVI error handling - empty data', () => {
    const emptyArray = new Float64Array(0);
    
    assert.throws(() => {
        wasm.nvi_js(emptyArray, emptyArray);
    }, {
        message: /Empty data/i
    }, 'Should throw error for empty data');
});

test('NVI error handling - all NaN values', () => {
    const nanClose = new Float64Array([NaN, NaN, NaN]);
    const nanVolume = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.nvi_js(nanClose, nanVolume);
    }, {
        message: /All close values are NaN/i
    }, 'Should throw error when all values are NaN');
});

test('NVI error handling - not enough valid data', () => {
    const close = new Float64Array([NaN, 100.0]);
    const volume = new Float64Array([NaN, 120.0]);
    
    assert.throws(() => {
        wasm.nvi_js(close, volume);
    }, {
        message: /Not enough valid data/i
    }, 'Should throw error when not enough valid data');
});

test('NVI fast API - basic operation', () => {
    const close = testData.close;
    const volume = testData.volume;
    const len = close.length;
    
    // Allocate buffers
    const closePtr = wasm.nvi_alloc(len);
    const volumePtr = wasm.nvi_alloc(len);
    const outPtr = wasm.nvi_alloc(len);
    
    try {
        // Copy data to WASM memory
        const closeMemory = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const volumeMemory = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        closeMemory.set(close);
        volumeMemory.set(volume);
        
        // Compute NVI using fast API
        wasm.nvi_into(closePtr, volumePtr, outPtr, len);
        
        // Read result from WASM memory
        const memory = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const result = Array.from(memory);
        
        // Compare with safe API
        const safeResult = wasm.nvi_js(close, volume);
        assertArrayClose(result, safeResult, 1e-9, 'Fast API should match safe API');
    } finally {
        // Clean up
        wasm.nvi_free(closePtr, len);
        wasm.nvi_free(volumePtr, len);
        wasm.nvi_free(outPtr, len);
    }
});

test('NVI fast API - in-place operation (aliasing)', () => {
    const close = testData.close.slice(0, 100); // Use smaller dataset
    const volume = testData.volume.slice(0, 100);
    const len = close.length;
    
    // First get expected result
    const expected = wasm.nvi_js(close, volume);
    
    // Allocate buffer and copy close data
    const dataPtr = wasm.nvi_alloc(len);
    const memory = new Float64Array(wasm.__wasm.memory.buffer, dataPtr, len);
    memory.set(close);
    
    // Allocate volume buffer
    const volumePtr = wasm.nvi_alloc(len);
    const volumeMemory = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
    volumeMemory.set(volume);
    
    try {
        // Compute NVI in-place (output overwrites close input)
        wasm.nvi_into(dataPtr, volumePtr, dataPtr, len);
        
        // Check result
        const result = Array.from(memory);
        assertArrayClose(result, expected, 1e-9, 'In-place operation should produce correct results');
    } finally {
        // Clean up
        wasm.nvi_free(dataPtr, len);
        wasm.nvi_free(volumePtr, len);
    }
});

test('NVI fast API - null pointer handling', () => {
    assert.throws(() => {
        wasm.nvi_into(0, 0, 0, 100);
    }, {
        message: /null pointer/i
    }, 'Should throw error for null pointers');
});

test('NVI memory management - no leaks', () => {
    // Test allocating and freeing multiple times
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const closePtrs = [];
        const volumePtrs = [];
        const outPtrs = [];
        
        // Allocate multiple buffers
        for (let i = 0; i < 10; i++) {
            closePtrs.push(wasm.nvi_alloc(size));
            volumePtrs.push(wasm.nvi_alloc(size));
            outPtrs.push(wasm.nvi_alloc(size));
        }
        
        // Verify all allocations succeeded
        for (let i = 0; i < 10; i++) {
            assert(closePtrs[i] !== 0, `Failed to allocate close buffer ${i}`);
            assert(volumePtrs[i] !== 0, `Failed to allocate volume buffer ${i}`);
            assert(outPtrs[i] !== 0, `Failed to allocate output buffer ${i}`);
        }
        
        // Free all buffers
        for (let i = 0; i < 10; i++) {
            wasm.nvi_free(closePtrs[i], size);
            wasm.nvi_free(volumePtrs[i], size);
            wasm.nvi_free(outPtrs[i], size);
        }
    }
    
    // If we get here without crashing, memory management is working
});

test.after(() => {
    console.log('NVI WASM tests completed');
});
