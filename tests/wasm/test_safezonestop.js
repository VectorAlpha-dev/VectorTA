/**
 * WASM binding tests for SafeZoneStop indicator.
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

test('SafeZoneStop partial params', () => {
    // Test with partial parameters - mirrors check_safezonestop_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    // Test with period=14, other params use defaults
    const result = wasm.safezonestop_js(high, low, 14, 2.5, 3, "short");
    assert.strictEqual(result.length, high.length);
});

test('SafeZoneStop accuracy', async () => {
    // Test SafeZoneStop matches expected values from Rust tests - mirrors check_safezonestop_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.safezonestop_js(high, low, 22, 2.5, 3, "long");
    
    assert.strictEqual(result.length, high.length);
    
    // Expected values from Rust tests
    const expectedLast5 = [
        45331.180007991,
        45712.94455308232,
        46019.94707339676,
        46461.767660969635,
        46461.767660969635,
    ];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-4,
        "SafeZoneStop last 5 values mismatch"
    );
});

test('SafeZoneStop default params', () => {
    // Test SafeZoneStop with default parameters - mirrors check_safezonestop_default_candles
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    // Default params: period=22, mult=2.5, max_lookback=3, direction="long"
    const result = wasm.safezonestop_js(high, low, 22, 2.5, 3, "long");
    assert.strictEqual(result.length, high.length);
});

test('SafeZoneStop zero period', () => {
    // Test SafeZoneStop fails with zero period - mirrors check_safezonestop_zero_period
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.safezonestop_js(high, low, 0, 2.5, 3, "long");
    }, /Invalid period/);
});

test('SafeZoneStop mismatched lengths', () => {
    // Test SafeZoneStop fails with mismatched lengths - mirrors check_safezonestop_mismatched_lengths
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]); // Different length
    
    assert.throws(() => {
        wasm.safezonestop_js(high, low, 22, 2.5, 3, "long");
    }, /Mismatched lengths/);
});

test('SafeZoneStop invalid direction', () => {
    // Test SafeZoneStop fails with invalid direction
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.safezonestop_js(high, low, 2, 2.5, 3, "invalid");
    }, /Invalid direction/);
});

test('SafeZoneStop nan handling', () => {
    // Test SafeZoneStop handles NaN values correctly - mirrors check_safezonestop_nan_handling
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.safezonestop_js(high, low, 22, 2.5, 3, "long");
    assert.strictEqual(result.length, high.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('SafeZoneStop fast API (into)', () => {
    // Test fast API with pre-allocated memory
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const len = high.length;
    
    // Allocate buffers for inputs and output
    const highPtr = wasm.safezonestop_alloc(len);
    const lowPtr = wasm.safezonestop_alloc(len);
    const outPtr = wasm.safezonestop_alloc(len);
    
    try {
        // Copy input data to WASM memory
        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        highMem.set(high);
        lowMem.set(low);
        
        // Call fast API with pointers
        wasm.safezonestop_into(
            highPtr,
            lowPtr,
            outPtr,
            len,
            22,
            2.5,
            3,
            "long"
        );
        
        // Read result
        const memory = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const result = Array.from(memory);
        
        // Compare with safe API
        const safeResult = wasm.safezonestop_js(high, low, 22, 2.5, 3, "long");
        assertArrayClose(result, safeResult, 1e-10, "Fast API mismatch");
    } finally {
        // Clean up
        wasm.safezonestop_free(highPtr, len);
        wasm.safezonestop_free(lowPtr, len);
        wasm.safezonestop_free(outPtr, len);
    }
});

test('SafeZoneStop fast API aliasing - high', () => {
    // Test fast API with aliasing (output same as high input)
    const high = new Float64Array(testData.high.slice(0, 100)); // Use smaller dataset
    const low = new Float64Array(testData.low.slice(0, 100));
    const len = high.length;
    
    // Create a copy for verification
    const highCopy = new Float64Array(high);
    
    // Allocate buffers
    const highPtr = wasm.safezonestop_alloc(len);
    const lowPtr = wasm.safezonestop_alloc(len);
    
    try {
        // Copy input data to WASM memory
        const highMem = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowMem = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        highMem.set(high);
        lowMem.set(low);
        
        // Call fast API with output aliasing high input
        wasm.safezonestop_into(
            highPtr,
            lowPtr,
            highPtr, // Output aliases with high input
            len,
            22,
            2.5,
            3,
            "long"
        );
        
        // Read result
        const result = Array.from(new Float64Array(wasm.__wasm.memory.buffer, highPtr, len));
        
        // Verify result by comparing with safe API
        const expected = wasm.safezonestop_js(highCopy, low, 22, 2.5, 3, "long");
        assertArrayClose(result, expected, 1e-10, "Aliasing test failed");
    } finally {
        // Clean up
        wasm.safezonestop_free(highPtr, len);
        wasm.safezonestop_free(lowPtr, len);
    }
});

test('SafeZoneStop batch API', () => {
    // Test batch processing
    const high = new Float64Array(testData.high.slice(0, 100)); // Use smaller dataset
    const low = new Float64Array(testData.low.slice(0, 100));
    
    const config = {
        period_range: [14, 30, 8],
        mult_range: [2.0, 3.0, 0.5],
        max_lookback_range: [2, 4, 1],
        direction: "long"
    };
    
    const result = wasm.safezonestop_batch(high, low, config);
    
    // Check structure
    assert(result.values, "Missing values in batch result");
    assert(result.combos, "Missing combos in batch result");
    assert(result.rows > 0, "Invalid rows count");
    assert(result.cols === 100, "Invalid cols count");
    
    // Verify dimensions
    assert.strictEqual(result.values.length, result.rows * result.cols);
    assert.strictEqual(result.combos.length, result.rows);
});

test('SafeZoneStop batch single parameter set', () => {
    // Test batch with single parameter combination
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    
    const config = {
        period_range: [22, 22, 0],
        mult_range: [2.5, 2.5, 0],
        max_lookback_range: [3, 3, 0],
        direction: "long"
    };
    
    const batchResult = wasm.safezonestop_batch(high, low, config);
    
    // Should match single calculation
    const singleResult = wasm.safezonestop_js(high, low, 22, 2.5, 3, "long");
    
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, 50);
    
    const batchValues = batchResult.values.slice(0, 50);
    assertArrayClose(
        batchValues,
        singleResult,
        1e-10,
        "Batch vs single mismatch"
    );
});

test('SafeZoneStop memory allocation/deallocation', () => {
    // Test memory management functions
    const len = 1000;
    
    // Allocate
    const ptr = wasm.safezonestop_alloc(len);
    assert(ptr > 0, "Invalid pointer returned");
    
    // Write some data
    const memory = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
    for (let i = 0; i < len; i++) {
        memory[i] = i * 1.5;
    }
    
    // Read back
    assert.strictEqual(memory[0], 0);
    assert.strictEqual(memory[999], 999 * 1.5);
    
    // Free
    wasm.safezonestop_free(ptr, len);
    // No assertion for free - just ensure it doesn't crash
});

test('SafeZoneStop reinput', () => {
    // Test SafeZoneStop applied to its own output - mirrors ALMA reinput test
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    // First pass
    const firstResult = wasm.safezonestop_js(high, low, 22, 2.5, 3, "long");
    assert.strictEqual(firstResult.length, high.length);
    
    // Second pass - apply SafeZoneStop to its own output (use as both high and low)
    const secondResult = wasm.safezonestop_js(firstResult, firstResult, 22, 2.5, 3, "long");
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Values should be different from first pass
    // Find indices where both are not NaN
    let hasDifference = false;
    for (let i = 0; i < firstResult.length; i++) {
        if (!isNaN(firstResult[i]) && !isNaN(secondResult[i])) {
            if (Math.abs(firstResult[i] - secondResult[i]) > 1e-10) {
                hasDifference = true;
                break;
            }
        }
    }
    assert(hasDifference, "Reinput should produce different values");
});

test('SafeZoneStop all NaN input', () => {
    // Test SafeZoneStop with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.safezonestop_js(allNaN, allNaN, 22, 2.5, 3, "long");
    }, /All values are NaN/);
});