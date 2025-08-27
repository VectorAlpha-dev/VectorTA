/**
 * WASM binding tests for MEDPRICE indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 * Tests follow the same quality and coverage standards as ALMA tests.
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
    // Load WASM module
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

// ============================================================================
// Basic Functionality Tests
// ============================================================================

test('MEDPRICE default params', () => {
    // Test with default parameters - mirrors Rust test
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.medprice_js(high, low);
    assert(result instanceof Float64Array);
    assert.strictEqual(result.length, high.length);
});

test('MEDPRICE accuracy', () => {
    // Test MEDPRICE matches expected values from Rust tests
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.medprice_js(high, low);
    
    assert.strictEqual(result.length, high.length, "Output length should match input");
    
    // Expected values from Rust test (medprice = (high + low) / 2)
    const expectedLastFive = [59166.0, 59244.5, 59118.0, 59146.5, 58767.5];
    
    const actualLastFive = result.slice(-5);
    
    for (let i = 0; i < 5; i++) {
        assertClose(actualLastFive[i], expectedLastFive[i], 1e-10, 
            `MEDPRICE mismatch at last_five[${i}]`);
    }
});

test('MEDPRICE formula verification', () => {
    // Test MEDPRICE formula: (high + low) / 2 with various inputs
    
    // Simple integer values
    const high = new Float64Array([100.0, 200.0, 300.0, 400.0, 500.0]);
    const low = new Float64Array([50.0, 100.0, 150.0, 200.0, 250.0]);
    
    const result = wasm.medprice_js(high, low);
    
    // Verify the formula for each value
    for (let i = 0; i < high.length; i++) {
        const expected = (high[i] + low[i]) / 2.0;
        assertClose(result[i], expected, 1e-15, 
            `Formula verification failed at index ${i}`);
    }
    
    // Test with fractional values
    const highFrac = new Float64Array([10.5, 20.25, 30.75, 40.125]);
    const lowFrac = new Float64Array([5.25, 10.125, 15.375, 20.0625]);
    
    const resultFrac = wasm.medprice_js(highFrac, lowFrac);
    
    for (let i = 0; i < highFrac.length; i++) {
        const expected = (highFrac[i] + lowFrac[i]) / 2.0;
        assertClose(resultFrac[i], expected, 1e-15,
            `Fractional formula verification failed at index ${i}`);
    }
});

test('MEDPRICE no warmup period', () => {
    // Test that MEDPRICE has no warmup period - first valid value appears immediately
    const high = new Float64Array([100.0, 110.0, 120.0, 130.0]);
    const low = new Float64Array([80.0, 90.0, 100.0, 110.0]);
    
    const result = wasm.medprice_js(high, low);
    
    // First value should be valid (not NaN)
    assert(!isNaN(result[0]), "MEDPRICE should not have warmup period");
    assertClose(result[0], 90.0, 1e-10); // (100 + 80) / 2
});

// ============================================================================
// Error Handling Tests
// ============================================================================

test('MEDPRICE empty data', () => {
    // Test MEDPRICE fails with empty data
    const high = new Float64Array([]);
    const low = new Float64Array([]);
    
    assert.throws(() => {
        wasm.medprice_js(high, low);
    }, /empty/i);
});

test('MEDPRICE different length', () => {
    // Test MEDPRICE fails with different slice lengths
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]);
    
    assert.throws(() => {
        wasm.medprice_js(high, low);
    }, /different|length/i);
});

test('MEDPRICE all values NaN', () => {
    // Test MEDPRICE fails with all NaN data
    const high = new Float64Array([NaN, NaN, NaN]);
    const low = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.medprice_js(high, low);
    }, /nan/i);
});

// ============================================================================
// NaN Handling Tests
// ============================================================================

test('MEDPRICE NaN handling basic', () => {
    // Test MEDPRICE handling of NaN values
    const high = new Float64Array([NaN, 100.0, 110.0]);
    const low = new Float64Array([NaN, 80.0, 90.0]);
    
    const result = wasm.medprice_js(high, low);
    
    assert.strictEqual(result.length, 3);
    assert(isNaN(result[0]), "First value should be NaN");
    assertClose(result[1], 90.0, 1e-10);  // (100 + 80) / 2
    assertClose(result[2], 100.0, 1e-10);  // (110 + 90) / 2
});

test('MEDPRICE late NaN handling', () => {
    // Test MEDPRICE handling of late NaN values
    const high = new Float64Array([100.0, 110.0, NaN]);
    const low = new Float64Array([80.0, 90.0, NaN]);
    
    const result = wasm.medprice_js(high, low);
    
    assert.strictEqual(result.length, 3);
    assertClose(result[0], 90.0, 1e-10);
    assertClose(result[1], 100.0, 1e-10);
    assert(isNaN(result[2]), "Last value should be NaN");
});

test('MEDPRICE NaN patterns', () => {
    // Test MEDPRICE with various NaN patterns
    
    // Alternating NaN pattern
    const highAlt = new Float64Array([100.0, NaN, 120.0, NaN, 140.0]);
    const lowAlt = new Float64Array([80.0, NaN, 100.0, NaN, 120.0]);
    
    const resultAlt = wasm.medprice_js(highAlt, lowAlt);
    
    assertClose(resultAlt[0], 90.0, 1e-10);
    assert(isNaN(resultAlt[1]));
    assertClose(resultAlt[2], 110.0, 1e-10);
    assert(isNaN(resultAlt[3]));
    assertClose(resultAlt[4], 130.0, 1e-10);
    
    // NaN cluster in middle
    const highCluster = new Float64Array([100.0, 110.0, NaN, NaN, NaN, 150.0, 160.0]);
    const lowCluster = new Float64Array([80.0, 90.0, NaN, NaN, NaN, 130.0, 140.0]);
    
    const resultCluster = wasm.medprice_js(highCluster, lowCluster);
    
    assertClose(resultCluster[0], 90.0, 1e-10);
    assertClose(resultCluster[1], 100.0, 1e-10);
    assert(isNaN(resultCluster[2]));
    assert(isNaN(resultCluster[3]));
    assert(isNaN(resultCluster[4]));
    assertClose(resultCluster[5], 140.0, 1e-10);
    assertClose(resultCluster[6], 150.0, 1e-10);
});

test('MEDPRICE partial NaN', () => {
    // Test MEDPRICE when only one input has NaN
    const high = new Float64Array([100.0, NaN, 120.0, 130.0]);
    const low = new Float64Array([80.0, 90.0, 100.0, 110.0]);
    
    const result = wasm.medprice_js(high, low);
    
    assertClose(result[0], 90.0, 1e-10);
    assert(isNaN(result[1]), "Should be NaN when either input is NaN");
    assertClose(result[2], 110.0, 1e-10);
    assertClose(result[3], 120.0, 1e-10);
});

// ============================================================================
// Boundary and Edge Case Tests
// ============================================================================

test('MEDPRICE boundary values', () => {
    // Test MEDPRICE with extreme values
    
    // Very large values
    const highLarge = new Float64Array([1e10, 1e11, 1e12]);
    const lowLarge = new Float64Array([5e9, 5e10, 5e11]);
    
    const resultLarge = wasm.medprice_js(highLarge, lowLarge);
    
    assertClose(resultLarge[0], 7.5e9, 1e-1);
    assertClose(resultLarge[1], 7.5e10, 1e-1);
    assertClose(resultLarge[2], 7.5e11, 1e-1);
    
    // Very small values
    const highSmall = new Float64Array([1e-10, 1e-11, 1e-12]);
    const lowSmall = new Float64Array([5e-11, 5e-12, 5e-13]);
    
    const resultSmall = wasm.medprice_js(highSmall, lowSmall);
    
    assertClose(resultSmall[0], 7.5e-11, 1e-20);
    assertClose(resultSmall[1], 7.5e-12, 1e-21);
    assertClose(resultSmall[2], 7.5e-13, 1e-22);
    
    // Mixed sign values
    const highMixed = new Float64Array([100.0, 50.0, -25.0]);
    const lowMixed = new Float64Array([-100.0, -50.0, -75.0]);
    
    const resultMixed = wasm.medprice_js(highMixed, lowMixed);
    
    assertClose(resultMixed[0], 0.0, 1e-10);  // (100 + -100) / 2
    assertClose(resultMixed[1], 0.0, 1e-10);  // (50 + -50) / 2
    assertClose(resultMixed[2], -50.0, 1e-10);  // (-25 + -75) / 2
});

test('MEDPRICE single value', () => {
    // Test MEDPRICE with single value input
    const high = new Float64Array([100.0]);
    const low = new Float64Array([80.0]);
    
    const result = wasm.medprice_js(high, low);
    
    assert.strictEqual(result.length, 1);
    assertClose(result[0], 90.0, 1e-10);
});

test('MEDPRICE identical values', () => {
    // Test MEDPRICE when high equals low
    const high = new Float64Array([100.0, 100.0, 100.0]);
    const low = new Float64Array([100.0, 100.0, 100.0]);
    
    const result = wasm.medprice_js(high, low);
    
    for (let i = 0; i < result.length; i++) {
        assertClose(result[i], 100.0, 1e-10);
    }
});

// ============================================================================
// Batch Processing Tests
// ============================================================================

test('MEDPRICE batch - single parameter combination', () => {
    // Test batch API with medprice (which has no parameters)
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    
    // Call batch with empty config or dummy config
    const batchResult = wasm.medprice_batch(high, low, {
        dummy_range: [0, 0, 0]
    });
    
    // Verify structure
    assert(batchResult.values, 'Should have values array');
    assert(batchResult.combos, 'Should have combos array');
    assert(typeof batchResult.rows === 'number', 'Should have rows count');
    assert(typeof batchResult.cols === 'number', 'Should have cols count');
    
    // Should have 1 row, 100 columns
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.values.length, 100);
    assert.strictEqual(batchResult.combos.length, 1);
    
    // Values should match regular calculation
    const regularResult = wasm.medprice_js(high, low);
    assertArrayClose(batchResult.values, regularResult, 1e-10, "Batch vs regular mismatch");
});

test('MEDPRICE batch - no config', () => {
    // Test batch API without config
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    
    // Call batch without config
    const batchResult = wasm.medprice_batch(high, low, null);
    
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 50);
});

test('MEDPRICE batch - error handling', () => {
    // Test batch error conditions
    
    // Empty data
    assert.throws(() => {
        wasm.medprice_batch(new Float64Array([]), new Float64Array([]), null);
    }, /empty/i, "Should throw on empty data");
    
    // Different lengths
    assert.throws(() => {
        wasm.medprice_batch(
            new Float64Array([1.0, 2.0]),
            new Float64Array([1.0]),
            null
        );
    }, /different|length/i, "Should throw on different lengths");
});

// ============================================================================
// Zero-copy / Fast API Tests
// ============================================================================

test('MEDPRICE zero-copy basic', () => {
    // Test fast/unsafe API
    const data = testData.high.slice(0, 100);
    const high = new Float64Array(data);
    const low = new Float64Array(testData.low.slice(0, 100));
    
    // Allocate output buffer
    const ptr = wasm.medprice_alloc(high.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        high.length
    );
    
    // Compute MEDPRICE
    try {
        // First copy high data into memory for potential in-place operation
        memView.set(high);
        
        // For medprice, we need separate high/low pointers
        const highPtr = wasm.medprice_alloc(high.length);
        const lowPtr = wasm.medprice_alloc(low.length);
        
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, high.length);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, low.length);
        
        highView.set(high);
        lowView.set(low);
        
        wasm.medprice_into(highPtr, lowPtr, ptr, high.length);
        
        // Verify results match regular API
        const regularResult = wasm.medprice_js(high, low);
        for (let i = 0; i < high.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
        
        // Cleanup
        wasm.medprice_free(highPtr, high.length);
        wasm.medprice_free(lowPtr, low.length);
    } finally {
        wasm.medprice_free(ptr, high.length);
    }
});

test('MEDPRICE zero-copy aliasing', () => {
    // Test aliasing detection when output pointer equals input pointer
    const high = new Float64Array([100.0, 110.0, 120.0, 130.0, 140.0]);
    const low = new Float64Array([80.0, 90.0, 100.0, 110.0, 120.0]);
    
    // Allocate buffers
    const highPtr = wasm.medprice_alloc(high.length);
    const lowPtr = wasm.medprice_alloc(low.length);
    
    try {
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, high.length);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, low.length);
        
        highView.set(high);
        lowView.set(low);
        
        // Use high pointer as output (aliasing)
        wasm.medprice_into(highPtr, lowPtr, highPtr, high.length);
        
        // Recreate view in case memory grew
        const resultView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, high.length);
        
        // Check results
        const expected = [90.0, 100.0, 110.0, 120.0, 130.0];
        for (let i = 0; i < high.length; i++) {
            assertClose(resultView[i], expected[i], 1e-10, 
                `Aliasing result mismatch at index ${i}`);
        }
    } finally {
        wasm.medprice_free(highPtr, high.length);
        wasm.medprice_free(lowPtr, low.length);
    }
});

test('MEDPRICE zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.medprice_into(0, 0, 0, 10);
    }, /null pointer|invalid memory/i);
    
    // Test with allocated memory but empty data
    const ptr = wasm.medprice_alloc(0);
    const highPtr = wasm.medprice_alloc(0);
    const lowPtr = wasm.medprice_alloc(0);
    try {
        assert.throws(() => {
            wasm.medprice_into(highPtr, lowPtr, ptr, 0);
        }, /empty/i);
    } finally {
        wasm.medprice_free(ptr, 0);
        wasm.medprice_free(highPtr, 0);
        wasm.medprice_free(lowPtr, 0);
    }
});

test('MEDPRICE zero-copy edge cases', () => {
    // Test with various array sizes including prime numbers
    const sizes = [1, 7, 13, 31, 97, 100, 256, 1000];
    
    for (const size of sizes) {
        const high = new Float64Array(size);
        const low = new Float64Array(size);
        
        // Fill with test data
        for (let i = 0; i < size; i++) {
            high[i] = 100.0 + i;
            low[i] = 80.0 + i;
        }
        
        const highPtr = wasm.medprice_alloc(size);
        const lowPtr = wasm.medprice_alloc(size);
        const outPtr = wasm.medprice_alloc(size);
        
        try {
            const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
            const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
            
            highView.set(high);
            lowView.set(low);
            
            wasm.medprice_into(highPtr, lowPtr, outPtr, size);
            
            const resultView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, size);
            
            // Verify formula
            for (let i = 0; i < size; i++) {
                const expected = (high[i] + low[i]) / 2.0;
                assertClose(resultView[i], expected, 1e-10,
                    `Size ${size} mismatch at index ${i}`);
            }
        } finally {
            wasm.medprice_free(highPtr, size);
            wasm.medprice_free(lowPtr, size);
            wasm.medprice_free(outPtr, size);
        }
    }
});

// ============================================================================
// Memory Management Tests
// ============================================================================

test('MEDPRICE zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.medprice_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Write pattern to verify memory
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, 
                `Memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.medprice_free(ptr, size);
    }
});

test('MEDPRICE memory stress test', () => {
    // Rapid allocation and deallocation
    const iterations = 100;
    const size = 1000;
    
    for (let iter = 0; iter < iterations; iter++) {
        const ptrs = [];
        
        // Allocate multiple buffers
        for (let i = 0; i < 5; i++) {
            const ptr = wasm.medprice_alloc(size);
            assert(ptr !== 0, `Allocation failed at iteration ${iter}, buffer ${i}`);
            ptrs.push(ptr);
        }
        
        // Free in reverse order
        while (ptrs.length > 0) {
            wasm.medprice_free(ptrs.pop(), size);
        }
    }
});

// ============================================================================
// Performance Tests
// ============================================================================

test('MEDPRICE large dataset', () => {
    // Test with full dataset for performance
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.medprice_js(high, low);
    
    assert.strictEqual(result.length, high.length);
    
    // Verify formula on a sample
    const sampleIdx = Math.floor(high.length / 2);
    if (!isNaN(high[sampleIdx]) && !isNaN(low[sampleIdx])) {
        const expected = (high[sampleIdx] + low[sampleIdx]) / 2.0;
        assertClose(result[sampleIdx], expected, 1e-10);
    }
});

test('MEDPRICE repeated calls consistency', () => {
    // Test consistency with repeated calls
    const high = new Float64Array(100);
    const low = new Float64Array(100);
    
    // Fill with random data
    for (let i = 0; i < 100; i++) {
        high[i] = Math.random() * 100;
        low[i] = high[i] * 0.8; // Low is 80% of high
    }
    
    const results = [];
    for (let i = 0; i < 10; i++) {
        results.push(wasm.medprice_js(high, low));
    }
    
    // All results should be identical
    for (let i = 1; i < results.length; i++) {
        assertArrayClose(results[0], results[i], 0,
            `Result ${i} differs from first result`);
    }
});