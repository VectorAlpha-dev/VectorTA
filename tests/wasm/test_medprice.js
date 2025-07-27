/**
 * WASM binding tests for MEDPRICE indicator.
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

test('MEDPRICE default params', () => {
    // Test with default parameters - mirrors Rust test
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.medprice_js(high, low);
    assert(result instanceof Float64Array);
    assert.strictEqual(result.length, high.length);
});

test('MEDPRICE accuracy', async () => {
    // Test MEDPRICE matches expected values from Rust tests
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.medprice_js(high, low);
    
    assert.strictEqual(result.length, high.length);
    
    // Expected values from Rust test (medprice = (high + low) / 2)
    const expectedLastFive = [59166.0, 59244.5, 59118.0, 59146.5, 58767.5];
    
    const actualLastFive = result.slice(-5);
    
    for (let i = 0; i < 5; i++) {
        assertClose(actualLastFive[i], expectedLastFive[i], 1e-1, 
            `MEDPRICE mismatch at index ${i}`);
    }
    
    // Compare with Rust implementation
    await compareWithRust('medprice', result, 'high,low', { high, low });
});

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

test('MEDPRICE NaN handling', () => {
    // Test MEDPRICE handling of NaN values
    const high = new Float64Array([NaN, 100.0, 110.0]);
    const low = new Float64Array([NaN, 80.0, 90.0]);
    
    const result = wasm.medprice_js(high, low);
    
    assert.strictEqual(result.length, 3);
    assert(isNaN(result[0]));
    assert.strictEqual(result[1], 90.0);
    assert.strictEqual(result[2], 100.0);
});

test('MEDPRICE late NaN handling', () => {
    // Test MEDPRICE handling of late NaN values
    const high = new Float64Array([100.0, 110.0, NaN]);
    const low = new Float64Array([80.0, 90.0, NaN]);
    
    const result = wasm.medprice_js(high, low);
    
    assert.strictEqual(result.length, 3);
    assert.strictEqual(result[0], 90.0);
    assert.strictEqual(result[1], 100.0);
    assert(isNaN(result[2]));
});

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

// Zero-copy / Fast API tests

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

// Memory leak prevention test
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