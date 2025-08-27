/**
 * WASM binding tests for ULTOSC indicator.
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
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('ULTOSC basic calculation', () => {
    // Test with default parameters - mirrors Rust tests
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.ultosc_js(high, low, close, 7, 14, 28);
    assert.strictEqual(result.length, close.length);
    
    // Expected values from Rust tests
    const expectedLastFive = [
        41.25546890298435,
        40.83865967175865,
        48.910324164909625,
        45.43113094857947,
        42.163165136766295,
    ];
    
    // Check last 5 values
    for (let i = 0; i < 5; i++) {
        assertClose(result[result.length - 5 + i], expectedLastFive[i], 1e-8);
    }
});

test('ULTOSC custom parameters', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Test with custom parameters
    const result = wasm.ultosc_js(high, low, close, 5, 10, 20);
    
    assert.strictEqual(result.length, close.length);
    assert.ok(!isNaN(result[result.length - 1])); // Last value should be valid
    
    // First 19 values should be NaN (warmup period for timeperiod3=20)
    for (let i = 0; i < 19; i++) {
        assert.ok(isNaN(result[i]));
    }
});

test('ULTOSC memory allocation/deallocation', () => {
    const size = 1000;
    const ptr = wasm.ultosc_alloc(size);
    assert.ok(ptr !== 0, 'Should allocate non-null pointer');
    
    // Clean up
    wasm.ultosc_free(ptr, size);
});

test('ULTOSC fast API', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate memory for all arrays
    const highPtr = wasm.ultosc_alloc(len);
    const lowPtr = wasm.ultosc_alloc(len);
    const closePtr = wasm.ultosc_alloc(len);
    const outPtr = wasm.ultosc_alloc(len);
    
    try {
        // Create views and copy data
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        
        // Test non-aliasing case
        wasm.ultosc_into(
            highPtr,
            lowPtr,
            closePtr,
            outPtr,
            len,
            7,
            14,
            28
        );
        
        // Read output
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const output = new Float64Array(outView);
        
        // Should match safe API result
        const safeResult = wasm.ultosc_js(high, low, close, 7, 14, 28);
        assertArrayClose(output, safeResult, 1e-10);
    } finally {
        // Clean up
        wasm.ultosc_free(highPtr, len);
        wasm.ultosc_free(lowPtr, len);
        wasm.ultosc_free(closePtr, len);
        wasm.ultosc_free(outPtr, len);
    }
});

test('ULTOSC fast API with aliasing', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Create a copy for comparison
    const closeCopy = new Float64Array(close);
    
    // Allocate memory
    const highPtr = wasm.ultosc_alloc(len);
    const lowPtr = wasm.ultosc_alloc(len);
    const closePtr = wasm.ultosc_alloc(len);
    
    try {
        // Create views and copy data
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        
        // Test aliasing - output overwrites close array
        wasm.ultosc_into(
            highPtr,
            lowPtr,
            closePtr,
            closePtr, // Same as input!
            len,
            7,
            14,
            28
        );
        
        // Read the modified close array
        const modifiedClose = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const result = new Float64Array(modifiedClose);
        
        // Should produce correct results despite aliasing
        const expectedResult = wasm.ultosc_js(high, low, closeCopy, 7, 14, 28);
        assertArrayClose(result, expectedResult, 1e-10);
    } finally {
        // Clean up
        wasm.ultosc_free(highPtr, len);
        wasm.ultosc_free(lowPtr, len);
        wasm.ultosc_free(closePtr, len);
    }
});

test('ULTOSC batch calculation', async () => {
    const high = new Float64Array(testData.high.slice(0, 100)); // Use smaller dataset for batch
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const config = {
        timeperiod1_range: [5, 9, 2],    // 5, 7, 9
        timeperiod2_range: [12, 16, 2],  // 12, 14, 16
        timeperiod3_range: [26, 30, 2]   // 26, 28, 30
    };
    
    const result = await wasm.ultosc_batch(high, low, close, config);
    
    assert.ok(result.values);
    assert.ok(result.combos);
    assert.strictEqual(result.rows, 27); // 3*3*3 combinations
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.values.length, result.rows * result.cols);
    assert.strictEqual(result.combos.length, result.rows);
    
    // Find the row for (7, 14, 28)
    let targetIdx = -1;
    for (let i = 0; i < result.combos.length; i++) {
        const combo = result.combos[i];
        if (combo.timeperiod1 === 7 && combo.timeperiod2 === 14 && combo.timeperiod3 === 28) {
            targetIdx = i;
            break;
        }
    }
    
    assert.ok(targetIdx >= 0, 'Should find (7, 14, 28) combination');
    
    // Extract the row for this combination
    const rowValues = new Float64Array(result.cols);
    for (let j = 0; j < result.cols; j++) {
        rowValues[j] = result.values[targetIdx * result.cols + j];
    }
    
    // Should match single calculation
    const singleResult = wasm.ultosc_js(high, low, close, 7, 14, 28);
    assertArrayClose(rowValues, singleResult, 1e-10);
});

test('ULTOSC error handling - empty data', () => {
    assert.throws(() => {
        wasm.ultosc_js(new Float64Array([]), new Float64Array([]), new Float64Array([]), 7, 14, 28);
    }, /EmptyData|empty/i);
});

test('ULTOSC error handling - mismatched lengths', () => {
    assert.throws(() => {
        wasm.ultosc_js(
            new Float64Array([1, 2]), 
            new Float64Array([0.5, 1.5, 2.5]), 
            new Float64Array([0.8, 1.8]),
            7, 14, 28
        );
    });
});

test('ULTOSC error handling - zero period', () => {
    const high = new Float64Array([10.0, 11.0, 12.0]);
    const low = new Float64Array([9.0, 10.0, 11.0]);
    const close = new Float64Array([9.5, 10.5, 11.5]);
    
    assert.throws(() => {
        wasm.ultosc_js(high, low, close, 0, 14, 28);
    }, /Invalid period/);
});

test('ULTOSC error handling - period exceeding data length', () => {
    const high = new Float64Array([10.0, 11.0, 12.0]);
    const low = new Float64Array([9.0, 10.0, 11.0]);
    const close = new Float64Array([9.5, 10.5, 11.5]);
    
    assert.throws(() => {
        wasm.ultosc_js(high, low, close, 7, 14, 50);
    }, /Period exceeds data length|Invalid periods/i);
});

test('ULTOSC NaN handling', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Insert some NaN values
    high[0] = NaN;
    high[1] = NaN;
    low[0] = NaN;
    low[1] = NaN;
    close[0] = NaN;
    close[1] = NaN;
    
    const result = wasm.ultosc_js(high, low, close, 7, 14, 28);
    
    assert.strictEqual(result.length, close.length);
    // First several values should be NaN due to input NaN and warmup
    for (let i = 0; i < 30; i++) {
        assert.ok(isNaN(result[i]));
    }
});

test('ULTOSC consistency', () => {
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    // Calculate multiple times
    const result1 = wasm.ultosc_js(high, low, close, 7, 14, 28);
    const result2 = wasm.ultosc_js(high, low, close, 7, 14, 28);
    
    // Results should be identical
    assertArrayClose(result1, result2, 0.0);
});

test('ULTOSC edge case - minimum data', () => {
    // Test with exactly the minimum required data
    // Need largest period (28) + 1 because first_valid requires both i-1 and i to be valid
    const size = 29; // Largest period + 1
    const high = new Float64Array(size);
    const low = new Float64Array(size); 
    const close = new Float64Array(size);
    
    // Fill with simple pattern
    for (let i = 0; i < size; i++) {
        high[i] = 10 + i * 0.1;
        low[i] = 9 + i * 0.1;
        close[i] = 9.5 + i * 0.1;
    }
    
    const result = wasm.ultosc_js(high, low, close, 7, 14, 28);
    assert.strictEqual(result.length, size);
    
    // Last value should be valid (not NaN)
    assert.ok(!isNaN(result[size - 1]));
});

// Optional: Compare with Rust implementation if available
test.skip('ULTOSC WASM vs Rust comparison', async () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const wasmResult = wasm.ultosc_js(high, low, close, 7, 14, 28);
    const rustResult = await compareWithRust('ultosc', { high, low, close, timeperiod1: 7, timeperiod2: 14, timeperiod3: 28 });
    
    assertArrayClose(wasmResult, rustResult, 1e-10);
});