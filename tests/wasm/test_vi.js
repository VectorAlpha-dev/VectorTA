/**
 * WASM binding tests for VI (Vortex Indicator).
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

test('VI partial params', () => {
    // Test with default parameters - mirrors check_vi_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.vi_js(high, low, close, 14);
    
    assert(result.plus, 'Result should have plus array');
    assert(result.minus, 'Result should have minus array');
    assert.strictEqual(result.plus.length, high.length);
    assert.strictEqual(result.minus.length, high.length);
});

test('VI accuracy', async () => {
    // Test VI matches expected values from Rust tests - mirrors check_vi_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.vi_js(high, low, close, 14);
    
    assert.strictEqual(result.plus.length, high.length);
    assert.strictEqual(result.minus.length, high.length);
    
    // Expected values from Rust tests
    const expected_last_five_plus = [
        0.9970238095238095,
        0.9871071716357775,
        0.9464453759945247,
        0.890897412369242,
        0.9206478557604156,
    ];
    const expected_last_five_minus = [
        1.0097117794486214,
        1.04174053182917,
        1.1152365471811105,
        1.181684712791338,
        1.1894672506875827,
    ];
    
    // Check last 5 values match expected
    const last5Plus = result.plus.slice(-5);
    const last5Minus = result.minus.slice(-5);
    
    assertArrayClose(
        last5Plus,
        expected_last_five_plus,
        1e-8,
        "VI plus last 5 values mismatch"
    );
    assertArrayClose(
        last5Minus,
        expected_last_five_minus,
        1e-8,
        "VI minus last 5 values mismatch"
    );
});

test('VI default candles', () => {
    // Test VI with default parameters - mirrors check_vi_default_candles
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Default params: period=14
    const result = wasm.vi_js(high, low, close, 14);
    assert.strictEqual(result.plus.length, high.length);
    assert.strictEqual(result.minus.length, high.length);
});

test('VI zero period', () => {
    // Test VI fails with zero period - mirrors check_vi_zero_period
    const high = new Float64Array([10.0, 11.0, 12.0]);
    const low = new Float64Array([9.0, 10.0, 11.0]);
    const close = new Float64Array([9.5, 10.5, 11.5]);
    
    assert.throws(() => {
        wasm.vi_js(high, low, close, 0);
    }, /Invalid period/);
});

test('VI period exceeds data', () => {
    // Test VI fails when period exceeds data length - mirrors check_vi_period_exceeds_length
    const high = new Float64Array([10.0, 11.0, 12.0]);
    const low = new Float64Array([9.0, 10.0, 11.0]);
    const close = new Float64Array([9.5, 10.5, 11.5]);
    
    assert.throws(() => {
        wasm.vi_js(high, low, close, 10);
    }, /Invalid period/);
});

test('VI very small dataset', () => {
    // Test VI fails with insufficient data - mirrors check_vi_very_small_data_set
    const high = new Float64Array([42.0]);
    const low = new Float64Array([41.0]);
    const close = new Float64Array([41.5]);
    
    assert.throws(() => {
        wasm.vi_js(high, low, close, 14);
    }, /Invalid period/);
});

test('VI NaN handling', () => {
    // Test VI handles NaN values correctly - mirrors check_vi_nan_handling
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.vi_js(high, low, close, 14);
    assert.strictEqual(result.plus.length, high.length);
    assert.strictEqual(result.minus.length, high.length);
    
    // Check that NaN values are handled (should not crash)
    // First few values should be NaN due to warmup period
    assert(isNaN(result.plus[0]), 'First plus value should be NaN');
    assert(isNaN(result.minus[0]), 'First minus value should be NaN');
    
    // After warmup period (14-1=13), values should not be NaN
    if (result.plus.length > 20) {
        for (let i = 20; i < Math.min(result.plus.length, 240); i++) {
            assert(!isNaN(result.plus[i]), `Found unexpected NaN in plus at index ${i}`);
            assert(!isNaN(result.minus[i]), `Found unexpected NaN in minus at index ${i}`);
        }
    }
});

test('VI empty input', () => {
    // Test VI fails with empty input - mirrors check for empty data
    const high = new Float64Array([]);
    const low = new Float64Array([]);
    const close = new Float64Array([]);
    
    assert.throws(() => {
        wasm.vi_js(high, low, close, 14);
    }, /Empty data/);
});

test('VI mismatched lengths', () => {
    // Test VI fails with mismatched input lengths
    const high = new Float64Array([10.0, 11.0, 12.0]);
    const low = new Float64Array([9.0, 10.0]);  // Different length
    const close = new Float64Array([9.5, 10.5, 11.5]);
    
    assert.throws(() => {
        wasm.vi_js(high, low, close, 2);
    }, /Empty data/);  // Will be caught by length mismatch check
});

test('VI batch single parameter set', () => {
    // Test batch with single parameter combination
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Using the new ergonomic batch API for single parameter
    const batchResult = wasm.vi_batch(high, low, close, {
        period_range: [14, 14, 0]
    });
    
    // Should match single calculation
    const singleResult = wasm.vi_js(high, low, close, 14);
    
    assert.strictEqual(batchResult.plus.length, singleResult.plus.length);
    assert.strictEqual(batchResult.minus.length, singleResult.minus.length);
    assertArrayClose(batchResult.plus, singleResult.plus, 1e-10, "Batch vs single plus mismatch");
    assertArrayClose(batchResult.minus, singleResult.minus, 1e-10, "Batch vs single minus mismatch");
});

test('VI batch multiple periods', () => {
    // Test batch with multiple period values
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Multiple periods: 10, 12, 14, 16, 18, 20
    const batchResult = wasm.vi_batch(high, low, close, {
        period_range: [10, 20, 2]
    });
    
    // Should have 6 rows * 100 cols = 600 values per output
    assert.strictEqual(batchResult.plus.length, 6 * 100);
    assert.strictEqual(batchResult.minus.length, 6 * 100);
    assert.strictEqual(batchResult.rows, 6);
    assert.strictEqual(batchResult.cols, 100);
    
    // Check that periods are correct
    const expectedPeriods = [10, 12, 14, 16, 18, 20];
    assert.deepStrictEqual(batchResult.periods, expectedPeriods);
    
    // Verify each row matches individual calculation
    for (let i = 0; i < expectedPeriods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowDataPlus = batchResult.plus.slice(rowStart, rowEnd);
        const rowDataMinus = batchResult.minus.slice(rowStart, rowEnd);
        
        const singleResult = wasm.vi_js(high, low, close, expectedPeriods[i]);
        assertArrayClose(
            rowDataPlus, 
            singleResult.plus, 
            1e-10, 
            `Period ${expectedPeriods[i]} plus mismatch`
        );
        assertArrayClose(
            rowDataMinus, 
            singleResult.minus, 
            1e-10, 
            `Period ${expectedPeriods[i]} minus mismatch`
        );
    }
});

test('VI batch edge cases', () => {
    // Test edge cases for batch processing
    const high = new Float64Array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    const low = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const close = new Float64Array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]);
    
    // Single value sweep
    const singleBatch = wasm.vi_batch(high, low, close, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleBatch.plus.length, 10);
    assert.strictEqual(singleBatch.minus.length, 10);
    assert.strictEqual(singleBatch.periods.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.vi_batch(high, low, close, {
        period_range: [5, 7, 10] // Step larger than range
    });
    
    // Should only have period=5
    assert.strictEqual(largeBatch.plus.length, 10);
    assert.strictEqual(largeBatch.minus.length, 10);
    assert.strictEqual(largeBatch.periods.length, 1);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.vi_batch(new Float64Array([]), new Float64Array([]), new Float64Array([]), {
            period_range: [14, 14, 0]
        });
    }, /Empty data/);
});

// Zero-copy API tests
test('VI zero-copy API', () => {
    const high = new Float64Array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    const low = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const close = new Float64Array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]);
    const period = 5;
    
    // Allocate buffers for outputs
    const plusPtr = wasm.vi_alloc(high.length);
    const minusPtr = wasm.vi_alloc(high.length);
    
    assert(plusPtr !== 0, 'Failed to allocate plus memory');
    assert(minusPtr !== 0, 'Failed to allocate minus memory');
    
    // Create views into WASM memory for inputs
    const highPtr = wasm.vi_alloc(high.length);
    const lowPtr = wasm.vi_alloc(high.length);
    const closePtr = wasm.vi_alloc(high.length);
    
    try {
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, high.length);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, low.length);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, close.length);
        
        // Copy data into WASM memory
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        
        // Compute VI
        wasm.vi_into(highPtr, lowPtr, closePtr, plusPtr, minusPtr, high.length, period);
        
        // Create views for outputs
        const plusView = new Float64Array(wasm.__wasm.memory.buffer, plusPtr, high.length);
        const minusView = new Float64Array(wasm.__wasm.memory.buffer, minusPtr, high.length);
        
        // Verify results match regular API
        const regularResult = wasm.vi_js(high, low, close, period);
        for (let i = 0; i < high.length; i++) {
            if (isNaN(regularResult.plus[i]) && isNaN(plusView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult.plus[i] - plusView[i]) < 1e-10,
                   `Zero-copy plus mismatch at index ${i}`);
            assert(Math.abs(regularResult.minus[i] - minusView[i]) < 1e-10,
                   `Zero-copy minus mismatch at index ${i}`);
        }
    } finally {
        // Always free memory
        wasm.vi_free(highPtr, high.length);
        wasm.vi_free(lowPtr, low.length);
        wasm.vi_free(closePtr, close.length);
        wasm.vi_free(plusPtr, high.length);
        wasm.vi_free(minusPtr, high.length);
    }
});

test('VI zero-copy with aliasing', () => {
    // Test aliasing detection
    const high = new Float64Array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    const low = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const close = new Float64Array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]);
    const period = 5;
    
    // Use high buffer as output (aliasing)
    const highPtr = wasm.vi_alloc(high.length);
    const lowPtr = wasm.vi_alloc(low.length);
    const closePtr = wasm.vi_alloc(close.length);
    
    try {
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, high.length);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, low.length);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, close.length);
        
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        
        // Use highPtr as plus output (aliasing!)
        wasm.vi_into(highPtr, lowPtr, closePtr, highPtr, lowPtr, high.length, period);
        
        // Should still produce correct results
        const regularResult = wasm.vi_js(high, low, close, period);
        const plusView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, high.length);
        const minusView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, low.length);
        
        // Plus output should match despite aliasing
        for (let i = 0; i < high.length; i++) {
            if (isNaN(regularResult.plus[i]) && isNaN(plusView[i])) {
                continue;
            }
            assert(Math.abs(regularResult.plus[i] - plusView[i]) < 1e-10,
                   `Aliasing plus mismatch at index ${i}`);
        }
    } finally {
        wasm.vi_free(highPtr, high.length);
        wasm.vi_free(lowPtr, low.length);
        wasm.vi_free(closePtr, close.length);
    }
});

test('VI zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.vi_into(0, 0, 0, 0, 0, 10, 14);
    }, /null pointer|Null pointer/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.vi_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.vi_into(ptr, ptr, ptr, ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        // Period exceeds length
        assert.throws(() => {
            wasm.vi_into(ptr, ptr, ptr, ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.vi_free(ptr, 10);
    }
});

test('VI zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const plusPtr = wasm.vi_alloc(size);
        const minusPtr = wasm.vi_alloc(size);
        
        assert(plusPtr !== 0, `Failed to allocate plus ${size} elements`);
        assert(minusPtr !== 0, `Failed to allocate minus ${size} elements`);
        
        // Write pattern to verify memory
        const plusView = new Float64Array(wasm.__wasm.memory.buffer, plusPtr, size);
        const minusView = new Float64Array(wasm.__wasm.memory.buffer, minusPtr, size);
        
        for (let i = 0; i < Math.min(10, size); i++) {
            plusView[i] = i * 1.5;
            minusView[i] = i * 2.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(plusView[i], i * 1.5, `Plus memory corruption at index ${i}`);
            assert.strictEqual(minusView[i], i * 2.5, `Minus memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.vi_free(plusPtr, size);
        wasm.vi_free(minusPtr, size);
    }
});

test.after(() => {
    console.log('VI WASM tests completed');
});