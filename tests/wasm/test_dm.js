/**
 * WASM binding tests for DM (Directional Movement) indicator.
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

test('DM partial params', () => {
    // Test with default parameters - mirrors check_dm_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.dm(high, low, 14);
    assert.strictEqual(result.rows, 2); // plus and minus
    assert.strictEqual(result.cols, high.length);
    assert.strictEqual(result.values.length, 2 * high.length);
});

test('DM accuracy', () => {
    // Test DM matches expected values from Rust tests - mirrors check_dm_known_values
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    // Expected values from Rust test check_dm_known_values
    const expectedPlus = [
        1410.819956368491,
        1384.04710234217,
        1285.186595032015,
        1199.3875525297283,
        1113.7170130633192,
    ];
    const expectedMinus = [
        3602.8631384045057,
        3345.5157713756125,
        3258.5503591344973,
        3025.796762053462,
        3493.668421906786,
    ];
    
    const result = wasm.dm(high, low, 14);
    
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, high.length);
    
    // Extract plus and minus from flattened array
    const plus = result.values.slice(0, result.cols);
    const minus = result.values.slice(result.cols, 2 * result.cols);
    
    // Check last 5 values match expected
    const plusLast5 = plus.slice(-5);
    const minusLast5 = minus.slice(-5);
    
    assertArrayClose(
        plusLast5,
        expectedPlus,
        1e-6,
        "DM plus last 5 values mismatch"
    );
    assertArrayClose(
        minusLast5,
        expectedMinus,
        1e-6,
        "DM minus last 5 values mismatch"
    );
});

test('DM default candles', () => {
    // Test DM with default parameters - mirrors check_dm_default_candles
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.dm(high, low, 14);
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, high.length);
});

test('DM zero period', () => {
    // Test DM fails with zero period - mirrors check_dm_zero_period
    const highData = new Float64Array([100.0, 110.0, 120.0]);
    const lowData = new Float64Array([90.0, 100.0, 110.0]);
    
    assert.throws(() => {
        wasm.dm(highData, lowData, 0);
    }, /Invalid period/);
});

test('DM period exceeds length', () => {
    // Test DM fails when period exceeds data length - mirrors check_dm_period_exceeds_length
    const highSmall = new Float64Array([100.0, 110.0, 120.0]);
    const lowSmall = new Float64Array([90.0, 100.0, 110.0]);
    
    assert.throws(() => {
        wasm.dm(highSmall, lowSmall, 10);
    }, /Invalid period/);
});

test('DM very small dataset', () => {
    // Test DM fails with insufficient data - mirrors check_dm_period_exceeds_length
    const singleHigh = new Float64Array([42.0]);
    const singleLow = new Float64Array([40.0]);
    
    assert.throws(() => {
        wasm.dm(singleHigh, singleLow, 14);
    }, /Invalid period/);
});

test('DM empty input', () => {
    // Test DM fails with empty input - mirrors check_dm_all_values_nan
    const emptyHigh = new Float64Array([]);
    const emptyLow = new Float64Array([]);
    
    assert.throws(() => {
        wasm.dm(emptyHigh, emptyLow, 14);
    }, /Input data is empty|Empty data/i);
});

test('DM all NaN input', () => {
    // Test DM with all NaN values - mirrors check_dm_all_values_nan
    const allNaNHigh = new Float64Array(100);
    const allNaNLow = new Float64Array(100);
    allNaNHigh.fill(NaN);
    allNaNLow.fill(NaN);
    
    assert.throws(() => {
        wasm.dm(allNaNHigh, allNaNLow, 14);
    }, /All values are NaN/);
});

test('DM NaN handling', () => {
    // Test DM handles NaN values correctly
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.dm(high, low, 14);
    
    // Extract plus and minus
    const plus = result.values.slice(0, result.cols);
    const minus = result.values.slice(result.cols, 2 * result.cols);
    
    // First period-1 values should be NaN (warmup period)
    // For DM, warmup = first_valid_idx + period - 1
    // Since our test data doesn't have leading NaN, first_valid_idx = 0
    // So warmup = 0 + 14 - 1 = 13
    assertAllNaN(plus.slice(0, 13), "Expected NaN in warmup period for plus");
    assertAllNaN(minus.slice(0, 13), "Expected NaN in warmup period for minus");
    
    // After warmup period (240), no NaN values should exist
    if (result.cols > 240) {
        for (let i = 240; i < result.cols; i++) {
            assert(!isNaN(plus[i]), `Found unexpected NaN at index ${i} in plus`);
            assert(!isNaN(minus[i]), `Found unexpected NaN at index ${i} in minus`);
        }
    }
});

test('DM batch single parameter set', () => {
    // Test batch with single parameter combination
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const batchResult = wasm.dm_batch(high, low, {
        period_range: [14, 14, 0]
    });
    
    // Should have 2 rows (plus and minus) for 1 period
    assert.strictEqual(batchResult.rows, 2);
    assert.strictEqual(batchResult.cols, high.length);
    assert.strictEqual(batchResult.periods.length, 1);
    assert.strictEqual(batchResult.periods[0], 14);
    
    // Extract plus and minus for the single period
    const plus = batchResult.values.slice(0, batchResult.cols);
    const minus = batchResult.values.slice(batchResult.cols, 2 * batchResult.cols);
    
    // Should match single calculation
    const singleResult = wasm.dm(high, low, 14);
    const singlePlus = singleResult.values.slice(0, singleResult.cols);
    const singleMinus = singleResult.values.slice(singleResult.cols, 2 * singleResult.cols);
    
    assertArrayClose(plus, singlePlus, 1e-10, "Batch vs single plus mismatch");
    assertArrayClose(minus, singleMinus, 1e-10, "Batch vs single minus mismatch");
});

test('DM batch multiple periods', () => {
    // Test batch with multiple period values
    const high = new Float64Array(testData.high.slice(0, 100)); // Use smaller dataset for speed
    const low = new Float64Array(testData.low.slice(0, 100));
    
    // Multiple periods: 10, 15, 20
    const batchResult = wasm.dm_batch(high, low, {
        period_range: [10, 20, 5] // 10, 15, 20
    });
    
    // Should have 2 * 3 = 6 rows (plus and minus for each period)
    assert.strictEqual(batchResult.rows, 6);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.periods.length, 3);
    assert.deepStrictEqual(batchResult.periods, [10, 15, 20]);
    
    // Verify each period matches individual calculation
    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        // Extract plus and minus for this period from batch result
        // Layout: [plus_p1, plus_p2, plus_p3, minus_p1, minus_p2, minus_p3]
        const plusStart = i * 100;
        const minusStart = (3 + i) * 100;
        const batchPlus = batchResult.values.slice(plusStart, plusStart + 100);
        const batchMinus = batchResult.values.slice(minusStart, minusStart + 100);
        
        // Compare with single calculation
        const singleResult = wasm.dm(high, low, periods[i]);
        const singlePlus = singleResult.values.slice(0, 100);
        const singleMinus = singleResult.values.slice(100, 200);
        
        assertArrayClose(
            batchPlus, 
            singlePlus, 
            1e-10, 
            `Period ${periods[i]} plus mismatch`
        );
        assertArrayClose(
            batchMinus, 
            singleMinus, 
            1e-10, 
            `Period ${periods[i]} minus mismatch`
        );
    }
});

test('DM batch metadata', () => {
    // Test that batch result includes correct parameter combinations
    const high = new Float64Array(20);
    const low = new Float64Array(20);
    high.fill(100);
    low.fill(90);
    
    const result = wasm.dm_batch(high, low, {
        period_range: [5, 10, 5] // periods: 5, 10
    });
    
    // Should have 2 periods
    assert.strictEqual(result.periods.length, 2);
    assert.strictEqual(result.periods[0], 5);
    assert.strictEqual(result.periods[1], 10);
    
    // Should have 2 * 2 = 4 rows (plus and minus for each period)
    assert.strictEqual(result.rows, 4);
    assert.strictEqual(result.cols, 20);
    assert.strictEqual(result.values.length, 4 * 20);
});

test('DM batch edge cases', () => {
    // Test edge cases for batch processing
    const high = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const low = new Float64Array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]);
    
    // Single value sweep
    const singleBatch = wasm.dm_batch(high, low, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleBatch.rows, 2);
    assert.strictEqual(singleBatch.periods.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.dm_batch(high, low, {
        period_range: [5, 7, 10] // Step larger than range
    });
    
    // Should only have period=5
    assert.strictEqual(largeBatch.rows, 2);
    assert.strictEqual(largeBatch.periods.length, 1);
    assert.strictEqual(largeBatch.periods[0], 5);
});

test('DM output structure', () => {
    // Verify the flattened array structure: [plus..., minus...]
    const high = new Float64Array([100, 110, 105, 115, 120, 110, 125, 130, 120, 135]);
    const low = new Float64Array([95, 100, 95, 105, 110, 100, 115, 120, 110, 125]);
    
    const result = wasm.dm(high, low, 3);
    
    // Verify structure
    assert.strictEqual(result.rows, 2, 'Should have 2 rows (plus and minus)');
    assert.strictEqual(result.cols, 10, 'Should have 10 columns (data length)');
    assert.strictEqual(result.values.length, 20, 'Should have 20 values total');
    
    // Extract plus and minus
    const plus = result.values.slice(0, 10);
    const minus = result.values.slice(10, 20);
    
    // Check warmup period (first 2 values should be NaN for period=3)
    assert(isNaN(plus[0]), 'Plus[0] should be NaN');
    assert(isNaN(plus[1]), 'Plus[1] should be NaN');
    assert(isNaN(minus[0]), 'Minus[0] should be NaN');
    assert(isNaN(minus[1]), 'Minus[1] should be NaN');
    
    // After warmup should have values
    assert(!isNaN(plus[2]), 'Plus[2] should have value');
    assert(!isNaN(minus[2]), 'Minus[2] should have value');
});

test('DM mismatched lengths', () => {
    // Test DM fails when high/low lengths differ
    const high = new Float64Array([100.0, 110.0, 120.0, 130.0]);
    const low = new Float64Array([90.0, 100.0, 110.0]); // Different length
    
    assert.throws(() => {
        wasm.dm(high, low, 2);
    }, /length mismatch/);
});

// Zero-copy API tests
test('DM zero-copy API', () => {
    const high = new Float64Array([100, 110, 105, 115, 120, 110, 125, 130, 120, 135]);
    const low = new Float64Array([95, 100, 95, 105, 110, 100, 115, 120, 110, 125]);
    const period = 5;
    
    // Allocate buffers for plus and minus
    const plusPtr = wasm.dm_alloc(high.length);
    const minusPtr = wasm.dm_alloc(low.length);
    assert(plusPtr !== 0, 'Failed to allocate plus memory');
    assert(minusPtr !== 0, 'Failed to allocate minus memory');
    
    // Create views into WASM memory
    const memoryBuffer = wasm.__wasm.memory.buffer;
    const highPtr = wasm.dm_alloc(high.length);
    const lowPtr = wasm.dm_alloc(low.length);
    
    const highView = new Float64Array(memoryBuffer, highPtr, high.length);
    const lowView = new Float64Array(memoryBuffer, lowPtr, low.length);
    const plusView = new Float64Array(memoryBuffer, plusPtr, high.length);
    const minusView = new Float64Array(memoryBuffer, minusPtr, low.length);
    
    // Copy data into WASM memory
    highView.set(high);
    lowView.set(low);
    
    try {
        // Compute DM using zero-copy API
        wasm.dm_into(highPtr, lowPtr, plusPtr, minusPtr, high.length, period);
        
        // Verify results match regular API
        const regularResult = wasm.dm(high, low, period);
        const regularPlus = regularResult.values.slice(0, high.length);
        const regularMinus = regularResult.values.slice(high.length, 2 * high.length);
        
        for (let i = 0; i < high.length; i++) {
            if (isNaN(regularPlus[i]) && isNaN(plusView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularPlus[i] - plusView[i]) < 1e-10,
                   `Plus zero-copy mismatch at index ${i}: regular=${regularPlus[i]}, zerocopy=${plusView[i]}`);
            assert(Math.abs(regularMinus[i] - minusView[i]) < 1e-10,
                   `Minus zero-copy mismatch at index ${i}: regular=${regularMinus[i]}, zerocopy=${minusView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.dm_free(highPtr, high.length);
        wasm.dm_free(lowPtr, low.length);
        wasm.dm_free(plusPtr, high.length);
        wasm.dm_free(minusPtr, low.length);
    }
});

test('DM zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.dm_into(0, 0, 0, 0, 10, 5);
    }, /null pointer/i);
    
    // Test invalid parameters with allocated memory
    const ptr1 = wasm.dm_alloc(10);
    const ptr2 = wasm.dm_alloc(10);
    const ptr3 = wasm.dm_alloc(10);
    const ptr4 = wasm.dm_alloc(10);
    
    try {
        // Invalid period
        assert.throws(() => {
            wasm.dm_into(ptr1, ptr2, ptr3, ptr4, 10, 0);
        }, /Invalid period/);
    } finally {
        wasm.dm_free(ptr1, 10);
        wasm.dm_free(ptr2, 10);
        wasm.dm_free(ptr3, 10);
        wasm.dm_free(ptr4, 10);
    }
});

test('DM zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr1 = wasm.dm_alloc(size);
        const ptr2 = wasm.dm_alloc(size);
        assert(ptr1 !== 0, `Failed to allocate ${size} elements for plus`);
        assert(ptr2 !== 0, `Failed to allocate ${size} elements for minus`);
        
        // Write pattern to verify memory
        const memoryBuffer = wasm.__wasm.memory.buffer;
        const view1 = new Float64Array(memoryBuffer, ptr1, size);
        const view2 = new Float64Array(memoryBuffer, ptr2, size);
        
        for (let i = 0; i < Math.min(10, size); i++) {
            view1[i] = i * 1.5;
            view2[i] = i * 2.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(view1[i], i * 1.5, `Memory corruption in plus at index ${i}`);
            assert.strictEqual(view2[i], i * 2.5, `Memory corruption in minus at index ${i}`);
        }
        
        // Free memory
        wasm.dm_free(ptr1, size);
        wasm.dm_free(ptr2, size);
    }
});

test('DM with slice data', () => {
    // Test DM with slice data - mirrors check_dm_with_slice_data
    const highValues = new Float64Array([8000.0, 8050.0, 8100.0, 8075.0, 8110.0, 8050.0]);
    const lowValues = new Float64Array([7800.0, 7900.0, 7950.0, 7950.0, 8000.0, 7950.0]);
    
    const result = wasm.dm(highValues, lowValues, 3);
    
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, 6);
    
    const plus = result.values.slice(0, 6);
    const minus = result.values.slice(6, 12);
    
    // First period-1 values should be NaN
    for (let i = 0; i < 2; i++) {
        assert(isNaN(plus[i]), `Plus[${i}] should be NaN`);
        assert(isNaN(minus[i]), `Minus[${i}] should be NaN`);
    }
});

test.after(() => {
    console.log('DM WASM tests completed');
});
