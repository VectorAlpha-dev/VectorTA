/**
 * WASM binding tests for MAB (Moving Average Bands) indicator.
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

test('MAB partial params', () => {
    // Test with default parameters - mirrors check_mab_partial_params
    const close = new Float64Array(testData.close);
    
    // Returns flattened array [upper..., middle..., lower...]
    const result = wasm.mab_js(close, 10, 50, 1.0, 1.0, "sma", "sma");
    assert.strictEqual(result.length, close.length * 3);
    
    // Split back into bands
    const upper = result.slice(0, close.length);
    const middle = result.slice(close.length, close.length * 2);
    const lower = result.slice(close.length * 2);
    
    assert.strictEqual(upper.length, close.length);
    assert.strictEqual(middle.length, close.length);
    assert.strictEqual(lower.length, close.length);
});

test('MAB accuracy', async () => {
    // Test MAB matches expected values from Rust tests - mirrors check_mab_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.mab_js(close, 10, 50, 1.0, 1.0, "sma", "sma");
    
    // Split flattened result
    const upper = result.slice(0, close.length);
    const middle = result.slice(close.length, close.length * 2);
    const lower = result.slice(close.length * 2);
    
    // Expected values from Rust unit tests
    const expectedUpperLast5 = [
        64002.843463352016,
        63976.62699738246,
        63949.00496307154,
        63912.13708526151,
        63828.40371728143,
    ];
    const expectedMiddleLast5 = [
        59213.89999999991,
        59180.79999999991,
        59161.39999999991,
        59131.99999999991,
        59042.39999999991,
    ];
    const expectedLowerLast5 = [
        59350.676536647945,
        59296.93300261751,
        59252.75503692843,
        59190.30291473845,
        59070.11628271853,
    ];
    
    // Check last 5 values match expected
    assertArrayClose(
        upper.slice(-5),
        expectedUpperLast5,
        1e-6,
        "MAB upper band last 5 values mismatch"
    );
    assertArrayClose(
        middle.slice(-5),
        expectedMiddleLast5,
        1e-6,
        "MAB middle band last 5 values mismatch"
    );
    assertArrayClose(
        lower.slice(-5),
        expectedLowerLast5,
        1e-6,
        "MAB lower band last 5 values mismatch"
    );
    
    // Note: MAB is not in the reference generator yet
    // Skipping compareWithRust for MAB
});

test('MAB default candles', () => {
    // Test MAB with default parameters - mirrors check_mab_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.mab_js(close, 10, 50, 1.0, 1.0, "sma", "sma");
    assert.strictEqual(result.length, close.length * 3);
});

test('MAB zero period', () => {
    // Test MAB fails with zero period - mirrors check_mab_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.mab_js(inputData, 0, 5, 1.0, 1.0, "sma", "sma");
    }, /Invalid period/);
});

test('MAB period exceeds length', () => {
    // Test MAB fails when period exceeds data length - mirrors check_mab_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.mab_js(dataSmall, 2, 10, 1.0, 1.0, "sma", "sma");
    }, /Invalid period/);
});

test('MAB very small dataset', () => {
    // Test MAB fails with insufficient data - mirrors check_mab_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.mab_js(singlePoint, 10, 20, 1.0, 1.0, "sma", "sma");
    }, /Invalid period|Not enough valid data/);
});

test('MAB all NaN', () => {
    // Test MAB fails with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.mab_js(allNaN, 10, 50, 1.0, 1.0, "sma", "sma");
    }, /All values are NaN/);
});

test('MAB empty input', () => {
    // Test MAB fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.mab_js(empty, 10, 50, 1.0, 1.0, "sma", "sma");
    }, /Input data slice is empty|EmptyData/);
});

test('MAB NaN handling', () => {
    // Test MAB NaN handling - mirrors check_mab_nan_handling
    const close = new Float64Array(testData.close);
    const fastPeriod = 10;
    const slowPeriod = 50;
    
    const result = wasm.mab_js(close, fastPeriod, slowPeriod, 1.0, 1.0, "sma", "sma");
    const upper = result.slice(0, close.length);
    const middle = result.slice(close.length, close.length * 2);
    const lower = result.slice(close.length * 2);
    
    // MAB warmup behavior (after fix):
    // - NaN values from 0 to 58 (inclusive)
    // - Real values start at 59
    // Calculation: warmup = first + max(fast, slow) + fast - 2
    //              warmup = 0 + 50 + 10 - 2 = 58
    const warmupLastNaN = Math.max(fastPeriod, slowPeriod) + fastPeriod - 2;  // 58
    const realValuesStart = warmupLastNaN + 1;                                // 59
    
    // Check NaN values up to index 58 (inclusive)
    for (let i = 0; i < Math.min(realValuesStart, upper.length); i++) {
        assert(isNaN(upper[i]), `Expected NaN at index ${i}`);
        assert(isNaN(middle[i]), `Expected NaN at index ${i}`);
        assert(isNaN(lower[i]), `Expected NaN at index ${i}`);
    }
    
    // After index 59, should have real non-zero values
    if (upper.length > realValuesStart) {
        for (let i = realValuesStart; i < Math.min(realValuesStart + 10, upper.length); i++) {
            assert(!isNaN(upper[i]), `Unexpected NaN at index ${i}`);
            assert(!isNaN(middle[i]), `Unexpected NaN at index ${i}`);
            assert(!isNaN(lower[i]), `Unexpected NaN at index ${i}`);
            // Verify they're not zero (real values)
            assert(Math.abs(upper[i]) > 1e-10, `Expected non-zero value at index ${i}`);
            assert(Math.abs(middle[i]) > 1e-10, `Expected non-zero value at index ${i}`);
            assert(Math.abs(lower[i]) > 1e-10, `Expected non-zero value at index ${i}`);
        }
    }
});

test('MAB batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Using the new ergonomic batch API for single parameter
    const batchResult = wasm.mab_batch(close, {
        fast_period_range: [10, 10, 0],
        slow_period_range: [50, 50, 0],
        devup_range: [1.0, 1.0, 0.0],
        devdn_range: [1.0, 1.0, 0.0],
        fast_ma_type: "sma",
        slow_ma_type: "sma"
    });
    
    // Verify structure
    assert(batchResult.upperbands, 'Should have upperbands array');
    assert(batchResult.middlebands, 'Should have middlebands array');
    assert(batchResult.lowerbands, 'Should have lowerbands array');
    assert(batchResult.combos, 'Should have combos array');
    assert(typeof batchResult.rows === 'number', 'Should have rows count');
    assert(typeof batchResult.cols === 'number', 'Should have cols count');
    
    // Should have 1 row
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, close.length);
    assert.strictEqual(batchResult.combos.length, 1);
    
    // Expected values from actual implementation
    const expectedUpper = [
        64002.843463352016,
        63976.62699738246,
        63949.00496307154,
        63912.13708526151,
        63828.40371728143,
    ];
    const expectedMiddle = [
        59213.90000000002,
        59180.800000000025,
        59161.40000000002,
        59132.00000000002,
        59042.40000000002,
    ];
    const expectedLower = [
        59350.676536647945,
        59296.93300261751,
        59252.75503692843,
        59190.30291473845,
        59070.11628271853,
    ];
    
    assertArrayClose(
        batchResult.upperbands.slice(-5),
        expectedUpper,
        1e-6,
        "MAB batch upper band mismatch"
    );
    assertArrayClose(
        batchResult.middlebands.slice(-5),
        expectedMiddle,
        1e-6,
        "MAB batch middle band mismatch"
    );
    assertArrayClose(
        batchResult.lowerbands.slice(-5),
        expectedLower,
        1e-6,
        "MAB batch lower band mismatch"
    );
});

test('MAB batch multiple periods', () => {
    // Test batch with multiple periods
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Test batch with multiple fast periods
    const batchResult = wasm.mab_batch(close, {
        fast_period_range: [10, 15, 5],  // 10, 15
        slow_period_range: [50, 50, 0],  // 50
        devup_range: [1.0, 2.0, 0.5],    // 1.0, 1.5, 2.0
        devdn_range: [1.0, 1.0, 0],      // 1.0
        fast_ma_type: "sma",
        slow_ma_type: "sma"
    });
    
    // Should have 2 * 1 * 3 * 1 = 6 rows
    assert.strictEqual(batchResult.rows, 6);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.combos.length, 6);
    
    // Verify each row has appropriate NaN prefix
    for (let i = 0; i < batchResult.rows; i++) {
        const rowStart = i * 100;
        const upperRow = batchResult.upperbands.slice(rowStart, rowStart + 100);
        const fastP = batchResult.combos[i].fast_period;
        const slowP = batchResult.combos[i].slow_period;
        // After fix: NaN values up to max(fast, slow) + fast - 2
        // First non-NaN appears at max(fast, slow) + fast - 1
        const firstNonNaN = Math.max(fastP, slowP) + fastP - 1;
        
        // Find first non-NaN value
        let firstValid = -1;
        for (let j = 0; j < upperRow.length; j++) {
            if (!isNaN(upperRow[j])) {
                firstValid = j;
                break;
            }
        }
        
        // Should have NaN values before firstNonNaN
        assert.strictEqual(firstValid, firstNonNaN, 
            `Row ${i}: first non-NaN at ${firstValid}, expected ${firstNonNaN} (fast=${fastP}, slow=${slowP})`);
    }
});

test('MAB different MA types', () => {
    // Test MAB with different moving average types
    const close = new Float64Array(testData.close);
    
    // Test with EMA - returns flattened array [upper..., middle..., lower...]
    const resultEMA = wasm.mab_js(close, 10, 50, 1.0, 1.0, "ema", "ema");
    assert.strictEqual(resultEMA.length, close.length * 3, 'Should have 3 bands flattened');
    
    // Test with mixed types
    const resultMixed = wasm.mab_js(close, 10, 50, 1.0, 1.0, "sma", "ema");
    assert.strictEqual(resultMixed.length, close.length * 3, 'Should have 3 bands flattened');
    
    // Results should be different
    const upperEMA = resultEMA.slice(0, close.length);
    const upperMixed = resultMixed.slice(0, close.length);
    
    // Check a few values after warmup
    let foundDifference = false;
    for (let i = 100; i < 110; i++) {
        if (Math.abs(upperEMA[i] - upperMixed[i]) > 1e-10) {
            foundDifference = true;
            break;
        }
    }
    assert(foundDifference, "Expected different values for different MA types");
});

test('MAB parameter boundaries', () => {
    // Test MAB with boundary values for devup/devdn parameters
    const close = new Float64Array(testData.close);
    
    // Test with zero deviations
    const resultZero = wasm.mab_js(close, 10, 50, 0.0, 0.0, "sma", "sma");
    assert.strictEqual(resultZero.length, close.length * 3, 'Should have 3 bands flattened');
    const upperZero = resultZero.slice(0, close.length);
    const middleZero = resultZero.slice(close.length, close.length * 2);
    const lowerZero = resultZero.slice(close.length * 2);
    
    // With zero deviations, upper and lower should equal each other (both equal fast MA)
    // They collapse to the fast MA, not the middle (slow MA)
    for (let i = 100; i < 110; i++) {
        if (!isNaN(upperZero[i])) {
            assertClose(upperZero[i], lowerZero[i], 1e-10,
                       `Upper should equal lower with devup=devdn=0 at index ${i}`);
            // Verify they are different from middle (slow MA)
            assert(Math.abs(upperZero[i] - middleZero[i]) > 1e-10,
                  `Upper/lower should NOT equal middle at index ${i}`);
        }
    }
    
    // Test with large deviations
    const resultLarge = wasm.mab_js(close, 10, 50, 5.0, 5.0, "sma", "sma");
    const upperLarge = resultLarge.slice(0, close.length);
    const lowerLarge = resultLarge.slice(close.length * 2);
    
    // Bands should be much wider - check that at least most are significantly wider
    let widerCount = 0;
    for (let i = 100; i < 110; i++) {
        if (!isNaN(upperLarge[i])) {
            const bandWidthLarge = upperLarge[i] - lowerLarge[i];
            const bandWidthNormal = 500; // Conservative expected width with dev=1.0
            if (bandWidthLarge > bandWidthNormal * 4) {
                widerCount++;
            }
        }
    }
    assert(widerCount >= 5, `Expected at least 5 indices with significantly wider bands, got ${widerCount}`);
    
    // Test with negative deviations (should work, creates inverted bands)
    const resultNeg = wasm.mab_js(close, 10, 50, -1.0, -1.0, "sma", "sma");
    const upperNeg = resultNeg.slice(0, close.length);
    const middleNeg = resultNeg.slice(close.length, close.length * 2);
    const lowerNeg = resultNeg.slice(close.length * 2);
    
    // Upper band should be below middle, lower band above middle
    for (let i = 100; i < 110; i++) {
        if (!isNaN(upperNeg[i])) {
            assert(upperNeg[i] < middleNeg[i],
                  `Negative devup should put upper below middle at index ${i}`);
            assert(lowerNeg[i] > middleNeg[i],
                  `Negative devdn should put lower above middle at index ${i}`);
        }
    }
});

// Zero-copy API tests
test('MAB zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const fast_period = 5;
    const slow_period = 10;
    const devup = 1.0;
    const devdn = 1.0;
    
    // Allocate buffers
    const inPtr = wasm.mab_alloc(data.length);
    const upperPtr = wasm.mab_alloc(data.length);
    const middlePtr = wasm.mab_alloc(data.length);
    const lowerPtr = wasm.mab_alloc(data.length);
    
    assert(inPtr !== 0, 'Failed to allocate input memory');
    assert(upperPtr !== 0, 'Failed to allocate upper memory');
    assert(middlePtr !== 0, 'Failed to allocate middle memory');
    assert(lowerPtr !== 0, 'Failed to allocate lower memory');
    
    try {
        // Create views into WASM memory
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
        const upperView = new Float64Array(wasm.__wasm.memory.buffer, upperPtr, data.length);
        const middleView = new Float64Array(wasm.__wasm.memory.buffer, middlePtr, data.length);
        const lowerView = new Float64Array(wasm.__wasm.memory.buffer, lowerPtr, data.length);
        
        // Copy data into WASM memory
        inView.set(data);
        
        // Compute MAB
        wasm.mab_into(inPtr, upperPtr, middlePtr, lowerPtr, data.length, 
                      fast_period, slow_period, devup, devdn, "sma", "sma");
        
        // Verify results match regular API
        const regularResult = wasm.mab_js(data, fast_period, slow_period, devup, devdn, "sma", "sma");
        const regularUpper = regularResult.slice(0, data.length);
        const regularMiddle = regularResult.slice(data.length, data.length * 2);
        const regularLower = regularResult.slice(data.length * 2);
        
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularUpper[i]) && isNaN(upperView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularUpper[i] - upperView[i]) < 1e-10,
                   `Upper zero-copy mismatch at index ${i}`);
            assert(Math.abs(regularMiddle[i] - middleView[i]) < 1e-10,
                   `Middle zero-copy mismatch at index ${i}`);
            assert(Math.abs(regularLower[i] - lowerView[i]) < 1e-10,
                   `Lower zero-copy mismatch at index ${i}`);
        }
    } finally {
        // Always free memory
        wasm.mab_free(inPtr, data.length);
        wasm.mab_free(upperPtr, data.length);
        wasm.mab_free(middlePtr, data.length);
        wasm.mab_free(lowerPtr, data.length);
    }
});

test('MAB zero-copy with aliasing', () => {
    // Test in-place computation (input aliased with upper output)
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    
    const ptr = wasm.mab_alloc(data.length);
    const middlePtr = wasm.mab_alloc(data.length);
    const lowerPtr = wasm.mab_alloc(data.length);
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, data.length);
        memView.set(data);
        
        // Save original data
        const originalData = new Float64Array(memView);
        
        // Compute MAB with input aliased to upper output
        wasm.mab_into(ptr, ptr, middlePtr, lowerPtr, data.length, 5, 10, 1.0, 1.0, "sma", "sma");
        
        // Verify it worked (input should be overwritten with upper band)
        const regularResult = wasm.mab_js(originalData, 5, 10, 1.0, 1.0, "sma", "sma");
        const regularUpper = regularResult.slice(0, data.length);
        
        // Recreate views in case memory grew
        const upperView = new Float64Array(wasm.__wasm.memory.buffer, ptr, data.length);
        
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularUpper[i]) && isNaN(upperView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularUpper[i] - upperView[i]) < 1e-10,
                   `Aliased zero-copy mismatch at index ${i}`);
        }
    } finally {
        wasm.mab_free(ptr, data.length);
        wasm.mab_free(middlePtr, data.length);
        wasm.mab_free(lowerPtr, data.length);
    }
});

test('MAB batch zero-copy API', () => {
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = Math.sin(i * 0.1) * 100 + 100;
    }
    
    // Parameters for batch
    const fast_period_start = 10, fast_period_end = 12, fast_period_step = 2;
    const slow_period_start = 20, slow_period_end = 20, slow_period_step = 0;
    const devup_start = 1.0, devup_end = 2.0, devup_step = 1.0;
    const devdn_start = 1.0, devdn_end = 1.0, devdn_step = 0.0;
    
    // Calculate expected rows: 2 fast * 1 slow * 2 devup * 1 devdn = 4
    const expectedRows = 4;
    const totalSize = expectedRows * data.length;
    
    // Allocate buffers
    const inPtr = wasm.mab_alloc(data.length);
    const upperPtr = wasm.mab_alloc(totalSize);
    const middlePtr = wasm.mab_alloc(totalSize);
    const lowerPtr = wasm.mab_alloc(totalSize);
    
    try {
        // Copy data
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
        inView.set(data);
        
        // Run batch
        const rows = wasm.mab_batch_into(
            inPtr, upperPtr, middlePtr, lowerPtr, data.length,
            fast_period_start, fast_period_end, fast_period_step,
            slow_period_start, slow_period_end, slow_period_step,
            devup_start, devup_end, devup_step,
            devdn_start, devdn_end, devdn_step,
            "sma", "sma"
        );
        
        assert.strictEqual(rows, expectedRows, "Unexpected number of rows");
        
        // Verify first row matches single calculation
        const upperView = new Float64Array(wasm.__wasm.memory.buffer, upperPtr, totalSize);
        const firstRowUpper = upperView.slice(0, data.length);
        
        const singleResult = wasm.mab_js(data, 10, 20, 1.0, 1.0, "sma", "sma");
        const singleUpper = singleResult.slice(0, data.length);
        
        for (let i = 0; i < data.length; i++) {
            if (isNaN(singleUpper[i]) && isNaN(firstRowUpper[i])) {
                continue;
            }
            assert(Math.abs(singleUpper[i] - firstRowUpper[i]) < 1e-10,
                   `Batch first row mismatch at index ${i}`);
        }
    } finally {
        wasm.mab_free(inPtr, data.length);
        wasm.mab_free(upperPtr, totalSize);
        wasm.mab_free(middlePtr, totalSize);
        wasm.mab_free(lowerPtr, totalSize);
    }
});

// Error handling for zero-copy API
test('MAB zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.mab_into(0, 0, 0, 0, 10, 5, 10, 1.0, 1.0, "sma", "sma");
    }, /Null pointer/);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.mab_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.mab_into(ptr, ptr, ptr, ptr, 10, 0, 10, 1.0, 1.0, "sma", "sma");
        }, /Invalid period/);
    } finally {
        wasm.mab_free(ptr, 10);
    }
});

// Memory leak prevention test
test('MAB zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr1 = wasm.mab_alloc(size);
        const ptr2 = wasm.mab_alloc(size);
        const ptr3 = wasm.mab_alloc(size);
        const ptr4 = wasm.mab_alloc(size);
        
        assert(ptr1 !== 0, `Failed to allocate ${size} elements`);
        assert(ptr2 !== 0, `Failed to allocate ${size} elements`);
        assert(ptr3 !== 0, `Failed to allocate ${size} elements`);
        assert(ptr4 !== 0, `Failed to allocate ${size} elements`);
        
        // Write pattern to verify memory
        const view1 = new Float64Array(wasm.__wasm.memory.buffer, ptr1, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            view1[i] = i * 1.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(view1[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.mab_free(ptr1, size);
        wasm.mab_free(ptr2, size);
        wasm.mab_free(ptr3, size);
        wasm.mab_free(ptr4, size);
    }
});

test.after(() => {
    console.log('MAB WASM tests completed');
});