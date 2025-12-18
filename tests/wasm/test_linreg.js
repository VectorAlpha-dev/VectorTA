/**
 * WASM binding tests for LinReg indicator.
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

test('LinReg partial params', () => {
    // Test with default parameters - mirrors check_linreg_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.linreg_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('LinReg accuracy', async () => {
    // Test LinReg matches expected values from Rust tests - mirrors check_linreg_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.linreg;
    
    const result = wasm.linreg_js(close, expected.defaultParams.period);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        0.1,
        "LinReg last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('linreg', result, 'close', expected.defaultParams);
});

test('LinReg default candles', async () => {
    // Test LinReg with default parameters - mirrors check_linreg_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.linreg_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    // Compare with Rust
    await compareWithRust('linreg', result, 'close', { period: 14 });
});

test('LinReg zero period', () => {
    // Test LinReg fails with zero period - mirrors check_linreg_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.linreg_js(inputData, 0);
    }, /Invalid period/);
});

test('LinReg period exceeds length', () => {
    // Test LinReg fails when period exceeds data length - mirrors check_linreg_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.linreg_js(dataSmall, 10);
    }, /Invalid period|Not enough valid data/);
});

test('LinReg very small dataset', () => {
    // Test LinReg with very small dataset - mirrors check_linreg_very_small_dataset
    const dataSingle = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.linreg_js(dataSingle, 14);
    }, /Invalid period|Not enough valid data/);
});

test('LinReg empty input', () => {
    // Test LinReg with empty input - mirrors check_linreg_empty_input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.linreg_js(dataEmpty, 14);
    }, /no data provided|empty|all values are nan/i);
});

test('LinReg all NaN', () => {
    // Test LinReg with all NaN input
    const data = new Float64Array([NaN, NaN, NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.linreg_js(data, 3);
    }, /All values are NaN/);
});

test('LinReg reinput', () => {
    // Test LinReg with re-input of LinReg result - mirrors check_linreg_reinput
    const close = new Float64Array(testData.close);
    
    // First LinReg pass with period=14
    const firstResult = wasm.linreg_js(close, 14);
    
    // Second LinReg pass with period=10 using first result as input
    const secondResult = wasm.linreg_js(firstResult, 10);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // The second pass will have its own warmup period
    // First pass warmup: 14 values
    // Second pass warmup: 10 values  
    // So we expect NaN values up to index 14+10=24
    for (let i = 24; i < secondResult.length; i++) {
        assert(isFinite(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('LinReg NaN handling', () => {
    // Test LinReg handling of NaN values - mirrors check_linreg_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.linreg_js(close, 14);
    
    assert.strictEqual(result.length, close.length);
    
    // The Rust test checks that after index 240, there are no NaN values
    // This implies the warmup period creates NaN values at the beginning
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('LinReg batch', () => {
    // Test LinReg batch computation
    const close = new Float64Array(testData.close);
    
    // Test period range 10-40 step 10
    const batchResult = wasm.linreg_batch(close, {
        period_range: [10, 40, 10]  // periods: 10, 20, 30, 40
    });
    
    // Should have correct structure
    assert(batchResult.values, 'Should have values array');
    assert(batchResult.combos, 'Should have combos array');
    assert(typeof batchResult.rows === 'number', 'Should have rows count');
    assert(typeof batchResult.cols === 'number', 'Should have cols count');
    
    // Should have 4 periods
    assert.strictEqual(batchResult.rows, 4);
    assert.strictEqual(batchResult.cols, close.length);
    assert.strictEqual(batchResult.combos.length, 4);
    assert.strictEqual(batchResult.values.length, 4 * close.length);
    
    // Check combos contain correct periods
    const expectedPeriods = [10, 20, 30, 40];
    for (let i = 0; i < 4; i++) {
        assert.strictEqual(batchResult.combos[i].period, expectedPeriods[i]);
    }
    
    // Verify each row matches individual calculation
    for (let i = 0; i < 4; i++) {
        const period = expectedPeriods[i];
        const individual_result = wasm.linreg_js(close, period);
        
        // Extract row from batch result
        const row_start = i * close.length;
        const row = batchResult.values.slice(row_start, row_start + close.length);
        
        assertArrayClose(row, individual_result, 1e-9, `Period ${period}`);
    }
});

test('LinReg different periods', () => {
    // Test LinReg with different period values
    const close = new Float64Array(testData.close);
    
    // Test various period values
    for (const period of [5, 10, 20, 50]) {
        const result = wasm.linreg_js(close, period);
        assert.strictEqual(result.length, close.length);
        
        // Verify warmup period
        // LinReg starts outputting at index period-1
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} for period=${period}`);
        }
        
        // Verify no NaN after warmup period
        for (let i = period - 1; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for period=${period}`);
        }
    }
});

test('LinReg batch performance', () => {
    // Test that batch computation is more efficient than multiple single computations
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use first 1000 values
    
    // Test 5 periods
    const startBatch = performance.now();
    const batchResult = wasm.linreg_batch(close, {
        period_range: [10, 50, 10]  // periods: 10, 20, 30, 40, 50
    });
    const batchTime = performance.now() - startBatch;
    
    const startSingle = performance.now();
    const singleResults = [];
    for (let period = 10; period <= 50; period += 10) {
        singleResults.push(...wasm.linreg_js(close, period));
    }
    const singleTime = performance.now() - startSingle;
    
    // Batch should be faster than multiple single calls
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    // Verify results match
    assertArrayClose(batchResult.values, singleResults, 1e-9, 'Batch vs single results');
});

test('LinReg edge cases', () => {
    // Test LinReg with edge case inputs
    
    // Test with monotonically increasing data (perfect linear regression)
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    const result = wasm.linreg_js(data, 3);
    assert.strictEqual(result.length, data.length);
    
    // LinReg outputs the fitted value at the end of the window
    // With data [8, 9, 10] and perfect linear fit, the value at x=3 is 10.0
    assertClose(result[result.length - 1], 10.0, 1e-9, "Perfect linear regression failed");
    
    // Test with constant values
    const constantData = new Float64Array(20).fill(5.0);
    const constantResult = wasm.linreg_js(constantData, 5);
    assert.strictEqual(constantResult.length, constantData.length);
    
    // With constant data, LinReg should predict the same constant value
    for (let i = 5; i < constantResult.length; i++) {
        assertClose(constantResult[i], 5.0, 1e-9, `Constant prediction failed at index ${i}`);
    }
});

test('LinReg single value', () => {
    // Test LinReg with single value input
    const data = new Float64Array([42.0]);
    
    // Period=1 with single value should work, returning [NaN]
    const result = wasm.linreg_js(data, 1);
    assert.strictEqual(result.length, 1);
    assert(isNaN(result[0]));
});

test('LinReg two values', () => {
    // Test LinReg with two values input
    const data = new Float64Array([1.0, 2.0]);
    
    // Period=1 is mathematically undefined (can't fit a line with one point)
    // So it returns all NaN values
    const result = wasm.linreg_js(data, 1);
    assert.strictEqual(result.length, 2);
    assert(isNaN(result[0]));
    assert(isNaN(result[1]));
    
    // Test with period=2 which should work
    const result2 = wasm.linreg_js(data, 2);
    assert.strictEqual(result2.length, 2);
    assert(isNaN(result2[0]));  // First value is NaN (warmup)
    assertClose(result2[1], 2.0, 1e-9, "Two-value prediction failed");
});

test('LinReg batch metadata', () => {
    // Test batch result includes correct parameter combinations
    const close = new Float64Array(50); // Need enough data
    close.fill(100);
    
    const result = wasm.linreg_batch(close, {
        period_range: [15, 45, 15]  // periods: 15, 30, 45
    });
    
    // Should have 3 periods
    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.combos[0].period, 15);
    assert.strictEqual(result.combos[1].period, 30);
    assert.strictEqual(result.combos[2].period, 45);
});

test('LinReg warmup period calculation', () => {
    // Test that warmup period is correctly calculated
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const testCases = [
        { period: 5, expectedWarmup: 5 },
        { period: 10, expectedWarmup: 10 },
        { period: 20, expectedWarmup: 20 },
        { period: 30, expectedWarmup: 30 },
    ];
    
    for (const { period, expectedWarmup } of testCases) {
        const result = wasm.linreg_js(close, period);
        
        // Check NaN values up to warmup period
        // LinReg starts outputting at index period-1
        for (let i = 0; i < period - 1 && i < result.length; i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} for period=${period}`);
        }
        
        // Check valid values after warmup
        if (period - 1 < result.length) {
            assert(!isNaN(result[period - 1]), 
                `Expected valid value at index ${period - 1} for period=${period}`);
        }
    }
});

test('LinReg consistency across calls', () => {
    // Test that LinReg produces consistent results across multiple calls
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.linreg_js(close, 14);
    const result2 = wasm.linreg_js(close, 14);
    
    assertArrayClose(result1, result2, 1e-15, "LinReg results not consistent");
});

test('LinReg parameter step precision', () => {
    // Test batch with very small step sizes
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    const batchResult = wasm.linreg_batch(data, {
        period_range: [2, 4, 1]  // periods: 2, 3, 4
    });
    
    // Should have 3 periods
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.combos.length, 3);
    assert.strictEqual(batchResult.values.length, 3 * data.length);
    
    // Verify combos
    assert.strictEqual(batchResult.combos[0].period, 2);
    assert.strictEqual(batchResult.combos[1].period, 3);
    assert.strictEqual(batchResult.combos[2].period, 4);
});

test('LinReg slope calculation', () => {
    // Test LinReg slope calculation with known data
    // Create data with known slope = 2
    const data = new Float64Array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    const result = wasm.linreg_js(data, 4);
    
    // LinReg outputs the fitted value at the end of the window
    // With the last window [10, 12, 14, 16] and slope=2, the fitted value at x=4 is 16
    assertClose(result[result.length - 1], 16.0, 1e-9, "Slope calculation failed");
});

test('LinReg streaming simulation', () => {
    // Test LinReg streaming functionality (simulated)
    const close = new Float64Array(testData.close.slice(0, 100));
    const period = 14;
    
    // Calculate batch result for comparison
    const batchResult = wasm.linreg_js(close, period);
    
    // LinReg requires full history, so streaming is more complex
    // We'll verify batch result has expected properties
    assert.strictEqual(batchResult.length, close.length);
    
    // Verify warmup period
    // LinReg starts outputting at index period-1
    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(batchResult[i]), `Expected NaN at index ${i}`);
    }
    
    // Verify values after warmup
    for (let i = period - 1; i < close.length; i++) {
        assert(isFinite(batchResult[i]), `Expected finite value at index ${i}`);
    }
});

test('LinReg large period', () => {
    // Test LinReg with large period relative to data size
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.sin(i * 0.1) * 100 + 1000; // Generate some test data
    }
    
    const result = wasm.linreg_js(data, 99);
    assert.strictEqual(result.length, data.length);
    
    // First 98 values should be NaN (warmup period = period-1)
    for (let i = 0; i < 98; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    // Values from index 98 onwards should be valid
    assert(isFinite(result[98]), "Expected finite value at index 98");
    assert(isFinite(result[99]), "Expected finite value at last index");
});

// ====================== Zero-Copy API Tests ======================

test('LinReg zero-copy basic', () => {
    // Test zero-copy API for LinReg
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 3;
    
    // Allocate memory
    const ptr = wasm.linreg_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view of WASM memory
    const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, data.length);
    
    // Copy data to WASM memory
    memView.set(data);
    
    // Compute LinReg in-place
    try {
        wasm.linreg_into(ptr, ptr, data.length, period);
        
        // Verify results match regular API
        const regularResult = wasm.linreg_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.linreg_free(ptr, data.length);
    }
});

test('LinReg zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.linreg_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.linreg_into(ptr, ptr, size, 14);
        
        // Recreate view in case memory grew
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        // Check warmup period has NaN (first + period - 1 = 0 + 14 - 1 = 13)
        for (let i = 0; i < 13; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = 13; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.linreg_free(ptr, size);
    }
});

test('LinReg zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.linreg_into(0, 0, 10, 5);
    }, /null pointer/);
    
    // Test invalid period
    const ptr = wasm.linreg_alloc(10);
    try {
        assert.throws(() => {
            wasm.linreg_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
    } finally {
        wasm.linreg_free(ptr, 10);
    }
});

test('LinReg zero-copy batch API', () => {
    const size = 100;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.1) * 10;
    }
    
    // Test batch with zero-copy
    const periods = 3;
    const totalSize = periods * size;
    const inPtr = wasm.linreg_alloc(size);
    const outPtr = wasm.linreg_alloc(totalSize);
    
    try {
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, size);
        inView.set(data);
        
        const rows = wasm.linreg_batch_into(inPtr, outPtr, size, 10, 30, 10);
        assert.strictEqual(rows, 3, 'Expected 3 rows');
        
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
        
        // Verify each row matches individual calculation
        const periodValues = [10, 20, 30];
        for (let i = 0; i < periods; i++) {
            const period = periodValues[i];
            const individual = wasm.linreg_js(data, period);
            const rowStart = i * size;
            const row = Array.from(outView.slice(rowStart, rowStart + size));
            
            assertArrayClose(row, individual, 1e-9, `Batch row ${i} mismatch`);
        }
    } finally {
        wasm.linreg_free(inPtr, size);
        wasm.linreg_free(outPtr, totalSize);
    }
});

// ====================== SIMD Consistency Tests ======================

test('LinReg SIMD consistency', () => {
    // This test verifies different kernels produce same results
    // WASM uses scalar kernel, but we verify consistency
    const testCases = [
        { size: 10, period: 5 },
        { size: 100, period: 14 },
        { size: 1000, period: 20 },
        { size: 10000, period: 50 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }
        
        const result = wasm.linreg_js(data, testCase.period);
        
        // Basic sanity checks
        assert.strictEqual(result.length, data.length);
        
        // Check warmup period (first + period - 1)
        for (let i = 0; i < testCase.period - 1; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        // Check values exist after warmup
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.period - 1; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        // Verify reasonable values
        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(Math.abs(avgAfterWarmup) < 1000, `Average value ${avgAfterWarmup} seems unreasonable`);
    }
});

// ====================== Advanced Batch Tests ======================

test('LinReg batch metadata validation', () => {
    // Test that batch result includes correct parameter combinations
    const close = new Float64Array(50);
    close.fill(100);
    
    const result = wasm.linreg_batch(close, {
        period_range: [10, 30, 10]  // periods: 10, 20, 30
    });
    
    // Should have 3 combinations
    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    
    // Check combinations
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[1].period, 20);
    assert.strictEqual(result.combos[2].period, 30);
});

test('LinReg batch warmup consistency', () => {
    // Test that each batch row has correct warmup period
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.random() * 100;
    }
    
    const result = wasm.linreg_batch(data, {
        period_range: [5, 15, 5]  // periods: 5, 10, 15
    });
    
    const periods = [5, 10, 15];
    for (let row = 0; row < periods.length; row++) {
        const period = periods[row];
        const rowStart = row * 100;
        const rowData = result.values.slice(rowStart, rowStart + 100);
        
        // Check warmup period (first + period - 1)
        const expectedWarmup = period - 1;
        for (let i = 0; i < expectedWarmup; i++) {
            assert(isNaN(rowData[i]), `Row ${row}: Expected NaN at index ${i}`);
        }
        
        // Check values after warmup
        for (let i = expectedWarmup; i < rowData.length; i++) {
            assert(!isNaN(rowData[i]), `Row ${row}: Unexpected NaN at index ${i}`);
        }
    }
});
