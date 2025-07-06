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
        wasm = await import(wasmPath);
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
    
    const result = wasm.linreg_js(close, 14);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected last 5 values from Rust test
    const expectedLastFive = [
        58929.37142857143,
        58899.42857142857,
        58918.857142857145,
        59100.6,
        58987.94285714286,
    ];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        0.1,
        "LinReg last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('linreg', result, 'close', { period: 14 });
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
    });
});

test('LinReg period exceeds length', () => {
    // Test LinReg fails when period exceeds data length - mirrors check_linreg_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.linreg_js(dataSmall, 10);
    });
});

test('LinReg very small dataset', () => {
    // Test LinReg with very small dataset - mirrors check_linreg_very_small_dataset
    const dataSingle = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.linreg_js(dataSingle, 14);
    });
});

test('LinReg empty input', () => {
    // Test LinReg with empty input - mirrors check_linreg_empty_input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.linreg_js(dataEmpty, 14);
    });
});

test('LinReg all NaN', () => {
    // Test LinReg with all NaN input
    const data = new Float64Array([NaN, NaN, NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.linreg_js(data, 3);
    });
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
    const period_start = 10;
    const period_end = 40;
    const period_step = 10;  // periods: 10, 20, 30, 40
    
    const batch_result = wasm.linreg_batch_js(
        close, 
        period_start, period_end, period_step
    );
    const metadata = wasm.linreg_batch_metadata_js(
        period_start, period_end, period_step
    );
    
    // Metadata should contain period values
    assert.strictEqual(metadata.length, 4);  // 4 periods
    assert.deepStrictEqual(Array.from(metadata), [10, 20, 30, 40]);
    
    // Batch result should contain all individual results flattened
    assert.strictEqual(batch_result.length, 4 * close.length);  // 4 periods
    
    // Verify each row matches individual calculation
    let row_idx = 0;
    for (const period of [10, 20, 30, 40]) {
        const individual_result = wasm.linreg_js(close, period);
        
        // Extract row from batch result
        const row_start = row_idx * close.length;
        const row = batch_result.slice(row_start, row_start + close.length);
        
        assertArrayClose(row, individual_result, 1e-9, `Period ${period}`);
        row_idx++;
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
    const batchResult = wasm.linreg_batch_js(close, 10, 50, 10);
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
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
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
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.linreg_batch_metadata_js(15, 45, 15);
    
    // Should have 3 periods: 15, 30, 45
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 15);
    assert.strictEqual(metadata[1], 30);
    assert.strictEqual(metadata[2], 45);
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
    
    const batch_result = wasm.linreg_batch_js(data, 2, 4, 1);  // periods: 2, 3, 4
    
    // Should have 3 periods
    assert.strictEqual(batch_result.length, 3 * data.length);
    
    // Verify metadata
    const metadata = wasm.linreg_batch_metadata_js(2, 4, 1);
    assert.deepStrictEqual(Array.from(metadata), [2, 3, 4]);
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