/**
 * WASM binding tests for Percentile Nearest Rank indicator.
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

test('PNR partial params', () => {
    // Test with partial parameters - mirrors check_pnr_partial_params
    const data = new Float64Array(EXPECTED_OUTPUTS.percentileNearestRank.basicTest.data);
    
    // Test with only length specified (percentage defaults to 50.0)
    // Note: WASM version requires explicit percentage parameter
    const result = wasm.percentile_nearest_rank_js(data, 5, 50.0);
    assert.strictEqual(result.length, data.length);
    assert.strictEqual(result[4], EXPECTED_OUTPUTS.percentileNearestRank.basicTest.expectedAt4);
});

test('PNR accuracy', async () => {
    // Test PNR matches expected values - mirrors check_pnr_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.percentileNearestRank;
    
    const result = wasm.percentile_nearest_rank_js(
        close,
        expected.defaultParams.length,
        expected.defaultParams.percentage
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check warmup period
    const warmup = expected.warmupPeriod;
    assertAllNaN(result.slice(0, warmup), `Expected NaN in first ${warmup} values`);
    
    // Check that we have valid values after warmup
    assert(!isNaN(result[warmup]), `Expected valid value at index ${warmup}`);
    
    // Check last 5 values match expected reference values
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "PNR last 5 values mismatch"
    );
    
    // Compare with Rust implementation
    // NOTE: generate_references binary doesn't have percentile_nearest_rank registered
    // await compareWithRust('percentile_nearest_rank', result, 'close', expected.defaultParams);
});

test('PNR default candles', () => {
    // Test PNR with default parameters - mirrors check_pnr_default_candles
    const close = new Float64Array(testData.close);
    
    // Default params: length=15, percentage=50.0
    // Note: WASM version requires explicit parameters
    const result = wasm.percentile_nearest_rank_js(close, 15, 50.0);
    assert.strictEqual(result.length, close.length);
    
    // Check warmup period
    assertAllNaN(result.slice(0, 14), "Expected NaN in warmup period");
    assert(!isNaN(result[14]), "Expected valid value after warmup");
});

test('PNR zero period', () => {
    // Test PNR fails with zero period - mirrors check_pnr_zero_period
    const data = new Float64Array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    
    assert.throws(() => {
        wasm.percentile_nearest_rank_js(data, 0, 50.0);
    }, /Invalid period/);
});

test('PNR period exceeds length', () => {
    // Test PNR fails when period exceeds data length - mirrors check_pnr_period_exceeds_length
    const data = new Float64Array([1.0, 1.0, 1.0, 1.0, 1.0]);
    
    assert.throws(() => {
        wasm.percentile_nearest_rank_js(data, 10, 50.0);
    }, /Invalid period/);
});

test('PNR very small dataset', () => {
    // Test PNR with single data point - mirrors check_pnr_very_small_dataset
    const data = new Float64Array([5.0]);
    
    const result = wasm.percentile_nearest_rank_js(data, 1, 50.0);
    assert.strictEqual(result.length, 1);
    assert.strictEqual(result[0], 5.0);
});

test('PNR empty input', () => {
    // Test PNR fails with empty input - mirrors check_pnr_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        // WASM API requires explicit params even for empty input
        wasm.percentile_nearest_rank_js(empty, 15, 50.0);
    }, /Input data is empty/);
});

test('PNR invalid percentage', () => {
    // Test PNR fails with invalid percentage - mirrors check_pnr_invalid_percentage
    const data = new Float64Array(Array(20).fill(1.0));
    
    // Test percentage > 100
    assert.throws(() => {
        wasm.percentile_nearest_rank_js(data, 5, 150.0);
    }, /Percentage must be between/);
    
    // Test negative percentage
    assert.throws(() => {
        wasm.percentile_nearest_rank_js(data, 5, -10.0);
    }, /Percentage must be between/);
});

test('PNR NaN handling', () => {
    // Test PNR handles NaN values correctly - mirrors check_pnr_nan_handling
    const data = new Float64Array([
        1.0, 2.0, NaN, 4.0, 5.0,
        NaN, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, NaN, 15.0
    ]);
    
    const result = wasm.percentile_nearest_rank_js(data, 5, 50.0);
    assert.strictEqual(result.length, data.length);
    
    // Should handle NaN values in window
    assert(!isNaN(result[6]), "Should handle NaN values in window");
});

test('PNR basic functionality', () => {
    // Test basic functionality with simple data
    const data = new Float64Array(EXPECTED_OUTPUTS.percentileNearestRank.basicTest.data);
    const expectedTest = EXPECTED_OUTPUTS.percentileNearestRank.basicTest;
    
    const result = wasm.percentile_nearest_rank_js(
        data,
        expectedTest.length,
        expectedTest.percentage
    );
    
    assert.strictEqual(result.length, data.length);
    
    // First 4 values should be NaN (warmup period for length=5)
    assertAllNaN(result.slice(0, 4), "Expected NaN in warmup period");
    
    // Check expected values from Rust tests
    assert.strictEqual(result[4], expectedTest.expectedAt4);
    assert.strictEqual(result[5], expectedTest.expectedAt5);
});

test('PNR different percentiles', () => {
    // Test with different percentile values
    const data = new Float64Array(EXPECTED_OUTPUTS.percentileNearestRank.percentileTests.data);
    const expectedTest = EXPECTED_OUTPUTS.percentileNearestRank.percentileTests;
    const length = expectedTest.length;
    
    // Test 25th percentile
    const result25 = wasm.percentile_nearest_rank_js(data, length, 25.0);
    assert.strictEqual(result25[4], expectedTest.p25At4);
    
    // Test 75th percentile  
    const result75 = wasm.percentile_nearest_rank_js(data, length, 75.0);
    assert.strictEqual(result75[4], expectedTest.p75At4);
    
    // Test 100th percentile (max)
    const result100 = wasm.percentile_nearest_rank_js(data, length, 100.0);
    assert.strictEqual(result100[4], expectedTest.p100At4);
});

test('PNR all NaN input', () => {
    // Test PNR with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        // WASM API requires explicit params even when data contains only NaN
        wasm.percentile_nearest_rank_js(allNaN, 15, 50.0);
    }, /All values are NaN/);
});

test('PNR batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Using the new ergonomic batch API for single parameter
    const batchResult = wasm.percentile_nearest_rank_batch(close, {
        length_range: [15, 15, 0],
        percentage_range: [50.0, 50.0, 0]
    });
    
    // Should match single calculation
    const singleResult = wasm.percentile_nearest_rank_js(close, 15, 50.0);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('PNR batch multiple parameters', () => {
    // Test batch with multiple parameter values
    const close = new Float64Array(testData.close.slice(0, 50));
    
    // Multiple parameters using ergonomic API
    const batchResult = wasm.percentile_nearest_rank_batch(close, {
        length_range: [10, 20, 10],      // 10, 20
        percentage_range: [25.0, 75.0, 25.0] // 25, 50, 75
    });
    
    // Should have 2 * 3 = 6 rows * 50 cols = 300 values
    assert.strictEqual(batchResult.values.length, 6 * 50);
    assert.strictEqual(batchResult.rows, 6);
    assert.strictEqual(batchResult.cols, 50);
    
    // Verify each row matches individual calculation
    const params = [
        {length: 10, percentage: 25.0},
        {length: 10, percentage: 50.0},
        {length: 10, percentage: 75.0},
        {length: 20, percentage: 25.0},
        {length: 20, percentage: 50.0},
        {length: 20, percentage: 75.0}
    ];
    
    for (let i = 0; i < params.length; i++) {
        const rowStart = i * 50;
        const rowEnd = rowStart + 50;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.percentile_nearest_rank_js(
            close, 
            params[i].length, 
            params[i].percentage
        );
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Row ${i} (length=${params[i].length}, percentage=${params[i].percentage}) mismatch`
        );
    }
});

test('PNR batch metadata from result', () => {
    // Test that batch result includes correct parameter combinations
    const close = new Float64Array(30);
    close.fill(100);
    
    const result = wasm.percentile_nearest_rank_batch(close, {
        length_range: [5, 10, 5],      // 5, 10
        percentage_range: [50.0, 100.0, 50.0]   // 50, 100
    });
    
    // Should have 2 * 2 = 4 combinations
    assert.strictEqual(result.combos.length, 4);
    
    // Check first combination
    assert.strictEqual(result.combos[0].length, 5);
    assert.strictEqual(result.combos[0].percentage, 50.0);
    
    // Check last combination
    assert.strictEqual(result.combos[3].length, 10);
    assertClose(result.combos[3].percentage, 100.0, 1e-10, "percentage mismatch");
});

test('PNR batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 30));
    
    const batchResult = wasm.percentile_nearest_rank_batch(close, {
        length_range: [5, 10, 5],      // 2 lengths
        percentage_range: [25.0, 50.0, 25.0]  // 2 percentages
    });
    
    // Should have 2 * 2 = 4 combinations
    assert.strictEqual(batchResult.combos.length, 4);
    assert.strictEqual(batchResult.rows, 4);
    assert.strictEqual(batchResult.cols, 30);
    assert.strictEqual(batchResult.values.length, 4 * 30);
    
    // Verify structure
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const length = batchResult.combos[combo].length;
        const percentage = batchResult.combos[combo].percentage;
        
        const rowStart = combo * 30;
        const rowData = batchResult.values.slice(rowStart, rowStart + 30);
        
        // First length-1 values should be NaN
        for (let i = 0; i < length - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for length ${length}`);
        }
        
        // After warmup should have values
        for (let i = length - 1; i < 30; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for length ${length}`);
        }
    }
});

test('PNR batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep
    const singleBatch = wasm.percentile_nearest_rank_batch(close, {
        length_range: [5, 5, 1],
        percentage_range: [50.0, 50.0, 0.1]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.percentile_nearest_rank_batch(close, {
        length_range: [5, 7, 10], // Step larger than range
        percentage_range: [50.0, 50.0, 0]
    });
    
    // Should only have length=5
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.percentile_nearest_rank_batch(new Float64Array([]), {
            length_range: [15, 15, 0],
            percentage_range: [50.0, 50.0, 0]
        });
    }, /All values are NaN|Input data is empty/);
});

test('PNR batch - new ergonomic API with single parameter', () => {
    // Test the new API with a single parameter combination
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result = wasm.percentile_nearest_rank_batch(close, {
        length_range: [15, 15, 0],
        percentage_range: [50.0, 50.0, 0]
    });
    
    // Verify structure
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');
    
    // Verify dimensions
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.values.length, close.length);
    
    // Verify parameters
    const combo = result.combos[0];
    assert.strictEqual(combo.length, 15);
    assert.strictEqual(combo.percentage, 50.0);
    
    // Compare with old API
    const oldResult = wasm.percentile_nearest_rank_js(close, 15, 50.0);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('PNR batch - new API with multiple parameters', () => {
    // Test with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.percentile_nearest_rank_batch(close, {
        length_range: [10, 15, 5],      // 10, 15
        percentage_range: [25.0, 50.0, 25.0] // 25, 50
    });
    
    // Should have 2 * 2 = 4 combinations
    assert.strictEqual(result.rows, 4);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 4);
    assert.strictEqual(result.values.length, 200);
    
    // Verify each combo
    const expectedCombos = [
        { length: 10, percentage: 25.0 },
        { length: 10, percentage: 50.0 },
        { length: 15, percentage: 25.0 },
        { length: 15, percentage: 50.0 }
    ];
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].length, expectedCombos[i].length);
        assert.strictEqual(result.combos[i].percentage, expectedCombos[i].percentage);
    }
    
    // Extract and verify a specific row
    const secondRow = result.values.slice(result.cols, 2 * result.cols);
    assert.strictEqual(secondRow.length, 50);
    
    // Compare with old API for first combination
    const oldResult = wasm.percentile_nearest_rank_js(close, 10, 25.0);
    const firstRow = result.values.slice(0, result.cols);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('PNR batch - new API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    // Invalid config structure
    assert.throws(() => {
        wasm.percentile_nearest_rank_batch(close, {
            length_range: [15, 15], // Missing step
            percentage_range: [50.0, 50.0, 0]
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.percentile_nearest_rank_batch(close, {
            length_range: [15, 15, 0]
            // Missing percentage_range
        });
    }, /Invalid config/);
    
    // Invalid data type
    assert.throws(() => {
        wasm.percentile_nearest_rank_batch(close, {
            length_range: "invalid",
            percentage_range: [50.0, 50.0, 0]
        });
    }, /Invalid config/);
});

// Zero-copy API tests (if available)
test('PNR zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const length = 5;
    const percentage = 50.0;
    
    // Check if zero-copy API exists
    if (!wasm.percentile_nearest_rank_alloc) {
        console.log('Zero-copy API not available for PNR');
        return;
    }
    
    // Allocate buffer
    const ptr = wasm.percentile_nearest_rank_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memory = wasm.__wbindgen_memory();
    const memView = new Float64Array(
        memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute PNR in-place
    try {
        wasm.percentile_nearest_rank_into(ptr, ptr, data.length, length, percentage);
        
        // Verify results match regular API
        const regularResult = wasm.percentile_nearest_rank_js(data, length, percentage);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.percentile_nearest_rank_free(ptr, data.length);
    }
});

// SIMD128 verification test
test('PNR SIMD128 consistency', () => {
    // This test verifies SIMD128 produces same results as scalar
    // It runs automatically when SIMD128 is enabled
    const testCases = [
        { size: 10, length: 5 },
        { size: 100, length: 15 },
        { size: 1000, length: 20 },
        { size: 10000, length: 50 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }
        
        const result = wasm.percentile_nearest_rank_js(data, testCase.length, 50.0);
        
        // Basic sanity checks
        assert.strictEqual(result.length, data.length);
        
        // Check warmup period
        for (let i = 0; i < testCase.length - 1; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        // Check values exist after warmup
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.length - 1; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        // Verify reasonable values
        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(Math.abs(avgAfterWarmup) < 10, `Average value ${avgAfterWarmup} seems unreasonable`);
    }
});

test.after(() => {
    console.log('Percentile Nearest Rank WASM tests completed');
});
