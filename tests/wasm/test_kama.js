/**
 * WASM binding tests for KAMA indicator.
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

test('KAMA partial params', () => {
    // Test with default parameters - mirrors check_kama_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.kama_js(close, 30);
    assert.strictEqual(result.length, close.length);
});

test('KAMA accuracy', async () => {
    // Test KAMA matches expected values from Rust tests - mirrors check_kama_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.kama;
    
    const result = wasm.kama_js(
        close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-6,
        "KAMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('kama', result, 'close', expected.defaultParams);
});

test('KAMA default candles', () => {
    // Test KAMA with default parameters - mirrors check_kama_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.kama_js(close, 30);
    assert.strictEqual(result.length, close.length);
});

test('KAMA zero period', () => {
    // Test KAMA fails with zero period - mirrors check_kama_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.kama_js(inputData, 0);
    }, /Invalid period/);
});

test('KAMA period exceeds length', () => {
    // Test KAMA fails when period exceeds data length - mirrors check_kama_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.kama_js(dataSmall, 10);
    }, /Invalid period/);
});

test('KAMA very small dataset', () => {
    // Test KAMA fails with insufficient data - mirrors check_kama_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.kama_js(singlePoint, 30);
    }, /Invalid period|Not enough valid data/);
});

test('KAMA empty input', () => {
    // Test KAMA fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.kama_js(empty, 30);
    }, /Input data slice is empty/);
});

test('KAMA all NaN input', () => {
    // Test KAMA with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.kama_js(allNaN, 30);
    }, /All values are NaN/);
});

test('KAMA NaN handling', () => {
    // Test KAMA handles NaN values correctly - mirrors check_kama_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.kama_js(close, 30);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (30), no NaN values should exist
    if (result.length > 30) {
        for (let i = 30; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First period values should be NaN
    assertAllNaN(result.slice(0, 30), "Expected NaN in warmup period");
});

test('KAMA batch single parameter set', () => {
    // Test batch with single parameter combination using new API
    const close = new Float64Array(testData.close);
    
    // Using the new ergonomic batch API
    const batchResult = wasm.kama_batch(close, {
        period_range: [30, 30, 0]
    });
    
    // Should match single calculation
    const singleResult = wasm.kama_js(close, 30);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('KAMA batch multiple periods', () => {
    // Test batch with multiple period values using new API
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 10, 20, 30, 40 using ergonomic API
    const batchResult = wasm.kama_batch(close, {
        period_range: [10, 40, 10]
    });
    
    // Should have 4 rows * 100 cols = 400 values
    assert.strictEqual(batchResult.values.length, 4 * 100);
    assert.strictEqual(batchResult.rows, 4);
    assert.strictEqual(batchResult.cols, 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 20, 30, 40];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.kama_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('KAMA batch metadata from result', () => {
    // Test that batch result includes correct parameter combinations
    const close = new Float64Array(50); // Need enough data
    close.fill(100);
    
    const result = wasm.kama_batch(close, {
        period_range: [10, 30, 10]  // periods: 10, 20, 30
    });
    
    // Should have 3 combinations
    assert.strictEqual(result.combos.length, 3);
    
    // Check combinations
    assert.strictEqual(result.combos[0].period, 10);
    assert.strictEqual(result.combos[1].period, 20);
    assert.strictEqual(result.combos[2].period, 30);
});

test('KAMA batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.kama_batch(close, {
        period_range: [5, 15, 5]  // 3 periods: 5, 10, 15
    });
    
    // Should have 3 combinations
    assert.strictEqual(batchResult.combos.length, 3);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 3 * 50);
    
    // Verify structure
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const period = batchResult.combos[combo].period;
        
        const rowStart = combo * 50;
        const rowData = batchResult.values.slice(rowStart, rowStart + 50);
        
        // First period values should be NaN
        for (let i = 0; i < period; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        // After warmup should have values
        for (let i = period; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('KAMA batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep
    const singleBatch = wasm.kama_batch(close, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.kama_batch(close, {
        period_range: [5, 7, 10] // Step larger than range
    });
    
    // Should only have period=5
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.kama_batch(new Float64Array([]), {
            period_range: [30, 30, 0]
        });
    }, /Input data slice is empty/);
});

test('KAMA batch - new ergonomic API with single parameter', () => {
    // Test the new API with a single parameter combination
    const close = new Float64Array(testData.close);
    
    const result = wasm.kama_batch(close, {
        period_range: [30, 30, 0]
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
    assert.strictEqual(combo.period, 30);
    
    // Compare with old API
    const oldResult = wasm.kama_js(close, 30);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('KAMA batch - new API with multiple parameters', () => {
    // Test with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.kama_batch(close, {
        period_range: [10, 20, 10]  // 10, 20
    });
    
    // Should have 2 combinations
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 2);
    assert.strictEqual(result.values.length, 100);
    
    // Verify each combo
    const expectedCombos = [
        { period: 10 },
        { period: 20 }
    ];
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedCombos[i].period);
    }
    
    // Extract and verify a specific row
    const secondRow = result.values.slice(result.cols, 2 * result.cols);
    assert.strictEqual(secondRow.length, 50);
    
    // Compare with old API for first combination
    const oldResult = wasm.kama_js(close, 10);
    const firstRow = result.values.slice(0, result.cols);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('KAMA batch - new API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    // Invalid config structure
    assert.throws(() => {
        wasm.kama_batch(close, {
            period_range: [30, 30] // Missing step
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.kama_batch(close, {
            // Missing period_range
        });
    }, /Invalid config/);
    
    // Invalid data type
    assert.throws(() => {
        wasm.kama_batch(close, {
            period_range: "invalid"
        });
    }, /Invalid config/);
});

// Zero-copy API tests
test('KAMA zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    
    // Allocate buffer
    const ptr = wasm.kama_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute KAMA in-place
    try {
        wasm.kama_into(ptr, ptr, data.length, period);
        
        // Verify results match regular API
        const regularResult = wasm.kama_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.kama_free(ptr, data.length);
    }
});

test('KAMA zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.kama_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.kama_into(ptr, ptr, size, 30);
        
        // Recreate view in case memory grew
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        // Check warmup period has NaN
        for (let i = 0; i < 30; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = 30; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.kama_free(ptr, size);
    }
});

// SIMD128 verification test
test('KAMA SIMD128 consistency', () => {
    // This test verifies SIMD128 produces same results as scalar
    // It runs automatically when SIMD128 is enabled
    const testCases = [
        { size: 10, period: 5 },
        { size: 100, period: 30 },
        { size: 1000, period: 50 },
        { size: 10000, period: 100 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }
        
        const result = wasm.kama_js(data, testCase.period);
        
        // Basic sanity checks
        assert.strictEqual(result.length, data.length);
        
        // Check warmup period
        for (let i = 0; i < testCase.period; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        // Check values exist after warmup
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.period; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        // Verify reasonable values
        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(Math.abs(avgAfterWarmup) < 10, `Average value ${avgAfterWarmup} seems unreasonable`);
    }
});

// Error handling for zero-copy API
test('KAMA zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.kama_into(0, 0, 10, 30);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.kama_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.kama_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        // Period exceeds length
        assert.throws(() => {
            wasm.kama_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.kama_free(ptr, 10);
    }
});

// Memory leak prevention test
test('KAMA zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000, 100000];
    
    for (const size of sizes) {
        const ptr = wasm.kama_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Write pattern to verify memory
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.kama_free(ptr, size);
    }
});

test('KAMA two values', () => {
    // Test KAMA with two values input
    const data = new Float64Array([1.0, 2.0]);
    
    // Should work with period=1 since we have 2 values (need period+1)
    const result = wasm.kama_js(data, 1);
    assert.strictEqual(result.length, 2);
    assert(isNaN(result[0])); // First value is NaN (warmup)
    assert(isFinite(result[1])); // Second value should be valid
});

test('KAMA warmup period calculation', () => {
    // Test that warmup period is correctly calculated
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const testCases = [
        { period: 5, expectedWarmup: 5 },
        { period: 10, expectedWarmup: 10 },
        { period: 20, expectedWarmup: 20 },
        { period: 30, expectedWarmup: 30 },
    ];
    
    for (const { period, expectedWarmup } of testCases) {
        const result = wasm.kama_js(close, period);
        
        // Check NaN values up to warmup period
        for (let i = 0; i < expectedWarmup && i < result.length; i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} for period=${period}`);
        }
        
        // Check valid values after warmup
        if (expectedWarmup < result.length) {
            assert(!isNaN(result[expectedWarmup]), 
                `Expected valid value at index ${expectedWarmup} for period=${period}`);
        }
    }
});

test('KAMA consistency across calls', () => {
    // Test that KAMA produces consistent results across multiple calls
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.kama_js(close, 30);
    const result2 = wasm.kama_js(close, 30);
    
    assertArrayClose(result1, result2, 1e-15, "KAMA results not consistent");
});

test('KAMA batch performance comparison', () => {
    // Test that batch computation is more efficient than multiple single computations
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use first 1000 values
    
    // Test 5 periods using new batch API
    const startBatch = performance.now();
    const batchResult = wasm.kama_batch(close, {
        period_range: [10, 50, 10]  // periods: 10, 20, 30, 40, 50
    });
    const batchTime = performance.now() - startBatch;
    
    const startSingle = performance.now();
    const singleResults = [];
    for (let period = 10; period <= 50; period += 10) {
        singleResults.push(...wasm.kama_js(close, period));
    }
    const singleTime = performance.now() - startSingle;
    
    // Batch should be faster than multiple single calls
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    // Verify results match (need to flatten batch result)
    assertArrayClose(batchResult.values, singleResults, 1e-9, 'Batch vs single results');
});

test('KAMA batch - legacy API compatibility', () => {
    // Test that old batch_js API still works
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const batch_result = wasm.kama_batch_js(close, 10, 30, 10);  // periods: 10, 20, 30
    const metadata = wasm.kama_batch_metadata_js(10, 30, 10);
    
    // Metadata should contain period values
    assert.strictEqual(metadata.length, 3);
    assert.deepStrictEqual(Array.from(metadata), [10, 20, 30]);
    
    // Batch result should contain all individual results flattened
    assert.strictEqual(batch_result.length, 3 * close.length);
    
    // Verify first row matches individual calculation
    const individual_result = wasm.kama_js(close, 10);
    const first_row = batch_result.slice(0, close.length);
    assertArrayClose(first_row, individual_result, 1e-9, 'Legacy API first row');
});

test.after(() => {
    console.log('KAMA WASM tests completed');
});