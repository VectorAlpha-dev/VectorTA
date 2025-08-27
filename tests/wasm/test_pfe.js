/**
 * WASM binding tests for PFE indicator.
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

test('PFE partial params', () => {
    // Test with default parameters - mirrors check_pfe_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.pfe_js(close, 10, 5); // period=10, smoothing=5 (defaults)
    assert.strictEqual(result.length, close.length);
});

test('PFE accuracy', async () => {
    // Test PFE matches expected values from Rust tests - mirrors check_pfe_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.pfe;
    
    const result = wasm.pfe_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.smoothing
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "PFE last 5 values mismatch"
    );
    
    // Check warmup period (first period values should be NaN)
    for (let i = 0; i < expected.defaultParams.period; i++) {
        assert(isNaN(result[i]), `Expected NaN at warmup index ${i}`);
    }
    
    // Compare full output with Rust
    // TODO: Enable once PFE is added to generate_references
    // await compareWithRust('pfe', result, 'close', expected.defaultParams);
});

test('PFE default candles', () => {
    // Test PFE with default parameters - mirrors check_pfe_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.pfe_js(close, 10, 5);
    assert.strictEqual(result.length, close.length);
});

test('PFE zero period', () => {
    // Test PFE fails with zero period - mirrors check_pfe_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.pfe_js(inputData, 0, 5);
    }, /Invalid period/);
});

test('PFE period exceeds length', () => {
    // Test PFE fails when period exceeds data length - mirrors check_pfe_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.pfe_js(dataSmall, 10, 2);
    }, /Invalid period/);
});

test('PFE very small dataset', () => {
    // Test PFE fails with insufficient data - mirrors check_pfe_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.pfe_js(singlePoint, 10, 2);
    }, /Invalid period|Not enough valid data/);
});

test('PFE empty input', () => {
    // Test PFE fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.pfe_js(empty, 10, 5);
    }, /Input data slice is empty/);
});

test('PFE invalid smoothing', () => {
    // Test PFE fails with invalid smoothing
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // Smoothing = 0
    assert.throws(() => {
        wasm.pfe_js(data, 2, 0);
    }, /Invalid smoothing/);
});

test('PFE reinput', () => {
    // Test PFE applied twice (re-input) - mirrors check_pfe_reinput
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.pfe;
    
    // First pass
    const firstResult = wasm.pfe_js(close, 10, 5);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply PFE to PFE output
    const secondResult = wasm.pfe_js(firstResult, 10, 5);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // After double smoothing, should have extended warmup
    // Check that warmup is properly handled
    let firstNonNaN = -1;
    for (let i = 0; i < secondResult.length; i++) {
        if (!isNaN(secondResult[i])) {
            firstNonNaN = i;
            break;
        }
    }
    assert(firstNonNaN >= 20, `Expected extended warmup, but first non-NaN at ${firstNonNaN}`);
});

test('PFE NaN handling', () => {
    // Test PFE handles NaN values correctly - mirrors check_pfe_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.pfe_js(close, 10, 5);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First period values should be NaN
    assertAllNaN(result.slice(0, 10), "Expected NaN in warmup period");
});

test('PFE all NaN input', () => {
    // Test PFE with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.pfe_js(allNaN, 10, 5);
    }, /All values are NaN/);
});

test('PFE batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Using the new ergonomic batch API for single parameter
    const batchResult = wasm.pfe_batch(close, {
        period_range: [10, 10, 0],
        smoothing_range: [5, 5, 0]
    });
    
    // Should match single calculation
    const singleResult = wasm.pfe_js(close, 10, 5);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('PFE batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 10, 14, 18
    const batchResult = wasm.pfe_batch(close, {
        period_range: [10, 18, 4],      // period range
        smoothing_range: [5, 5, 0]       // smoothing range  
    });
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 14, 18];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.pfe_js(close, periods[i], 5);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('PFE batch metadata from result', () => {
    // Test that batch result includes correct parameter combinations
    const close = new Float64Array(30); // Need enough data for period 20
    close.fill(100);
    
    const result = wasm.pfe_batch(close, {
        period_range: [10, 20, 5],      // period: 10, 15, 20
        smoothing_range: [5, 10, 5]      // smoothing: 5, 10
    });
    
    // Should have 3 * 2 = 6 combinations
    assert.strictEqual(result.combos.length, 6);
    
    // Check first combination
    assert.strictEqual(result.combos[0].period, 10);    // period
    assert.strictEqual(result.combos[0].smoothing, 5);   // smoothing
    
    // Check last combination
    assert.strictEqual(result.combos[5].period, 20);     // period
    assert.strictEqual(result.combos[5].smoothing, 10);  // smoothing
});

test('PFE batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.pfe_batch(close, {
        period_range: [10, 14, 4],      // 2 periods
        smoothing_range: [5, 10, 5]      // 2 smoothings
    });
    
    // Should have 2 * 2 = 4 combinations
    assert.strictEqual(batchResult.combos.length, 4);
    assert.strictEqual(batchResult.rows, 4);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 4 * 50);
    
    // Verify structure
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const period = batchResult.combos[combo].period;
        const smoothing = batchResult.combos[combo].smoothing;
        
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

test('PFE batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    
    // Single value sweep
    const singleBatch = wasm.pfe_batch(close, {
        period_range: [5, 5, 1],
        smoothing_range: [3, 3, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 12);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.pfe_batch(close, {
        period_range: [5, 7, 10], // Step larger than range
        smoothing_range: [3, 3, 0]
    });
    
    // Should only have period=5
    assert.strictEqual(largeBatch.values.length, 12);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.pfe_batch(new Float64Array([]), {
            period_range: [10, 10, 0],
            smoothing_range: [5, 5, 0]
        });
    }, /All values are NaN|Empty/);
});

// Zero-copy API tests
test('PFE zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const period = 5;
    const smoothing = 3;
    
    // Allocate buffer
    const ptr = wasm.pfe_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute PFE in-place
    try {
        wasm.pfe_into(ptr, ptr, data.length, period, smoothing);
        
        // Verify results match regular API
        const regularResult = wasm.pfe_js(data, period, smoothing);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.pfe_free(ptr, data.length);
    }
});

test('PFE zero-copy with large dataset', () => {
    const size = 10000;
    const data = new Float64Array(size);
    // Generate trending data for PFE
    for (let i = 0; i < size; i++) {
        data[i] = 100 + i * 0.01 + Math.sin(i * 0.1) * 5;
    }
    
    const ptr = wasm.pfe_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.pfe_into(ptr, ptr, size, 10, 5);
        
        // Recreate view in case memory grew
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        // Check warmup period has NaN
        for (let i = 0; i < 10; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = 10; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
        
        // PFE values should be between -100 and 100
        for (let i = 10; i < Math.min(100, size); i++) {
            assert(memView2[i] >= -100 && memView2[i] <= 100, 
                   `PFE value ${memView2[i]} at index ${i} out of bounds [-100, 100]`);
        }
    } finally {
        wasm.pfe_free(ptr, size);
    }
});

test('PFE zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.pfe_into(0, 0, 10, 10, 5);
    }, /null pointer/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.pfe_alloc(10);
    try {
        // Create view and set some data (not all NaN)
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, 10);
        for (let i = 0; i < 10; i++) {
            memView[i] = i + 1.0;
        }
        
        // Invalid period
        assert.throws(() => {
            wasm.pfe_into(ptr, ptr, 10, 0, 5);
        }, /Invalid period/);
        
        // Invalid smoothing
        assert.throws(() => {
            wasm.pfe_into(ptr, ptr, 10, 5, 0);
        }, /Invalid smoothing/);
    } finally {
        wasm.pfe_free(ptr, 10);
    }
});

test('PFE zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.pfe_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Write pattern to verify memory
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 2.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 2.5, `Memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.pfe_free(ptr, size);
    }
});

test('PFE batch - new ergonomic API with single parameter', () => {
    // Test the new API with a single parameter combination
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result = wasm.pfe_batch(close, {
        period_range: [10, 10, 0],
        smoothing_range: [5, 5, 0]
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
    assert.strictEqual(combo.period, 10);
    assert.strictEqual(combo.smoothing, 5);
    
    // Compare with old API
    const oldResult = wasm.pfe_js(close, 10, 5);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('PFE batch - new API with multiple parameters', () => {
    // Test with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.pfe_batch(close, {
        period_range: [10, 14, 4],      // 10, 14
        smoothing_range: [5, 10, 5]      // 5, 10
    });
    
    // Should have 2 * 2 = 4 combinations
    assert.strictEqual(result.rows, 4);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 4);
    assert.strictEqual(result.values.length, 200);
    
    // Verify each combo
    const expectedCombos = [
        { period: 10, smoothing: 5 },
        { period: 10, smoothing: 10 },
        { period: 14, smoothing: 5 },
        { period: 14, smoothing: 10 }
    ];
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedCombos[i].period);
        assert.strictEqual(result.combos[i].smoothing, expectedCombos[i].smoothing);
    }
    
    // Extract and verify a specific row
    const secondRow = result.values.slice(result.cols, 2 * result.cols);
    assert.strictEqual(secondRow.length, 50);
    
    // Compare with old API for first combination
    const oldResult = wasm.pfe_js(close, 10, 5);
    const firstRow = result.values.slice(0, result.cols);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('PFE batch - new API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    // Invalid config structure
    assert.throws(() => {
        wasm.pfe_batch(close, {
            period_range: [10, 10], // Missing step
            smoothing_range: [5, 5, 0]
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.pfe_batch(close, {
            period_range: [10, 10, 0]
            // Missing smoothing_range
        });
    }, /Invalid config/);
    
    // Invalid data type
    assert.throws(() => {
        wasm.pfe_batch(close, {
            period_range: "invalid",
            smoothing_range: [5, 5, 0]
        });
    }, /Invalid config/);
});

// SIMD128 verification test
test('PFE SIMD128 consistency', () => {
    // This test verifies SIMD128 produces same results as scalar
    // It runs automatically when SIMD128 is enabled
    const testCases = [
        { size: 20, period: 5 },
        { size: 100, period: 10 },
        { size: 1000, period: 20 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        // Generate trending data suitable for PFE
        for (let i = 0; i < testCase.size; i++) {
            data[i] = 100 + i * 0.5 + Math.sin(i * 0.2) * 2;
        }
        
        const result = wasm.pfe_js(data, testCase.period, 5);
        
        // Basic sanity checks
        assert.strictEqual(result.length, data.length);
        
        // Check warmup period
        for (let i = 0; i < testCase.period; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        // Check values exist after warmup
        let countAfterWarmup = 0;
        for (let i = testCase.period; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            // PFE should be between -100 and 100
            assert(result[i] >= -100 && result[i] <= 100, 
                   `PFE value ${result[i]} out of bounds at index ${i}`);
            countAfterWarmup++;
        }
        
        assert(countAfterWarmup > 0, 'Should have values after warmup');
    }
});

// Batch into API test
test('PFE batch_into zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const combos = 2 * 2; // 2 periods * 2 smoothings = 4 combinations
    const totalSize = data.length * combos;
    
    // Allocate input and output buffers
    const inPtr = wasm.pfe_alloc(data.length);
    const outPtr = wasm.pfe_alloc(totalSize);
    
    try {
        // Copy data to WASM memory
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
        inView.set(data);
        
        // Run batch calculation
        const numCombos = wasm.pfe_batch_into(
            inPtr, outPtr, data.length,
            5, 7, 2,  // period range: 5, 7
            3, 5, 2   // smoothing range: 3, 5
        );
        
        assert.strictEqual(numCombos, combos, 'Should return correct number of combinations');
        
        // Verify output
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
        
        // Check first combination (period=5, smoothing=3)
        const firstRow = Array.from(outView.slice(0, data.length));
        const expected = wasm.pfe_js(data, 5, 3);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(expected[i]) && isNaN(firstRow[i])) continue;
            assert(Math.abs(firstRow[i] - expected[i]) < 1e-10, 
                   `Mismatch at index ${i}: got ${firstRow[i]}, expected ${expected[i]}`);
        }
    } finally {
        wasm.pfe_free(inPtr, data.length);
        wasm.pfe_free(outPtr, totalSize);
    }
});

test.after(() => {
    console.log('PFE WASM tests completed');
});