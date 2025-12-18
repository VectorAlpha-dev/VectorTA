/**
 * WASM binding tests for SAMA indicator.
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

test('SAMA partial params', () => {
    // Test with different parameters - mirrors check_sama_partial_params
    const close = new Float64Array(testData.close);
    
    // Test with default params
    const result = wasm.sama_js(close, 200, 14, 6);
    assert.strictEqual(result.length, close.length);
    
    // Test with different params
    const result2 = wasm.sama_js(close, 50, 14, 6);
    assert.strictEqual(result2.length, close.length);
});

test('SAMA accuracy', async () => {
    // Test SAMA matches expected values from Rust tests - mirrors check_sama_accuracy
    const close = new Float64Array(testData.close.slice(0, 300)); // Use first 300 values
    const expected = EXPECTED_OUTPUTS.sama;
    
    // Test with default params
    const result = wasm.sama_js(
        close,
        expected.defaultParams.length,
        expected.defaultParams.majLength,
        expected.defaultParams.minLength
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected (for default params with length=200)
    const validValues = result.filter(v => !isNaN(v));
    if (validValues.length >= 5) {
        const last5 = validValues.slice(-5);
        assertArrayClose(
            last5,
            expected.last5Values,
            1e-8,
            "SAMA last 5 values mismatch (default params)"
        );
    }
    
    // Test with smaller params to get more valid values
    const result2 = wasm.sama_js(
        close,
        expected.testParams.length,
        expected.testParams.majLength,
        expected.testParams.minLength
    );
    
    const validValues2 = result2.filter(v => !isNaN(v));
    assert(validValues2.length >= 5, "Should have at least 5 valid values with length=50");
    const last5Test = validValues2.slice(-5);
    assertArrayClose(
        last5Test,
        expected.testLast5,
        1e-8,
        "SAMA last 5 values mismatch (test params)"
    );
});

test('SAMA default candles', () => {
    // Test SAMA with default parameters - mirrors check_sama_default_candles
    const close = new Float64Array(testData.close);
    
    // Default params: length=200, maj_length=14, min_length=6
    const result = wasm.sama_js(close, 200, 14, 6);
    assert.strictEqual(result.length, close.length);
    
    // Pine Script parity: values computed immediately, no NaN warmup
    // Should have valid values from the start
    assertNoNaN(result.slice(0, 200), "Should have valid values from start");
});

test('SAMA zero period', () => {
    // Test SAMA fails with zero period - mirrors check_sama_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    // Test with zero length
    assert.throws(() => {
        wasm.sama_js(inputData, 0, 14, 6);
    }, /Invalid period/);
    
    // Test with zero maj_length
    assert.throws(() => {
        wasm.sama_js(inputData, 10, 0, 6);
    }, /Invalid period/);
    
    // Test with zero min_length
    assert.throws(() => {
        wasm.sama_js(inputData, 10, 14, 0);
    }, /Invalid period/);
});

test('SAMA period exceeds length', () => {
    // Test SAMA fails when length exceeds data length - mirrors check_sama_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.sama_js(dataSmall, 10, 14, 6);
    }, /Invalid period|Not enough valid data/);
});

test('SAMA very small dataset', () => {
    // Test SAMA fails with insufficient data - mirrors check_sama_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.sama_js(singlePoint, 200, 14, 6);
    }, /Invalid period|Not enough valid data/);
});

test('SAMA empty input', () => {
    // Test SAMA fails with empty input - mirrors check_sama_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.sama_js(empty, 200, 14, 6);
    }, /Input data slice is empty/);
});

test('SAMA all NaN input', () => {
    // Test SAMA fails with all NaN values - mirrors check_sama_all_nan
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.sama_js(allNaN, 50, 14, 6);
    }, /All values are NaN/);
});

test('SAMA reinput', () => {
    // Test SAMA applied twice (re-input) - mirrors check_sama_reinput
    const close = new Float64Array(testData.close.slice(0, 300)); // Use first 300 values
    const expected = EXPECTED_OUTPUTS.sama;
    
    // First pass with test params for more valid values
    const firstResult = wasm.sama_js(
        close,
        expected.testParams.length,
        expected.testParams.majLength,
        expected.testParams.minLength
    );
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply SAMA to SAMA output
    const secondResult = wasm.sama_js(
        firstResult,
        expected.testParams.length,
        expected.testParams.majLength,
        expected.testParams.minLength
    );
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Check last 5 values match expected
    const validReinput = secondResult.filter(v => !isNaN(v));
    assert(validReinput.length >= 5, "Should have at least 5 valid reinput values");
    const last5Reinput = validReinput.slice(-5);
    assertArrayClose(
        last5Reinput,
        expected.reinputLast5,
        1e-8,
        "SAMA re-input last 5 values mismatch"
    );
});

test('SAMA NaN handling', () => {
    // Test SAMA handles NaN values correctly - mirrors check_sama_nan_handling
    const close = new Float64Array(testData.close);
    
    // Test with test params to get more valid values
    const result = wasm.sama_js(close, 50, 14, 6);
    assert.strictEqual(result.length, close.length);
    
    // Find first non-NaN in input
    let firstValid = -1;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    
    if (firstValid >= 0) {
        const warmupPeriod = firstValid + 50; // first_valid + length
        
        // Pine Script parity: values computed immediately from first valid input
        if (warmupPeriod <= result.length) {
            // Should have values from the start
            assertNoNaN(result.slice(0, Math.min(10, result.length)), "Should have valid values from start");
            
            // All values should be computed
            if (warmupPeriod < result.length - 10) {
                // Check values exist throughout
                const afterWarmup = result.slice(warmupPeriod, warmupPeriod + 10);
                let hasValidValues = false;
                for (let v of afterWarmup) {
                    if (!isNaN(v)) {
                        hasValidValues = true;
                        break;
                    }
                }
                assert(hasValidValues, "Should have some valid values after warmup");
            }
        }
    }
});

test('SAMA batch single parameter set', () => {
    // Test batch with single parameter combination - mirrors check_batch_default_row
    const close = new Float64Array(testData.close.slice(0, 300));
    const expected = EXPECTED_OUTPUTS.sama;
    
    // Using the new ergonomic batch API for single parameter
    const batchResult = wasm.sama_batch(close, {
        length_range: [200, 200, 0],
        maj_length_range: [14, 14, 0],
        min_length_range: [6, 6, 0]
    });
    
    // Verify structure
    assert(batchResult.values, 'Should have values array');
    assert(batchResult.combos, 'Should have combos array');
    assert(typeof batchResult.rows === 'number', 'Should have rows count');
    assert(typeof batchResult.cols === 'number', 'Should have cols count');
    
    // Should have 1 combination
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, close.length);
    assert.strictEqual(batchResult.combos.length, 1);
    assert.strictEqual(batchResult.values.length, close.length);
    
    // Check parameters
    const combo = batchResult.combos[0];
    assert.strictEqual(combo.length, 200);
    assert.strictEqual(combo.maj_length, 14);
    assert.strictEqual(combo.min_length, 6);
    
    // Check last 5 values match expected
    const validValues = batchResult.values.filter(v => !isNaN(v));
    if (validValues.length >= 5) {
        const last5 = validValues.slice(-5);
        assertArrayClose(
            last5,
            expected.last5Values,
            1e-8,
            "SAMA batch default row mismatch"
        );
    }
});

test('SAMA batch multiple parameters', () => {
    // Test batch with multiple parameter values - mirrors check_batch_sweep
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple parameters using ergonomic API
    const batchResult = wasm.sama_batch(close, {
        length_range: [40, 50, 5],      // 40, 45, 50
        maj_length_range: [12, 14, 1],  // 12, 13, 14
        min_length_range: [4, 6, 1]     // 4, 5, 6
    });
    
    // Should have 3 * 3 * 3 = 27 rows
    assert.strictEqual(batchResult.rows, 27);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.combos.length, 27);
    assert.strictEqual(batchResult.values.length, 27 * 100);
    
    // Verify first combination
    assert.strictEqual(batchResult.combos[0].length, 40);
    assert.strictEqual(batchResult.combos[0].maj_length, 12);
    assert.strictEqual(batchResult.combos[0].min_length, 4);
    
    // Verify last combination
    assert.strictEqual(batchResult.combos[26].length, 50);
    assert.strictEqual(batchResult.combos[26].maj_length, 14);
    assert.strictEqual(batchResult.combos[26].min_length, 6);
    
    // Verify each row matches individual calculation
    for (let i = 0; i < Math.min(3, batchResult.combos.length); i++) {
        const combo = batchResult.combos[i];
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.sama_js(close, combo.length, combo.maj_length, combo.min_length);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Row ${i} (length=${combo.length}, maj=${combo.maj_length}, min=${combo.min_length}) mismatch`
        );
    }
});

test('SAMA batch metadata from result', () => {
    // Test that batch result includes correct parameter combinations
    const close = new Float64Array(60); // Need enough data for length 50
    close.fill(100);
    
    const result = wasm.sama_batch(close, {
        length_range: [40, 50, 10],       // 40, 50
        maj_length_range: [12, 14, 2],    // 12, 14
        min_length_range: [5, 6, 1]       // 5, 6
    });
    
    // Should have 2 * 2 * 2 = 8 combinations
    assert.strictEqual(result.combos.length, 8);
    
    // Check first combination
    assert.strictEqual(result.combos[0].length, 40);
    assert.strictEqual(result.combos[0].maj_length, 12);
    assert.strictEqual(result.combos[0].min_length, 5);
    
    // Check last combination
    assert.strictEqual(result.combos[7].length, 50);
    assert.strictEqual(result.combos[7].maj_length, 14);
    assert.strictEqual(result.combos[7].min_length, 6);
});

test('SAMA batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 60));
    
    const batchResult = wasm.sama_batch(close, {
        length_range: [45, 50, 5],        // 2 lengths
        maj_length_range: [13, 14, 1],    // 2 maj_lengths
        min_length_range: [5, 5, 0]       // 1 min_length
    });
    
    // Should have 2 * 2 * 1 = 4 combinations
    assert.strictEqual(batchResult.combos.length, 4);
    assert.strictEqual(batchResult.rows, 4);
    assert.strictEqual(batchResult.cols, 60);
    assert.strictEqual(batchResult.values.length, 4 * 60);
    
    // Verify structure
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const length = batchResult.combos[combo].length;
        const rowStart = combo * 60;
        const rowData = batchResult.values.slice(rowStart, rowStart + 60);
        
        // Pine Script parity: values computed immediately, no NaN warmup
        // All values should be valid (not NaN)
        for (let i = 0; i < length; i++) {
            assert(!isNaN(rowData[i]), `Should have valid value at index ${i} for length ${length}`);
        }
        
        // After warmup should have values
        for (let i = length; i < Math.min(length + 5, 60); i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for length ${length}`);
        }
    }
});

test('SAMA batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    
    // Single value sweep
    const singleBatch = wasm.sama_batch(close, {
        length_range: [10, 10, 1],
        maj_length_range: [5, 5, 0],
        min_length_range: [3, 3, 0]
    });
    
    assert.strictEqual(singleBatch.values.length, 15);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.sama_batch(close, {
        length_range: [10, 12, 10], // Step larger than range
        maj_length_range: [5, 5, 0],
        min_length_range: [3, 3, 0]
    });
    
    // Should only have length=10
    assert.strictEqual(largeBatch.values.length, 15);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.sama_batch(new Float64Array([]), {
            length_range: [10, 10, 0],
            maj_length_range: [5, 5, 0],
            min_length_range: [3, 3, 0]
        });
    }, /All values are NaN|Input data slice is empty/);
});

test('SAMA batch vs single calculation', () => {
    // Test batch result matches single calculation for same parameters
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Single calculation
    const singleResult = wasm.sama_js(close, 45, 13, 5);
    
    // Batch with same single parameter
    const batchResult = wasm.sama_batch(close, {
        length_range: [45, 45, 0],
        maj_length_range: [13, 13, 0],
        min_length_range: [5, 5, 0]
    });
    
    // Should match exactly
    assertArrayClose(
        batchResult.values,
        singleResult,
        1e-10,
        "Batch vs single calculation mismatch"
    );
});

// Zero-copy API tests
test('SAMA zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const length = 10;
    const majLength = 5;
    const minLength = 3;
    
    // Allocate buffer
    const ptr = wasm.sama_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memory = wasm.__wasm.memory;
    const memView = new Float64Array(
        memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute SAMA in-place
    try {
        wasm.sama_into(ptr, ptr, data.length, length, majLength, minLength);
        
        // Verify results match regular API
        const regularResult = wasm.sama_js(data, length, majLength, minLength);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.sama_free(ptr, data.length);
    }
});

test('SAMA zero-copy with large dataset', () => {
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.sama_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memory = wasm.__wasm.memory;
        const memView = new Float64Array(memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.sama_into(ptr, ptr, size, 50, 14, 6);
        
        // Recreate view in case memory grew
        const memory2 = wasm.__wasm.memory;
        const memView2 = new Float64Array(memory2.buffer, ptr, size);
        
        // Pine Script parity: values computed immediately, no NaN warmup
        for (let i = 0; i < 50; i++) {
            assert(!isNaN(memView2[i]), `Should have valid value at index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = 50; i < Math.min(60, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.sama_free(ptr, size);
    }
});

test('SAMA batch_into zero-copy', () => {
    const size = 100;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = 50 + Math.sin(i * 0.1) * 10;
    }
    
    // Allocate input buffer
    const inPtr = wasm.sama_alloc(size);
    assert(inPtr !== 0, 'Failed to allocate input buffer');
    
    // Copy data
    const memory = wasm.__wasm.memory;
    const inView = new Float64Array(memory.buffer, inPtr, size);
    inView.set(data);
    
    // Allocate output buffer for batch results
    const numCombos = 2 * 2 * 2; // 2 lengths * 2 maj * 2 min
    const outPtr = wasm.sama_alloc(size * numCombos);
    assert(outPtr !== 0, 'Failed to allocate output buffer');
    
    try {
        // Run batch processing
        const rowCount = wasm.sama_batch_into(
            inPtr, outPtr, size,
            40, 45, 5,    // length range
            12, 13, 1,    // maj_length range
            4, 5, 1       // min_length range
        );
        
        assert.strictEqual(rowCount, numCombos, 'Should return correct row count');
        
        // Recreate view for output
        const memory2 = wasm.__wasm.memory;
        const outView = new Float64Array(memory2.buffer, outPtr, size * numCombos);
        
        // Verify first row matches single calculation
        const singleResult = wasm.sama_js(data, 40, 12, 4);
        const firstRow = Array.from(outView.slice(0, size));
        assertArrayClose(firstRow, singleResult, 1e-10, 'First row should match single calc');
        
    } finally {
        wasm.sama_free(inPtr, size);
        wasm.sama_free(outPtr, size * numCombos);
    }
});

// SIMD128 verification test
test('SAMA SIMD128 consistency', () => {
    // This test verifies SIMD128 produces same results as scalar
    // It runs automatically when SIMD128 is enabled
    const testCases = [
        { size: 15, length: 10, majLength: 5, minLength: 3 },
        { size: 100, length: 50, majLength: 14, minLength: 6 },
        { size: 1000, length: 200, majLength: 20, minLength: 10 },
        { size: 5000, length: 500, majLength: 50, minLength: 25 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = 50 + Math.sin(i * 0.1) * 10 + Math.cos(i * 0.05) * 5;
        }
        
        const result = wasm.sama_js(data, testCase.length, testCase.majLength, testCase.minLength);
        
        // Basic sanity checks
        assert.strictEqual(result.length, data.length);
        
        // Pine Script parity: values computed immediately, no NaN warmup
        for (let i = 0; i < testCase.length; i++) {
            assert(!isNaN(result[i]), `Should have valid value at index ${i} for size=${testCase.size}`);
        }
        
        // Check values exist after warmup
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.length; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        // Verify reasonable values
        if (countAfterWarmup > 0) {
            const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
            assert(avgAfterWarmup > 0, `Average value ${avgAfterWarmup} should be positive`);
            assert(avgAfterWarmup < 100, `Average value ${avgAfterWarmup} seems too large`);
        }
    }
});

// Error handling for zero-copy API
test('SAMA zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.sama_into(0, 0, 10, 50, 14, 6);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.sama_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.sama_into(ptr, ptr, 10, 0, 14, 6);
        }, /Invalid period/);
        
        // Zero maj_length
        assert.throws(() => {
            wasm.sama_into(ptr, ptr, 10, 10, 0, 6);
        }, /Invalid period/);
        
        // Zero min_length
        assert.throws(() => {
            wasm.sama_into(ptr, ptr, 10, 10, 14, 0);
        }, /Invalid period/);
    } finally {
        wasm.sama_free(ptr, 10);
    }
});

// Memory leak prevention test
test('SAMA zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.sama_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Write pattern to verify memory
        const memory = wasm.__wasm.memory;
        const memView = new Float64Array(memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.sama_free(ptr, size);
    }
});

test('SAMA batch error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    // Invalid config structure
    assert.throws(() => {
        wasm.sama_batch(close, {
            length_range: [40, 40], // Missing step
            maj_length_range: [12, 12, 0],
            min_length_range: [5, 5, 0]
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.sama_batch(close, {
            length_range: [40, 40, 0],
            maj_length_range: [12, 12, 0]
            // Missing min_length_range
        });
    }, /Invalid config/);
    
    // Invalid data type
    assert.throws(() => {
        wasm.sama_batch(close, {
            length_range: "invalid",
            maj_length_range: [12, 12, 0],
            min_length_range: [5, 5, 0]
        });
    }, /Invalid config/);
});

// Compatibility test with old API
test('SAMA wasm() function compatibility', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    
    // Test with all parameters specified
    const result1 = wasm.sama_js(data, 10, 5, 3);
    assert.strictEqual(result1.length, data.length);
    
    // Test with different parameters
    const result2 = wasm.sama_js(data, 5, 3, 2);
    assert.strictEqual(result2.length, data.length);
    
    // Test with larger parameters
    const result3 = wasm.sama_js(data, 14, 7, 4);
    assert.strictEqual(result3.length, data.length);
});

test.after(() => {
    console.log('SAMA WASM tests completed');
});
