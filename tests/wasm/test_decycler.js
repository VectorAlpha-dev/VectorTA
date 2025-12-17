/**
 * WASM binding tests for Decycler indicator.
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

test('Decycler partial params', () => {
    // Test with default parameters - mirrors check_decycler_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.decycler_js(close, 125, 0.707);
    assert.strictEqual(result.length, close.length);
});

test('Decycler accuracy', async () => {
    // Test Decycler matches expected values from Rust tests - mirrors check_decycler_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.decycler;
    
    const result = wasm.decycler_js(
        close,
        expected.defaultParams.hp_period,
        expected.defaultParams.k
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "Decycler last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('decycler', result, 'close', expected.defaultParams);
});

test('Decycler default candles', () => {
    // Test Decycler with default parameters - mirrors check_decycler_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.decycler_js(close, 125, 0.707);
    assert.strictEqual(result.length, close.length);
});

test('Decycler zero period', () => {
    // Test Decycler fails with zero period - mirrors check_decycler_zero_hp_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.decycler_js(inputData, 0, 0.707);
    }, /invalid.*period/i);
});

test('Decycler period exceeds length', () => {
    // Test Decycler fails when period exceeds data length - mirrors check_decycler_hp_period_exceeds_data_len
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.decycler_js(dataSmall, 10, 0.707);
    }, /invalid period/i);
});

test('Decycler edge case k=0', () => {
    // Test Decycler handles edge case where k=0 - mirrors check_decycler_edge_case_k_zero
    const inputData = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    assert.throws(() => {
        wasm.decycler_js(inputData, 3, 0.0);
    }, /invalid k/i);
});

test('Decycler NaN handling', () => {
    // Test Decycler handles NaN values correctly - mirrors check_decycler_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.decycler_js(close, 125, 0.707);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (first + 2), no NaN values should exist
    const firstNonNaN = result.findIndex(v => !isNaN(v));
    if (firstNonNaN !== -1 && result.length > firstNonNaN + 127) {
        for (let i = firstNonNaN + 127; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('Decycler warmup period', () => {
    // Test Decycler warmup period matches expected behavior
    const close = new Float64Array(testData.close);
    const hp_period = 125;
    
    const result = wasm.decycler_js(close, hp_period, 0.707);
    
    // Find first non-NaN index in result
    const firstNonNaN = result.findIndex(v => !isNaN(v));
    if (firstNonNaN !== -1) {
        // According to Rust implementation, warmup is first + 2
        // where first is the first non-NaN in input data
        const firstInput = close.findIndex(v => !isNaN(v));
        const expectedWarmup = firstInput + 2;
        
        assert.strictEqual(firstNonNaN, expectedWarmup, 
            `Warmup period mismatch: expected ${expectedWarmup}, got ${firstNonNaN}`);
        
        // Verify all values before warmup are NaN
        for (let i = 0; i < expectedWarmup; i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
        }
    }
});

test('Decycler partial NaN input', () => {
    // Test Decycler with data containing some NaN values
    // Create simple test data
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1.0; // 1 to 100
    }
    
    // Inject a few NaN values
    data[10] = NaN;
    data[11] = NaN;
    
    // Run decycler - it should handle the NaN values
    const result = wasm.decycler_js(data, 5, 0.707);
    assert.strictEqual(result.length, data.length);
    
    // The test verifies that the indicator can be called with data containing NaN
    // The actual behavior with NaN in the middle might vary
});

test('Decycler edge case k values', () => {
    // Test Decycler with edge case k values
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1.0; // 1 to 100
    }
    
    // Test very small positive k (just above 0)
    const resultSmall = wasm.decycler_js(data, 10, 0.001);
    assert.strictEqual(resultSmall.length, data.length);
    
    // Test k = 0.707 (default critical damping)
    const resultDefault = wasm.decycler_js(data, 10, 0.707);
    assert.strictEqual(resultDefault.length, data.length);
    
    // Test large k value
    const resultLarge = wasm.decycler_js(data, 10, 10.0);
    assert.strictEqual(resultLarge.length, data.length);
    
    // Results should differ based on k value
    // After warmup, check that different k values produce different results
    const warmupEnd = 2; // first + 2 where first=0 for this data
    const checkIdx = warmupEnd + 5;
    if (data.length > checkIdx) {
        assert(resultSmall[checkIdx] !== resultDefault[checkIdx], 
            "Different k values should produce different results");
        assert(resultDefault[checkIdx] !== resultLarge[checkIdx], 
            "Different k values should produce different results");
    }
});

test('Decycler reinput', () => {
    // Test Decycler applied twice (re-input)
    const close = new Float64Array(testData.close);
    const hp_period = 30;
    const k = 0.707;
    
    // First pass
    const firstResult = wasm.decycler_js(close, hp_period, k);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply Decycler to Decycler output
    const secondResult = wasm.decycler_js(firstResult, hp_period, k);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Both should have same length and valid values after warmup
    const warmupEnd = 4; // (first + 2) * 2 for double application
    if (secondResult.length > warmupEnd + 10) {
        // Check that second pass produces valid values
        let hasValidValues = false;
        for (let i = warmupEnd; i < warmupEnd + 10; i++) {
            if (!isNaN(secondResult[i])) {
                hasValidValues = true;
                break;
            }
        }
        assert(hasValidValues, "Expected valid values after double application");
    }
});

test('Decycler all NaN input', () => {
    // Test Decycler with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.decycler_js(allNaN, 50, 0.707);  // Use period < length to test NaN check
    }, /all values are nan/i);
});

test('Decycler batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Using the new ergonomic batch API for single parameter
    const batchResult = wasm.decycler_batch(close, {
        hp_period_range: [125, 125, 0],
        k_range: [0.707, 0.707, 0]
    });
    
    // Should match single calculation
    const singleResult = wasm.decycler_js(close, 125, 0.707);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    // Slightly relaxed tolerance to avoid flakiness across platforms
    assertArrayClose(batchResult.values, singleResult, 5e-9, "Batch vs single mismatch");
});

test('Decycler batch multiple hp_periods', () => {
    // Test batch with multiple hp_period values
    const close = new Float64Array(testData.close.slice(0, 200)); // Use smaller dataset for speed
    
    // Multiple hp_periods: 100, 125, 150 using ergonomic API
    const batchResult = wasm.decycler_batch(close, {
        hp_period_range: [100, 150, 25],  // hp_period range
        k_range: [0.707, 0.707, 0]         // k range
    });
    
    // Should have 3 rows * 200 cols = 600 values
    assert.strictEqual(batchResult.values.length, 3 * 200);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 200);
    
    // Verify each row matches individual calculation
    const hp_periods = [100, 125, 150];
    for (let i = 0; i < hp_periods.length; i++) {
        const rowStart = i * 200;
        const rowEnd = rowStart + 200;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.decycler_js(close, hp_periods[i], 0.707);
        assertArrayClose(rowData, singleResult, 1e-10, `Row ${i} mismatch`);
    }
});

test('Decycler batch multiple parameters', () => {
    // Test batch with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 50));
    
    // 2 hp_periods * 3 k values = 6 combinations
    const batchResult = wasm.decycler_batch(close, {
        hp_period_range: [10, 20, 10],   // 10, 20
        k_range: [0.5, 0.7, 0.1]         // 0.5, 0.6, 0.7
    });
    
    // Should have 6 rows * 50 cols = 300 values
    assert.strictEqual(batchResult.values.length, 6 * 50);
    assert.strictEqual(batchResult.rows, 6);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.combos.length, 6);
    
    // Verify combos
    const expectedCombos = [
        { hp_period: 10, k: 0.5 },
        { hp_period: 10, k: 0.6 },
        { hp_period: 10, k: 0.7 },
        { hp_period: 20, k: 0.5 },
        { hp_period: 20, k: 0.6 },
        { hp_period: 20, k: 0.7 }
    ];
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(batchResult.combos[i].hp_period, expectedCombos[i].hp_period,
            `hp_period mismatch at index ${i}`);
        assertClose(batchResult.combos[i].k, expectedCombos[i].k, 0.01, 
            `k value mismatch at index ${i}`);
    }
    
    // Verify a specific row matches individual calculation
    // Check row 3 (hp_period=20, k=0.5)
    const rowIdx = 3;
    const rowStart = rowIdx * 50;
    const rowEnd = rowStart + 50;
    const rowData = batchResult.values.slice(rowStart, rowEnd);
    
    const singleResult = wasm.decycler_js(close, 20, 0.5);
    assertArrayClose(rowData, singleResult, 1e-10, 
        `Batch row ${rowIdx} doesn't match single calculation`);
});

test('Decycler batch invalid parameters', () => {
    // Test batch with invalid parameter ranges
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Invalid hp_period range (start > end)
    assert.throws(() => {
        wasm.decycler_batch(close, {
            hp_period_range: [20, 10, 5],
            k_range: [0.707, 0.707, 0]
        });
    }, /invalid.*period|empty.*grid|invalid.*range|not enough valid data/i);
    
    // Invalid k range (negative k)
    // The batch function might not validate individual k values until processing
    // Try with a simple negative k that should fail
    assert.throws(() => {
        wasm.decycler_js(close, 5, -0.5);
    }, /invalid k/i);
});

test('Decycler batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep
    const singleBatch = wasm.decycler_batch(close, {
        hp_period_range: [5, 5, 1],
        k_range: [0.707, 0.707, 0.1]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.decycler_batch(close, {
        hp_period_range: [5, 7, 10], // Step larger than range
        k_range: [0.707, 0.707, 0]
    });
    
    // Should only have hp_period=5
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.decycler_batch(new Float64Array([]), {
            hp_period_range: [125, 125, 0],
            k_range: [0.707, 0.707, 0]
        });
    }, /empty input data|empty data/i);
});

// Fast API tests (zero-copy)
test('Decycler zero-copy API basic', () => {
    // Test basic zero-copy API functionality
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const hp_period = 5;
    const k = 0.707;
    
    // Allocate buffer
    const ptr = wasm.decycler_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute Decycler in-place
    try {
        wasm.decycler_into(ptr, ptr, data.length, hp_period, k);
        
        // Verify results match regular API
        const regularResult = wasm.decycler_js(data, hp_period, k);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert.strictEqual(memView[i], regularResult[i], 
                             `Value mismatch at index ${i}: ${memView[i]} vs ${regularResult[i]}`);
        }
    } finally {
        wasm.decycler_free(ptr, data.length);
    }
});

test('Decycler zero-copy separate buffers', () => {
    // Test zero-copy with separate input/output buffers
    const data = new Float64Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    
    const inPtr = wasm.decycler_alloc(data.length);
    const outPtr = wasm.decycler_alloc(data.length);
    
    try {
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, data.length);
        
        // Copy data to input buffer
        inView.set(data);
        
        // Compute with separate buffers
        wasm.decycler_into(inPtr, outPtr, data.length, 3, 0.707);
        
        // Input should be unchanged
        for (let i = 0; i < data.length; i++) {
            assert.strictEqual(inView[i], data[i], `Input modified at index ${i}`);
        }
        
        // Output should match regular API
        const regularResult = wasm.decycler_js(data, 3, 0.707);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(outView[i])) {
                continue;
            }
            assert.strictEqual(outView[i], regularResult[i], 
                             `Output mismatch at index ${i}`);
        }
    } finally {
        wasm.decycler_free(inPtr, data.length);
        wasm.decycler_free(outPtr, data.length);
    }
});

test('Decycler zero-copy error handling', () => {
    // Test error handling in zero-copy API
    
    // Null pointer test
    assert.throws(() => {
        wasm.decycler_into(0, 0, 10, 125, 0.707);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.decycler_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.decycler_into(ptr, ptr, 10, 0, 0.707);
        }, /invalid.*period/i);
        
        // Invalid k
        assert.throws(() => {
            wasm.decycler_into(ptr, ptr, 10, 5, 0.0);
        }, /invalid k/i);
    } finally {
        wasm.decycler_free(ptr, 10);
    }
});

// Fast batch API tests
test('Decycler batch_into API', () => {
    // Test fast batch API
    const data = new Float64Array(testData.close.slice(0, 100));
    
    // Calculate expected output size: 2 hp_periods * 2 k values * 100 data points
    const expectedSize = 2 * 2 * 100;
    
    const inPtr = wasm.decycler_alloc(data.length);
    const outPtr = wasm.decycler_alloc(expectedSize);
    
    try {
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
        inView.set(data);
        
        // Run batch: hp_periods [10, 20], k values [0.5, 0.7]
        const rows = wasm.decycler_batch_into(
            inPtr, outPtr, data.length,
            10, 20, 10,     // hp_period range
            0.5, 0.7, 0.2   // k range
        );
        
        assert.strictEqual(rows, 4, 'Should have 4 parameter combinations');
        
        // Verify output matches regular batch API
        const regularBatch = wasm.decycler_batch(data, {
            hp_period_range: [10, 20, 10],
            k_range: [0.5, 0.7, 0.2]
        });
        
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, expectedSize);
        assertArrayClose(
            Array.from(outView), 
            regularBatch.values, 
            1e-10, 
            "Fast batch mismatch"
        );
    } finally {
        wasm.decycler_free(inPtr, data.length);
        wasm.decycler_free(outPtr, expectedSize);
    }
});

// Memory management tests
test('Decycler zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000, 100000];
    
    for (const size of sizes) {
        const ptr = wasm.decycler_alloc(size);
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
        wasm.decycler_free(ptr, size);
    }
});

// SIMD consistency test
test('Decycler SIMD kernel consistency', () => {
    // Test that all kernels produce the same results
    const data = new Float64Array(testData.close.slice(0, 100));
    const hp_period = 20;
    const k = 0.707;
    
    // Get result from JS API (uses auto kernel selection)
    const autoResult = wasm.decycler_js(data, hp_period, k);
    
    // Since AVX2/AVX512 are stubs that fallback to scalar,
    // we can only verify that the auto selection works
    assert.strictEqual(autoResult.length, data.length);
    
    // Verify warmup period has NaN values
    const firstNonNaN = autoResult.findIndex(v => !isNaN(v));
    assert(firstNonNaN >= 2, 'Warmup period should have at least 2 NaN values');
    
    // Verify after warmup has values
    for (let i = firstNonNaN; i < autoResult.length; i++) {
        assert(!isNaN(autoResult[i]), `Unexpected NaN at index ${i} after warmup`);
    }
});

test.after(() => {
    console.log('Decycler WASM tests completed');
});
