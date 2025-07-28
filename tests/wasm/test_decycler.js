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
    }, /Invalid.*period/);
});

test('Decycler period exceeds length', () => {
    // Test Decycler fails when period exceeds data length - mirrors check_decycler_hp_period_exceeds_data_len
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.decycler_js(dataSmall, 10, 0.707);
    }, /Not enough valid data/);
});

test('Decycler edge case k=0', () => {
    // Test Decycler handles edge case where k=0 - mirrors check_decycler_edge_case_k_zero
    const inputData = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    assert.throws(() => {
        wasm.decycler_js(inputData, 3, 0.0);
    }, /Invalid k/);
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

test('Decycler all NaN input', () => {
    // Test Decycler with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.decycler_js(allNaN, 125, 0.707);
    }, /All values are NaN/);
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
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
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
        assert.strictEqual(batchResult.combos[i].hp_period, expectedCombos[i].hp_period);
        assertClose(batchResult.combos[i].k, expectedCombos[i].k, 0.01, "k value mismatch");
    }
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
    }, /Empty data/);
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
        }, /Invalid.*period/);
        
        // Invalid k
        assert.throws(() => {
            wasm.decycler_into(ptr, ptr, 10, 5, 0.0);
        }, /Invalid k/);
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