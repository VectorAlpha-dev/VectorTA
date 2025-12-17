/**
 * WASM binding tests for APO indicator.
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

test('APO partial params', () => {
    // Test with default parameters - mirrors check_apo_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.apo_js(close, 10, 20);
    assert.strictEqual(result.length, close.length);
});

test('APO accuracy', async () => {
    // Test APO matches expected values from Rust tests - mirrors check_apo_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.apo;
    
    const result = wasm.apo_js(
        close,
        expected.defaultParams.short_period,
        expected.defaultParams.long_period
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected with 1e-1 tolerance (as in Rust tests)
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-1,
        "APO last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('apo', result, 'single', expected.defaultParams);
});

test('APO default params', () => {
    // Test APO with default parameters
    const close = new Float64Array(testData.close);
    
    const result = wasm.apo_js(close, 10, 20);
    assert.strictEqual(result.length, close.length);
});

test('APO zero period', () => {
    // Test APO fails with zero period - mirrors check_apo_zero_period
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.apo_js(data, 0, 20);
    }, /Invalid period/);
});

test('APO period invalid', () => {
    // Test APO fails when short_period >= long_period - mirrors check_apo_period_invalid
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.apo_js(data, 20, 10);
    }, /short_period not less than long_period/);
    
    assert.throws(() => {
        wasm.apo_js(data, 10, 10);
    }, /short_period not less than long_period/);
});

test('APO very small dataset', () => {
    // Test APO fails with insufficient data - mirrors check_apo_very_small_dataset
    const single_point = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.apo_js(single_point, 9, 10);
    }, /Not enough valid data/);
});

test('APO reinput', () => {
    // Test APO applied twice (re-input) - mirrors check_apo_reinput
    const close = new Float64Array(testData.close);
    
    // First pass with specific parameters
    const firstResult = wasm.apo_js(close, 10, 20);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass with different parameters using first result as input
    const secondResult = wasm.apo_js(
        new Float64Array(firstResult),
        5,
        15
    );
    assert.strictEqual(secondResult.length, firstResult.length);
});

test('APO NaN handling', () => {
    // Test APO handles NaN values correctly - mirrors check_apo_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.apo_js(close, 10, 20);
    assert.strictEqual(result.length, close.length);
    
    // After index 30, no NaN values should exist
    if (result.length > 30) {
        for (let i = 30; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('APO all NaN input', () => {
    // Test APO with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.apo_js(allNaN, 10, 20);
    }, /All values are NaN/);
});

test('APO batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Single parameter set: short=10, long=20
    const batchResult = wasm.apo_batch_js(
        close,
        10, 10, 0,    // short_period range
        20, 20, 0     // long_period range
    );
    
    // Should match single calculation
    const singleResult = wasm.apo_js(close, 10, 20);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('APO batch multiple parameters', () => {
    // Test batch with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Multiple periods: short=5,10,15 and long=20,30
    const batchResult = wasm.apo_batch_js(
        close,
        5, 15, 5,     // short: 5, 10, 15
        20, 30, 10    // long: 20, 30
    );
    
    // Should have 3 * 2 = 6 valid combinations
    const expectedCombos = 6;
    assert.strictEqual(batchResult.length, expectedCombos * 100);
    
    // Verify first combination (short=5, long=20)
    const firstRow = batchResult.slice(0, 100);
    const singleResult = wasm.apo_js(close, 5, 20);
    assertArrayClose(firstRow, singleResult, 1e-10, "First combination mismatch");
});

test('APO batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.apo_batch_metadata_js(
        5, 15, 5,     // short: 5, 10, 15
        20, 30, 10    // long: 20, 30
    );
    
    // Should have 6 combinations * 2 values each = 12 values
    assert.strictEqual(metadata.length, 12);
    
    // Check values (short, long pairs)
    const expected = [
        5, 20,   // combo 0
        5, 30,   // combo 1
        10, 20,  // combo 2
        10, 30,  // combo 3
        15, 20,  // combo 4
        15, 30   // combo 5
    ];
    
    for (let i = 0; i < expected.length; i++) {
        assert.strictEqual(metadata[i], expected[i], `Metadata mismatch at index ${i}`);
    }
});

test('APO batch invalid combinations', () => {
    // Test batch processing with some invalid combinations
    const close = new Float64Array(testData.close.slice(0, 50));
    
    // This range includes invalid combos where short >= long
    const batchResult = wasm.apo_batch_js(
        close,
        10, 30, 10,   // short: 10, 20, 30
        15, 25, 10    // long: 15, 25
    );
    
    // Only valid combinations: (10,15), (10,25), (20,25)
    assert.strictEqual(batchResult.length, 3 * 50);
});

test('APO batch empty result', () => {
    // Test batch processing with no valid combinations
    const close = new Float64Array([1, 2, 3, 4, 5]);
    
    // All combinations would have short >= long
    assert.throws(() => {
        wasm.apo_batch_js(
            close,
            20, 30, 10,   // short: 20, 30
            10, 15, 5     // long: 10, 15
        );
    }, /Invalid period|invalid range/);
});

// New API tests
test('APO batch - new ergonomic API with single parameter', () => {
    // Test the new API with a single parameter combination
    const close = new Float64Array(testData.close);
    
    const result = wasm.apo_batch(close, {
        short_period_range: [10, 10, 0],
        long_period_range: [20, 20, 0]
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
    assert.strictEqual(combo.short_period, 10);
    assert.strictEqual(combo.long_period, 20);
    
    // Compare with old API
    const oldResult = wasm.apo_js(close, 10, 20);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10, 
               `Value mismatch at index ${i}`);
    }
});

test('APO batch - new API with multiple parameters', () => {
    // Test with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.apo_batch(close, {
        short_period_range: [5, 15, 5],   // 5, 10, 15
        long_period_range: [20, 30, 10]   // 20, 30
    });
    
    // Should have 6 combinations
    assert.strictEqual(result.rows, 6);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 6);
    assert.strictEqual(result.values.length, 300);
    
    // Verify each combo
    const expectedCombos = [
        [5, 20], [5, 30],
        [10, 20], [10, 30],
        [15, 20], [15, 30]
    ];
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].short_period, expectedCombos[i][0]);
        assert.strictEqual(result.combos[i].long_period, expectedCombos[i][1]);
    }
    
    // Extract and verify a specific row
    const secondRow = result.values.slice(result.cols, 2 * result.cols);
    assert.strictEqual(secondRow.length, 50);
    
    // Compare with old API for first combination
    const oldResult = wasm.apo_js(close, 5, 20);
    const firstRow = result.values.slice(0, result.cols);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('APO batch - new API matches old API results', () => {
    // Comprehensive comparison test
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const params = {
        short_period_range: [5, 15, 5],
        long_period_range: [20, 30, 5]
    };
    
    // Old API
    const oldValues = wasm.apo_batch_js(
        close,
        params.short_period_range[0], params.short_period_range[1], params.short_period_range[2],
        params.long_period_range[0], params.long_period_range[1], params.long_period_range[2]
    );
    
    // New API
    const newResult = wasm.apo_batch(close, params);
    
    // Should produce identical values
    assert.strictEqual(oldValues.length, newResult.values.length);
    
    for (let i = 0; i < oldValues.length; i++) {
        if (isNaN(oldValues[i]) && isNaN(newResult.values[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldValues[i] - newResult.values[i]) < 1e-10,
               `Value mismatch at index ${i}: old=${oldValues[i]}, new=${newResult.values[i]}`);
    }
});

test('APO batch - new API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    // Invalid config structure
    assert.throws(() => {
        wasm.apo_batch(close, {
            short_period_range: [10, 10]  // Missing step
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.apo_batch(close, {
            short_period_range: [5, 15, 5]
            // Missing long_period_range
        });
    }, /Invalid config/);
    
    // Invalid data type
    assert.throws(() => {
        wasm.apo_batch(close, {
            short_period_range: "invalid",
            long_period_range: [20, 30, 5]
        });
    }, /Invalid config/);
});

test('APO edge cases', () => {
    // Minimum valid data
    const minData = new Float64Array(20).fill(1.0); // Exactly long_period length
    const result = wasm.apo_js(minData, 10, 20);
    assert.strictEqual(result.length, minData.length);
    
    // First value should be 0.0 (short_ema - long_ema at initialization)
    assert.strictEqual(result[0], 0.0);
    
    // With constant prices, APO should converge to 0
    assert(Math.abs(result[result.length - 1]) < 1e-10, "APO should converge to 0 with constant prices");
});

// Fast API tests
test('APO fast API - basic operation', () => {
    const data = new Float64Array(testData.close);
    const len = data.length;
    
    // Allocate memory for input and output
    const inPtr = wasm.apo_alloc(len);
    const outPtr = wasm.apo_alloc(len);
    assert(inPtr !== 0, 'Failed to allocate input memory');
    assert(outPtr !== 0, 'Failed to allocate output memory');
    
    try {
        // Copy data to WASM memory
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(data);
        
        // Compute using fast API
        wasm.apo_into(inPtr, outPtr, len, 10, 20);
        
        // Read results back (convert to array for comparison)
        const resultView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const result = Array.from(resultView);
        
        // Compare with safe API
        const safeResult = wasm.apo_js(data, 10, 20);
        assertArrayClose(result, safeResult, 1e-10, "Fast API vs safe API mismatch");
    } finally {
        // Always free memory
        wasm.apo_free(inPtr, len);
        wasm.apo_free(outPtr, len);
    }
});

test('APO fast API - in-place operation (aliasing)', () => {
    const data = new Float64Array(testData.close);
    const len = data.length;
    
    // Create a copy for comparison
    const originalData = new Float64Array(data);
    
    // Allocate memory and copy data
    const ptr = wasm.apo_alloc(len);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    try {
        // Copy data to WASM memory
        const wasmData = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        wasmData.set(data);
        
        // Compute in-place (input and output are the same)
        wasm.apo_into(ptr, ptr, len, 10, 20);
        
        // Re-create view after computation (memory might have changed)
        const resultView = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        const result = Array.from(resultView);
        
        // Compare with safe API result
        const expectedResult = wasm.apo_js(originalData, 10, 20);
        assertArrayClose(result, expectedResult, 1e-10, "In-place operation mismatch");
    } finally {
        wasm.apo_free(ptr, len);
    }
});

test('APO fast API - error handling', () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    const len = data.length;
    
    // Test with null pointers
    assert.throws(() => {
        wasm.apo_into(0, 0, len, 10, 20);
    }, /Null pointer/);
    
    // Test with invalid parameters
    const inPtr = wasm.apo_alloc(len);
    const outPtr = wasm.apo_alloc(len);
    
    try {
        // Copy data to WASM memory
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(data);
        
        assert.throws(() => {
            wasm.apo_into(inPtr, outPtr, len, 0, 20);
        }, /Invalid period/);
        
        assert.throws(() => {
            wasm.apo_into(inPtr, outPtr, len, 20, 10);
        }, /short_period not less than long_period/);
    } finally {
        wasm.apo_free(inPtr, len);
        wasm.apo_free(outPtr, len);
    }
});

test('APO fast API - memory management', () => {
    const sizes = [100, 1000, 10000];
    
    sizes.forEach(size => {
        const ptr = wasm.apo_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Verify we can write to the memory
        const mem = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        mem[0] = 42.0;
        mem[size - 1] = 99.0;
        assert.strictEqual(mem[0], 42.0);
        assert.strictEqual(mem[size - 1], 99.0);
        
        // Free the memory
        wasm.apo_free(ptr, size);
    });
});

test.after(() => {
    console.log('APO WASM tests completed');
});
