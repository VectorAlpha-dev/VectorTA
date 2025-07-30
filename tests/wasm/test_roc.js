/**
 * WASM binding tests for ROC indicator.
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

test('ROC partial params', () => {
    // Test with default parameters - mirrors check_roc_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.roc_js(close, 10);
    assert.strictEqual(result.length, close.length);
});

test('ROC accuracy', async () => {
    // Test ROC matches expected values from Rust tests - mirrors check_roc_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.roc;
    
    const result = wasm.roc_js(
        close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-7,
        "ROC last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('roc', result, 'close', expected.defaultParams);
});

test('ROC default candles', () => {
    // Test ROC with default parameters - mirrors check_roc_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.roc_js(close, 10);
    assert.strictEqual(result.length, close.length);
});

test('ROC zero period', () => {
    // Test ROC fails with zero period - mirrors check_roc_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.roc_js(inputData, 0);
    }, /Invalid period/);
});

test('ROC period exceeds length', () => {
    // Test ROC fails when period exceeds data length - mirrors check_roc_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.roc_js(dataSmall, 10);
    }, /Invalid period/);
});

test('ROC very small dataset', () => {
    // Test ROC fails with insufficient data - mirrors check_roc_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.roc_js(singlePoint, 9);
    }, /Invalid period|Not enough valid data/);
});

test('ROC empty input', () => {
    // Test ROC fails with empty input - mirrors check_roc_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.roc_js(empty, 10);
    }, /Input data slice is empty/);
});

test('ROC reinput', () => {
    // Test ROC applied twice (re-input) - mirrors check_roc_reinput
    const close = new Float64Array(testData.close);
    
    // First pass with period 14 (matching Rust test)
    const firstResult = wasm.roc_js(close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply ROC to ROC output
    const secondResult = wasm.roc_js(firstResult, 14);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Check that there are no NaN values after index 28 (matching Rust test)
    for (let i = 28; i < secondResult.length; i++) {
        assert(!isNaN(secondResult[i]), 
               `Expected no NaN after index 28, found NaN at ${i}`);
    }
});

test('ROC NaN handling', () => {
    // Test ROC handles NaN values correctly - mirrors check_roc_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.roc_js(close, 10);
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

test('ROC all NaN input', () => {
    // Test ROC with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.roc_js(allNaN, 10);
    }, /All values are NaN/);
});

test('ROC batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Using the new ergonomic batch API for single parameter
    const batchResult = wasm.roc_batch(close, {
        period_range: [10, 10, 0]
    });
    
    // Should match single calculation
    const singleResult = wasm.roc_js(close, 10);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('ROC batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 10, 12, 14 using ergonomic API
    const batchResult = wasm.roc_batch(close, {
        period_range: [10, 14, 2]  // period range
    });
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 12, 14];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.roc_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('ROC batch metadata from result', () => {
    // Test that batch result includes correct parameter combinations
    const close = new Float64Array(20); // Need enough data for period 14
    close.fill(100);
    
    const result = wasm.roc_batch(close, {
        period_range: [10, 14, 2]  // period: 10, 12, 14
    });
    
    // Should have 3 combinations
    assert.strictEqual(result.combos.length, 3);
    
    // Check first combination
    assert.strictEqual(result.combos[0].period, 10);
    
    // Check last combination
    assert.strictEqual(result.combos[2].period, 14);
});

test('ROC batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.roc_batch(close, {
        period_range: [10, 14, 2]  // 3 periods
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

test('ROC batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    
    // Single value sweep
    const singleBatch = wasm.roc_batch(close, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 15);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.roc_batch(close, {
        period_range: [5, 7, 10] // Step larger than range
    });
    
    // Should only have period=5
    assert.strictEqual(largeBatch.values.length, 15);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.roc_batch(new Float64Array([]), {
            period_range: [10, 10, 0]
        });
    }, /All values are NaN/);
});

// Zero-copy API tests
test('ROC zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    
    // Allocate buffer
    const ptr = wasm.roc_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute ROC in-place
    try {
        wasm.roc_into(ptr, ptr, data.length, period);
        
        // Verify results match regular API
        const regularResult = wasm.roc_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.roc_free(ptr, data.length);
    }
});

test('ROC zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1 + 100; // Ensure positive values
    }
    
    const ptr = wasm.roc_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.roc_into(ptr, ptr, size, 10);
        
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
    } finally {
        wasm.roc_free(ptr, size);
    }
});

// Error handling for zero-copy API
test('ROC zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.roc_into(0, 0, 10, 10);
    }, /null pointer|Null pointer/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.roc_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.roc_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        // Period larger than data
        assert.throws(() => {
            wasm.roc_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.roc_free(ptr, 10);
    }
});

// Memory leak prevention test
test('ROC zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000, 100000];
    
    for (const size of sizes) {
        const ptr = wasm.roc_alloc(size);
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
        wasm.roc_free(ptr, size);
    }
});

test.after(() => {
    console.log('ROC WASM tests completed');
});