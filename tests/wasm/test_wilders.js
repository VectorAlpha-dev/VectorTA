/**
 * WASM binding tests for WILDERS indicator.
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

test('Wilders partial params', () => {
    // Test with default parameters - mirrors check_wilders_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.wilders_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('Wilders accuracy', async () => {
    // Test WILDERS matches expected values from Rust tests - mirrors check_wilders_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.wilders;
    
    const result = wasm.wilders_js(
        close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "Wilders last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('wilders', result, 'close', expected.defaultParams);
});

test('Wilders default candles', () => {
    // Test Wilders with default parameters - mirrors check_wilders_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.wilders_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('Wilders zero period', () => {
    // Test Wilders fails with zero period - mirrors check_wilders_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.wilders_js(inputData, 0);
    }, /Invalid period/);
});

test('Wilders period exceeds length', () => {
    // Test Wilders fails when period exceeds data length - mirrors check_wilders_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.wilders_js(dataSmall, 10);
    }, /Invalid period/);
});

test('Wilders very small dataset', () => {
    // Test Wilders with very small dataset - mirrors check_wilders_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    // Should work with period=1
    const result = wasm.wilders_js(singlePoint, 1);
    assert.strictEqual(result.length, 1);
});

test('Wilders empty input', () => {
    // Test Wilders fails with empty input - mirrors check_wilders_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.wilders_js(empty, 5);
    }, /Input data slice is empty/);
});

test('Wilders NaN handling', () => {
    // Test Wilders handles NaN values correctly - mirrors check_wilders_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.wilders_js(close, 5);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First period-1 values should be NaN
    const warmupEnd = 4;
    assertAllNaN(result.slice(0, warmupEnd), "Expected NaN in warmup period");
    
    // Value at warmupEnd should be the first valid output
    assert(!isNaN(result[warmupEnd]), `Expected valid value at index ${warmupEnd} (first output)`);
});

test('Wilders all NaN input', () => {
    // Test Wilders with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.wilders_js(allNaN, 5);
    }, /All values are NaN/);
});

test('Wilders batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Single parameter set: period=5
    const batchResult = wasm.wilders_batch_js(
        close,
        5, 5, 0      // period range
    );
    
    // Should match single calculation
    const singleResult = wasm.wilders_js(close, 5);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('Wilders batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 5, 6, 7, 8, 9, 10
    const batchResult = wasm.wilders_batch_js(
        close,
        5, 10, 1      // period range
    );
    
    // Should have 6 rows * 100 cols = 600 values
    assert.strictEqual(batchResult.length, 6 * 100);
    
    // Verify each row matches individual calculation
    const periods = [5, 6, 7, 8, 9, 10];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.wilders_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('Wilders batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.wilders_batch_metadata_js(
        5, 10, 1      // period: 5, 6, 7, 8, 9, 10
    );
    
    // Should have 6 periods
    assert.strictEqual(metadata.length, 6);
    
    // Check values
    for (let i = 0; i < 6; i++) {
        assert.strictEqual(metadata[i], 5 + i);
    }
});

test('Wilders batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.wilders_batch_js(
        close,
        5, 7, 2      // 2 periods: 5, 7
    );
    
    const metadata = wasm.wilders_batch_metadata_js(
        5, 7, 2
    );
    
    // Should have 2 combinations
    assert.strictEqual(metadata.length, 2);
    assert.strictEqual(batchResult.length, 2 * 50);
    
    // Verify structure
    for (let combo = 0; combo < 2; combo++) {
        const period = metadata[combo];
        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);
        const warmupEnd = period - 1;
        
        // First period-1 values should be NaN
        for (let i = 0; i < warmupEnd; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        // Value at warmupEnd should be first valid output
        assert(!isNaN(rowData[warmupEnd]), `Expected valid value at index ${warmupEnd} for period ${period}`);
        
        // After warmup should have values
        for (let i = warmupEnd; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('Wilders batch unified API', () => {
    // Test the new unified batch API (if exposed)
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Check if unified API exists (it may be called wilders_batch without _js suffix)
    if (typeof wasm.wilders_batch === 'function') {
        const result = wasm.wilders_batch(close, {
            period_range: [5, 7, 1]  // periods 5, 6, 7
        });
        
        // Verify structure
        assert(result.values, 'Should have values array');
        assert(result.combos, 'Should have combos array');
        assert(typeof result.rows === 'number', 'Should have rows count');
        assert(typeof result.cols === 'number', 'Should have cols count');
        
        // Verify dimensions
        assert.strictEqual(result.rows, 3);
        assert.strictEqual(result.cols, 100);
        assert.strictEqual(result.combos.length, 3);
        assert.strictEqual(result.values.length, 300);
        
        // Verify parameters
        for (let i = 0; i < 3; i++) {
            assert.strictEqual(result.combos[i].period, 5 + i);
        }
        
        // Compare with old API for first combination
        const oldResult = wasm.wilders_js(close, 5);
        const firstRow = result.values.slice(0, 100);
        for (let i = 0; i < oldResult.length; i++) {
            if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
                   `Value mismatch at index ${i}`);
        }
    } else {
        // If unified API doesn't exist, just verify the old API still works
        const batchResult = wasm.wilders_batch_js(close, 5, 7, 1);
        assert.strictEqual(batchResult.length, 3 * 100);
    }
});

test('Wilders batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep
    const singleBatch = wasm.wilders_batch_js(
        close,
        5, 5, 1
    );
    
    assert.strictEqual(singleBatch.length, 10);
    
    // Step larger than range
    const largeBatch = wasm.wilders_batch_js(
        close,
        5, 7, 10 // Step larger than range
    );
    
    // Should only have period=5
    assert.strictEqual(largeBatch.length, 10);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.wilders_batch_js(
            new Float64Array([]),
            5, 5, 0
        );
    }, /All values are NaN/);
});

// Zero-copy API tests
test('Wilders zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    
    // Allocate buffer
    const ptr = wasm.wilders_alloc(data.length);
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
    
    // Compute Wilders in-place
    try {
        wasm.wilders_into(ptr, ptr, data.length, period);
        
        // Verify results match regular API
        const regularResult = wasm.wilders_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.wilders_free(ptr, data.length);
    }
});

test('Wilders zero-copy with large dataset', () => {
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.wilders_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memory = wasm.__wbindgen_memory();
        const memView = new Float64Array(memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.wilders_into(ptr, ptr, size, 5);
        
        // Recreate view in case memory grew
        const memory2 = wasm.__wbindgen_memory();
        const memView2 = new Float64Array(memory2.buffer, ptr, size);
        
        // Check warmup period has NaN
        for (let i = 0; i < 4; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = 4; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.wilders_free(ptr, size);
    }
});

test('Wilders zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.wilders_into(0, 0, 10, 5);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.wilders_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.wilders_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        // Period exceeds length
        assert.throws(() => {
            wasm.wilders_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.wilders_free(ptr, 10);
    }
});

test('Wilders memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.wilders_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Write pattern to verify memory
        const memory = wasm.__wbindgen_memory();
        const memView = new Float64Array(memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.wilders_free(ptr, size);
    }
});

test('Wilders NaN in initial window', () => {
    // Test that NaN in initial window is properly handled
    const data = new Float64Array([1.0, 2.0, NaN, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    assert.throws(() => {
        wasm.wilders_js(data, 5);
    }, /Not enough valid data/);
});

test.after(() => {
    console.log('Wilders WASM tests completed');
});