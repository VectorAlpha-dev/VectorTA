/**
 * WASM binding tests for SuperSmoother indicator.
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

test('SuperSmoother partial params', () => {
    // Test with default parameters - mirrors check_supersmoother_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.supersmoother_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('SuperSmoother accuracy', async () => {
    // Test SuperSmoother matches expected values from Rust tests - mirrors check_supersmoother_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.supersmoother;
    
    const result = wasm.supersmoother_js(
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
        "SuperSmoother last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('supersmoother', result, 'close', expected.defaultParams);
});

test('SuperSmoother default candles', () => {
    // Test SuperSmoother with default parameters - mirrors check_supersmoother_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.supersmoother_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('SuperSmoother zero period', () => {
    // Test SuperSmoother fails with zero period - mirrors check_supersmoother_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.supersmoother_js(inputData, 0);
    }, /Invalid period/);
});

test('SuperSmoother period exceeds length', () => {
    // Test SuperSmoother fails when period exceeds data length - mirrors check_supersmoother_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.supersmoother_js(dataSmall, 10);
    }, /Invalid period/);
});

test('SuperSmoother very small dataset', () => {
    // Test SuperSmoother fails with insufficient data - mirrors check_supersmoother_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.supersmoother_js(singlePoint, 14);
    }, /Invalid period|Not enough valid data/);
});

test('SuperSmoother empty input', () => {
    // Test SuperSmoother fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.supersmoother_js(empty, 14);
    }, /Empty data/);
});

test('SuperSmoother reinput', () => {
    // Test SuperSmoother applied twice (re-input) - mirrors check_supersmoother_reinput
    const close = new Float64Array(testData.close);
    
    // First pass with period=14
    const firstResult = wasm.supersmoother_js(close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass with period=10 - apply SuperSmoother to SuperSmoother output
    const secondResult = wasm.supersmoother_js(firstResult, 10);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // The Rust test only verifies that re-input works and produces same length
    // It doesn't check specific values
});

test('SuperSmoother NaN handling', () => {
    // Test SuperSmoother handles NaN values correctly - mirrors check_supersmoother_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.supersmoother_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First 13 values should be NaN (warmup_period = first + period - 1 = 0 + 14 - 1 = 13)
    assertAllNaN(result.slice(0, 13), "Expected NaN in warmup period");
    
    // Values at indices 13 and 14 should be initialized
    assert(!isNaN(result[13]), "Value at index 13 should be initialized");
    if (result.length > 14) {
        assert(!isNaN(result[14]), "Value at index 14 should be initialized");
    }
});

test('SuperSmoother all NaN input', () => {
    // Test SuperSmoother with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.supersmoother_js(allNaN, 14);
    }, /All values are NaN/);
});

test('SuperSmoother batch single parameter', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Get batch result
    const batchResult = wasm.supersmoother_batch_js(close, 14, 14, 0);
    
    // Get metadata
    const metadata = wasm.supersmoother_batch_metadata_js(14, 14, 0);
    assert.strictEqual(metadata.length, 1);
    assert.strictEqual(metadata[0], 14);
    
    // Should match single calculation
    const singleResult = wasm.supersmoother_js(close, 14);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('SuperSmoother batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 10, 15, 20
    const batchResult = wasm.supersmoother_batch_js(close, 10, 20, 5);
    
    // Get metadata
    const metadata = wasm.supersmoother_batch_metadata_js(10, 20, 5);
    assert.strictEqual(metadata.length, 3);
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.length, 3 * 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.supersmoother_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('SuperSmoother batch metadata', () => {
    // Test metadata generation
    const metadata = wasm.supersmoother_batch_metadata_js(5, 15, 5);
    
    // Should return [5, 10, 15]
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 5);
    assert.strictEqual(metadata[1], 10);
    assert.strictEqual(metadata[2], 15);
    
    // Test with step = 0 (single value)
    const singleMeta = wasm.supersmoother_batch_metadata_js(7, 7, 0);
    assert.strictEqual(singleMeta.length, 1);
    assert.strictEqual(singleMeta[0], 7);
});

test('SuperSmoother batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep
    const singleBatch = wasm.supersmoother_batch_js(close, 5, 5, 0);
    const singleMeta = wasm.supersmoother_batch_metadata_js(5, 5, 0);
    
    assert.strictEqual(singleBatch.length, 10);
    assert.strictEqual(singleMeta.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.supersmoother_batch_js(close, 5, 7, 10); // Step larger than range
    const largeMeta = wasm.supersmoother_batch_metadata_js(5, 7, 10);
    
    // Should only have period=5
    assert.strictEqual(largeBatch.length, 10);
    assert.strictEqual(largeMeta.length, 1);
    assert.strictEqual(largeMeta[0], 5);
});

test('SuperSmoother leading NaNs', () => {
    // Test handling of leading NaN values
    const data = new Float64Array(20);
    for (let i = 0; i < 5; i++) {
        data[i] = NaN;
    }
    for (let i = 5; i < 20; i++) {
        data[i] = i - 4; // 1, 2, 3, ...
    }
    
    const period = 3;
    const result = wasm.supersmoother_js(data, period);
    
    // For 2-pole supersmoother with leading NaNs:
    // first_non_nan = 5
    // warmup = first_non_nan + period - 1 = 5 + 3 - 1 = 7
    // Initial values at indices 7 and 8
    // Main calculation starts at index 9
    
    // Check that NaN input produces NaN output
    for (let i = 0; i < 5; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} where input is NaN`);
    }
    
    // Due to warmup, values remain NaN
    for (let i = 5; i < 7; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} due to warmup`);
    }
    
    // Initial values should be set from data
    assert.strictEqual(result[7], data[7], "Expected initial value at index 7");
    assert.strictEqual(result[8], data[8], "Expected initial value at index 8");
});

test('SuperSmoother consistency', () => {
    // Test that multiple runs produce identical results
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 3;
    
    // Run multiple times to ensure consistency
    const result1 = wasm.supersmoother_js(data, period);
    const result2 = wasm.supersmoother_js(data, period);
    
    // Results should be identical
    for (let i = 0; i < result1.length; i++) {
        if (isNaN(result1[i]) && isNaN(result2[i])) continue;
        assert.strictEqual(result1[i], result2[i], `Inconsistent result at index ${i}`);
    }
});

// Zero-copy API tests
test('SuperSmoother zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    
    // Allocate buffer
    const ptr = wasm.supersmoother_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute SuperSmoother in-place
    try {
        wasm.supersmoother_into(ptr, ptr, data.length, period);
        
        // Verify results match regular API
        const regularResult = wasm.supersmoother_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.supersmoother_free(ptr, data.length);
    }
});

test('SuperSmoother zero-copy with large dataset', () => {
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.supersmoother_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.supersmoother_into(ptr, ptr, size, 14);
        
        // Recreate view in case memory grew
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        // Check warmup period has NaN
        for (let i = 0; i < 13; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = 13; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.supersmoother_free(ptr, size);
    }
});

// Error handling for zero-copy API
test('SuperSmoother zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.supersmoother_into(0, 0, 10, 14);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.supersmoother_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.supersmoother_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        // Period > length
        assert.throws(() => {
            wasm.supersmoother_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.supersmoother_free(ptr, 10);
    }
});

// Memory management test
test('SuperSmoother zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.supersmoother_alloc(size);
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
        wasm.supersmoother_free(ptr, size);
    }
});

// New ergonomic batch API tests (if implemented)
test('SuperSmoother batch - new unified API', () => {
    // Test the new unified API if it exists
    const close = new Float64Array(testData.close.slice(0, 50));
    
    // Check if new API exists
    if (typeof wasm.supersmoother_batch !== 'function') {
        console.log('New unified batch API not yet implemented');
        return;
    }
    
    // Test with single parameter
    const result = wasm.supersmoother_batch(close, {
        period_range: [14, 14, 0]
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
    assert.strictEqual(combo.period, 14);
});

test('SuperSmoother batch full parameter sweep', () => {
    // Test full parameter sweep
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.supersmoother_batch_js(close, 10, 20, 10);
    const metadata = wasm.supersmoother_batch_metadata_js(10, 20, 10);
    
    // Should have 2 periods: 10, 20
    assert.strictEqual(metadata.length, 2);
    assert.strictEqual(batchResult.length, 2 * 50);
    
    // Verify structure
    for (let combo = 0; combo < metadata.length; combo++) {
        const period = metadata[combo];
        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);
        
        // First period-1 values should be NaN
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        // After warmup should have values
        for (let i = period - 1; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('Compare SuperSmoother with Rust implementation', async () => {
    const close = new Float64Array(testData.close);
    const period = 14;
    
    // Get WASM result
    const wasmResult = wasm.supersmoother_js(close, period);
    
    // Compare with Rust
    const result = await compareWithRust('supersmoother', wasmResult, 'close', { period });
    
    // compareWithRust will throw if there's a mismatch
    assert.ok(result, 'Comparison with Rust succeeded');
});

test.after(() => {
    console.log('SuperSmoother WASM tests completed');
});