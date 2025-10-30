/**
 * WASM binding tests for RSI indicator.
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
        // Support both ESM and CJS outputs from wasm-pack (nodejs target is CJS)
        if (wasm && wasm.default) {
            wasm = wasm.default;
        }
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('RSI partial params', () => {
    // Test with default parameters - mirrors check_rsi_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.rsi_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('RSI accuracy', async () => {
    // Test RSI matches expected values from Rust tests - mirrors check_rsi_accuracy
    const close = new Float64Array(testData.close);
    const expectedLastFive = [43.42, 42.68, 41.62, 42.86, 39.01];
    
    const result = wasm.rsi_js(close, 14);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-2,  // Using 1e-2 as per Rust test
        "RSI last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('rsi', result, 'close', { period: 14 });
});

test('RSI default candles', () => {
    // Test RSI with default parameters - mirrors check_rsi_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.rsi_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('RSI zero period', () => {
    // Test RSI fails with zero period - mirrors check_rsi_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.rsi_js(inputData, 0);
    }, /Invalid period/);
});

test('RSI period exceeds length', () => {
    // Test RSI fails when period exceeds data length - mirrors check_rsi_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.rsi_js(dataSmall, 10);
    }, /Invalid period/);
});

test('RSI very small dataset', () => {
    // Test RSI fails with insufficient data - mirrors check_rsi_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.rsi_js(singlePoint, 14);
    }, /Invalid period|Not enough valid data/);
});

test('RSI empty input', () => {
    // Test RSI fails with empty input - mirrors check_rsi_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.rsi_js(empty, 14);
    }, /Input data slice is empty|Invalid period/);
});

test('RSI reinput', () => {
    // Test RSI applied twice (re-input) - mirrors check_rsi_reinput
    const close = new Float64Array(testData.close);
    
    // First pass with period 14
    const firstResult = wasm.rsi_js(close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass with period 5 - apply RSI to RSI output
    const secondResult = wasm.rsi_js(firstResult, 5);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // After warmup period (240), no NaN values should exist
    if (secondResult.length > 240) {
        for (let i = 240; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Found unexpected NaN at index ${i} after warmup period`);
        }
    }
});

test('RSI nan handling', () => {
    // Test RSI handles NaN values correctly - mirrors check_rsi_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.rsi_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i} after warmup period`);
        }
    }
});

test('RSI all nan input', () => {
    // Test RSI with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.rsi_js(allNaN, 14);
    }, /All values are NaN/);
});

test('RSI batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Using the new ergonomic batch API for single parameter
    const batchResult = wasm.rsi_batch(close, {
        period_range: [14, 14, 0]
    });
    
    // Should match single calculation
    const singleResult = wasm.rsi_js(close, 14);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('RSI batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 10, 14, 18
    const batchResult = wasm.rsi_batch(close, {
        period_range: [10, 18, 4]  // period range
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
        
        const singleResult = wasm.rsi_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('RSI batch metadata from result', () => {
    // Test that batch result includes correct parameter combinations
    const close = new Float64Array(20); // Need enough data for period 15
    close.fill(100);
    
    const result = wasm.rsi_batch(close, {
        period_range: [5, 15, 5]  // period: 5, 10, 15
    });
    
    // Should have 3 combinations
    assert.strictEqual(result.combos.length, 3);
    
    // Check combinations
    assert.strictEqual(result.combos[0].period, 5);
    assert.strictEqual(result.combos[1].period, 10);
    assert.strictEqual(result.combos[2].period, 15);
});

test('RSI batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.rsi_batch(close, {
        period_range: [10, 20, 10]  // 2 periods
    });
    
    // Should have 2 combinations
    assert.strictEqual(batchResult.combos.length, 2);
    assert.strictEqual(batchResult.rows, 2);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 2 * 50);
    
    // Verify structure
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const period = batchResult.combos[combo].period;
        const rowStart = combo * 50;
        const rowData = batchResult.values.slice(rowStart, rowStart + 50);
        
        // First period-1 values should be NaN
        for (let i = 0; i < period; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        // After warmup should have values
        for (let i = period; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('RSI batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep
    const singleBatch = wasm.rsi_batch(close, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.rsi_batch(close, {
        period_range: [5, 7, 10]  // Step larger than range
    });
    
    // Should only have period=5
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.rsi_batch(new Float64Array([]), {
            period_range: [14, 14, 0]
        });
    }, /Invalid period|empty/i);
});

// Zero-copy API tests
test('RSI zero-copy API', () => {
    const data = new Float64Array([
        45.15, 46.26, 46.50, 46.23, 46.08, 46.03, 46.83, 47.69,
        47.54, 49.25, 49.23, 48.20, 47.57, 47.61, 48.08, 47.21,
        46.76, 46.68, 46.21, 47.47, 47.98, 47.13, 46.58, 46.03,
        46.54, 46.79, 47.05, 47.49, 47.27, 47.96, 47.24
    ]);
    const period = 14;
    
    // Allocate buffer
    const ptr = wasm.rsi_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute RSI in-place
    try {
        wasm.rsi_into(ptr, ptr, data.length, period);
        
        // Verify results match regular API
        const regularResult = wasm.rsi_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.rsi_free(ptr, data.length);
    }
});

test('RSI zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = 50 + Math.sin(i * 0.01) * 20 + Math.random() * 5;
    }
    
    const ptr = wasm.rsi_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.rsi_into(ptr, ptr, size, 14);
        
        // Recreate view in case memory grew
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        // Check warmup period has NaN
        for (let i = 0; i < 14; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = 14; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
            // RSI should be between 0 and 100
            assert(memView2[i] >= 0 && memView2[i] <= 100, 
                   `RSI value out of range at ${i}: ${memView2[i]}`);
        }
    } finally {
        wasm.rsi_free(ptr, size);
    }
});

test('RSI zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.rsi_into(0, 0, 10, 14);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.rsi_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.rsi_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        // Period exceeds length
        assert.throws(() => {
            wasm.rsi_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.rsi_free(ptr, 10);
    }
});

// Memory leak prevention test
test('RSI zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000, 100000];
    
    for (const size of sizes) {
        const ptr = wasm.rsi_alloc(size);
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
        wasm.rsi_free(ptr, size);
    }
});

test.after(() => {
    console.log('RSI WASM tests completed');
});
