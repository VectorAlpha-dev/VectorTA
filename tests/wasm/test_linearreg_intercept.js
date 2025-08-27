/**
 * WASM binding tests for Linear Regression Intercept indicator.
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

test('Linear Regression Intercept partial params', () => {
    // Test with default parameters
    const close = new Float64Array(testData.close);
    
    const result = wasm.linearreg_intercept_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('Linear Regression Intercept accuracy', async () => {
    // Test matches expected values from Rust tests
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.linearreg_intercept;
    
    const result = wasm.linearreg_intercept_js(
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
        "Linear Regression Intercept last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('linearreg_intercept', result, 'close', expected.defaultParams);
});

test('Linear Regression Intercept default candles', () => {
    // Test with default parameters
    const close = new Float64Array(testData.close);
    
    const result = wasm.linearreg_intercept_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('Linear Regression Intercept zero period', () => {
    // Test fails with zero period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.linearreg_intercept_js(inputData, 0);
    }, /Invalid period/);
});

test('Linear Regression Intercept period exceeds length', () => {
    // Test fails when period exceeds data length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.linearreg_intercept_js(dataSmall, 10);
    }, /Invalid period/);
});

test('Linear Regression Intercept very small dataset', () => {
    // Test fails with insufficient data
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.linearreg_intercept_js(singlePoint, 14);
    }, /Invalid period|Not enough valid data/);
});

test('Linear Regression Intercept empty input', () => {
    // Test fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.linearreg_intercept_js(empty, 14);
    }, /Input data slice is empty/);
});

test('Linear Regression Intercept reinput', () => {
    // Test applied twice (re-input)
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.linearreg_intercept;
    
    // First pass
    const firstResult = wasm.linearreg_intercept_js(close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply to output
    const secondResult = wasm.linearreg_intercept_js(firstResult, 14);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Check last 5 values match expected (if provided)
    if (expected.reinputLast5) {
        const last5 = secondResult.slice(-5);
        assertArrayClose(
            last5,
            expected.reinputLast5,
            1e-8,
            "Linear Regression Intercept re-input last 5 values mismatch"
        );
    }
});

test('Linear Regression Intercept NaN handling', () => {
    // Test handles NaN values correctly
    const close = new Float64Array(testData.close);
    
    const result = wasm.linearreg_intercept_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    // First period-1 values should be NaN (warmup = first + period - 1)
    assertAllNaN(result.slice(0, 13), "Expected NaN in warmup period");
    
    // After warmup period, no NaN values should exist
    for (let i = 13; i < Math.min(100, result.length); i++) {
        assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
    }
});

test('Linear Regression Intercept all NaN input', () => {
    // Test with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.linearreg_intercept_js(allNaN, 14);
    }, /All values are NaN/);
});

test('Linear Regression Intercept period=1 edge case', () => {
    // Test period=1 returns input values directly
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    const result = wasm.linearreg_intercept_js(data, 1);
    
    // Period=1 should return the input values
    assertArrayClose(result, data, 1e-10, "Period=1 should return input values");
});

test('Linear Regression Intercept linear trend property', () => {
    // Test with perfect linear data: y = 2x + 10
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = 2.0 * i + 10.0;
    }
    const period = 10;
    
    const result = wasm.linearreg_intercept_js(data, period);
    
    // For perfect linear data, the intercept equals the value at window start
    const warmup = period - 1;
    for (let i = warmup + 5; i < warmup + 10; i++) {
        const windowStart = i - period + 1;
        const expected = data[windowStart];
        assertClose(result[i], expected, 1e-9, 
                   `Linear trend mismatch at index ${i}`);
    }
});

test('Linear Regression Intercept batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Using the new ergonomic batch API for single parameter
    const batchResult = wasm.linearreg_intercept_batch(close, {
        period_range: [14, 14, 0]
    });
    
    // Should match single calculation
    const singleResult = wasm.linearreg_intercept_js(close, 14);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('Linear Regression Intercept batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 10, 12, 14
    const batchResult = wasm.linearreg_intercept_batch(close, {
        period_range: [10, 14, 2]      // period range
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
        
        const singleResult = wasm.linearreg_intercept_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('Linear Regression Intercept batch metadata from result', () => {
    // Test that batch result includes correct parameter combinations
    const close = new Float64Array(20); // Need enough data for period 14
    close.fill(100);
    
    const result = wasm.linearreg_intercept_batch(close, {
        period_range: [10, 14, 2]      // period: 10, 12, 14
    });
    
    // Should have 3 combinations
    assert.strictEqual(result.combos.length, 3);
    
    // Check first combination
    assert.strictEqual(result.combos[0].period, 10);
    
    // Check last combination
    assert.strictEqual(result.combos[2].period, 14);
});

test('Linear Regression Intercept batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.linearreg_intercept_batch(close, {
        period_range: [10, 15, 5]      // 2 periods: 10, 15
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
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        // After warmup should have values
        for (let i = period - 1; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('Linear Regression Intercept batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep
    const singleBatch = wasm.linearreg_intercept_batch(close, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.linearreg_intercept_batch(close, {
        period_range: [5, 7, 10] // Step larger than range
    });
    
    // Should only have period=5
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.linearreg_intercept_batch(new Float64Array([]), {
            period_range: [10, 10, 0]
        });
    }, /Input data slice is empty/);
});

// Zero-copy API tests
test('Linear Regression Intercept zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    
    // Allocate buffer
    const ptr = wasm.linearreg_intercept_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute in-place
    try {
        wasm.linearreg_intercept_into(ptr, ptr, data.length, period);
        
        // Verify results match regular API
        const regularResult = wasm.linearreg_intercept_js(data, period);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.linearreg_intercept_free(ptr, data.length);
    }
});

test('Linear Regression Intercept zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.linearreg_intercept_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.linearreg_intercept_into(ptr, ptr, size, 12);
        
        // Recreate view in case memory grew
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        // Check warmup period has NaN
        for (let i = 0; i < 11; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = 11; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.linearreg_intercept_free(ptr, size);
    }
});

// Error handling for zero-copy API
test('Linear Regression Intercept zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.linearreg_intercept_into(0, 0, 10, 10);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.linearreg_intercept_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.linearreg_intercept_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        // Period exceeds length
        assert.throws(() => {
            wasm.linearreg_intercept_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.linearreg_intercept_free(ptr, 10);
    }
});

// Memory leak prevention test
test('Linear Regression Intercept zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000, 100000];
    
    for (const size of sizes) {
        const ptr = wasm.linearreg_intercept_alloc(size);
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
        wasm.linearreg_intercept_free(ptr, size);
    }
});

test('Linear Regression Intercept batch into API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const periods = 3; // 2, 4, 6
    const totalSize = periods * data.length;
    
    // Allocate buffers
    const inPtr = wasm.linearreg_intercept_alloc(data.length);
    const outPtr = wasm.linearreg_intercept_alloc(totalSize);
    
    try {
        // Copy input data
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, data.length);
        inView.set(data);
        
        // Run batch - should return number of rows
        const rows = wasm.linearreg_intercept_batch_into(
            inPtr, outPtr, data.length,
            2, 6, 2  // period range: 2, 4, 6
        );
        
        // Verify returned rows count
        assert.strictEqual(rows, 3, "batch_into should return number of rows");
        
        // Check results
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, totalSize);
        
        // Verify each batch matches individual calculation
        const expectedPeriods = [2, 4, 6];
        for (let i = 0; i < expectedPeriods.length; i++) {
            const period = expectedPeriods[i];
            const rowStart = i * data.length;
            const rowData = Array.from(outView.slice(rowStart, rowStart + data.length));
            
            const singleResult = wasm.linearreg_intercept_js(data, period);
            
            for (let j = 0; j < data.length; j++) {
                if (isNaN(singleResult[j]) && isNaN(rowData[j])) {
                    continue;
                }
                assert(Math.abs(singleResult[j] - rowData[j]) < 1e-10,
                       `Batch mismatch for period ${period} at index ${j}`);
            }
        }
    } finally {
        wasm.linearreg_intercept_free(inPtr, data.length);
        wasm.linearreg_intercept_free(outPtr, totalSize);
    }
});

test.after(() => {
    console.log('Linear Regression Intercept WASM tests completed');
});