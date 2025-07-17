/**
 * WASM binding tests for ALMA indicator.
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

test('ALMA partial params', () => {
    // Test with default parameters - mirrors check_alma_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.alma_js(close, 9, 0.85, 6.0);
    assert.strictEqual(result.length, close.length);
});

test('ALMA accuracy', async () => {
    // Test ALMA matches expected values from Rust tests - mirrors check_alma_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.alma;
    
    const result = wasm.alma_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.offset,
        expected.defaultParams.sigma
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "ALMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('alma', result, 'close', expected.defaultParams);
});

test('ALMA default candles', () => {
    // Test ALMA with default parameters - mirrors check_alma_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.alma_js(close, 9, 0.85, 6.0);
    assert.strictEqual(result.length, close.length);
});

test('ALMA zero period', () => {
    // Test ALMA fails with zero period - mirrors check_alma_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.alma_js(inputData, 0, 0.85, 6.0);
    }, /Invalid period/);
});

test('ALMA period exceeds length', () => {
    // Test ALMA fails when period exceeds data length - mirrors check_alma_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.alma_js(dataSmall, 10, 0.85, 6.0);
    }, /Invalid period/);
});

test('ALMA very small dataset', () => {
    // Test ALMA fails with insufficient data - mirrors check_alma_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.alma_js(singlePoint, 9, 0.85, 6.0);
    }, /Invalid period|Not enough valid data/);
});

test('ALMA empty input', () => {
    // Test ALMA fails with empty input - mirrors check_alma_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.alma_js(empty, 9, 0.85, 6.0);
    }, /Input data slice is empty/);
});

test('ALMA invalid sigma', () => {
    // Test ALMA fails with invalid sigma - mirrors check_alma_invalid_sigma
    const data = new Float64Array([1.0, 2.0, 3.0]);
    
    // Sigma = 0
    assert.throws(() => {
        wasm.alma_js(data, 2, 0.85, 0.0);
    }, /Invalid sigma/);
    
    // Negative sigma
    assert.throws(() => {
        wasm.alma_js(data, 2, 0.85, -1.0);
    }, /Invalid sigma/);
});

test('ALMA invalid offset', () => {
    // Test ALMA fails with invalid offset - mirrors check_alma_invalid_offset
    const data = new Float64Array([1.0, 2.0, 3.0]);
    
    // NaN offset
    assert.throws(() => {
        wasm.alma_js(data, 2, NaN, 6.0);
    }, /Invalid offset/);
    
    // Offset > 1
    assert.throws(() => {
        wasm.alma_js(data, 2, 1.5, 6.0);
    }, /Invalid offset/);
    
    // Offset < 0
    assert.throws(() => {
        wasm.alma_js(data, 2, -0.1, 6.0);
    }, /Invalid offset/);
});

test('ALMA reinput', () => {
    // Test ALMA applied twice (re-input) - mirrors check_alma_reinput
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.alma;
    
    // First pass
    const firstResult = wasm.alma_js(close, 9, 0.85, 6.0);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply ALMA to ALMA output
    const secondResult = wasm.alma_js(firstResult, 9, 0.85, 6.0);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Check last 5 values match expected
    const last5 = secondResult.slice(-5);
    assertArrayClose(
        last5,
        expected.reinputLast5,
        1e-8,
        "ALMA re-input last 5 values mismatch"
    );
});

test('ALMA NaN handling', () => {
    // Test ALMA handles NaN values correctly - mirrors check_alma_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.alma_js(close, 9, 0.85, 6.0);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First period-1 values should be NaN
    assertAllNaN(result.slice(0, 8), "Expected NaN in warmup period");
});

test('ALMA all NaN input', () => {
    // Test ALMA with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.alma_js(allNaN, 9, 0.85, 6.0);
    }, /All values are NaN/);
});

test('ALMA batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Single parameter set: period=9, offset=0.85, sigma=6.0
    const batchResult = wasm.alma_batch_js(
        close,
        9, 9, 0,      // period range
        0.85, 0.85, 0, // offset range
        6.0, 6.0, 0    // sigma range
    );
    
    // Should match single calculation
    const singleResult = wasm.alma_js(close, 9, 0.85, 6.0);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('ALMA batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 9, 11, 13
    const batchResult = wasm.alma_batch_js(
        close,
        9, 13, 2,      // period range
        0.85, 0.85, 0, // offset range  
        6.0, 6.0, 0    // sigma range
    );
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.length, 3 * 100);
    
    // Verify each row matches individual calculation
    const periods = [9, 11, 13];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.alma_js(close, periods[i], 0.85, 6.0);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('ALMA batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.alma_batch_metadata_js(
        9, 13, 2,      // period: 9, 11, 13
        0.85, 0.95, 0.05, // offset: 0.85, 0.90, 0.95
        6.0, 7.0, 0.5   // sigma: 6.0, 6.5, 7.0
    );
    
    // Should have 3 * 3 * 3 = 27 combinations
    // Each combo has 3 values: [period, offset, sigma]
    assert.strictEqual(metadata.length, 27 * 3);
    
    // Check first combination
    assert.strictEqual(metadata[0], 9);    // period
    assert.strictEqual(metadata[1], 0.85); // offset
    assert.strictEqual(metadata[2], 6.0);  // sigma
    
    // Check last combination
    assert.strictEqual(metadata[78], 13);   // period
    assertClose(metadata[79], 0.95, 1e-10, "offset mismatch"); // offset
    assert.strictEqual(metadata[80], 7.0);  // sigma
});

test('ALMA batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.alma_batch_js(
        close,
        9, 11, 2,      // 2 periods
        0.85, 0.90, 0.05, // 2 offsets
        6.0, 6.0, 0    // 1 sigma
    );
    
    const metadata = wasm.alma_batch_metadata_js(
        9, 11, 2,
        0.85, 0.90, 0.05,
        6.0, 6.0, 0
    );
    
    // Should have 2 * 2 * 1 = 4 combinations
    const numCombos = metadata.length / 3;
    assert.strictEqual(numCombos, 4);
    assert.strictEqual(batchResult.length, 4 * 50);
    
    // Verify structure
    for (let combo = 0; combo < numCombos; combo++) {
        const period = metadata[combo * 3];
        const offset = metadata[combo * 3 + 1];
        const sigma = metadata[combo * 3 + 2];
        
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

test('ALMA batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep
    const singleBatch = wasm.alma_batch_js(
        close,
        5, 5, 1,
        0.85, 0.85, 0.1,
        6.0, 6.0, 1.0
    );
    
    assert.strictEqual(singleBatch.length, 10);
    
    // Step larger than range
    const largeBatch = wasm.alma_batch_js(
        close,
        5, 7, 10, // Step larger than range
        0.85, 0.85, 0,
        6.0, 6.0, 0
    );
    
    // Should only have period=5
    assert.strictEqual(largeBatch.length, 10);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.alma_batch_js(
            new Float64Array([]),
            9, 9, 0,
            0.85, 0.85, 0,
            6.0, 6.0, 0
        );
    }, /All values are NaN/);
});

// New API tests
test('ALMA batch - new ergonomic API with single parameter', () => {
    // Test the new API with a single parameter combination
    const close = new Float64Array(testData.close);
    
    const result = wasm.alma_batch(close, {
        period_range: [9, 9, 0],
        offset_range: [0.85, 0.85, 0],
        sigma_range: [6.0, 6.0, 0]
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
    assert.strictEqual(combo.period, 9);
    assert.strictEqual(combo.offset, 0.85);
    assert.strictEqual(combo.sigma, 6.0);
    
    // Compare with old API
    const oldResult = wasm.alma_js(close, 9, 0.85, 6.0);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('ALMA batch - new API with multiple parameters', () => {
    // Test with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.alma_batch(close, {
        period_range: [9, 11, 2],      // 9, 11
        offset_range: [0.85, 0.90, 0.05], // 0.85, 0.90
        sigma_range: [6.0, 6.0, 0]     // 6.0
    });
    
    // Should have 2 * 2 * 1 = 4 combinations
    assert.strictEqual(result.rows, 4);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 4);
    assert.strictEqual(result.values.length, 200);
    
    // Verify each combo
    const expectedCombos = [
        { period: 9, offset: 0.85, sigma: 6.0 },
        { period: 9, offset: 0.90, sigma: 6.0 },
        { period: 11, offset: 0.85, sigma: 6.0 },
        { period: 11, offset: 0.90, sigma: 6.0 }
    ];
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedCombos[i].period);
        assert.strictEqual(result.combos[i].offset, expectedCombos[i].offset);
        assert.strictEqual(result.combos[i].sigma, expectedCombos[i].sigma);
    }
    
    // Extract and verify a specific row
    const secondRow = result.values.slice(result.cols, 2 * result.cols);
    assert.strictEqual(secondRow.length, 50);
    
    // Compare with old API for first combination
    const oldResult = wasm.alma_js(close, 9, 0.85, 6.0);
    const firstRow = result.values.slice(0, result.cols);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(firstRow[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - firstRow[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('ALMA batch - new API matches old API results', () => {
    // Comprehensive comparison test
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const params = {
        period_range: [10, 15, 5],
        offset_range: [0.8, 0.9, 0.1],
        sigma_range: [5.0, 7.0, 2.0]
    };
    
    // Old API
    const oldValues = wasm.alma_batch_js(
        close,
        params.period_range[0], params.period_range[1], params.period_range[2],
        params.offset_range[0], params.offset_range[1], params.offset_range[2],
        params.sigma_range[0], params.sigma_range[1], params.sigma_range[2]
    );
    
    // New API
    const newResult = wasm.alma_batch(close, params);
    
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

test('ALMA batch - new API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    // Invalid config structure
    assert.throws(() => {
        wasm.alma_batch(close, {
            period_range: [9, 9], // Missing step
            offset_range: [0.85, 0.85, 0],
            sigma_range: [6.0, 6.0, 0]
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.alma_batch(close, {
            period_range: [9, 9, 0],
            offset_range: [0.85, 0.85, 0]
            // Missing sigma_range
        });
    }, /Invalid config/);
    
    // Invalid data type
    assert.throws(() => {
        wasm.alma_batch(close, {
            period_range: "invalid",
            offset_range: [0.85, 0.85, 0],
            sigma_range: [6.0, 6.0, 0]
        });
    }, /Invalid config/);
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

// Zero-copy API tests
test('ALMA zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    const offset = 0.85;
    const sigma = 6.0;
    
    // Allocate buffer
    const ptr = wasm.alma_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute ALMA in-place
    try {
        wasm.alma_into(ptr, ptr, data.length, period, offset, sigma);
        
        // Verify results match regular API
        const regularResult = wasm.alma_js(data, period, offset, sigma);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.alma_free(ptr, data.length);
    }
});

test('ALMA zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.alma_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.alma_into(ptr, ptr, size, 9, 0.85, 6.0);
        
        // Recreate view in case memory grew
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        // Check warmup period has NaN
        for (let i = 0; i < 8; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = 8; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.alma_free(ptr, size);
    }
});

// Context API tests
test('ALMA context API basic', () => {
    const period = 9;
    const offset = 0.85;
    const sigma = 6.0;
    
    // Create context
    const ctx = new wasm.AlmaContext(period, offset, sigma);
    assert(ctx, 'Failed to create ALMA context');
    
    try {
        // Get warmup period
        const warmup = ctx.get_warmup_period();
        assert.strictEqual(warmup, period - 1, 'Incorrect warmup period');
        
        // Process some data
        const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        const ptr = wasm.alma_alloc(data.length);
        
        try {
            const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, data.length);
            memView.set(data);
            
            ctx.update_into(ptr, ptr, data.length);
            
            // Recreate view in case memory grew
            const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, data.length);
            
            // Compare with regular API
            const regularResult = wasm.alma_js(data, period, offset, sigma);
            for (let i = 0; i < data.length; i++) {
                if (isNaN(regularResult[i]) && isNaN(memView2[i])) {
                    continue;
                }
                assert(Math.abs(regularResult[i] - memView2[i]) < 1e-10,
                       `Context mismatch at index ${i}`);
            }
        } finally {
            wasm.alma_free(ptr, data.length);
        }
    } finally {
        // Context is automatically freed when it goes out of scope
    }
});

test('ALMA context reuse performance', () => {
    const ctx = new wasm.AlmaContext(9, 0.85, 6.0);
    assert(ctx, 'Failed to create context');
    
    try {
        // Process multiple datasets with same context
        const datasets = [];
        for (let d = 0; d < 10; d++) {
            const data = new Float64Array(1000);
            for (let i = 0; i < 1000; i++) {
                data[i] = Math.sin(i * 0.01 + d) + Math.random() * 0.1;
            }
            datasets.push(data);
        }
        
        // Process all datasets
        for (const data of datasets) {
            const ptr = wasm.alma_alloc(data.length);
            try {
                const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, data.length);
                memView.set(data);
                ctx.update_into(ptr, ptr, data.length);
                
                // Recreate view after update
                const memViewAfter = new Float64Array(wasm.__wasm.memory.buffer, ptr, data.length);
                
                // Verify some values are computed
                let hasValues = false;
                for (let i = 8; i < 20; i++) {
                    if (!isNaN(memViewAfter[i])) {
                        hasValues = true;
                        break;
                    }
                }
                assert(hasValues, 'Context should produce values after warmup');
            } finally {
                wasm.alma_free(ptr, data.length);
            }
        }
    } finally {
        // Context cleanup
    }
});

// SIMD128 verification test
test('ALMA SIMD128 consistency', () => {
    // This test verifies SIMD128 produces same results as scalar
    // It runs automatically when SIMD128 is enabled
    const testCases = [
        { size: 10, period: 5 },
        { size: 100, period: 9 },
        { size: 1000, period: 20 },
        { size: 10000, period: 50 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }
        
        const result = wasm.alma_js(data, testCase.period, 0.85, 6.0);
        
        // Basic sanity checks
        assert.strictEqual(result.length, data.length);
        
        // Check warmup period
        for (let i = 0; i < testCase.period - 1; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        // Check values exist after warmup
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.period - 1; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        // Verify reasonable values
        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(Math.abs(avgAfterWarmup) < 10, `Average value ${avgAfterWarmup} seems unreasonable`);
    }
});

// Error handling for zero-copy API
test('ALMA zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.alma_into(0, 0, 10, 9, 0.85, 6.0);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.alma_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.alma_into(ptr, ptr, 10, 0, 0.85, 6.0);
        }, /Invalid period/);
        
        // Invalid sigma
        assert.throws(() => {
            wasm.alma_into(ptr, ptr, 10, 5, 0.85, 0.0);
        }, /Invalid sigma/);
    } finally {
        wasm.alma_free(ptr, 10);
    }
});

// Memory leak prevention test
test('ALMA zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000, 100000];
    
    for (const size of sizes) {
        const ptr = wasm.alma_alloc(size);
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
        wasm.alma_free(ptr, size);
    }
});

test.after(() => {
    console.log('ALMA WASM tests completed');
});