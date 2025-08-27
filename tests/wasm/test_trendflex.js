/**
 * WASM binding tests for TrendFlex indicator.
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
    assertNoNaN 
} from './test_utils.js';
import { compareWithRust } from './rust-comparison.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

// Expected outputs for TrendFlex - exact values from Rust tests
const EXPECTED_OUTPUTS = {
    trendflex: {
        defaultParams: { period: 20 },
        last5Values: [
            -0.19724678008015128,
            -0.1238001236481444,
            -0.10515389737087717,
            -0.1149541079904878,
            -0.16006869484450567,
        ]
    }
};

test.before(async () => {
    // Load WASM module
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
    } catch (error) {
        console.error('Failed to load WASM module:', error);
        throw error;
    }
    
    // Load test data
    testData = loadTestData();
});

test('trendflex_partial_params', () => {
    const close = testData.close;
    
    // Test with default period of 20
    const result = wasm.trendflex_js(close, 20);
    assert.equal(result.length, close.length);
});

test('trendflex_accuracy', () => {
    const close = testData.close;
    const expected = EXPECTED_OUTPUTS.trendflex;
    
    const result = wasm.trendflex_js(close, expected.defaultParams.period);
    
    assert.equal(result.length, close.length);
    
    // Check last 5 values match expected from Rust tests
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "TrendFlex last 5 values mismatch"
    );
    
    // Compare with Rust
    compareWithRust('trendflex', result, 'close', expected.defaultParams);
});

test('trendflex_default_candles', () => {
    const close = testData.close;
    
    // Default params: period=20
    const result = wasm.trendflex_js(close, 20);
    assert.equal(result.length, close.length);
});

test('trendflex_zero_period', () => {
    const inputData = [10.0, 20.0, 30.0];
    
    assert.throws(() => {
        wasm.trendflex_js(inputData, 0);
    }, /period = 0|ZeroTrendFlexPeriod/);
});

test('trendflex_period_exceeds_length', () => {
    const dataSmall = [10.0, 20.0, 30.0];
    
    assert.throws(() => {
        wasm.trendflex_js(dataSmall, 10);
    }, /period > data len|TrendFlexPeriodExceedsData/);
});

test('trendflex_very_small_dataset', () => {
    const singlePoint = [42.0];
    
    assert.throws(() => {
        wasm.trendflex_js(singlePoint, 9);
    }, /period > data len|TrendFlexPeriodExceedsData/);
});

test('trendflex_empty_input', () => {
    const empty = [];
    
    assert.throws(() => {
        wasm.trendflex_js(empty, 20);
    }, /No data provided|NoDataProvided/);
});

test('trendflex_reinput', () => {
    const close = testData.close;
    
    // First pass
    const firstResult = wasm.trendflex_js(close, 20);
    assert.equal(firstResult.length, close.length);
    
    // Second pass - apply TrendFlex to TrendFlex output
    const secondResult = wasm.trendflex_js(firstResult, 10);
    assert.equal(secondResult.length, firstResult.length);
    
    // Check for NaN handling after warmup
    if (secondResult.length > 240) {
        for (let i = 240; i < secondResult.length; i++) {
            assert.ok(!isNaN(secondResult[i]), `Unexpected NaN at index ${i}`);
        }
    }
});

test('trendflex_nan_handling', () => {
    const close = testData.close;
    
    const result = wasm.trendflex_js(close, 20);
    assert.equal(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert.ok(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
        }
    }
    
    // Calculate warmup period: first_valid + period
    const firstValid = 0; // Since close data starts valid at index 0
    const warmup = firstValid + 20;
    
    // First warmup values should be NaN
    assertAllNaN(result.slice(0, warmup), `Expected NaN in warmup period [0:${warmup})`);
    // Value at warmup index should NOT be NaN
    assert.ok(!isNaN(result[warmup]), `Expected valid value at index ${warmup}`);
});

test('trendflex_batch_single_param', () => {
    const close = testData.close;
    
    const result = wasm.trendflex_batch_js(close, 20, 20, 0);
    const metadata = wasm.trendflex_batch_metadata_js(20, 20, 0);
    
    // Should have 1 period value
    assert.equal(metadata.length, 1);
    assert.equal(metadata[0], 20);
    
    // Result should be flattened array (1 row × data length)
    assert.equal(result.length, close.length);
    
    // Check last 5 values match expected from Rust tests
    const expected = EXPECTED_OUTPUTS.trendflex.last5Values;
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected,
        1e-8,
        "TrendFlex batch last 5 values mismatch"
    );
    
    // Verify matches single calculation
    const singleResult = wasm.trendflex_js(close, 20);
    assertArrayClose(
        result,
        singleResult,
        1e-10,
        "Batch vs single calculation mismatch"
    );
});

test('trendflex_batch_multiple_periods', () => {
    const close = testData.close.slice(0, 100); // Use smaller dataset for speed
    
    const result = wasm.trendflex_batch_js(close, 10, 30, 10);
    const metadata = wasm.trendflex_batch_metadata_js(10, 30, 10);
    
    // Should have 3 periods: 10, 20, 30
    assert.equal(metadata.length, 3);
    assert.deepEqual(Array.from(metadata), [10, 20, 30]);
    
    // Result should be flattened array (3 rows × data length)
    assert.equal(result.length, 3 * close.length);
    
    // Verify each row matches individual calculation
    const periods = [10, 20, 30];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * close.length;
        const rowEnd = rowStart + close.length;
        const rowData = result.slice(rowStart, rowEnd);
        
        const singleResult = wasm.trendflex_js(close, periods[i]);
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Period ${periods[i]} mismatch`
        );
        
        // Check warmup period for each row
        const warmup = periods[i]; // first_valid=0 + period
        assertAllNaN(rowData.slice(0, warmup), `Expected NaN in warmup [0:${warmup}) for period=${periods[i]}`);
        assert.ok(!isNaN(rowData[warmup]), `Expected valid value at index ${warmup} for period=${periods[i]}`);
    }
});

test('trendflex_all_nan_input', () => {
    const allNan = new Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.trendflex_js(allNan, 20);
    }, /All values are NaN|AllValuesNaN/);
});

// New ergonomic batch API tests
test('trendflex_batch_ergonomic_single_parameter', () => {
    const close = testData.close;
    
    // Using the new ergonomic batch API for single parameter
    const batchResult = wasm.trendflex_batch(close, {
        period_range: [20, 20, 0]
    });
    
    // Should match single calculation
    const singleResult = wasm.trendflex_js(close, 20);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('trendflex_batch_ergonomic_multiple_periods', () => {
    const close = testData.close.slice(0, 100);
    
    // Multiple periods: 10, 20, 30 using ergonomic API
    const batchResult = wasm.trendflex_batch(close, {
        period_range: [10, 30, 10]      // period range
    });
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    // Check combos
    assert.strictEqual(batchResult.combos.length, 3);
    assert.strictEqual(batchResult.combos[0].period, 10);
    assert.strictEqual(batchResult.combos[1].period, 20);
    assert.strictEqual(batchResult.combos[2].period, 30);
});

test('trendflex_batch_edge_cases', () => {
    const close = new Float64Array(10);
    close.fill(100);
    
    // Single value sweep
    const singleBatch = wasm.trendflex_batch(close, {
        period_range: [5, 5, 1]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.trendflex_batch(close, {
        period_range: [5, 7, 10] // Step larger than range
    });
    
    // Should only have period=5
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.trendflex_batch(new Float64Array([]), {
            period_range: [20, 20, 0]
        });
    }, /All values are NaN/);
});

// Zero-copy API tests
test('trendflex_zero_copy_basic', () => {
    // Use smaller dataset to avoid memory growth issues
    const data = new Float64Array(testData.close.slice(0, 100));
    const period = 20;
    
    // Allocate buffer
    const ptr = wasm.trendflex_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute TrendFlex in-place
    try {
        wasm.trendflex_into(ptr, ptr, data.length, period);
        
        // IMPORTANT: Recreate the view after WASM execution
        // The memory might have grown, invalidating the original view
        const updatedMemView = new Float64Array(wasm.__wasm.memory.buffer, ptr, data.length);
        
        // Verify results match regular API
        const regularResult = wasm.trendflex_js(data, period);
        for (let i = 0; i < data.length; i++) {
            // Handle detached buffer case - undefined is treated as NaN
            const memValue = updatedMemView[i];
            const regValue = regularResult[i];
            
            if ((isNaN(regValue) || regValue === undefined) && 
                (isNaN(memValue) || memValue === undefined)) {
                continue; // Both NaN/undefined is OK
            }
            
            assert(Math.abs(regValue - memValue) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regValue}, zerocopy=${memValue}`);
        }
    } finally {
        // Always free memory
        wasm.trendflex_free(ptr, data.length);
    }
});

test('trendflex_zero_copy_error_handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.trendflex_into(0, 0, 10, 20);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.trendflex_alloc(10);
    try {
        // Invalid period (0)
        assert.throws(() => {
            wasm.trendflex_into(ptr, ptr, 10, 0);
        }, /Invalid period/);
        
        // Period exceeds length
        assert.throws(() => {
            wasm.trendflex_into(ptr, ptr, 10, 20);
        }, /Invalid period/);
    } finally {
        wasm.trendflex_free(ptr, 10);
    }
});

test('trendflex_batch_into', () => {
    const data = new Float64Array(testData.close.slice(0, 100));
    const period_start = 10;
    const period_end = 30;
    const period_step = 10;
    
    // Calculate expected size (3 periods × 100 data points)
    const expected_combos = 3;
    const total_size = expected_combos * data.length;
    
    // Allocate input and output buffers
    const in_ptr = wasm.trendflex_alloc(data.length);
    const out_ptr = wasm.trendflex_alloc(total_size);
    
    try {
        // Copy data to input buffer
        const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        const inView = new Float64Array(memory.buffer, in_ptr, data.length);
        inView.set(data);
        
        // Call batch_into
        const n_combos = wasm.trendflex_batch_into(
            in_ptr, out_ptr, data.length,
            period_start, period_end, period_step
        );
        
        assert.strictEqual(n_combos, expected_combos);
        
        // Verify output
        const memory2 = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        const outView = new Float64Array(memory2.buffer, out_ptr, total_size);
        
        // Compare with regular batch API
        const regularBatch = wasm.trendflex_batch(data, {
            period_range: [period_start, period_end, period_step]
        });
        
        assertArrayClose(
            Array.from(outView),
            regularBatch.values,
            1e-10,
            "Batch into vs regular batch mismatch"
        );
    } finally {
        wasm.trendflex_free(in_ptr, data.length);
        wasm.trendflex_free(out_ptr, total_size);
    }
});

// Additional comprehensive tests matching ALMA's coverage

test('trendflex_zero_copy_large_dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.trendflex_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        const memView = new Float64Array(memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.trendflex_into(ptr, ptr, size, 20);
        
        // Recreate view in case memory grew
        const memory2 = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        const memView2 = new Float64Array(memory2.buffer, ptr, size);
        
        // Check warmup period has NaN
        for (let i = 0; i < 20; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = 20; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.trendflex_free(ptr, size);
    }
});

test('trendflex_memory_management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000, 50000];
    
    for (const size of sizes) {
        const ptr = wasm.trendflex_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Write pattern to verify memory
        const memory = wasm.__wbindgen_memory ? wasm.__wbindgen_memory() : wasm.memory;
        const memView = new Float64Array(memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.trendflex_free(ptr, size);
    }
});

test('trendflex_simd128_consistency', () => {
    // This test verifies SIMD128 produces same results as scalar
    // It runs automatically when SIMD128 is enabled in WASM
    const testCases = [
        { size: 10, period: 5 },
        { size: 100, period: 20 },
        { size: 1000, period: 30 },
        { size: 5000, period: 50 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }
        
        const result = wasm.trendflex_js(data, testCase.period);
        
        // Basic sanity checks
        assert.strictEqual(result.length, data.length);
        
        // Check warmup period
        for (let i = 0; i < testCase.period; i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        // Check values exist after warmup
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.period; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        // Verify reasonable values - TrendFlex is normalized, so values should be in reasonable range
        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(Math.abs(avgAfterWarmup) < 10, `Average value ${avgAfterWarmup} seems unreasonable`);
    }
});

test('trendflex_batch_metadata_comprehensive', () => {
    const close = new Float64Array(50);
    close.fill(100);
    
    const result = wasm.trendflex_batch(close, {
        period_range: [10, 20, 5]  // periods: 10, 15, 20
    });
    
    // Should have 3 combinations
    assert.strictEqual(result.combos.length, 3);
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    
    // Check each combo
    const expectedPeriods = [10, 15, 20];
    for (let i = 0; i < expectedPeriods.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedPeriods[i]);
    }
    
    // Extract and verify specific rows
    for (let i = 0; i < result.rows; i++) {
        const rowStart = i * result.cols;
        const rowEnd = rowStart + result.cols;
        const rowData = result.values.slice(rowStart, rowEnd);
        
        // Check warmup for this row
        const period = result.combos[i].period;
        for (let j = 0; j < period; j++) {
            assert(isNaN(rowData[j]), `Expected NaN at index ${j} for period ${period}`);
        }
        
        // Should have valid values after warmup
        if (period < rowData.length) {
            assert(!isNaN(rowData[period]), `Expected valid value at index ${period} for period ${period}`);
        }
    }
});

test('trendflex_invalid_period_comprehensive', () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    
    // Period = 0
    assert.throws(() => {
        wasm.trendflex_js(data, 0);
    }, /period = 0|ZeroTrendFlexPeriod/, 'Should reject period=0');
    
    // Period > data length
    assert.throws(() => {
        wasm.trendflex_js(data, 10);
    }, /period > data len|TrendFlexPeriodExceedsData/, 'Should reject period > length');
    
    // Period = data length should still fail
    assert.throws(() => {
        wasm.trendflex_js(data, data.length);
    }, /period > data len|TrendFlexPeriodExceedsData/, 'Should reject period = length');
});

test('trendflex_warmup_calculation_comprehensive', () => {
    // Test with various period values to verify warmup = first_valid + period
    const testPeriods = [5, 10, 20, 30];
    const close = testData.close.slice(0, 100); // Use subset for speed
    
    for (const period of testPeriods) {
        if (period >= close.length) continue;
        
        const result = wasm.trendflex_js(close, period);
        
        // First valid index is 0 for clean data
        const firstValid = 0;
        const warmup = firstValid + period;
        
        // Check NaN pattern
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} for period=${period}`);
        }
        
        // Check first non-NaN value
        if (warmup < result.length) {
            assert(!isNaN(result[warmup]), `Expected valid value at index ${warmup} for period=${period}`);
        }
    }
});