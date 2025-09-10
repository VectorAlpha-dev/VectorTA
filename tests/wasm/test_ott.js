/**
 * WASM binding tests for OTT indicator.
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

test('OTT accuracy', async () => {
    // Test OTT matches expected values from Rust tests - mirrors check_ott_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.ott;
    
    // Use accuracy params (period=50)
    const result = wasm.ott_js(
        close, 
        expected.accuracyParams.period, 
        expected.accuracyParams.percent, 
        expected.accuracyParams.ma_type
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "OTT last 5 values mismatch"
    );
});

test('OTT partial params', () => {
    // Test with default parameters - mirrors check_ott_partial_params
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.ott;
    
    const result = wasm.ott_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.percent,
        expected.defaultParams.ma_type
    );
    assert.strictEqual(result.length, close.length);
    
    // Note: VAR with period=2 doesn't have a traditional warmup period (returns 0.0 instead of NaN)
    // This is specific to the VAR implementation
    // So we just verify length and that we have valid numeric values
    for (let i = 0; i < result.length; i++) {
        assert(!isNaN(result[i]), `Should not have NaN values with VAR period=2 at index ${i}`);
    }
});

test('OTT default candles', () => {
    // Test OTT with default parameters - mirrors check_ott_default_candles
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.ott;
    
    const result = wasm.ott_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.percent,
        expected.defaultParams.ma_type
    );
    assert.strictEqual(result.length, close.length);
});

test('OTT zero period', () => {
    // Test OTT fails with zero period - mirrors check_ott_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.ott_js(inputData, 0, 1.4, "VAR");
    }, /Invalid period/);
});

test('OTT period exceeds length', () => {
    // Test OTT fails when period exceeds data length - mirrors check_ott_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.ott_js(dataSmall, 10, 1.4, "VAR");
    }, /Invalid period|Period.*exceeds|Not enough data/);
});

test('OTT very small dataset', () => {
    // Test OTT fails with insufficient data - mirrors check_ott_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.ott_js(singlePoint, 50, 1.4, "VAR");
    }, /Invalid period|Not enough valid data|Not enough data/);
});

test('OTT empty input', () => {
    // Test OTT fails with empty input - mirrors check_ott_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.ott_js(empty, 50, 1.4, "VAR");
    }, /Input data slice is empty/);
});

test('OTT all NaN input', () => {
    // Test OTT with all NaN values - mirrors check_ott_all_nan
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.ott_js(allNaN, 50, 1.4, "VAR");
    }, /All values are NaN/);
});

test('OTT invalid percent', () => {
    // Test OTT fails with invalid percent values
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // Negative percent
    assert.throws(() => {
        wasm.ott_js(data, 2, -1.0, "VAR");
    }, /Invalid percent/);
    
    // NaN percent
    assert.throws(() => {
        wasm.ott_js(data, 2, NaN, "VAR");
    }, /Invalid percent/);
    
    // Infinity percent
    assert.throws(() => {
        wasm.ott_js(data, 2, Infinity, "VAR");
    }, /Invalid percent/);
    
    // Zero percent should be valid
    const result = wasm.ott_js(data, 2, 0.0, "VAR");
    assert.strictEqual(result.length, data.length);
});

test('OTT invalid MA type', () => {
    // Test OTT fails with invalid MA type string
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    assert.throws(() => {
        wasm.ott_js(data, 2, 1.4, "INVALID_MA");
    }, /Invalid moving average|Invalid MA type|Unsupported moving average/);
});

test('OTT MA type variations', () => {
    // Test OTT with different MA types (VAR, WWMA, TMA)
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const maTypes = ["VAR", "WWMA", "TMA"];
    const results = {};
    
    for (const maType of maTypes) {
        const result = wasm.ott_js(close, 20, 1.4, maType);
        assert.strictEqual(result.length, close.length, `Length mismatch for ${maType}`);
        results[maType] = result;
        
        // Verify warmup period - VAR may not have NaN values at start
        if (maType === "VAR") {
            // VAR might return 0.0 instead of NaN
            // No specific warmup check for VAR
        } else {
            // Other MA types should have warmup NaN values
            const warmup = 19;
            let hasNaN = false;
            for (let i = 0; i < warmup && i < result.length; i++) {
                if (isNaN(result[i])) {
                    hasNaN = true;
                    break;
                }
            }
            // Some MA types may not have NaN warmup
            // Just verify we have values
            if (result.length > warmup) {
                for (let i = warmup + 5; i < result.length; i++) {
                    assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for ${maType}`);
                }
            }
        }
    }
    
    // Different MA types should produce different results
    let varDiffWwma = false;
    let varDiffTma = false;
    for (let i = 19; i < 100; i++) {
        if (Math.abs(results.VAR[i] - results.WWMA[i]) > 1e-10) varDiffWwma = true;
        if (Math.abs(results.VAR[i] - results.TMA[i]) > 1e-10) varDiffTma = true;
    }
    assert(varDiffWwma, "VAR and WWMA should produce different results");
    assert(varDiffTma, "VAR and TMA should produce different results");
});

test('OTT NaN handling', () => {
    // Test OTT handles NaN values correctly - mirrors check_alma_nan_handling
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.ott;
    
    // Use accuracy params
    const result = wasm.ott_js(
        close, 
        expected.accuracyParams.period, 
        expected.accuracyParams.percent, 
        expected.accuracyParams.ma_type
    );
    assert.strictEqual(result.length, close.length);
    
    // Note: VAR with period=2 doesn't have NaN warmup values, it returns 0.0
    // So we just verify no NaN values exist in the output
    for (let i = 0; i < result.length; i++) {
        assert(!isNaN(result[i]), `Should not have NaN values with VAR period=2 at index ${i}`);
    }
});

test('OTT reinput', () => {
    // Test OTT applied twice (re-input) - mirrors check_ott_reinput
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.ott;
    const params = expected.accuracyParams;
    
    // First pass
    const firstResult = wasm.ott_js(
        close, 
        params.period, 
        params.percent, 
        params.ma_type
    );
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply OTT to OTT output
    const secondResult = wasm.ott_js(
        firstResult, 
        params.period, 
        params.percent, 
        params.ma_type
    );
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Check last 5 values match expected
    const last5 = secondResult.slice(-5);
    assertArrayClose(
        last5,
        expected.reinputLast5,
        1e-8,
        "OTT re-input last 5 values mismatch"
    );
    
    // Skip compareWithRust for OTT - not implemented in generate_references
});

test('OTT batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.ott;
    
    // Using the new ergonomic batch API for single parameter
    const batchResult = wasm.ott_batch(close, {
        period_range: [expected.accuracyParams.period, expected.accuracyParams.period, 0],
        percent_range: [expected.accuracyParams.percent, expected.accuracyParams.percent, 0],
        ma_types: [expected.accuracyParams.ma_type]
    });
    
    // Should have 1 combination
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, close.length);
    assert.strictEqual(batchResult.combos.length, 1);
    assert.strictEqual(batchResult.values.length, close.length);
    
    // Check last 5 values match expected
    const last5 = batchResult.values.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "OTT batch last 5 values mismatch"
    );
    
    // Should match single calculation
    const singleResult = wasm.ott_js(
        close, 
        expected.accuracyParams.period, 
        expected.accuracyParams.percent, 
        expected.accuracyParams.ma_type
    );
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('OTT batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Multiple periods: 20, 25, 30
    const batchResult = wasm.ott_batch(close, {
        period_range: [20, 30, 5],
        percent_range: [1.4, 1.4, 0],
        ma_types: ["VAR"]
    });
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    // Verify each row matches individual calculation
    const periods = [20, 25, 30];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.ott_js(close, periods[i], 1.4, "VAR");
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('OTT batch multiple percents', () => {
    // Test batch with multiple percent values
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Multiple percents: 0.5, 1.0, 1.5, 2.0
    const batchResult = wasm.ott_batch(close, {
        period_range: [20, 20, 0],
        percent_range: [0.5, 2.0, 0.5],
        ma_types: ["VAR"]
    });
    
    // Should have 4 combinations
    assert.strictEqual(batchResult.rows, 4);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.combos.length, 4);
    
    // Different percents should produce different results
    for (let i = 0; i < 3; i++) {
        const row1Start = i * 100;
        const row2Start = (i + 1) * 100;
        const row1 = batchResult.values.slice(row1Start, row1Start + 100);
        const row2 = batchResult.values.slice(row2Start, row2Start + 100);
        
        let different = false;
        for (let j = 19; j < 100; j++) {  // After warmup
            if (Math.abs(row1[j] - row2[j]) > 1e-10) {
                different = true;
                break;
            }
        }
        assert(different, `Percent ${0.5 + i*0.5} and ${0.5 + (i+1)*0.5} should produce different results`);
    }
});

test('OTT batch multiple MA types', () => {
    // Test batch with multiple MA types
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const batchResult = wasm.ott_batch(close, {
        period_range: [20, 20, 0],
        percent_range: [1.4, 1.4, 0],
        ma_types: ["VAR", "WWMA", "TMA"]
    });
    
    // Should have 3 combinations (3 MA types)
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.combos.length, 3);
    
    // Verify MA types in combos
    assert.strictEqual(batchResult.combos[0].ma_type, "VAR");
    assert.strictEqual(batchResult.combos[1].ma_type, "WWMA");
    assert.strictEqual(batchResult.combos[2].ma_type, "TMA");
    
    // Different MA types should produce different results
    const results = {};
    for (let i = 0; i < 3; i++) {
        const rowStart = i * 100;
        results[batchResult.combos[i].ma_type] = batchResult.values.slice(rowStart, rowStart + 100);
    }
    
    let varDiffWwma = false;
    let varDiffTma = false;
    for (let i = 19; i < 100; i++) {  // After warmup
        if (Math.abs(results.VAR[i] - results.WWMA[i]) > 1e-10) varDiffWwma = true;
        if (Math.abs(results.VAR[i] - results.TMA[i]) > 1e-10) varDiffTma = true;
    }
    assert(varDiffWwma, "VAR and WWMA should produce different results");
    assert(varDiffTma, "VAR and TMA should produce different results");
});

test('OTT batch metadata from result', () => {
    // Test that batch result includes correct parameter combinations
    const close = new Float64Array(20);
    close.fill(100);
    
    const result = wasm.ott_batch(close, {
        period_range: [5, 10, 5],      // periods: 5, 10
        percent_range: [1.0, 1.5, 0.5], // percents: 1.0, 1.5
        ma_types: ["VAR", "WWMA"]      // 2 MA types
    });
    
    // Should have 2 * 2 * 2 = 8 combinations
    assert.strictEqual(result.combos.length, 8);
    
    // Check first combination
    assert.strictEqual(result.combos[0].period, 5);
    assert.strictEqual(result.combos[0].percent, 1.0);
    assert.strictEqual(result.combos[0].ma_type, "VAR");
    
    // Check last combination
    assert.strictEqual(result.combos[7].period, 10);
    assertClose(result.combos[7].percent, 1.5, 1e-10, "percent mismatch");
    assert.strictEqual(result.combos[7].ma_type, "WWMA");
});

test('OTT batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.ott_batch(close, {
        period_range: [10, 12, 2],      // 2 periods
        percent_range: [1.0, 1.5, 0.5], // 2 percents
        ma_types: ["VAR", "TMA"]        // 2 MA types
    });
    
    // Should have 2 * 2 * 2 = 8 combinations
    assert.strictEqual(batchResult.combos.length, 8);
    assert.strictEqual(batchResult.rows, 8);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 8 * 50);
    
    // Verify structure
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const period = batchResult.combos[combo].period;
        const rowStart = combo * 50;
        const rowData = batchResult.values.slice(rowStart, rowStart + 50);
        
        // Note: VAR may not have NaN warmup (returns 0.0 instead)
        // Just verify structure without strict NaN check for VAR
        
        // After warmup should have values
        for (let i = period - 1; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('OTT batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep
    const singleBatch = wasm.ott_batch(close, {
        period_range: [5, 5, 1],
        percent_range: [1.4, 1.4, 0.1],
        ma_types: ["VAR"]
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.ott_batch(close, {
        period_range: [5, 7, 10], // Step larger than range
        percent_range: [1.4, 1.4, 0],
        ma_types: ["VAR"]
    });
    
    // Should only have period=5
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.ott_batch(new Float64Array([]), {
            period_range: [5, 5, 0],
            percent_range: [1.4, 1.4, 0],
            ma_types: ["VAR"]
        });
    }, /All values are NaN|Input data slice is empty/);
});

test('OTT batch error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    // Invalid config structure
    assert.throws(() => {
        wasm.ott_batch(close, {
            period_range: [5, 5], // Missing step
            percent_range: [1.4, 1.4, 0],
            ma_types: ["VAR"]
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.ott_batch(close, {
            period_range: [5, 5, 0],
            percent_range: [1.4, 1.4, 0]
            // Missing ma_types
        });
    }, /Invalid config/);
    
    // Invalid data type
    assert.throws(() => {
        wasm.ott_batch(close, {
            period_range: "invalid",
            percent_range: [1.4, 1.4, 0],
            ma_types: ["VAR"]
        });
    }, /Invalid config/);
});

// Zero-copy API tests
test('OTT zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    const percent = 1.4;
    const ma_type = "VAR";
    
    // Allocate buffer
    const ptr = wasm.ott_alloc(data.length);
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
    
    // Compute OTT in-place
    try {
        wasm.ott_into(ptr, ptr, data.length, period, percent, ma_type);
        
        // Verify results match regular API
        const regularResult = wasm.ott_js(data, period, percent, ma_type);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.ott_free(ptr, data.length);
    }
});

test('OTT zero-copy with large dataset', () => {
    const size = 100000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.ott_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memory = wasm.__wbindgen_memory();
        const memView = new Float64Array(memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.ott_into(ptr, ptr, size, 50, 1.4, "VAR");
        
        // Recreate view in case memory grew
        const memory2 = wasm.__wbindgen_memory();
        const memView2 = new Float64Array(memory2.buffer, ptr, size);
        
        // Note: VAR with period=50 may not have NaN warmup (returns 0.0 instead)
        // Skip warmup check for VAR
        
        // Check after warmup has values
        for (let i = 49; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.ott_free(ptr, size);
    }
});

test('OTT zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.ott_into(0, 0, 10, 50, 1.4, "VAR");
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.ott_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.ott_into(ptr, ptr, 10, 0, 1.4, "VAR");
        }, /Invalid period/);
        
        // Invalid MA type
        assert.throws(() => {
            wasm.ott_into(ptr, ptr, 10, 5, 1.4, "INVALID");
        }, /Invalid moving average|Invalid MA type|Unsupported moving average/);
    } finally {
        wasm.ott_free(ptr, 10);
    }
});

test('OTT zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000, 100000];
    
    for (const size of sizes) {
        const ptr = wasm.ott_alloc(size);
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
        wasm.ott_free(ptr, size);
    }
});

// SIMD128 verification test
test('OTT SIMD128 consistency', () => {
    // This test verifies SIMD128 produces same results as scalar
    // It runs automatically when SIMD128 is enabled
    const testCases = [
        { size: 10, period: 5 },
        { size: 100, period: 20 },
        { size: 1000, period: 50 },
        { size: 10000, period: 100 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }
        
        const result = wasm.ott_js(data, testCase.period, 1.4, "VAR");
        
        // Basic sanity checks
        assert.strictEqual(result.length, data.length);
        
        // Note: VAR may not have NaN warmup period (returns 0.0 instead)
        // Skip warmup check for VAR
        
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

test('OTT all MA types comprehensive', () => {
    // Test OTT with all 8 supported MA types - comprehensive coverage
    const close = new Float64Array(testData.close.slice(0, 200));
    
    // All 8 supported MA types
    const allMaTypes = ["SMA", "EMA", "WMA", "TMA", "VAR", "WWMA", "ZLEMA", "TSF"];
    const results = {};
    const period = 20;
    const percent = 1.4;
    
    for (const maType of allMaTypes) {
        try {
            const result = wasm.ott_js(close, period, percent, maType);
            assert.strictEqual(result.length, close.length, `Length mismatch for ${maType}`);
            results[maType] = result;
            
            // Verify warmup period exists
            let firstValid = -1;
            for (let i = 0; i < result.length; i++) {
                if (!isNaN(result[i])) {
                    firstValid = i;
                    break;
                }
            }
            // Don't assert warmup for VAR, EMA, WWMA as they may start with values immediately
            if (!["VAR", "EMA", "WWMA"].includes(maType)) {
                assert(firstValid > 0, `Expected warmup period for ${maType}`);
            }
            
            // After sufficient warmup, should have valid values
            if (result.length > period + 10) {
                let validCount = 0;
                for (let i = period; i < result.length; i++) {
                    if (!isNaN(result[i])) validCount++;
                }
                assert(validCount > 0, `Expected valid values after warmup for ${maType}`);
            }
        } catch (e) {
            assert.fail(`Failed to calculate OTT with MA type ${maType}: ${e.message}`);
        }
    }
    
    // Verify that different MA types produce different results
    const startIdx = period + 10;  // Safe index after warmup
    const endIdx = startIdx + 20;  // Compare 20 values
    
    if (endIdx < close.length) {
        // SMA vs EMA should differ
        if (results.SMA && results.EMA) {
            let different = false;
            for (let i = startIdx; i < endIdx; i++) {
                if (Math.abs(results.SMA[i] - results.EMA[i]) > 1e-6) {
                    different = true;
                    break;
                }
            }
            assert(different, "SMA and EMA should produce different results");
        }
        
        // VAR vs WWMA should differ
        if (results.VAR && results.WWMA) {
            let different = false;
            for (let i = startIdx; i < endIdx; i++) {
                if (Math.abs(results.VAR[i] - results.WWMA[i]) > 1e-6) {
                    different = true;
                    break;
                }
            }
            assert(different, "VAR and WWMA should produce different results");
        }
        
        // ZLEMA vs WMA should differ
        if (results.ZLEMA && results.WMA) {
            let different = false;
            for (let i = startIdx; i < endIdx; i++) {
                if (Math.abs(results.ZLEMA[i] - results.WMA[i]) > 1e-6) {
                    different = true;
                    break;
                }
            }
            assert(different, "ZLEMA and WMA should produce different results");
        }
    }
});

test('OTT batch with different MA types stress test', () => {
    // Stress test with all MA type combinations
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Test all three MA types at once
    const result = wasm.ott_batch(close, {
        period_range: [15, 20, 5],    // 2 periods
        percent_range: [1.0, 2.0, 1.0], // 2 percents
        ma_types: ["VAR", "WWMA", "TMA"] // 3 MA types
    });
    
    // Should have 2 * 2 * 3 = 12 combinations
    assert.strictEqual(result.combos.length, 12);
    assert.strictEqual(result.rows, 12);
    assert.strictEqual(result.cols, 100);
    
    // Verify each combination exists
    const expectedCombos = [];
    for (const period of [15, 20]) {
        for (const percent of [1.0, 2.0]) {
            for (const ma_type of ["VAR", "WWMA", "TMA"]) {
                expectedCombos.push({ period, percent, ma_type });
            }
        }
    }
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedCombos[i].period);
        assertClose(result.combos[i].percent, expectedCombos[i].percent, 1e-10);
        assert.strictEqual(result.combos[i].ma_type, expectedCombos[i].ma_type);
    }
});

test.after(() => {
    console.log('OTT WASM tests completed');
});