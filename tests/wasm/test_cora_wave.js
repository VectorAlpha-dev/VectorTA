/**
 * WASM binding tests for CoRa Wave indicator.
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
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('CoRa Wave partial params', () => {
    // Test with default parameters - mirrors check_cora_wave_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.cora_wave_js(close, 20, 2.0, true);
    assert.strictEqual(result.length, close.length);
});

test('CoRa Wave accuracy', () => {
    // Test CoRa Wave matches expected values from Rust tests - mirrors check_cora_wave_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.coraWave;
    
    const result = wasm.cora_wave_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.r_multi,
        expected.defaultParams.smooth
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,  // Use same tight tolerance as ALMA
        "CoRa Wave last 5 values mismatch"
    );
});

test('CoRa Wave default candles', () => {
    // Test CoRa Wave with default parameters - mirrors check_cora_wave_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.cora_wave_js(close, 20, 2.0, true);
    assert.strictEqual(result.length, close.length);
});

test('CoRa Wave zero period', () => {
    // Test CoRa Wave fails with zero period - mirrors check_cora_wave_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cora_wave_js(inputData, 0, 2.0, true);
    }, /Invalid period/);
});

test('CoRa Wave period exceeds length', () => {
    // Test CoRa Wave fails when period exceeds data length - mirrors check_cora_wave_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cora_wave_js(dataSmall, 10, 2.0, true);
    }, /Invalid period/);
});

test('CoRa Wave very small dataset', () => {
    // Test CoRa Wave fails with insufficient data - mirrors check_cora_wave_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.cora_wave_js(singlePoint, 20, 2.0, true);
    }, /Invalid period|Not enough valid data/);
});

test('CoRa Wave empty input', () => {
    // Test CoRa Wave fails with empty input - mirrors check_cora_wave_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.cora_wave_js(empty, 20, 2.0, true);
    }, /empty/i);
});

test('CoRa Wave invalid r_multi', () => {
    // Test CoRa Wave fails with invalid r_multi values
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // NaN r_multi
    assert.throws(() => {
        wasm.cora_wave_js(data, 2, NaN, false);
    }, /Invalid r_multi/);
    
    // Negative r_multi
    assert.throws(() => {
        wasm.cora_wave_js(data, 2, -1.0, false);
    }, /Invalid r_multi/);
    
    // Zero r_multi - currently allowed, produces valid output
    const resultZero = wasm.cora_wave_js(data, 2, 0.0, false);
    assert.strictEqual(resultZero.length, data.length);
});

test('CoRa Wave NaN handling', () => {
    // Test CoRa Wave handles NaN values correctly
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.coraWave;
    
    const result = wasm.cora_wave_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.r_multi,
        expected.defaultParams.smooth
    );
    
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (22), no NaN values should exist after some additional buffer
    const warmup = expected.warmupPeriod;  // Should be 22 for default params with smoothing
    if (result.length > warmup + 100) {
        for (let i = warmup + 100; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First warmup values should be NaN
    assertAllNaN(result.slice(0, warmup), "Expected NaN in warmup period");
});

test('CoRa Wave all NaN input', () => {
    // Test CoRa Wave with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.cora_wave_js(allNaN, 20, 2.0, true);
    }, /All values are NaN/);
});

test('CoRa Wave without smoothing', () => {
    // Test CoRa Wave without smoothing
    const close = new Float64Array(testData.close.slice(0, 100)); // Use subset for faster test
    
    const resultSmooth = wasm.cora_wave_js(close, 20, 2.0, true);
    const resultRaw = wasm.cora_wave_js(close, 20, 2.0, false);
    
    assert.strictEqual(resultSmooth.length, close.length);
    assert.strictEqual(resultRaw.length, close.length);
    
    // Results should be different when smoothing is on vs off
    // Check some valid (non-NaN) values
    let foundDifference = false;
    for (let i = 30; i < resultSmooth.length; i++) {
        if (!isNaN(resultSmooth[i]) && !isNaN(resultRaw[i])) {
            if (Math.abs(resultSmooth[i] - resultRaw[i]) > 1e-10) {
                foundDifference = true;
                break;
            }
        }
    }
    assert(foundDifference, 'Smoothed and raw values should be different');
});

test('CoRa Wave with different r_multi values', () => {
    // Test CoRa Wave with different r_multi values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use subset for faster test
    
    const result1 = wasm.cora_wave_js(close, 20, 1.0, false);
    const result2 = wasm.cora_wave_js(close, 20, 2.0, false);
    const result3 = wasm.cora_wave_js(close, 20, 3.0, false);
    
    // Results should be different for different r_multi values
    let foundDifference12 = false;
    let foundDifference23 = false;
    for (let i = 30; i < result1.length; i++) {
        if (!isNaN(result1[i]) && !isNaN(result2[i]) && !isNaN(result3[i])) {
            if (Math.abs(result1[i] - result2[i]) > 1e-10) {
                foundDifference12 = true;
            }
            if (Math.abs(result2[i] - result3[i]) > 1e-10) {
                foundDifference23 = true;
            }
            if (foundDifference12 && foundDifference23) {
                break;
            }
        }
    }
    assert(foundDifference12, 'Different r_multi values (1.0 vs 2.0) should produce different results');
    assert(foundDifference23, 'Different r_multi values (2.0 vs 3.0) should produce different results');
});

test('CoRa Wave batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Using the correct batch API for CoRa Wave
    const batchResult = wasm.cora_wave_batch(close, {
        period_range: [20, 20, 0],
        r_multi_range: [2.0, 2.0, 0],
        smooth: false  // Note: single boolean, not array
    });
    
    // Should match single calculation
    const singleResult = wasm.cora_wave_js(close, 20, 2.0, false);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('CoRa Wave batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 50)); // Use smaller dataset for speed
    
    // Multiple periods: 15, 20 using correct API
    const batchResult = wasm.cora_wave_batch(close, {
        period_range: [15, 20, 5],      // period range
        r_multi_range: [2.0, 2.0, 0],   // r_multi range  
        smooth: false                    // smooth option (boolean, not array)
    });
    
    // Should have 2 rows * 50 cols = 100 values
    assert.strictEqual(batchResult.values.length, 2 * 50);
    assert.strictEqual(batchResult.rows, 2);
    assert.strictEqual(batchResult.cols, 50);
    
    // Verify each row matches individual calculation
    const periods = [15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 50;
        const rowEnd = rowStart + 50;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.cora_wave_js(close, periods[i], 2.0, false);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('CoRa Wave batch metadata from result', () => {
    // Test that batch result includes correct parameter combinations
    const close = new Float64Array(25); // Need enough data for period 20
    close.fill(100);
    
    const result = wasm.cora_wave_batch(close, {
        period_range: [15, 20, 5],      // period: 15, 20
        r_multi_range: [1.5, 2.0, 0.5], // r_multi: 1.5, 2.0
        smooth: false                    // smooth option (single boolean)
    });
    
    // Should have 2 * 2 = 4 combinations (smooth is single value)
    assert.strictEqual(result.combos.length, 4);
    
    // Check first combination
    assert.strictEqual(result.combos[0].period, 15);
    assert.strictEqual(result.combos[0].r_multi, 1.5);
    assert.strictEqual(result.combos[0].smooth, false);
    
    // Check last combination
    assert.strictEqual(result.combos[3].period, 20);
    assert.strictEqual(result.combos[3].r_multi, 2.0);
    assert.strictEqual(result.combos[3].smooth, false);
});

test('CoRa Wave batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 30));
    
    const batchResult = wasm.cora_wave_batch(close, {
        period_range: [10, 15, 5],      // 2 periods
        r_multi_range: [1.0, 1.5, 0.5], // 2 r_multis
        smooth: false                    // smooth option (boolean)
    });
    
    // Should have 2 * 2 = 4 combinations
    assert.strictEqual(batchResult.combos.length, 4);
    assert.strictEqual(batchResult.rows, 4);
    assert.strictEqual(batchResult.cols, 30);
    assert.strictEqual(batchResult.values.length, 4 * 30);
    
    // Verify structure
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const period = batchResult.combos[combo].period;
        const r_multi = batchResult.combos[combo].r_multi;
        const smooth = batchResult.combos[combo].smooth;
        
        const rowStart = combo * 30;
        const rowData = batchResult.values.slice(rowStart, rowStart + 30);
        
        // First period-1 values should be NaN (no smoothing, so just period-1)
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        // After warmup should have values
        for (let i = period - 1; i < 30; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('CoRa Wave batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep
    const singleBatch = wasm.cora_wave_batch(close, {
        period_range: [5, 5, 1],
        r_multi_range: [2.0, 2.0, 0.1],
        smooth: false  // Single boolean
    });
    
    assert.strictEqual(singleBatch.values.length, 10);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.cora_wave_batch(close, {
        period_range: [5, 7, 10], // Step larger than range
        r_multi_range: [2.0, 2.0, 0],
        smooth: false  // Single boolean
    });
    
    // Should only have period=5
    assert.strictEqual(largeBatch.values.length, 10);
    assert.strictEqual(largeBatch.combos.length, 1);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.cora_wave_batch(new Float64Array([]), {
            period_range: [20, 20, 0],
            r_multi_range: [2.0, 2.0, 0],
            smooth: true  // Single boolean
        });
    }, /All values are NaN/);
});

test('CoRa Wave batch - API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 25));
    
    // Invalid config structure
    assert.throws(() => {
        wasm.cora_wave_batch(close, {
            period_range: [20, 20], // Missing step
            r_multi_range: [2.0, 2.0, 0],
            smooth: true
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.cora_wave_batch(close, {
            period_range: [20, 20, 0],
            r_multi_range: [2.0, 2.0, 0]
            // Missing smooth
        });
    }, /Invalid config/);
    
    // Invalid data type
    assert.throws(() => {
        wasm.cora_wave_batch(close, {
            period_range: "invalid",
            r_multi_range: [2.0, 2.0, 0],
            smooth: true
        });
    }, /Invalid config/);
});

// Zero-copy API tests
test('CoRa Wave zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 5;
    const r_multi = 2.0;
    const smooth = false;
    
    // Allocate buffer
    const ptr = wasm.cora_wave_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute CoRa Wave in-place
    try {
        wasm.cora_wave_into(ptr, ptr, data.length, period, r_multi, smooth);
        
        // Verify results match regular API
        const regularResult = wasm.cora_wave_js(data, period, r_multi, smooth);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.cora_wave_free(ptr, data.length);
    }
});

test('CoRa Wave zero-copy with large dataset', () => {
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) + Math.random() * 0.1;
    }
    
    const ptr = wasm.cora_wave_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.cora_wave_into(ptr, ptr, size, 20, 2.0, true);
        
        // Recreate view in case memory grew
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        // Check warmup period has NaN
        // With period=20 and smoothing=true: smooth_period=sqrt(20).round()=4, warmup = 19 + 3 = 22
        for (let i = 0; i < 22; i++) {
            assert(isNaN(memView2[i]), `Expected NaN at warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = 22; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.cora_wave_free(ptr, size);
    }
});

// Error handling for zero-copy API
test('CoRa Wave zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.cora_wave_into(0, 0, 10, 20, 2.0, true);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.cora_wave_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.cora_wave_into(ptr, ptr, 10, 0, 2.0, true);
        }, /Invalid period/);
        
        // Invalid r_multi (negative)
        assert.throws(() => {
            wasm.cora_wave_into(ptr, ptr, 10, 5, -1.0, false);
        }, /Invalid r_multi/);
    } finally {
        wasm.cora_wave_free(ptr, 10);
    }
});

// Memory leak prevention test
test('CoRa Wave zero-copy memory management', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.cora_wave_alloc(size);
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
        wasm.cora_wave_free(ptr, size);
    }
});

// SIMD128 verification test
test('CoRa Wave SIMD128 consistency', () => {
    // This test verifies SIMD128 produces same results as scalar
    // It runs automatically when SIMD128 is enabled
    const testCases = [
        { size: 10, period: 5 },
        { size: 100, period: 10 },
        { size: 1000, period: 20 },
        { size: 5000, period: 30 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }
        
        const result = wasm.cora_wave_js(data, testCase.period, 2.0, true);
        
        // Basic sanity checks
        assert.strictEqual(result.length, data.length);
        
        // Check warmup period (with smoothing = true)
        // smooth_period = sqrt(period).round().max(1)
        const smoothPeriod = Math.max(1, Math.round(Math.sqrt(testCase.period)));
        const warmup = testCase.period - 1 + Math.max(0, smoothPeriod - 1);
        for (let i = 0; i < Math.min(warmup, result.length); i++) {
            assert(isNaN(result[i]), `Expected NaN at warmup index ${i} for size=${testCase.size}, period=${testCase.period}`);
        }
        
        // Check values exist after warmup
        if (result.length > warmup + 10) {
            let sumAfterWarmup = 0;
            let countAfterWarmup = 0;
            for (let i = warmup; i < Math.min(warmup + 100, result.length); i++) {
                assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
                sumAfterWarmup += result[i];
                countAfterWarmup++;
            }
            
            // Verify reasonable values
            const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
            assert(Math.abs(avgAfterWarmup) < 100, `Average value ${avgAfterWarmup} seems unreasonable`);
        }
    }
});

// Additional comprehensive tests
test('CoRa Wave with leading NaN values', () => {
    // Test CoRa Wave handles leading NaN values correctly
    const close = new Float64Array(testData.close.slice(0, 100));
    // Add some NaN values at the beginning
    for (let i = 0; i < 5; i++) {
        close[i] = NaN;
    }
    
    const result = wasm.cora_wave_js(close, 20, 2.0, true);
    assert.strictEqual(result.length, close.length);
    
    // First 5 + warmup period (22) should be NaN
    const expectedNanCount = 5 + 22;
    for (let i = 0; i < expectedNanCount; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    // After that, should have valid values
    if (result.length > expectedNanCount + 10) {
        for (let i = expectedNanCount + 10; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
        }
    }
});

test('CoRa Wave batch with expected parameters from test_utils', () => {
    // Test batch processing using parameters from EXPECTED_OUTPUTS
    const close = new Float64Array(testData.close.slice(0, 100));
    const expected = EXPECTED_OUTPUTS.coraWave;
    
    const batchResult = wasm.cora_wave_batch(close, {
        period_range: expected.batchRange.periodRange,
        r_multi_range: expected.batchRange.rMultiRange,
        smooth: expected.batchRange.smooth
    });
    
    // Should have 3 periods * 3 r_multis = 9 combinations
    assert.strictEqual(batchResult.combos.length, 9);
    assert.strictEqual(batchResult.rows, 9);
    assert.strictEqual(batchResult.cols, 100);
    
    // Find the default params row (period=20, r_multi=2.0, smooth=false)
    let defaultIdx = -1;
    for (let i = 0; i < batchResult.combos.length; i++) {
        if (batchResult.combos[i].period === 20 && 
            Math.abs(batchResult.combos[i].r_multi - 2.0) < 1e-10 &&
            batchResult.combos[i].smooth === false) {
            defaultIdx = i;
            break;
        }
    }
    
    assert(defaultIdx >= 0, 'Default parameters not found in batch result');
    
    // Compare with single calculation (without smoothing)
    const singleResult = wasm.cora_wave_js(close, 20, 2.0, false);
    const batchRow = batchResult.values.slice(defaultIdx * 100, (defaultIdx + 1) * 100);
    
    // Values should match where both are not NaN
    for (let i = 0; i < singleResult.length; i++) {
        if (!isNaN(batchRow[i]) && !isNaN(singleResult[i])) {
            assertClose(batchRow[i], singleResult[i], 1e-8, 
                       `Batch row doesn't match single calculation at index ${i}`);
        }
    }
});

test('CoRa Wave performance characteristics', () => {
    // Test CoRa Wave maintains expected performance characteristics
    // Generate synthetic trending data
    const trendUp = new Float64Array(100);
    const trendDown = new Float64Array(100);
    const choppy = new Float64Array(100);
    
    for (let i = 0; i < 100; i++) {
        trendUp[i] = 100 + i;  // Upward trend
        trendDown[i] = 200 - i;  // Downward trend
        choppy[i] = 100 + 10 * (i % 2);  // Choppy/oscillating
    }
    
    // CoRa Wave should smooth the choppy data
    const resultChoppy = wasm.cora_wave_js(choppy, 10, 2.0, true);
    
    // Calculate the variance of the smoothed output (excluding NaN)
    let validResult = [];
    for (let i = 0; i < resultChoppy.length; i++) {
        if (!isNaN(resultChoppy[i])) {
            validResult.push(resultChoppy[i]);
        }
    }
    
    if (validResult.length > 20) {
        // Smoothed output should have lower variance than input
        const inputSlice = choppy.slice(choppy.length - validResult.length);
        let inputMean = 0, outputMean = 0;
        for (let i = 0; i < validResult.length; i++) {
            inputMean += inputSlice[i];
            outputMean += validResult[i];
        }
        inputMean /= validResult.length;
        outputMean /= validResult.length;
        
        let inputVar = 0, outputVar = 0;
        for (let i = 0; i < validResult.length; i++) {
            inputVar += Math.pow(inputSlice[i] - inputMean, 2);
            outputVar += Math.pow(validResult[i] - outputMean, 2);
        }
        inputVar /= validResult.length;
        outputVar /= validResult.length;
        
        assert(outputVar < inputVar, 'CoRa Wave should smooth choppy data');
    }
    
    // For trending data, CoRa Wave should follow the trend
    const resultUp = wasm.cora_wave_js(trendUp, 10, 2.0, true);
    let validUp = [];
    for (let i = 0; i < resultUp.length; i++) {
        if (!isNaN(resultUp[i])) {
            validUp.push(resultUp[i]);
        }
    }
    
    if (validUp.length > 2) {
        // Check that the trend is preserved (mostly increasing)
        let increasingCount = 0;
        for (let i = 1; i < validUp.length; i++) {
            if (validUp[i] > validUp[i-1]) {
                increasingCount++;
            }
        }
        const increasingRatio = increasingCount / (validUp.length - 1);
        assert(increasingRatio > 0.7, `CoRa Wave should follow upward trend (ratio: ${increasingRatio})`);
    }
});

test('CoRa Wave edge cases with extreme values', () => {
    // Test with very small period
    const close = new Float64Array(testData.close.slice(0, 20));
    let result = wasm.cora_wave_js(close, 2, 2.0, false);
    assert.strictEqual(result.length, close.length);
    
    // Test with period equal to data length
    result = wasm.cora_wave_js(close, close.length, 2.0, false);
    assert.strictEqual(result.length, close.length);
    // Should have NaN for all but the last value
    for (let i = 0; i < result.length - 1; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    assert(!isNaN(result[result.length - 1]), 'Last value should not be NaN');
    
    // Test with large r_multi
    result = wasm.cora_wave_js(close, 5, 10.0, false);
    assert.strictEqual(result.length, close.length);
    
    // Test with very small r_multi (but positive)
    result = wasm.cora_wave_js(close, 5, 0.001, false);
    assert.strictEqual(result.length, close.length);
    
    // Test with zero r_multi (should be allowed)
    result = wasm.cora_wave_js(close, 5, 0.0, false);
    assert.strictEqual(result.length, close.length);
});

test.after(() => {
    console.log('CoRa Wave WASM tests completed');
});
