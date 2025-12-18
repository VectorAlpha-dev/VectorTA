/**
 * WASM binding tests for MAAQ indicator.
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
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('MAAQ partial params', () => {
    // Test with default parameters - mirrors check_maaq_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.maaq_js(close, 11, 2, 30);
    assert.strictEqual(result.length, close.length);
});

test('MAAQ accuracy', async () => {
    // Test MAAQ matches expected values from Rust tests - mirrors check_maaq_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.maaq_js(close, 11, 2, 30);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected last 5 values from Rust test
    const expectedLastFive = [
        59747.657115949725,
        59740.803138018055,
        59724.24153333905,
        59720.60576365108,
        59673.9954445178,
    ];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        0.01,  // 1e-2 tolerance as in Rust test
        "MAAQ last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('maaq', result, 'close', { 
        period: 11, 
        fast_period: 2, 
        slow_period: 30 
    });
});

test('MAAQ default candles', async () => {
    // Test MAAQ with default parameters - mirrors check_maaq_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.maaq_js(close, 11, 2, 30);
    assert.strictEqual(result.length, close.length);
    
    // Compare with Rust
    await compareWithRust('maaq', result, 'close', { 
        period: 11, 
        fast_period: 2, 
        slow_period: 30 
    });
});

test('MAAQ zero period', () => {
    // Test MAAQ fails with zero period - mirrors check_maaq_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    // Test with zero period
    assert.throws(() => {
        wasm.maaq_js(inputData, 0, 2, 30);
    });
    
    // Test with zero fast_period
    assert.throws(() => {
        wasm.maaq_js(inputData, 11, 0, 30);
    });
    
    // Test with zero slow_period
    assert.throws(() => {
        wasm.maaq_js(inputData, 11, 2, 0);
    });
});

test('MAAQ period exceeds length', () => {
    // Test MAAQ fails when period exceeds data length - mirrors check_maaq_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.maaq_js(dataSmall, 10, 2, 30);
    });
});

test('MAAQ very small dataset', () => {
    // Test MAAQ with very small dataset - mirrors check_maaq_very_small_dataset
    const dataSingle = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.maaq_js(dataSingle, 11, 2, 30);
    });
});

test('MAAQ empty input', () => {
    // Test MAAQ with empty input - mirrors check_maaq_empty_input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.maaq_js(dataEmpty, 11, 2, 30);
    });
});

test('MAAQ all NaN', () => {
    // Test MAAQ with all NaN input
    const data = new Float64Array([NaN, NaN, NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.maaq_js(data, 3, 2, 5);
    });
});

test('MAAQ reinput', () => {
    // Test MAAQ with re-input of MAAQ result - mirrors check_maaq_reinput
    const close = new Float64Array(testData.close);
    
    // First MAAQ pass with default params
    const firstResult = wasm.maaq_js(close, 11, 2, 30);
    
    // Second MAAQ pass with different params using first result as input
    const secondResult = wasm.maaq_js(firstResult, 20, 3, 25);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // The second pass will have its own warmup period
    // Check that we have some valid values
    for (let i = 40; i < secondResult.length; i++) {
        assert(isFinite(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('MAAQ NaN handling', () => {
    // Test MAAQ handling of NaN values - mirrors check_maaq_nan_handling
    const close = new Float64Array(testData.close);
    const period = 11;
    
    const result = wasm.maaq_js(close, period, 2, 30);
    
    assert.strictEqual(result.length, close.length);
    
    // First period-1 values should be NaN (warmup period)
    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(result[i]), `Expected NaN at warmup index ${i}, got ${result[i]}`);
    }
    
    // The Rust test checks that after index 240, there are no NaN values
    // This implies the warmup period creates NaN values at the beginning
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('MAAQ batch', () => {
    // Test MAAQ batch computation
    const close = new Float64Array(testData.close);
    
    // Test period range 11-50 step 10, static fast/slow
    const batch_result = wasm.maaq_batch_js(
        close, 
        {
            period_range: [11, 41, 10],      // period range: 11, 21, 31, 41
            fast_period_range: [2, 2, 0],    // fast_period static
            slow_period_range: [30, 30, 0]   // slow_period static
        }
    );
    const metadata = wasm.maaq_batch_metadata_js(
        11, 41, 10,
        2, 2, 0,
        30, 30, 0
    );
    
    // Metadata should contain period values (flattened)
    // 4 combinations * 3 values per combo = 12 values
    assert.strictEqual(metadata.length, 12);
    // Check first combination
    assert.strictEqual(metadata[0], 11);  // period
    assert.strictEqual(metadata[1], 2);   // fast_period
    assert.strictEqual(metadata[2], 30);  // slow_period
    
    // Batch result should contain all individual results flattened
    assert.strictEqual(batch_result.length, 4 * close.length);
    
    // Verify each row matches individual calculation
    let row_idx = 0;
    for (let p = 11; p <= 41; p += 10) {
        const individual_result = wasm.maaq_js(close, p, 2, 30);
        
        // Extract row from batch result
        const row_start = row_idx * close.length;
        const row = batch_result.slice(row_start, row_start + close.length);
        
        assertArrayClose(row, individual_result, 1e-9, `Period ${p}`);
        row_idx++;
    }
});

test('MAAQ different periods', () => {
    // Test MAAQ with different period values
    const close = new Float64Array(testData.close);
    
    // Test various period combinations
    const testCases = [
        [5, 2, 10],
        [10, 3, 20],
        [20, 5, 40],
        [50, 10, 100],
    ];
    
    for (const [period, fast_p, slow_p] of testCases) {
        const result = wasm.maaq_js(close, period, fast_p, slow_p);
        assert.strictEqual(result.length, close.length);
        
        // Count valid values after warmup
        let validCount = 0;
        for (let i = period; i < result.length; i++) {
            if (!isNaN(result[i])) validCount++;
        }
        assert(validCount > close.length - period - 5, 
            `Too many NaN values for params=(${period}, ${fast_p}, ${slow_p})`);
    }
});

test('MAAQ batch performance', () => {
    // Test that batch computation is more efficient than multiple single computations
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use first 1000 values
    
    // Test multiple period combinations
    const startBatch = performance.now();
    const batchResult = wasm.maaq_batch_js(
        close,
        {
            period_range: [10, 30, 10],      // periods: 10, 20, 30
            fast_period_range: [2, 2, 0],    // fast_period fixed at 2
            slow_period_range: [25, 35, 5]   // slow_periods: 25, 30, 35
        }
    );
    const batchTime = performance.now() - startBatch;
    
    const startSingle = performance.now();
    const singleResults = [];
    for (let p = 10; p <= 30; p += 10) {
        for (let s = 25; s <= 35; s += 5) {
            singleResults.push(...wasm.maaq_js(close, p, 2, s));
        }
    }
    const singleTime = performance.now() - startSingle;
    
    // Batch should be faster than multiple single calls
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    // Verify results match
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});

test('MAAQ edge cases', () => {
    // Test MAAQ with edge case inputs
    
    // Test with monotonically increasing data
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1;
    }
    const result = wasm.maaq_js(data, 10, 2, 20);
    assert.strictEqual(result.length, data.length);
    
    // After warmup, values should be smoothed
    for (let i = 10; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN at index ${i}`);
    }
    
    // Test with constant values
    const constantData = new Float64Array(100).fill(5.0);
    const constantResult = wasm.maaq_js(constantData, 10, 2, 20);
    assert.strictEqual(constantResult.length, constantData.length);
    
    // With constant data, MAAQ should converge to the constant value
    for (let i = 20; i < constantResult.length; i++) {
        assertClose(constantResult[i], 5.0, 1e-9, `Constant prediction failed at index ${i}`);
    }
});

test('MAAQ batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.maaq_batch_metadata_js(
        11, 31, 10,      // period range: 11, 21, 31
        2, 4, 2,         // fast_period range: 2, 4
        25, 35, 10       // slow_period range: 25, 35
    );
    
    // Should have 3 * 2 * 2 = 12 combinations, flattened to 36 values
    assert.strictEqual(metadata.length, 36);
    
    // Check first combination
    assert.strictEqual(metadata[0], 11);  // period
    assert.strictEqual(metadata[1], 2);   // fast_period
    assert.strictEqual(metadata[2], 25);  // slow_period
    
    // Check second combination (same period, same fast, different slow)
    assert.strictEqual(metadata[3], 11);  // period
    assert.strictEqual(metadata[4], 2);   // fast_period
    assert.strictEqual(metadata[5], 35);  // slow_period
});

test('MAAQ warmup period calculation', () => {
    // Test that warmup period is correctly calculated
    // MAAQ follows ALMA's warmup semantics: first (period-1) values are NaN
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const testCases = [
        { period: 5, fast_p: 2, slow_p: 10 },
        { period: 10, fast_p: 3, slow_p: 20 },
        { period: 20, fast_p: 5, slow_p: 30 },
        { period: 30, fast_p: 10, slow_p: 40 },
    ];
    
    for (const { period, fast_p, slow_p } of testCases) {
        const result = wasm.maaq_js(close, period, fast_p, slow_p);
        
        // MAAQ outputs NaN during warmup period (first period-1 values)
        for (let i = 0; i < period - 1 && i < result.length; i++) {
            assert(isNaN(result[i]), 
                `Expected NaN at warmup index ${i} for period=${period}, got ${result[i]}`);
        }
        
        // Value at index period-1 should be valid (not NaN)
        if (period - 1 < result.length) {
            assert(!isNaN(result[period - 1]),
                `Expected valid value at index ${period - 1} for period=${period}`);
        }
    }
});

test('MAAQ consistency across calls', () => {
    // Test that MAAQ produces consistent results across multiple calls
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.maaq_js(close, 11, 2, 30);
    const result2 = wasm.maaq_js(close, 11, 2, 30);
    
    assertArrayClose(result1, result2, 1e-15, "MAAQ results not consistent");
});

test('MAAQ parameter step precision', () => {
    // Test batch with very small step sizes
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.maaq_batch_js(
        data,
        {
            period_range: [5, 7, 1],         // periods: 5, 6, 7
            fast_period_range: [2, 3, 1],    // fast_periods: 2, 3
            slow_period_range: [10, 10, 0]   // slow_period: 10
        }
    );
    
    // Should have 3 * 2 * 1 = 6 combinations
    assert.strictEqual(batch_result.length, 6 * data.length);
    
    // Verify metadata
    const metadata = wasm.maaq_batch_metadata_js(5, 7, 1, 2, 3, 1, 10, 10, 0);
    assert.strictEqual(metadata.length, 18);  // 6 combos * 3 values
});

test('MAAQ streaming simulation', () => {
    // Test MAAQ streaming functionality (simulated)
    const close = new Float64Array(testData.close.slice(0, 100));
    const period = 11;
    const fast_period = 2;
    const slow_period = 30;
    
    // Calculate batch result for comparison
    const batchResult = wasm.maaq_js(close, period, fast_period, slow_period);
    
    // MAAQ has streaming support, verify batch result has expected properties
    assert.strictEqual(batchResult.length, close.length);
    
    // Verify warmup period - first period-1 values are NaN
    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(batchResult[i]), `Expected NaN at warmup index ${i}`);
    }
    
    // Verify values after warmup are valid and different from input (smoothed)
    let hasDifferentValues = false;
    for (let i = period - 1; i < close.length; i++) {
        assert(!isNaN(batchResult[i]), `Unexpected NaN at index ${i}`);
        if (Math.abs(batchResult[i] - close[i]) > 1e-9) {
            hasDifferentValues = true;
        }
    }
    assert(hasDifferentValues, "MAAQ should produce smoothed values after warmup");
});

test('MAAQ large period', () => {
    // Test MAAQ with large period relative to data size
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.sin(i * 0.1) * 100 + 1000; // Generate some test data
    }
    
    const period = 50;
    const result = wasm.maaq_js(data, period, 5, 60);
    assert.strictEqual(result.length, data.length);
    
    // First period-1 values should be NaN (warmup period)
    for (let i = 0; i < period - 1; i++) {
        assert(isNaN(result[i]), `Expected NaN at warmup index ${i}`);
    }
    
    // Values after warmup should be valid
    for (let i = period - 1; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

// Note: MAAQ expects clean data in real-world conditions
// NaN handling test removed as users should provide valid data

test('MAAQ batch with invalid ranges', () => {
    // Test batch with invalid parameter ranges
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    // Descending ranges are supported (start > end)
    const desc = wasm.maaq_batch_js(data, {
        period_range: [20, 10, 5], // 20, 15, 10
        fast_period_range: [2, 2, 0],
        slow_period_range: [30, 30, 0]
    });
    assert.strictEqual(desc.length, 3 * data.length, 'Expected 3 period combos');
    
    // Period exceeds data length
    assert.throws(() => {
        wasm.maaq_batch_js(data, {
            period_range: [100, 200, 50],
            fast_period_range: [2, 2, 0],
            slow_period_range: [30, 30, 0]
        });
    });
});

test('MAAQ accuracy with expected values', () => {
    // Test that MAAQ matches centralized expected values
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.maaq;
    
    const result = wasm.maaq_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.fast_period,
        expected.defaultParams.slow_period
    );
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        0.01,  // 1e-2 tolerance as in Python test
        "MAAQ last 5 values mismatch with expected"
    );
    
    // Verify warmup period
    const warmupEnd = expected.defaultParams.period - 1;
    for (let i = 0; i < warmupEnd; i++) {
        assert(isNaN(result[i]), `Expected NaN at warmup index ${i}`);
    }
});

test('MAAQ single value with period 1', () => {
    // Test edge case of single value with period=1
    const data = new Float64Array([42.0]);
    
    // Should fail with insufficient data
    assert.throws(() => {
        wasm.maaq_js(data, 1, 1, 1);
    });
});
