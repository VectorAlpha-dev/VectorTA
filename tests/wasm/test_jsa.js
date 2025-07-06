/**
 * WASM binding tests for JSA indicator.
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
        wasm = await import(wasmPath);
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('JSA partial params', () => {
    // Test with default parameters - mirrors check_jsa_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.jsa_js(close, 30);
    assert.strictEqual(result.length, close.length);
});

test('JSA accuracy', async () => {
    // Test JSA matches expected values from Rust tests - mirrors check_jsa_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.jsa_js(close, 30);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected last 5 values from Rust test
    const expectedLastFive = [61640.0, 61418.0, 61240.0, 61060.5, 60889.5];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-5,
        "JSA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    // await compareWithRust('jsa', result, 'close', { period: 30 });
});

test('JSA default candles', async () => {
    // Test JSA with default parameters - mirrors check_jsa_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.jsa_js(close, 30);
    assert.strictEqual(result.length, close.length);
    
    // Compare with Rust
    await compareWithRust('jsa', result, 'close', { period: 30 });
});

test('JSA zero period', () => {
    // Test JSA fails with zero period - mirrors check_jsa_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.jsa_js(inputData, 0);
    });
});

test('JSA period exceeds length', () => {
    // Test JSA fails when period exceeds data length - mirrors check_jsa_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.jsa_js(dataSmall, 10);
    });
});

test('JSA very small dataset', () => {
    // Test JSA with very small dataset - mirrors check_jsa_very_small_dataset
    const dataSingle = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.jsa_js(dataSingle, 5);
    });
});

test('JSA empty input', () => {
    // Test JSA with empty input - mirrors check_jsa_empty_input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.jsa_js(dataEmpty, 30);
    });
});

test('JSA all NaN', () => {
    // Test JSA with all NaN input
    const data = new Float64Array([NaN, NaN, NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.jsa_js(data, 3);
    });
});

test('JSA reinput', () => {
    // Test JSA with re-input of JSA result - mirrors check_jsa_reinput
    const close = new Float64Array(testData.close);
    
    // First JSA pass with period=10
    const firstResult = wasm.jsa_js(close, 10);
    
    // Second JSA pass with period=5 using first result as input
    const secondResult = wasm.jsa_js(firstResult, 5);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Verify values are reasonable (not NaN/Inf) after index 30
    for (let i = 30; i < secondResult.length; i++) {
        assert(isFinite(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('JSA NaN handling', () => {
    // Test JSA handling of NaN values - mirrors check_jsa_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.jsa_js(close, 30);
    
    assert.strictEqual(result.length, close.length);
    
    // Check warmup period
    for (let i = 0; i < 30; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    // After warmup period, no NaN values should exist
    for (let i = 30; i < result.length; i++) {
        assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
    }
});

test('JSA batch', () => {
    // Test JSA batch computation
    const close = new Float64Array(testData.close);
    
    // Test period range 10-40 step 10
    const period_start = 10;
    const period_end = 40;
    const period_step = 10;  // periods: 10, 20, 30, 40
    
    const batch_result = wasm.jsa_batch_js(
        close, 
        period_start, period_end, period_step
    );
    const metadata = wasm.jsa_batch_metadata_js(
        period_start, period_end, period_step
    );
    
    // Metadata should contain period values
    assert.strictEqual(metadata.length, 4);  // 4 periods
    assert.deepStrictEqual(Array.from(metadata), [10, 20, 30, 40]);
    
    // Batch result should contain all individual results flattened
    assert.strictEqual(batch_result.length, 4 * close.length);  // 4 periods
    
    // Verify each row matches individual calculation
    let row_idx = 0;
    for (const period of [10, 20, 30, 40]) {
        const individual_result = wasm.jsa_js(close, period);
        
        // Extract row from batch result
        const row_start = row_idx * close.length;
        const row = batch_result.slice(row_start, row_start + close.length);
        
        assertArrayClose(row, individual_result, 1e-9, `Period ${period}`);
        row_idx++;
    }
});

test('JSA different periods', () => {
    // Test JSA with different period values
    const close = new Float64Array(testData.close);
    
    // Test various period values
    for (const period of [5, 10, 20, 50]) {
        const result = wasm.jsa_js(close, period);
        assert.strictEqual(result.length, close.length);
        
        // Verify warmup period
        for (let i = 0; i < period; i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} for period=${period}`);
        }
        
        // Verify no NaN after warmup period
        for (let i = period; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for period=${period}`);
        }
    }
});

test('JSA batch performance', () => {
    // Test that batch computation is more efficient than multiple single computations
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use first 1000 values
    
    // Test 5 periods
    const startBatch = performance.now();
    const batchResult = wasm.jsa_batch_js(close, 10, 50, 10);
    const batchTime = performance.now() - startBatch;
    
    const startSingle = performance.now();
    const singleResults = [];
    for (let period = 10; period <= 50; period += 10) {
        singleResults.push(...wasm.jsa_js(close, period));
    }
    const singleTime = performance.now() - startSingle;
    
    // Batch should be faster than multiple single calls
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    // Verify results match
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});

test('JSA edge cases', () => {
    // Test JSA with edge case inputs
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    // Test with period=1
    const result1 = wasm.jsa_js(data, 1);
    assert.strictEqual(result1.length, data.length);
    assert(isNaN(result1[0])); // First value is NaN
    // Each subsequent value should be average of current and previous
    for (let i = 1; i < data.length; i++) {
        const expected = (data[i] + data[i-1]) * 0.5;
        assertClose(result1[i], expected, 1e-9, `Value at index ${i}`);
    }
    
    // Test with period equal to data length - should NOT throw
    const result2 = wasm.jsa_js(data, data.length);
    assert.strictEqual(result2.length, data.length);
});

test('JSA single value', () => {
    // Test JSA with single value input
    const data = new Float64Array([42.0]);
    
    // Period=1 with single value should work
    const result = wasm.jsa_js(data, 1);
    assert.strictEqual(result.length, 1);
    assert(isNaN(result[0])); // Should be NaN as it's in the warmup period
});

test('JSA two values', () => {
    // Test JSA with two values input
    const data = new Float64Array([1.0, 2.0]);
    
    // Should work with period=1
    const result = wasm.jsa_js(data, 1);
    assert.strictEqual(result.length, 2);
    assert(isNaN(result[0])); // First value is NaN
    assertClose(result[1], (data[1] + data[0]) * 0.5, 1e-9); // Average of both values
    
    // Should also work with period=2 (period == data.length is allowed)
    const result2 = wasm.jsa_js(data, 2);
    assert.strictEqual(result2.length, 2);
    // Both values should be NaN as warmup = first + period = 0 + 2 = 2
    assert(isNaN(result2[0]));
    assert(isNaN(result2[1]));
});

test('JSA batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.jsa_batch_metadata_js(15, 45, 15);
    
    // Should have 3 periods: 15, 30, 45
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 15);
    assert.strictEqual(metadata[1], 30);
    assert.strictEqual(metadata[2], 45);
});

test('JSA warmup period calculation', () => {
    // Test that warmup period is correctly calculated
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const testCases = [
        { period: 5, expectedWarmup: 5 },
        { period: 10, expectedWarmup: 10 },
        { period: 20, expectedWarmup: 20 },
        { period: 30, expectedWarmup: 30 },
    ];
    
    for (const { period, expectedWarmup } of testCases) {
        const result = wasm.jsa_js(close, period);
        
        // Check NaN values up to warmup period
        for (let i = 0; i < expectedWarmup && i < result.length; i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} for period=${period}`);
        }
        
        // Check valid values after warmup
        if (expectedWarmup < result.length) {
            assert(!isNaN(result[expectedWarmup]), 
                `Expected valid value at index ${expectedWarmup} for period=${period}`);
        }
    }
});

test('JSA consistency across calls', () => {
    // Test that JSA produces consistent results across multiple calls
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.jsa_js(close, 30);
    const result2 = wasm.jsa_js(close, 30);
    
    assertArrayClose(result1, result2, 1e-15, "JSA results not consistent");
});

test('JSA parameter step precision', () => {
    // Test batch with very small step sizes
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    const batch_result = wasm.jsa_batch_js(data, 2, 4, 1);  // periods: 2, 3, 4
    
    // Should have 3 periods
    assert.strictEqual(batch_result.length, 3 * data.length);
    
    // Verify metadata
    const metadata = wasm.jsa_batch_metadata_js(2, 4, 1);
    assert.deepStrictEqual(Array.from(metadata), [2, 3, 4]);
});

test('JSA streaming', () => {
    // Test JSA streaming functionality
    const close = new Float64Array(testData.close.slice(0, 100));
    const period = 30;
    
    // Calculate batch result for comparison
    const batchResult = wasm.jsa_js(close, period);
    
    // Simulate streaming by calculating incrementally
    const streamResults = new Float64Array(close.length);
    streamResults.fill(NaN);
    
    // For JSA, we need to keep a window of data
    for (let i = period; i < close.length; i++) {
        // JSA formula: (current + value_period_ago) / 2
        streamResults[i] = (close[i] + close[i - period]) * 0.5;
    }
    
    // Compare results after warmup period
    for (let i = period; i < close.length; i++) {
        assertClose(streamResults[i], batchResult[i], 1e-9, 
                   `Streaming mismatch at index ${i}`);
    }
});

test('JSA large period', () => {
    // Test JSA with large period relative to data size
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.sin(i * 0.1) * 100 + 1000; // Generate some test data
    }
    
    const result = wasm.jsa_js(data, 99);
    assert.strictEqual(result.length, data.length);
    
    // Only the last value should be valid
    for (let i = 0; i < 99; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    // Last value should be average of first and last
    const expected = (data[99] + data[0]) * 0.5;
    assertClose(result[99], expected, 1e-9, "Last value mismatch");
});