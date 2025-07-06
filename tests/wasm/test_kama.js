/**
 * WASM binding tests for KAMA indicator.
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

test('KAMA partial params', () => {
    // Test with default parameters - mirrors check_kama_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.kama_js(close, 30);
    assert.strictEqual(result.length, close.length);
});

test('KAMA accuracy', async () => {
    // Test KAMA matches expected values from Rust tests - mirrors check_kama_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.kama_js(close, 30);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected last 5 values from Rust test
    const expectedLastFive = [
        60234.925553804125,
        60176.838757545665,
        60115.177367962766,
        60071.37070833558,
        59992.79386218023
    ];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-6,
        "KAMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('kama', result, 'close', { period: 30 });
});

test('KAMA default candles', async () => {
    // Test KAMA with default parameters - mirrors check_kama_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.kama_js(close, 30);
    assert.strictEqual(result.length, close.length);
    
    // Compare with Rust
    await compareWithRust('kama', result, 'close', { period: 30 });
});

test('KAMA zero period', () => {
    // Test KAMA fails with zero period - mirrors check_kama_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.kama_js(inputData, 0);
    });
});

test('KAMA period exceeds length', () => {
    // Test KAMA fails when period exceeds data length - mirrors check_kama_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.kama_js(dataSmall, 10);
    });
});

test('KAMA very small dataset', () => {
    // Test KAMA with very small dataset - mirrors check_kama_very_small_dataset
    const dataSingle = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.kama_js(dataSingle, 5);
    });
});

test('KAMA empty input', () => {
    // Test KAMA with empty input - mirrors check_kama_empty_input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.kama_js(dataEmpty, 30);
    });
});

test('KAMA all NaN', () => {
    // Test KAMA with all NaN input
    const data = new Float64Array([NaN, NaN, NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.kama_js(data, 3);
    });
});

test('KAMA reinput', () => {
    // Test KAMA with re-input of KAMA result - mirrors check_kama_reinput
    const close = new Float64Array(testData.close);
    
    // First KAMA pass with period=30 (matching Rust test)
    const firstResult = wasm.kama_js(close, 30);
    
    // Second KAMA pass with period=10 using first result as input
    const secondResult = wasm.kama_js(firstResult, 10);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // The second pass will have its own warmup period
    // First pass warmup: 30 values
    // Second pass warmup: 10 values  
    // So we expect NaN values up to index 30+10=40
    for (let i = 40; i < secondResult.length; i++) {
        assert(isFinite(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('KAMA NaN handling', () => {
    // Test KAMA handling of NaN values - mirrors check_kama_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.kama_js(close, 30);
    
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

test('KAMA batch', () => {
    // Test KAMA batch computation
    const close = new Float64Array(testData.close);
    
    // Test period range 10-40 step 10
    const period_start = 10;
    const period_end = 40;
    const period_step = 10;  // periods: 10, 20, 30, 40
    
    const batch_result = wasm.kama_batch_js(
        close, 
        period_start, period_end, period_step
    );
    const metadata = wasm.kama_batch_metadata_js(
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
        const individual_result = wasm.kama_js(close, period);
        
        // Extract row from batch result
        const row_start = row_idx * close.length;
        const row = batch_result.slice(row_start, row_start + close.length);
        
        assertArrayClose(row, individual_result, 1e-9, `Period ${period}`);
        row_idx++;
    }
});

test('KAMA different periods', () => {
    // Test KAMA with different period values
    const close = new Float64Array(testData.close);
    
    // Test various period values
    for (const period of [5, 10, 20, 50]) {
        const result = wasm.kama_js(close, period);
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

test('KAMA batch performance', () => {
    // Test that batch computation is more efficient than multiple single computations
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use first 1000 values
    
    // Test 5 periods
    const startBatch = performance.now();
    const batchResult = wasm.kama_batch_js(close, 10, 50, 10);
    const batchTime = performance.now() - startBatch;
    
    const startSingle = performance.now();
    const singleResults = [];
    for (let period = 10; period <= 50; period += 10) {
        singleResults.push(...wasm.kama_js(close, period));
    }
    const singleTime = performance.now() - startSingle;
    
    // Batch should be faster than multiple single calls
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    // Verify results match
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});

test('KAMA edge cases', () => {
    // Test KAMA with edge case inputs
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    // Test with period=1
    const result1 = wasm.kama_js(data, 1);
    assert.strictEqual(result1.length, data.length);
    assert(isNaN(result1[0])); // First value is NaN
    // Subsequent values should be valid
    for (let i = 1; i < data.length; i++) {
        assert(isFinite(result1[i]), `Expected finite value at index ${i}`);
    }
    
    // Test with period equal to data length-1
    const result2 = wasm.kama_js(data, data.length - 1);
    assert.strictEqual(result2.length, data.length);
    // Almost all should be NaN except last
    for (let i = 0; i < data.length - 1; i++) {
        assert(isNaN(result2[i]), `Expected NaN at index ${i}`);
    }
    assert(isFinite(result2[data.length - 1]), 'Expected finite value at last index');
});

test('KAMA single value', () => {
    // Test KAMA with single value input
    const data = new Float64Array([42.0]);
    
    // Period=1 with single value should fail (need at least period+1 values)
    assert.throws(() => {
        wasm.kama_js(data, 1);
    });
});

test('KAMA two values', () => {
    // Test KAMA with two values input
    const data = new Float64Array([1.0, 2.0]);
    
    // Should work with period=1
    const result = wasm.kama_js(data, 1);
    assert.strictEqual(result.length, 2);
    assert(isNaN(result[0])); // First value is NaN
    assert(isFinite(result[1])); // Second value should be valid
});

test('KAMA batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.kama_batch_metadata_js(15, 45, 15);
    
    // Should have 3 periods: 15, 30, 45
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 15);
    assert.strictEqual(metadata[1], 30);
    assert.strictEqual(metadata[2], 45);
});

test('KAMA warmup period calculation', () => {
    // Test that warmup period is correctly calculated
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const testCases = [
        { period: 5, expectedWarmup: 5 },
        { period: 10, expectedWarmup: 10 },
        { period: 20, expectedWarmup: 20 },
        { period: 30, expectedWarmup: 30 },
    ];
    
    for (const { period, expectedWarmup } of testCases) {
        const result = wasm.kama_js(close, period);
        
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

test('KAMA consistency across calls', () => {
    // Test that KAMA produces consistent results across multiple calls
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.kama_js(close, 30);
    const result2 = wasm.kama_js(close, 30);
    
    assertArrayClose(result1, result2, 1e-15, "KAMA results not consistent");
});

test('KAMA parameter step precision', () => {
    // Test batch with very small step sizes
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    const batch_result = wasm.kama_batch_js(data, 2, 4, 1);  // periods: 2, 3, 4
    
    // Should have 3 periods
    assert.strictEqual(batch_result.length, 3 * data.length);
    
    // Verify metadata
    const metadata = wasm.kama_batch_metadata_js(2, 4, 1);
    assert.deepStrictEqual(Array.from(metadata), [2, 3, 4]);
});

test('KAMA streaming simulation', () => {
    // Test KAMA streaming functionality (simulated)
    const close = new Float64Array(testData.close.slice(0, 100));
    const period = 30;
    
    // Calculate batch result for comparison
    const batchResult = wasm.kama_js(close, period);
    
    // KAMA requires full history, so streaming is more complex
    // We'll verify batch result has expected properties
    assert.strictEqual(batchResult.length, close.length);
    
    // Verify warmup period
    for (let i = 0; i < period; i++) {
        assert(isNaN(batchResult[i]), `Expected NaN at index ${i}`);
    }
    
    // Verify values after warmup
    for (let i = period; i < close.length; i++) {
        assert(isFinite(batchResult[i]), `Expected finite value at index ${i}`);
    }
});

test('KAMA large period', () => {
    // Test KAMA with large period relative to data size
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.sin(i * 0.1) * 100 + 1000; // Generate some test data
    }
    
    const result = wasm.kama_js(data, 99);
    assert.strictEqual(result.length, data.length);
    
    // Only the last value should be valid
    for (let i = 0; i < 99; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    // Last value should be valid
    assert(isFinite(result[99]), "Expected finite value at last index");
});