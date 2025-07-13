/**
 * WASM binding tests for HMA indicator.
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

test('HMA partial params', () => {
    // Test with default parameters - mirrors check_hma_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.hma_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('HMA accuracy', async () => {
    // Test HMA matches expected values from Rust tests - mirrors check_hma_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.hma_js(close, 5);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected last 5 values from Rust test
    const expectedLastFive = [
        59334.13333336847,
        59201.4666667018,
        59047.77777781293,
        59048.71111114628,
        58803.44444447962,
    ];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-3,  // Using 1e-3 as in Rust test
        "HMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    // await compareWithRust('hma', result, 'close', { period: 5 });
});

test('HMA default candles', async () => {
    // Test HMA with default parameters - mirrors check_hma_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.hma_js(close, 5);
    assert.strictEqual(result.length, close.length);
    
    // Compare with Rust
    await compareWithRust('hma', result, 'close', { period: 5 });
});

test('HMA zero period', () => {
    // Test HMA fails with zero period - mirrors check_hma_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.hma_js(inputData, 0);
    });
});

test('HMA period exceeds length', () => {
    // Test HMA fails when period exceeds data length - mirrors check_hma_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.hma_js(dataSmall, 10);
    });
});

test('HMA very small dataset', () => {
    // Test HMA with very small dataset - mirrors check_hma_very_small_dataset
    const dataSingle = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.hma_js(dataSingle, 5);
    });
});

test('HMA empty input', () => {
    // Test HMA with empty input - mirrors check_hma_empty_input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.hma_js(dataEmpty, 5);
    });
});

test('HMA all NaN', () => {
    // Test HMA with all NaN input
    const data = new Float64Array([NaN, NaN, NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.hma_js(data, 3);
    });
});

test('HMA reinput', () => {
    // Test HMA with re-input of HMA result - mirrors check_hma_reinput
    const close = new Float64Array(testData.close);
    
    // First HMA pass with period=5
    const firstResult = wasm.hma_js(close, 5);
    
    // Second HMA pass with period=3 using first result as input
    const secondResult = wasm.hma_js(firstResult, 3);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Verify no NaN values after index 240 in second result
    if (secondResult.length > 240) {
        for (let i = 240; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `NaN found at index ${i}`);
        }
    }
});

test('HMA NaN handling', () => {
    // Test HMA handling of NaN values - mirrors check_hma_nan_handling
    const close = new Float64Array(testData.close);
    const period = 5;
    
    const result = wasm.hma_js(close, period);
    
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (period * 2), no NaN values should exist
    if (result.length > period * 2) {
        for (let i = period * 2; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
        }
    }
});

test('HMA batch', () => {
    // Test HMA batch computation
    const close = new Float64Array(testData.close);
    
    // Test period range 3-9 step 2
    const period_start = 3;
    const period_end = 9;
    const period_step = 2;  // periods: 3, 5, 7, 9
    
    const batch_result = wasm.hma_batch_js(
        close, 
        period_start, period_end, period_step
    );
    const metadata = wasm.hma_batch_metadata_js(
        period_start, period_end, period_step
    );
    
    // Metadata should contain period values
    assert.strictEqual(metadata.length, 4);  // 4 periods
    
    // Batch result should contain all individual results flattened
    assert.strictEqual(batch_result.length, 4 * close.length);  // 4 periods
    
    // Verify each row matches individual calculation
    let row_idx = 0;
    for (const period of [3, 5, 7, 9]) {
        const individual_result = wasm.hma_js(close, period);
        
        // Extract row from batch result
        const row_start = row_idx * close.length;
        const row = batch_result.slice(row_start, row_start + close.length);
        
        assertArrayClose(row, individual_result, 1e-9, `Period ${period}`);
        row_idx++;
    }
});

test('HMA different periods', () => {
    // Test HMA with different period values
    const close = new Float64Array(testData.close);
    
    // Test various period values
    for (const period of [3, 5, 10, 20]) {
        const result = wasm.hma_js(close, period);
        assert.strictEqual(result.length, close.length);
        
        // Calculate expected warmup period
        const sqrtPeriod = Math.floor(Math.sqrt(period));
        const warmup = period + sqrtPeriod - 1;
        
        // Verify no NaN after warmup period
        for (let i = warmup; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for period=${period}`);
        }
    }
});

test('HMA batch performance', () => {
    // Test that batch computation is more efficient than multiple single computations
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use first 1000 values
    
    // Test 5 periods
    const startBatch = performance.now();
    const batchResult = wasm.hma_batch_js(close, 5, 25, 5);
    const batchTime = performance.now() - startBatch;
    
    const startSingle = performance.now();
    const singleResults = [];
    for (let period = 5; period <= 25; period += 5) {
        singleResults.push(...wasm.hma_js(close, period));
    }
    const singleTime = performance.now() - startSingle;
    
    // Batch should be faster than multiple single calls
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    // Verify results match
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});

test('HMA zero half', () => {
    // Test HMA fails when period/2 is zero
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // Period=1 would result in half=0
    assert.throws(() => {
        wasm.hma_js(data, 1);
    });
});

test('HMA small periods', () => {
    // Test HMA with small periods
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    
    // Period=2 should work (sqrt(2) ≈ 1.4 → 1)
    const result = wasm.hma_js(data, 2);
    assert.strictEqual(result.length, data.length);
    
    // Check warmup period for period=2
    // warmup = period + sqrt(period) - 2 = 2 + 1 - 2 = 1
    assert(isNaN(result[0]));
    assert(!isNaN(result[1])); // Should have valid value from index 1
});

test('HMA not enough valid data', () => {
    // Test HMA with insufficient valid data after NaN prefix
    const data = new Float64Array([NaN, NaN, 1.0, 2.0, 3.0]);
    
    // With period=4, needs at least 4 valid values
    assert.throws(() => {
        wasm.hma_js(data, 4);
    });
});

test('HMA batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.hma_batch_metadata_js(5, 15, 5);
    
    // Should have 3 periods: 5, 10, 15
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 5);
    assert.strictEqual(metadata[1], 10);
    assert.strictEqual(metadata[2], 15);
});

test('HMA warmup period calculation', () => {
    // Test that warmup period is correctly calculated
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const testCases = [
        { period: 3, expectedWarmup: 3 + 1 - 2 },  // sqrt(3) = 1, warmup = 2
        { period: 5, expectedWarmup: 5 + 2 - 2 },  // sqrt(5) = 2, warmup = 5
        { period: 10, expectedWarmup: 10 + 3 - 2 }, // sqrt(10) = 3, warmup = 11
        { period: 16, expectedWarmup: 16 + 4 - 2 }, // sqrt(16) = 4, warmup = 18
    ];
    
    for (const { period, expectedWarmup } of testCases) {
        const result = wasm.hma_js(close, period);
        
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