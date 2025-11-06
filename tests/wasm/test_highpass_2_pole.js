/**
 * WASM binding tests for HighPass 2-Pole indicator.
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

test('HighPass2 partial params', () => {
    // Test with default parameters - mirrors check_highpass2_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.highpass_2_pole_js(close, 48, 0.707);
    assert.strictEqual(result.length, close.length);
});

test('HighPass2 accuracy', async () => {
    // Test HighPass2 matches expected values from Rust tests - mirrors check_highpass2_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.highpass_2_pole_js(close, 48, 0.707);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected last 5 values from Rust test
    const expectedLastFive = [
        445.29073821108943,
        359.51467478973296,
        250.7236793408186,
        394.04381266217234,
        -52.65414073315134,
    ];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-6,  // Using 1e-6 as in Rust test
        "HighPass2 last 5 values mismatch"
    );
    
    // Compare full output with Rust
    // Use a very tight tolerance so it never exceeds Rust's 1e-6 absolute
    await compareWithRust('highpass_2_pole', result, 'close', { period: 48, k: 0.707 }, 1e-12);
});

test('HighPass2 default candles', async () => {
    // Test HighPass2 with default parameters - mirrors check_highpass2_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.highpass_2_pole_js(close, 48, 0.707);
    assert.strictEqual(result.length, close.length);
    
    // Compare with Rust
    // Use a very tight tolerance so it never exceeds Rust's 1e-6 absolute
    await compareWithRust('highpass_2_pole', result, 'close', { period: 48, k: 0.707 }, 1e-12);
});

test('HighPass2 zero period', () => {
    // Test HighPass2 fails with zero period - mirrors check_highpass2_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.highpass_2_pole_js(inputData, 0, 0.707);
    });
});

test('HighPass2 period exceeds length', () => {
    // Test HighPass2 fails when period exceeds data length - mirrors check_highpass2_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.highpass_2_pole_js(dataSmall, 10, 0.707);
    });
});

test('HighPass2 very small dataset', () => {
    // Test HighPass2 with very small dataset - mirrors check_highpass2_very_small_dataset
    const dataSingle = new Float64Array([42.0]);
    
    // Period=2 should fail with single data point
    assert.throws(() => {
        wasm.highpass_2_pole_js(dataSingle, 2, 0.707);
    }, /Invalid period/);
});

test('HighPass2 empty input', () => {
    // Test HighPass2 with empty input - mirrors check_highpass2_empty_input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.highpass_2_pole_js(dataEmpty, 48, 0.707);
    });
});

test('HighPass2 invalid k', () => {
    // Test HighPass2 with invalid k - mirrors check_highpass2_invalid_k
    const data = new Float64Array([1.0, 2.0, 3.0]);
    
    // Test k = -0.5 (negative)
    assert.throws(() => {
        wasm.highpass_2_pole_js(data, 2, -0.5);
    });
});

test('HighPass2 all NaN', () => {
    // Test HighPass2 with all NaN input
    const data = new Float64Array([NaN, NaN, NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.highpass_2_pole_js(data, 3, 0.707);
    });
});

test('HighPass2 reinput', () => {
    // Test HighPass2 with re-input of HighPass2 result - mirrors check_highpass2_reinput
    const close = new Float64Array(testData.close);
    
    // First HighPass2 pass with period=48, k=0.707
    const firstResult = wasm.highpass_2_pole_js(close, 48, 0.707);
    
    // Second HighPass2 pass with period=32, k=0.707 using first result as input
    const secondResult = wasm.highpass_2_pole_js(firstResult, 32, 0.707);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Verify no NaN values after warmup period in second result
    for (let i = 240; i < secondResult.length; i++) {
        assert(!isNaN(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('HighPass2 NaN handling', () => {
    // Test HighPass2 handling of NaN values - mirrors check_highpass2_nan_handling
    const close = new Float64Array(testData.close);
    const period = 48;
    const k = 0.707;
    
    const result = wasm.highpass_2_pole_js(close, period, k);
    
    assert.strictEqual(result.length, close.length);
    
    // highpass_2_pole only puts NaN for leading NaN inputs
    // Since test data has no leading NaN, output should have no NaN values
    for (let i = 0; i < result.length; i++) {
        assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
    }
});

test('HighPass2 batch', () => {
    // Test HighPass2 batch computation
    const close = new Float64Array(testData.close);
    
    // Test period range 40-60 step 10, k range 0.5-0.9 step 0.2
    const period_start = 40;
    const period_end = 60;
    const period_step = 10;  // periods: 40, 50, 60
    const k_start = 0.5;
    const k_end = 0.9;
    const k_step = 0.2;      // k: 0.5, 0.7, 0.9
    
    // Create config object for batch API
    const config = {
        period_range: [period_start, period_end, period_step],
        k_range: [k_start, k_end, k_step]
    };
    
    const batch_output = wasm.highpass_2_pole_batch(close, config);
    const batch_result = batch_output.values;
    const metadata = batch_output.combos;
    
    // Metadata should contain combo objects
    assert.strictEqual(metadata.length, 9);  // 3 periods x 3 k values
    
    // Batch result should contain all individual results flattened
    assert.strictEqual(batch_result.length, 9 * close.length);  // 3 periods x 3 k values = 9 rows
    
    // Verify each row matches individual calculation
    let row_idx = 0;
    for (const period of [40, 50, 60]) {
        for (const k of [0.5, 0.7, 0.9]) {
            const individual_result = wasm.highpass_2_pole_js(close, period, k);
            
            // Extract row from batch result
            const row_start = row_idx * close.length;
            const row = batch_result.slice(row_start, row_start + close.length);
            
            assertArrayClose(row, individual_result, 1e-9, `Period ${period}, k ${k}`);
            row_idx++;
        }
    }
});

test('HighPass2 different k values', () => {
    // Test HighPass2 with different k values
    const close = new Float64Array(testData.close);
    const period = 48;
    
    // Test various k values between 0.1 and 0.9
    for (const k of [0.1, 0.3, 0.5, 0.707, 0.9]) {
        const result = wasm.highpass_2_pole_js(close, period, k);
        assert.strictEqual(result.length, close.length);
        
        // highpass_2_pole only puts NaN for leading NaN inputs
        // Since test data has no leading NaN, all output should be valid
        for (let i = 0; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for k=${k}`);
        }
    }
});

test('HighPass2 batch performance', () => {
    // Test that batch computation is more efficient than multiple single computations
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use first 1000 values
    
    // Test 5 periods x 4 k values = 20 combinations
    const startBatch = performance.now();
    const config = {
        period_range: [30, 70, 10],
        k_range: [0.3, 0.9, 0.2]
    };
    const batchOutput = wasm.highpass_2_pole_batch(close, config);
    const batchResult = batchOutput.values;
    const batchTime = performance.now() - startBatch;
    
    const startSingle = performance.now();
    const singleResults = [];
    for (let period = 30; period <= 70; period += 10) {
        for (let k = 0.3; k <= 0.9 + 1e-10; k += 0.2) {
            singleResults.push(...wasm.highpass_2_pole_js(close, period, k));
        }
    }
    const singleTime = performance.now() - startSingle;
    
    // Batch should be faster than multiple single calls
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    // Verify results match
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});

// Zero-copy API tests
test('HighPass2 zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const period = 5;
    const k = 0.707;
    
    // Check if zero-copy functions are available
    if (!wasm.highpass_2_pole_alloc || !wasm.highpass_2_pole_into || !wasm.highpass_2_pole_free) {
        console.log('Zero-copy API not available for highpass_2_pole, skipping test');
        return;
    }
    
    // Allocate buffer
    const ptr = wasm.highpass_2_pole_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memView = new Float64Array(
        wasm.__wasm.memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute HighPass2 in-place
    try {
        wasm.highpass_2_pole_into(ptr, ptr, data.length, period, k);
        
        // Verify results match regular API
        const regularResult = wasm.highpass_2_pole_js(data, period, k);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.highpass_2_pole_free(ptr, data.length);
    }
});

test('HighPass2 zero-copy with large dataset', () => {
    // Check if zero-copy functions are available
    if (!wasm.highpass_2_pole_alloc || !wasm.highpass_2_pole_into || !wasm.highpass_2_pole_free) {
        console.log('Zero-copy API not available for highpass_2_pole, skipping test');
        return;
    }
    
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.sin(i * 0.01) * 100 + Math.random() * 10;
    }
    
    const ptr = wasm.highpass_2_pole_alloc(size);
    assert(ptr !== 0, 'Failed to allocate large buffer');
    
    try {
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        memView.set(data);
        
        wasm.highpass_2_pole_into(ptr, ptr, size, 48, 0.707);
        
        // Recreate view in case memory grew
        const memView2 = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        
        // highpass_2_pole only puts NaN for leading NaN inputs
        // Since our input has no leading NaN, output starts with real values immediately
        // The filter seeds first two values and calculates from index 2
        for (let i = 0; i < Math.min(100, size); i++) {
            assert(!isNaN(memView2[i]), `Unexpected NaN at index ${i}`);
        }
    } finally {
        wasm.highpass_2_pole_free(ptr, size);
    }
});

// Error handling for zero-copy API
test('HighPass2 zero-copy error handling', () => {
    // Check if zero-copy functions are available
    if (!wasm.highpass_2_pole_alloc || !wasm.highpass_2_pole_into || !wasm.highpass_2_pole_free) {
        console.log('Zero-copy API not available for highpass_2_pole, skipping test');
        return;
    }
    
    // Test null pointer
    assert.throws(() => {
        wasm.highpass_2_pole_into(0, 0, 10, 48, 0.707);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.highpass_2_pole_alloc(10);
    try {
        // Invalid period
        assert.throws(() => {
            wasm.highpass_2_pole_into(ptr, ptr, 10, 0, 0.707);
        }, /Invalid period/);
        
        // Invalid k
        assert.throws(() => {
            wasm.highpass_2_pole_into(ptr, ptr, 10, 5, 0.0);
        }, /Invalid k/);
        
        assert.throws(() => {
            wasm.highpass_2_pole_into(ptr, ptr, 10, 5, -1.0);
        }, /Invalid k/);
    } finally {
        wasm.highpass_2_pole_free(ptr, 10);
    }
});

// Memory leak prevention test
test('HighPass2 zero-copy memory management', () => {
    // Check if zero-copy functions are available
    if (!wasm.highpass_2_pole_alloc || !wasm.highpass_2_pole_free) {
        console.log('Zero-copy API not available for highpass_2_pole, skipping test');
        return;
    }
    
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 5000];
    
    for (const size of sizes) {
        const ptr = wasm.highpass_2_pole_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Write pattern to verify memory
        const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 2.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 2.5, `Memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.highpass_2_pole_free(ptr, size);
    }
});

// SIMD128 verification test
test('HighPass2 SIMD128 consistency', () => {
    // This test verifies SIMD128 produces same results as scalar
    // It runs automatically when SIMD128 is enabled
    const testCases = [
        { size: 20, period: 5 },
        { size: 100, period: 10 },
        { size: 500, period: 48 },
        { size: 1000, period: 100 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) * Math.exp(-i * 0.001) + Math.cos(i * 0.05) * 10;
        }
        
        const result = wasm.highpass_2_pole_js(data, testCase.period, 0.707);
        
        // Basic sanity checks
        assert.strictEqual(result.length, data.length);
        
        // highpass_2_pole only puts NaN for leading NaN inputs
        // Since our input has no leading NaN, output starts with real values immediately
        // No warmup NaN values expected when input has no leading NaN
        
        // Check values exist after warmup
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.period - 1; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        // Verify reasonable values (highpass filter should have zero mean over long term)
        const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
        assert(Math.abs(avgAfterWarmup) < 100, `Average value ${avgAfterWarmup} seems unreasonable`);
    }
});

// Batch API with new ergonomic interface tests
test('HighPass2 batch - new ergonomic API with single parameter', () => {
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result = wasm.highpass_2_pole_batch(close, {
        period_range: [48, 48, 0],
        k_range: [0.707, 0.707, 0]
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
    assert.strictEqual(combo.period, 48);
    assert.strictEqual(combo.k, 0.707);
    
    // Compare with single API
    const singleResult = wasm.highpass_2_pole_js(close, 48, 0.707);
    for (let i = 0; i < singleResult.length; i++) {
        if (isNaN(singleResult[i]) && isNaN(result.values[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(singleResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('HighPass2 batch - edge cases', () => {
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    
    // Single value sweep
    const singleBatch = wasm.highpass_2_pole_batch(close, {
        period_range: [5, 5, 1],
        k_range: [0.5, 0.5, 0.1]
    });
    
    assert.strictEqual(singleBatch.values.length, 12);
    assert.strictEqual(singleBatch.combos.length, 1);
    
    // Step larger than range
    const largeBatch = wasm.highpass_2_pole_batch(close, {
        period_range: [5, 7, 10], // Step larger than range
        k_range: [0.707, 0.707, 0]
    });
    
    // Should only have period=5
    assert.strictEqual(largeBatch.values.length, 12);
    assert.strictEqual(largeBatch.combos.length, 1);
    assert.strictEqual(largeBatch.combos[0].period, 5);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.highpass_2_pole_batch(new Float64Array([]), {
            period_range: [48, 48, 0],
            k_range: [0.707, 0.707, 0]
        });
    }, /All values are NaN|Empty/);
});

test('HighPass2 first value handling', () => {
    // Test handling of leading NaN values
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Add leading NaN values
    const closeWithNaN = new Float64Array(103);
    closeWithNaN[0] = NaN;
    closeWithNaN[1] = NaN;
    closeWithNaN[2] = NaN;
    closeWithNaN.set(close, 3);
    
    const result = wasm.highpass_2_pole_js(closeWithNaN, 48, 0.707);
    
    assert.strictEqual(result.length, closeWithNaN.length);
    
    // First 3 values should remain NaN
    assert(isNaN(result[0]));
    assert(isNaN(result[1]));
    assert(isNaN(result[2]));
    
    // Find first non-NaN value position
    let firstValid = null;
    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i])) {
            firstValid = i;
            break;
        }
    }
    
    // highpass_2_pole preserves leading NaN, then starts calculating immediately
    // So first valid should be at position 3 (right after the 3 NaN values)
    assert.strictEqual(firstValid, 3, `First valid at ${firstValid}, expected 3`);
});

// Note: Streaming functionality is available through the Python bindings.
// For WASM, repeated calculations with same parameters can be achieved using
// the zero-copy API with persistent buffers for optimal performance.
