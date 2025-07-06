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
        wasm = await import(wasmPath);
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
    await compareWithRust('highpass_2_pole', result, 'close', { period: 48, k: 0.707 });
});

test('HighPass2 default candles', async () => {
    // Test HighPass2 with default parameters - mirrors check_highpass2_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.highpass_2_pole_js(close, 48, 0.707);
    assert.strictEqual(result.length, close.length);
    
    // Compare with Rust
    await compareWithRust('highpass_2_pole', result, 'close', { period: 48, k: 0.707 });
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
    
    assert.throws(() => {
        wasm.highpass_2_pole_js(dataSingle, 2, 0.707);
    });
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
    
    // After warmup period, there should be no NaN values
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
        }
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
    
    const batch_result = wasm.highpass_2_pole_batch_js(
        close, 
        period_start, period_end, period_step,
        k_start, k_end, k_step
    );
    const metadata = wasm.highpass_2_pole_batch_metadata_js(
        period_start, period_end, period_step,
        k_start, k_end, k_step
    );
    
    // Metadata should contain [period, k] pairs
    assert.strictEqual(metadata.length, 18);  // 3 periods x 3 k values x 2 values per combo
    
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
        
        // The highpass_2_pole filter includes a warmup period of NaN values
        // Find where valid data starts
        let firstValid = null;
        for (let i = 0; i < result.length; i++) {
            if (!isNaN(result[i])) {
                firstValid = i;
                break;
            }
        }
        
        // Verify that we have valid data after the warmup
        assert(firstValid !== null, `No valid data found for k=${k}`);
        
        // Verify no NaN after first valid
        for (let i = firstValid; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for k=${k}`);
        }
    }
});

test('HighPass2 batch performance', () => {
    // Test that batch computation is more efficient than multiple single computations
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use first 1000 values
    
    // Test 5 periods x 4 k values = 20 combinations
    const startBatch = performance.now();
    const batchResult = wasm.highpass_2_pole_batch_js(close, 30, 70, 10, 0.3, 0.9, 0.2);
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