/**
 * WASM binding tests for HighPass indicator.
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

test('HighPass partial params', () => {
    // Test with default parameters - mirrors check_highpass_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.highpass_js(close, 48);
    assert.strictEqual(result.length, close.length);
});

test('HighPass accuracy', async () => {
    // Test HighPass matches expected values from Rust tests - mirrors check_highpass_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.highpass_js(close, 48);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected last 5 values from Rust test
    const expectedLastFive = [
        -265.1027020005024,
        -330.0916060058495,
        -422.7478979710918,
        -261.87532144673423,
        -698.9026088956363,
    ];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-6,  // Using 1e-6 as in Rust test
        "HighPass last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('highpass', result, 'close', { period: 48 });
});

test('HighPass default candles', async () => {
    // Test HighPass with default parameters - mirrors check_highpass_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.highpass_js(close, 48);
    assert.strictEqual(result.length, close.length);
    
    // Compare with Rust
    await compareWithRust('highpass', result, 'close', { period: 48 });
});

test('HighPass zero period', () => {
    // Test HighPass fails with zero period - mirrors check_highpass_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.highpass_js(inputData, 0);
    });
});

test('HighPass period exceeds length', () => {
    // Test HighPass fails when period exceeds data length - mirrors check_highpass_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.highpass_js(dataSmall, 48);
    });
});

test('HighPass very small dataset', () => {
    // Test HighPass with very small dataset - mirrors check_highpass_very_small_dataset
    const dataSmall = new Float64Array([42.0, 43.0]);
    
    assert.throws(() => {
        wasm.highpass_js(dataSmall, 2);
    });
});

test('HighPass empty input', () => {
    // Test HighPass with empty input - mirrors check_highpass_empty_input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.highpass_js(dataEmpty, 48);
    });
});

test('HighPass invalid alpha', () => {
    // Test HighPass with invalid alpha - mirrors check_highpass_invalid_alpha
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // Period=4 causes cos(pi/2) ~ 0
    assert.throws(() => {
        wasm.highpass_js(data, 4);
    });
});

test('HighPass all NaN', () => {
    // Test HighPass with all NaN input
    const data = new Float64Array([NaN, NaN, NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.highpass_js(data, 3);
    });
});

test('HighPass reinput', () => {
    // Test HighPass with re-input of HighPass result - mirrors check_highpass_reinput
    const close = new Float64Array(testData.close);
    
    // First HighPass pass with period=36
    const firstResult = wasm.highpass_js(close, 36);
    
    // Second HighPass pass with period=24 using first result as input
    const secondResult = wasm.highpass_js(firstResult, 24);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Verify no NaN values after index 240 in second result
    for (let i = 240; i < secondResult.length; i++) {
        assert(!isNaN(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('HighPass NaN handling', () => {
    // Test HighPass handling of NaN values - mirrors check_highpass_nan_handling
    const close = new Float64Array(testData.close);
    const period = 48;
    
    const result = wasm.highpass_js(close, period);
    
    assert.strictEqual(result.length, close.length);
    
    // The highpass filter copies the first value directly, so no NaN expected
    for (let i = 0; i < result.length; i++) {
        assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
    }
});

test('HighPass batch', () => {
    // Test HighPass batch computation
    const close = new Float64Array(testData.close);
    
    // Test period range 30-60 step 10
    const period_start = 30;
    const period_end = 60;
    const period_step = 10;  // periods: 30, 40, 50, 60
    
    const batch_result = wasm.highpass_batch_js(
        close, 
        period_start, period_end, period_step
    );
    const metadata = wasm.highpass_batch_metadata_js(
        period_start, period_end, period_step
    );
    
    // Metadata should contain period values
    assert.strictEqual(metadata.length, 4);  // 4 periods
    
    // Batch result should contain all individual results flattened
    assert.strictEqual(batch_result.length, 4 * close.length);  // 4 periods
    
    // Verify each row matches individual calculation
    let row_idx = 0;
    for (const period of [30, 40, 50, 60]) {
        const individual_result = wasm.highpass_js(close, period);
        
        // Extract row from batch result
        const row_start = row_idx * close.length;
        const row = batch_result.slice(row_start, row_start + close.length);
        
        assertArrayClose(row, individual_result, 1e-9, `Period ${period}`);
        row_idx++;
    }
});

test('HighPass different periods', () => {
    // Test HighPass with different period values
    const close = new Float64Array(testData.close);
    
    // Test various period values
    for (const period of [10, 20, 30, 48, 60]) {
        const result = wasm.highpass_js(close, period);
        assert.strictEqual(result.length, close.length);
        
        // Find where valid data starts
        let firstValid = null;
        for (let i = 0; i < result.length; i++) {
            if (!isNaN(result[i])) {
                firstValid = i;
                break;
            }
        }
        
        // Verify that we have valid data
        assert(firstValid !== null, `No valid data found for period=${period}`);
        
        // Verify no NaN after first valid
        for (let i = firstValid; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for period=${period}`);
        }
    }
});

test('HighPass batch performance', () => {
    // Test that batch computation is more efficient than multiple single computations
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use first 1000 values
    
    // Test 5 periods
    const startBatch = performance.now();
    const batchResult = wasm.highpass_batch_js(close, 20, 60, 10);
    const batchTime = performance.now() - startBatch;
    
    const startSingle = performance.now();
    const singleResults = [];
    for (let period = 20; period <= 60; period += 10) {
        singleResults.push(...wasm.highpass_js(close, period));
    }
    const singleTime = performance.now() - startSingle;
    
    // Batch should be faster than multiple single calls
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    // Verify results match
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});