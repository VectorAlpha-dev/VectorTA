/**
 * WASM binding tests for FWMA indicator.
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

test('FWMA partial params', () => {
    // Test with default parameters - mirrors check_fwma_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.fwma_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('FWMA accuracy', async () => {
    // Test FWMA matches expected values from Rust tests - mirrors check_fwma_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.fwma_js(close, 5);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected last 5 values from Rust test
    const expectedLastFive = [
        59273.583333333336,
        59252.5,
        59167.083333333336,
        59151.0,
        58940.333333333336
    ];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        1e-8,
        "FWMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('fwma', result, 'close', { period: 5 });
});

test('FWMA default candles', async () => {
    // Test FWMA with default parameters - mirrors check_fwma_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.fwma_js(close, 5);
    assert.strictEqual(result.length, close.length);
    
    // Compare with Rust
    await compareWithRust('fwma', result, 'close', { period: 5 });
});

test('FWMA zero period', () => {
    // Test FWMA fails with zero period - mirrors check_fwma_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.fwma_js(inputData, 0);
    });
});

test('FWMA period exceeds length', () => {
    // Test FWMA fails when period exceeds data length - mirrors check_fwma_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.fwma_js(dataSmall, 5);
    });
});

test('FWMA very small dataset', () => {
    // Test FWMA with very small dataset - mirrors check_fwma_very_small_dataset
    const dataSingle = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.fwma_js(dataSingle, 5);
    });
});

test('FWMA reinput', () => {
    // Test FWMA with re-input of FWMA result - mirrors check_fwma_reinput
    const close = new Float64Array(testData.close);
    
    // First FWMA pass with period=5
    const firstResult = wasm.fwma_js(close, 5);
    
    // Second FWMA pass with period=3 using first result as input
    const secondResult = wasm.fwma_js(firstResult, 3);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Verify no NaN values after warmup period in second result
    for (let i = 240; i < secondResult.length; i++) {
        assert(!isNaN(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('FWMA NaN handling', () => {
    // Test FWMA handling of NaN values - mirrors check_fwma_nan_handling
    const data = new Float64Array([NaN, NaN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    const period = 3;
    
    const result = wasm.fwma_js(data, period);
    
    assert.strictEqual(result.length, data.length);
    
    // First 2 (NaN input) + period - 1 values should be NaN
    for (let i = 0; i < 2 + period - 1; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
    
    // Remaining should not be NaN
    for (let i = 2 + period - 1; i < result.length; i++) {
        assert(!isNaN(result[i]), `Unexpected NaN at index ${i}`);
    }
});

test('FWMA batch', () => {
    // Test FWMA batch computation
    const close = new Float64Array(testData.close);
    const period_start = 3;
    const period_end = 9;
    const period_step = 2; // periods: 3, 5, 7, 9
    
    const batch_result = wasm.fwma_batch_js(close, period_start, period_end, period_step);
    const metadata = wasm.fwma_batch_metadata_js(period_start, period_end, period_step);
    
    const expected_periods = [3, 5, 7, 9];
    assert.deepStrictEqual(Array.from(metadata), expected_periods, 'Metadata periods mismatch');
    
    // Batch result should contain all individual results flattened
    assert.strictEqual(batch_result.length, expected_periods.length * close.length, 'Batch result length mismatch');
    
    // Verify each row matches individual calculation
    for (let i = 0; i < expected_periods.length; i++) {
        const period = expected_periods[i];
        const individual_result = wasm.fwma_js(close, period);
        
        // Extract row from batch result
        const row_start = i * close.length;
        const row = batch_result.slice(row_start, row_start + close.length);
        
        assertArrayClose(row, individual_result, 1e-9, `Period ${period}`);
    }
});

test('FWMA Fibonacci weights calculation', () => {
    // For period=5, Fibonacci sequence is [1, 1, 2, 3, 5]
    // Normalized weights are [1/12, 1/12, 2/12, 3/12, 5/12]
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    const period = 5;
    
    const result = wasm.fwma_js(data, period);
    
    // Expected: (1*1 + 2*1 + 3*2 + 4*3 + 5*5) / 12 = 46/12 = 3.833...
    const expected = (1*1 + 2*1 + 3*2 + 4*3 + 5*5) / 12;
    assertClose(result[4], expected, 1e-9, 'FWMA calculation');
});

test('FWMA batch performance', () => {
    // Test that batch computation is more efficient than multiple single computations
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use first 1000 values
    const periods = Array.from({length: 20}, (_, i) => i + 3); // periods 3-22
    
    const startBatch = performance.now();
    const batchResult = wasm.fwma_batch_js(close, 3, 22, 1);
    const batchTime = performance.now() - startBatch;
    
    const startSingle = performance.now();
    const singleResults = [];
    for (const period of periods) {
        singleResults.push(...wasm.fwma_js(close, period));
    }
    const singleTime = performance.now() - startSingle;
    
    // Batch should be faster than multiple single calls
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    // Verify results match
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});