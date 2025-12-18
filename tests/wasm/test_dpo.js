/**
 * WASM binding tests for DPO (Detrended Price Oscillator) indicator.
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
    assertNoNaN
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

test('DPO partial params', () => {
    // Test with default period (5) - mirrors check_dpo_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.dpo_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('DPO accuracy', async () => {
    // Test DPO matches expected values from Rust tests - mirrors check_dpo_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.dpo_js(close, 5);
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected from Rust tests
    const expectedLastFive = [
        65.3999999999287,
        131.3999999999287,
        32.599999999925785,
        98.3999999999287,
        117.99999999992724,
    ];
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLastFive,
        0.1, // Using same tolerance as Rust tests
        "DPO last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('dpo', result, 'close', { period: 5 });
});

test('DPO default candles', () => {
    // Test DPO with default parameters - mirrors check_dpo_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.dpo_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('DPO zero period', () => {
    // Test DPO fails with zero period - mirrors check_dpo_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.dpo_js(inputData, 0);
    }, /Invalid period/);
});

test('DPO period exceeds length', () => {
    // Test DPO fails when period exceeds data length - mirrors check_dpo_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.dpo_js(dataSmall, 10);
    }, /Invalid period/);
});

test('DPO very small dataset', () => {
    // Test DPO fails with insufficient data - mirrors check_dpo_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.dpo_js(singlePoint, 5);
    }, /Invalid period|Not enough valid data/);
});

test('DPO nan handling', () => {
    // Test DPO handles NaN values correctly - mirrors check_dpo_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.dpo_js(close, 5);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period, no NaN values should exist
    // Based on Rust test which checks after index 20
    if (result.length > 20) {
        for (let i = 20; i < result.length; i++) {
            assert.ok(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('DPO fast API (in-place)', () => {
    // Test fast API with aliasing (in-place operation)
    const close = new Float64Array(testData.close);
    const period = 5;
    
    // Allocate output buffer
    const outputPtr = wasm.dpo_alloc(close.length);
    
    try {
        // Copy input to output buffer for in-place operation
        const outputInitial = new Float64Array(wasm.__wasm.memory.buffer, outputPtr, close.length);
        outputInitial.set(close);
        
        // Perform in-place DPO
        wasm.dpo_into(outputPtr, outputPtr, close.length, period);
        
        // Compare with regular API
        const expected = wasm.dpo_js(close, period);
        
        // Get fresh view of output after computation
        const outputResult = new Float64Array(wasm.__wasm.memory.buffer, outputPtr, close.length);
        
        assertArrayClose(outputResult, expected, 1e-10, "Fast API in-place mismatch");
    } finally {
        // Clean up
        wasm.dpo_free(outputPtr, close.length);
    }
});

test('DPO fast API (separate buffers)', () => {
    // Test fast API with separate input/output buffers
    const close = new Float64Array(testData.close);
    const period = 5;
    
    // Allocate buffers
    const inputPtr = wasm.dpo_alloc(close.length);
    const outputPtr = wasm.dpo_alloc(close.length);
    
    try {
        // Copy input data
        const input = new Float64Array(wasm.__wasm.memory.buffer, inputPtr, close.length);
        input.set(close);
        
        // Perform DPO
        wasm.dpo_into(inputPtr, outputPtr, close.length, period);
        
        // Compare with regular API
        const expected = wasm.dpo_js(close, period);
        
        // Get output (create view after computation)
        const outputResult = new Float64Array(wasm.__wasm.memory.buffer, outputPtr, close.length);
        
        assertArrayClose(outputResult, expected, 1e-10, "Fast API separate buffers mismatch");
    } finally {
        // Clean up
        wasm.dpo_free(inputPtr, close.length);
        wasm.dpo_free(outputPtr, close.length);
    }
});

test('DPO batch single parameter', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    const config = {
        period_range: [5, 5, 0] // Single period
    };
    
    const batchResult = wasm.dpo_batch(close, config);
    
    // Should match single calculation
    const singleResult = wasm.dpo_js(close, 5);
    
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, close.length);
    assert.strictEqual(batchResult.values.length, close.length);
    
    assertArrayClose(
        batchResult.values,
        singleResult,
        1e-8, // Batch uses prefix sums; single uses a sliding sum (tiny FP-order differences).
        "Batch vs single mismatch"
    );
});

test('DPO batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 200)); // Use smaller dataset for speed
    
    const config = {
        period_range: [5, 15, 5] // Periods: 5, 10, 15
    };
    
    const batchResult = wasm.dpo_batch(close, config);
    
    // Should have 3 rows * 200 cols
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 200);
    assert.strictEqual(batchResult.values.length, 3 * 200);
    
    // Verify each row matches individual calculation
    const periods = [5, 10, 15];
    for (let i = 0; i < periods.length; i++) {
        const period = periods[i];
        const rowStart = i * 200;
        const rowEnd = (i + 1) * 200;
        const row = batchResult.values.slice(rowStart, rowEnd);
        
        const expected = wasm.dpo_js(close, period);
        assertArrayClose(
            row,
            expected,
            1e-10,
            `Batch row ${i} (period=${period}) mismatch`
        );
    }
});

test('DPO all NaN input', () => {
    // Test DPO with all NaN values
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.dpo_js(allNaN, 5);
    }, /All values are NaN/);
});

test('DPO batch into (fast batch API)', () => {
    // Test fast batch API
    const close = new Float64Array(testData.close.slice(0, 100));
    const periodStart = 5;
    const periodEnd = 20;
    const periodStep = 5; // 5, 10, 15, 20
    
    const expectedRows = 4;
    const cols = close.length;
    
    // Allocate buffers
    const inputPtr = wasm.dpo_alloc(close.length);
    const outputPtr = wasm.dpo_alloc(expectedRows * cols);
    
    try {
        // Copy input data
        const input = new Float64Array(wasm.__wasm.memory.buffer, inputPtr, close.length);
        input.set(close);
        
        // Perform batch DPO
        const rows = wasm.dpo_batch_into(
            inputPtr, 
            outputPtr, 
            close.length,
            periodStart,
            periodEnd,
            periodStep
        );
        
        assert.strictEqual(rows, expectedRows);
        
        // Get output
        const output = new Float64Array(wasm.__wasm.memory.buffer, outputPtr, rows * cols);
        
        // Verify against regular batch API
        const config = {
            period_range: [periodStart, periodEnd, periodStep]
        };
        const expected = wasm.dpo_batch(close, config);
        
        assertArrayClose(
            output,
            expected.values,
            1e-10,
            "Batch into API mismatch"
        );
    } finally {
        // Clean up
        wasm.dpo_free(inputPtr, close.length);
        wasm.dpo_free(outputPtr, expectedRows * cols);
    }
});
