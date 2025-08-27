/**
 * WASM binding tests for ROCR indicator.
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
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('ROCR partial params', () => {
    // Test with default parameters - mirrors check_rocr_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.rocr_js(close, 10);
    assert.strictEqual(result.length, close.length);
});

test('ROCR accuracy', () => {
    // Test ROCR matches expected values from Rust tests - mirrors check_rocr_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.rocr;
    
    const result = wasm.rocr_js(close, expected.defaultParams.period);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "ROCR last 5 values mismatch"
    );
});

test('ROCR default period', () => {
    // Test ROCR with default period
    const close = new Float64Array(testData.close);
    
    const result = wasm.rocr_js(close, 10);
    assert.strictEqual(result.length, close.length);
});

test('ROCR zero period', () => {
    // Test ROCR fails with zero period - mirrors check_rocr_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.rocr_js(inputData, 0);
    }, /Invalid period/);
});

test('ROCR period exceeds length', () => {
    // Test ROCR fails when period exceeds data length - mirrors check_rocr_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.rocr_js(dataSmall, 10);
    }, /Invalid period/);
});

test('ROCR very small dataset', () => {
    // Test ROCR fails with insufficient data - mirrors check_rocr_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.rocr_js(singlePoint, 9);
    }, /Invalid period|Not enough/);
});

test('ROCR empty input', () => {
    // Test ROCR fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.rocr_js(empty, 9);
    }, /Empty data/);
});

test('ROCR all NaN input', () => {
    // Test ROCR with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.rocr_js(allNaN, 9);
    }, /All values are NaN/);
});

test('ROCR reinput', () => {
    // Test ROCR applied twice (re-input) - mirrors check_rocr_reinput
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.rocr_js(close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply ROCR to ROCR output
    const secondResult = wasm.rocr_js(firstResult, 14);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // After warmup period (28), values should be valid
    if (secondResult.length > 28) {
        for (let i = 28; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Found unexpected NaN at index ${i} after double warmup`);
        }
    }
});

test('ROCR NaN handling', () => {
    // Test ROCR handles NaN values correctly - mirrors check_rocr_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.rocr_js(close, 9);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First period values should be NaN
    assertAllNaN(result.slice(0, 9), "Expected NaN in warmup period");
});

test('ROCR batch single parameter', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Using the batch API for single parameter
    const batchResult = wasm.rocr_batch(close, {
        period_range: [10, 10, 0]
    });
    
    // Should match single calculation
    const singleResult = wasm.rocr_js(close, 10);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, close.length);
    
    // Extract the single row and compare
    const rowData = batchResult.values.slice(0, close.length);
    assertArrayClose(rowData, singleResult, 1e-10, "Batch vs single mismatch");
});

test('ROCR batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 5, 10, 15
    const batchResult = wasm.rocr_batch(close, {
        period_range: [5, 15, 5]
    });
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    // Verify each row matches individual calculation
    const periods = [5, 10, 15];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.rocr_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Batch row ${i} (period=${periods[i]}) mismatch`
        );
    }
});

test('ROCR edge cases', () => {
    // Test with data containing zeros (ROCR should handle division by zero)
    const dataWithZeros = new Float64Array([1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0, 10.0]);
    const result = wasm.rocr_js(dataWithZeros, 2);
    assert.strictEqual(result.length, dataWithZeros.length);
    
    // When past value is 0, ROCR should be 0 (not inf or nan)
    assert.strictEqual(result[2], 0.0, "Expected 0 when dividing by 0");
    assert.strictEqual(result[6], 0.0, "Expected 0 when dividing by 0");
    
    // Test with very small period
    const smallResult = wasm.rocr_js(new Float64Array(testData.close.slice(0, 20)), 1);
    assert.strictEqual(smallResult.length, 20);
    
    // Test with large period
    const largeData = new Float64Array(testData.close.slice(0, 200));
    const largeResult = wasm.rocr_js(largeData, 100);
    assert.strictEqual(largeResult.length, 200);
    
    // First 100 values should be NaN
    for (let i = 0; i < 100; i++) {
        assert(isNaN(largeResult[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    // After warmup, values should be valid
    for (let i = 100; i < 200; i++) {
        assert(!isNaN(largeResult[i]), `Expected valid value at index ${i} after warmup`);
    }
});

test('ROCR batch configuration', () => {
    // Test various batch configurations
    const close = new Float64Array(testData.close.slice(0, 50));
    
    // Test with step size
    const result1 = wasm.rocr_batch(close, {
        period_range: [5, 20, 5]  // 5, 10, 15, 20
    });
    assert.strictEqual(result1.rows, 4);
    assert.strictEqual(result1.combos.length, 4);
    assert.strictEqual(result1.combos[0].period, 5);
    assert.strictEqual(result1.combos[1].period, 10);
    assert.strictEqual(result1.combos[2].period, 15);
    assert.strictEqual(result1.combos[3].period, 20);
    
    // Test with single value (step = 0)
    const result2 = wasm.rocr_batch(close, {
        period_range: [7, 7, 0]
    });
    assert.strictEqual(result2.rows, 1);
    assert.strictEqual(result2.combos[0].period, 7);
    
    // Test with fine-grained step
    const result3 = wasm.rocr_batch(close, {
        period_range: [8, 10, 1]  // 8, 9, 10
    });
    assert.strictEqual(result3.rows, 3);
});

test.after(() => {
    console.log('ROCR WASM tests completed');
});