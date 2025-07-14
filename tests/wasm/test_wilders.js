/**
 * WASM binding tests for WILDERS indicator.
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
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('Wilders partial params', () => {
    // Test with default parameters - mirrors check_wilders_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.wilders_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('Wilders accuracy', async () => {
    // Test WILDERS matches expected values from Rust tests - mirrors check_wilders_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.wilders;
    
    const result = wasm.wilders_js(
        close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "Wilders last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('wilders', result, 'close', expected.defaultParams);
});

test('Wilders default candles', () => {
    // Test Wilders with default parameters - mirrors check_wilders_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.wilders_js(close, 5);
    assert.strictEqual(result.length, close.length);
});

test('Wilders zero period', () => {
    // Test Wilders fails with zero period - mirrors check_wilders_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.wilders_js(inputData, 0);
    }, /Invalid period/);
});

test('Wilders period exceeds length', () => {
    // Test Wilders fails when period exceeds data length - mirrors check_wilders_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.wilders_js(dataSmall, 10);
    }, /Invalid period/);
});

test('Wilders very small dataset', () => {
    // Test Wilders with very small dataset - mirrors check_wilders_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    // Should work with period=1
    const result = wasm.wilders_js(singlePoint, 1);
    assert.strictEqual(result.length, 1);
});

test('Wilders reinput', () => {
    // Test Wilders applied twice (re-input) - mirrors check_wilders_reinput
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.wilders_js(close, 5);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply Wilders to Wilders output with different period
    const secondResult = wasm.wilders_js(firstResult, 10);
    assert.strictEqual(secondResult.length, firstResult.length);
});

test('Wilders NaN handling', () => {
    // Test Wilders handles NaN values correctly - mirrors check_wilders_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.wilders_js(close, 5);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First period-1 values should be NaN
    assertAllNaN(result.slice(0, 4), "Expected NaN in warmup period");
});

test('Wilders all NaN input', () => {
    // Test Wilders with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.wilders_js(allNaN, 5);
    }, /All values are NaN/);
});

test('Wilders batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Single parameter set: period=5
    const batchResult = wasm.wilders_batch_js(
        close,
        5, 5, 0      // period range
    );
    
    // Should match single calculation
    const singleResult = wasm.wilders_js(close, 5);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('Wilders batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 5, 6, 7, 8, 9, 10
    const batchResult = wasm.wilders_batch_js(
        close,
        5, 10, 1      // period range
    );
    
    // Should have 6 rows * 100 cols = 600 values
    assert.strictEqual(batchResult.length, 6 * 100);
    
    // Verify each row matches individual calculation
    const periods = [5, 6, 7, 8, 9, 10];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.wilders_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('Wilders batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.wilders_batch_metadata_js(
        5, 10, 1      // period: 5, 6, 7, 8, 9, 10
    );
    
    // Should have 6 periods
    assert.strictEqual(metadata.length, 6);
    
    // Check values
    for (let i = 0; i < 6; i++) {
        assert.strictEqual(metadata[i], 5 + i);
    }
});

test('Wilders batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.wilders_batch_js(
        close,
        5, 7, 2      // 2 periods: 5, 7
    );
    
    const metadata = wasm.wilders_batch_metadata_js(
        5, 7, 2
    );
    
    // Should have 2 combinations
    assert.strictEqual(metadata.length, 2);
    assert.strictEqual(batchResult.length, 2 * 50);
    
    // Verify structure
    for (let combo = 0; combo < 2; combo++) {
        const period = metadata[combo];
        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);
        
        // First period-1 values should be NaN
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        // After warmup should have values
        for (let i = period - 1; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('Wilders batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep
    const singleBatch = wasm.wilders_batch_js(
        close,
        5, 5, 1
    );
    
    assert.strictEqual(singleBatch.length, 10);
    
    // Step larger than range
    const largeBatch = wasm.wilders_batch_js(
        close,
        5, 7, 10 // Step larger than range
    );
    
    // Should only have period=5
    assert.strictEqual(largeBatch.length, 10);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.wilders_batch_js(
            new Float64Array([]),
            5, 5, 0
        );
    }, /All values are NaN/);
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('Wilders WASM tests completed');
});