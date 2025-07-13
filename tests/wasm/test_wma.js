/**
 * WASM binding tests for WMA indicator.
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
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('WMA partial params', () => {
    // Test with default parameters - mirrors check_wma_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.wma_js(close, 30);
    assert.strictEqual(result.length, close.length);
});

test('WMA accuracy', async () => {
    // Test WMA matches expected values from Rust tests - mirrors check_wma_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.wma;
    
    const result = wasm.wma_js(
        close,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-6,  // Using 1e-6 as per Rust test
        "WMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('wma', result, 'close', expected.defaultParams);
});

test('WMA zero period', () => {
    // Test WMA fails with zero period - mirrors check_wma_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.wma_js(inputData, 0);
    }, /Invalid period/);
});

test('WMA period exceeds length', () => {
    // Test WMA fails when period exceeds data length - mirrors check_wma_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.wma_js(dataSmall, 10);
    }, /Invalid period/);
});

test('WMA very small dataset', () => {
    // Test WMA fails with insufficient data - mirrors check_wma_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.wma_js(singlePoint, 9);
    }, /Invalid period|Not enough valid data/);
});

test('WMA reinput', () => {
    // Test WMA applied twice (re-input) - mirrors check_wma_reinput
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.wma_js(close, 14);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply WMA to WMA output
    const secondResult = wasm.wma_js(firstResult, 5);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Check that values after warmup are not NaN
    if (secondResult.length > 50) {
        for (let i = 50; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Unexpected NaN at index ${i}`);
        }
    }
});

test('WMA NaN handling', () => {
    // Test WMA handles NaN values correctly - mirrors check_wma_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.wma_js(close, 14);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (50), no NaN values should exist
    if (result.length > 50) {
        for (let i = 50; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First period values should be NaN
    assertAllNaN(result.slice(0, 14), "Expected NaN in warmup period");
});

test('WMA all NaN input', () => {
    // Test WMA with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.wma_js(allNaN, 30);
    }, /All values are NaN/);
});

test('WMA batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Single parameter set: period=30
    const batchResult = wasm.wma_batch_js(
        close,
        30, 30, 0      // period range
    );
    
    // Should match single calculation
    const singleResult = wasm.wma_js(close, 30);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('WMA batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 10, 20, 30
    const batchResult = wasm.wma_batch_js(
        close,
        10, 30, 10      // period range
    );
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.length, 3 * 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 20, 30];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.wma_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('WMA batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.wma_batch_metadata_js(
        10, 30, 10      // period: 10, 20, 30
    );
    
    // Should have 3 values (one per period)
    assert.strictEqual(metadata.length, 3);
    
    // Check values
    assert.strictEqual(metadata[0], 10);   // period
    assert.strictEqual(metadata[1], 20);   // period
    assert.strictEqual(metadata[2], 30);   // period
});

test('WMA batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.wma_batch_js(
        close,
        10, 14, 2      // 3 periods
    );
    
    const metadata = wasm.wma_batch_metadata_js(
        10, 14, 2
    );
    
    // Should have 3 combinations
    const numCombos = metadata.length;
    assert.strictEqual(numCombos, 3);
    assert.strictEqual(batchResult.length, 3 * 50);
    
    // Verify structure
    for (let combo = 0; combo < numCombos; combo++) {
        const period = metadata[combo];
        
        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);
        
        // First period values should be NaN
        for (let i = 0; i < period; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        // After warmup should have values
        for (let i = period; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('WMA batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep
    const singleBatch = wasm.wma_batch_js(
        close,
        5, 5, 1
    );
    
    assert.strictEqual(singleBatch.length, 10);
    
    // Step larger than range
    const largeBatch = wasm.wma_batch_js(
        close,
        5, 7, 10 // Step larger than range
    );
    
    // Should only have period=5
    assert.strictEqual(largeBatch.length, 10);
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('WMA WASM tests completed');
});
