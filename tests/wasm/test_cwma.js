/**
 * WASM binding tests for CWMA indicator.
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
        wasm = await import(wasmPath);
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('CWMA partial params', () => {
    // Test with default parameters - mirrors check_cwma_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.cwma_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('CWMA accuracy', async () => {
    // Test CWMA matches expected values from Rust tests - mirrors check_cwma_accuracy
    const close = new Float64Array(testData.close);
    const expectedLast5 = [
        59224.641237300435,
        59213.64831277214,
        59171.21190130624,
        59167.01279027576,
        59039.413552249636
    ];
    
    const result = wasm.cwma_js(close, 14);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-9,
        "CWMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('cwma', result, 'close', {period: 14});
});

test('CWMA default candles', () => {
    // Test CWMA with default parameters - mirrors check_cwma_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.cwma_js(close, 14);
    assert.strictEqual(result.length, close.length);
});

test('CWMA zero period', () => {
    // Test CWMA fails with zero period - mirrors check_cwma_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cwma_js(inputData, 0);
    }, /Invalid period/);
});

test('CWMA period exceeds length', () => {
    // Test CWMA fails when period exceeds data length - mirrors check_cwma_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cwma_js(dataSmall, 10);
    }, /Invalid period/);
});

test('CWMA very small dataset', () => {
    // Test CWMA fails with insufficient data - mirrors check_cwma_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.cwma_js(singlePoint, 9);
    }, /Invalid period|Not enough valid data/);
});

test('CWMA empty input', () => {
    // Test CWMA fails with empty input - mirrors check_cwma_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.cwma_js(empty, 9);
    }, /Input data slice is empty/);
});

test('CWMA reinput', () => {
    // Test CWMA applied twice (re-input) - mirrors check_cwma_reinput
    const close = new Float64Array(testData.close);
    
    // First pass with period 80
    const firstResult = wasm.cwma_js(close, 80);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass with period 60 - apply CWMA to CWMA output
    const secondResult = wasm.cwma_js(firstResult, 60);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // After warmup period (240), no NaN values should exist
    if (secondResult.length > 240) {
        for (let i = 240; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('CWMA NaN handling', () => {
    // Test CWMA handles NaN values correctly - mirrors check_cwma_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.cwma_js(close, 9);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First period-1 values should be NaN
    assertAllNaN(result.slice(0, 8), "Expected NaN in warmup period");
});

test('CWMA all NaN input', () => {
    // Test CWMA with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.cwma_js(allNaN, 9);
    }, /All values are NaN/);
});

test('CWMA batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Single parameter set: period=14
    const batchResult = wasm.cwma_batch_js(
        close,
        14, 14, 0  // period range
    );
    
    // Should match single calculation
    const singleResult = wasm.cwma_js(close, 14);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('CWMA batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 10, 15, 20, 25
    const batchResult = wasm.cwma_batch_js(
        close,
        10, 25, 5  // period range
    );
    
    // Should have 4 rows * 100 cols = 400 values
    assert.strictEqual(batchResult.length, 4 * 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 15, 20, 25];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.cwma_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('CWMA batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.cwma_batch_metadata_js(
        10, 30, 5  // period: 10, 15, 20, 25, 30
    );
    
    // Should have 5 periods
    assert.strictEqual(metadata.length, 5);
    
    // Check values
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 15);
    assert.strictEqual(metadata[2], 20);
    assert.strictEqual(metadata[3], 25);
    assert.strictEqual(metadata[4], 30);
});

test('CWMA batch warmup validation', () => {
    // Test that batch correctly handles warmup periods
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.cwma_batch_js(
        close,
        5, 15, 5  // periods: 5, 10, 15
    );
    
    const metadata = wasm.cwma_batch_metadata_js(5, 15, 5);
    const numCombos = metadata.length;
    assert.strictEqual(numCombos, 3);
    
    // Verify warmup for each period
    for (let combo = 0; combo < numCombos; combo++) {
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

test('CWMA batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    
    // Single value sweep
    const singleBatch = wasm.cwma_batch_js(
        close,
        7, 7, 1
    );
    
    assert.strictEqual(singleBatch.length, 15);
    
    // Step = 0 with period that requires more data than available should throw
    assert.throws(() => {
        wasm.cwma_batch_js(
            close,
            10, 20, 0  // Period 10 needs 16 values (round_up8), but we only have 15
        );
    }, /Not enough valid data/);
    
    // Step larger than range
    const largeBatch = wasm.cwma_batch_js(
        close,
        5, 7, 10  // Step larger than range
    );
    
    // Should only have period=5
    assert.strictEqual(largeBatch.length, 15);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.cwma_batch_js(
            new Float64Array([]),
            9, 9, 0
        );
    }, /Input data slice is empty/);
});

test('CWMA batch performance test', () => {
    // Test that batch is more efficient than multiple single calls
    const close = new Float64Array(testData.close.slice(0, 200));
    
    // Batch calculation
    const startBatch = Date.now();
    const batchResult = wasm.cwma_batch_js(
        close,
        10, 50, 2  // 21 periods
    );
    const batchTime = Date.now() - startBatch;
    
    // Equivalent single calculations
    const startSingle = Date.now();
    const singleResults = [];
    for (let period = 10; period <= 50; period += 2) {
        singleResults.push(...wasm.cwma_js(close, period));
    }
    const singleTime = Date.now() - startSingle;
    
    // Batch should have same total length
    assert.strictEqual(batchResult.length, singleResults.length);
    
    // Log performance (batch should be faster)
    console.log(`  CWMA Batch time: ${batchTime}ms, Single calls time: ${singleTime}ms`);
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('CWMA WASM tests completed');
});