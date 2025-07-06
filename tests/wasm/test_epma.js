/**
 * WASM binding tests for EPMA indicator.
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

test('EPMA partial params', () => {
    // Test with default parameters - mirrors check_epma_partial_params
    const close = new Float64Array(testData.close);
    
    // Default parameters (period=11, offset=4)
    const result = wasm.epma_js(close, undefined, undefined);
    assert.strictEqual(result.length, close.length);
    
    // Partial custom parameters
    const resultCustomPeriod = wasm.epma_js(close, 15, undefined);
    assert.strictEqual(resultCustomPeriod.length, close.length);
    
    const resultCustomOffset = wasm.epma_js(close, undefined, 3);
    assert.strictEqual(resultCustomOffset.length, close.length);
});

test('EPMA accuracy', async () => {
    // Test accuracy matches expected values - mirrors check_epma_accuracy
    const close = new Float64Array(testData.close);
    const expectedLast5 = [59174.48, 59201.04, 59167.60, 59200.32, 59117.04];
    
    const result = wasm.epma_js(close, 11, 4);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        0.1,  // 1e-1 tolerance as in Rust test
        "EPMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('epma', result, 'close', { period: 11, offset: 4 });
});

test('EPMA empty input', () => {
    // Test error with empty input - mirrors check_epma_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.epma_js(empty, undefined, undefined);
    }, /Input data slice is empty/);
});

test('EPMA zero period', () => {
    // Test error with zero period - mirrors check_epma_zero_period
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    // With period=0 and default offset=4, it will fail with "Invalid offset"
    assert.throws(() => {
        wasm.epma_js(data, 0, undefined);
    }, /Invalid offset/);
    
    // With period=0 and offset=0, it will fail with "Invalid offset" because 0 >= 0
    assert.throws(() => {
        wasm.epma_js(data, 0, 0);
    }, /Invalid offset/);
});

test('EPMA period exceeds length', () => {
    // Test error when period exceeds data length - mirrors check_epma_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.epma_js(dataSmall, 10, undefined);
    }, /Invalid period/);
});

test('EPMA very small dataset', () => {
    // Test error with insufficient data - mirrors check_epma_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.epma_js(singlePoint, 9, undefined);
    }, /Invalid period|Not enough valid data/);
});

test('EPMA invalid offset', () => {
    // Test error with invalid offset - mirrors check_epma_invalid_offset
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0]);
    
    // Offset >= period
    assert.throws(() => {
        wasm.epma_js(data, 3, 3);
    }, /Invalid offset/);
    
    assert.throws(() => {
        wasm.epma_js(data, 3, 4);
    }, /Invalid offset/);
});

test('EPMA all NaN input', () => {
    // Test error with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.epma_js(allNaN, undefined, undefined);
    }, /All values are NaN/);
});

test('EPMA period too small', () => {
    // Test error with period < 2
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // With period=1 and default offset=4, it will fail with "Invalid offset"
    assert.throws(() => {
        wasm.epma_js(data, 1, undefined);
    }, /Invalid offset/);
    
    // With period=1 and offset=0, it will fail with "Invalid period"
    assert.throws(() => {
        wasm.epma_js(data, 1, 0);
    }, /Invalid period/);
});

test('EPMA not enough valid data', () => {
    // Test error when there's not enough valid data after NaN prefix
    const data = new Float64Array([NaN, NaN, 1.0, 2.0, 3.0]);
    
    // With period=3, offset=2, needs period+offset+1=6 valid values
    assert.throws(() => {
        wasm.epma_js(data, 3, 2);
    }, /Not enough valid data/);
});

test('EPMA reinput', () => {
    // Test applying indicator twice - mirrors check_epma_reinput
    const close = new Float64Array(testData.close);
    
    // First pass with period=9, offset will default to 4
    const firstResult = wasm.epma_js(close, 9, undefined);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass with period=3, offset=0 (explicitly set to avoid error)
    const secondResult = wasm.epma_js(firstResult, 3, 0);
    assert.strictEqual(secondResult.length, firstResult.length);
});

test('EPMA NaN handling', () => {
    // Test NaN handling - mirrors check_epma_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.epma_js(close, 11, 4);
    assert.strictEqual(result.length, close.length);
    
    // First period+offset+1 values should be NaN (11+4+1=16)
    assertAllNaN(result.slice(0, 16), "Expected NaN in warmup period");
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('EPMA batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Single parameter set
    const batchResult = wasm.epma_batch_js(
        close,
        11, 11, 0,    // period range
        4, 4, 0       // offset range
    );
    
    // Should match single calculation
    const singleResult = wasm.epma_js(close, 11, 4);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('EPMA batch multiple parameters', () => {
    // Test batch with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Multiple parameters: period 5,7,9 x offset 1,2,3
    const batchResult = wasm.epma_batch_js(
        close,
        5, 9, 2,      // period: 5, 7, 9
        1, 3, 1       // offset: 1, 2, 3
    );
    
    // Should have 3 x 3 = 9 rows * 100 cols = 900 values
    assert.strictEqual(batchResult.length, 9 * 100);
    
    // Verify first combination matches individual calculation
    const firstRow = batchResult.slice(0, 100);
    const singleResult = wasm.epma_js(close, 5, 1);
    assertArrayClose(
        firstRow, 
        singleResult, 
        1e-10, 
        "First batch row mismatch"
    );
});

test('EPMA batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.epma_batch_metadata_js(
        5, 9, 2,      // period: 5, 7, 9
        1, 3, 1       // offset: 1, 2, 3
    );
    
    // Should have 3 x 3 = 9 combinations, each with 2 values
    assert.strictEqual(metadata.length, 9 * 2);
    
    // Check first combination
    assert.strictEqual(metadata[0], 5);   // period
    assert.strictEqual(metadata[1], 1);   // offset
    
    // Check second combination
    assert.strictEqual(metadata[2], 5);   // period
    assert.strictEqual(metadata[3], 2);   // offset
    
    // Check fourth combination (second period)
    assert.strictEqual(metadata[6], 7);   // period
    assert.strictEqual(metadata[7], 1);   // offset
});

test('EPMA batch warmup validation', () => {
    // Test batch warmup period handling
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.epma_batch_js(
        close,
        5, 10, 5,     // period: 5, 10
        2, 4, 2       // offset: 2, 4
    );
    
    const metadata = wasm.epma_batch_metadata_js(5, 10, 5, 2, 4, 2);
    const numCombos = metadata.length / 2;
    assert.strictEqual(numCombos, 4);  // 2 periods x 2 offsets
    
    // Check warmup periods for each combination
    for (let combo = 0; combo < numCombos; combo++) {
        const period = metadata[combo * 2];
        const offset = metadata[combo * 2 + 1];
        const warmup = period + offset + 1;
        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);
        
        // First warmup values should be NaN
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(rowData[i]), 
                `Expected NaN at index ${i} for period=${period}, offset=${offset}`);
        }
        
        // After warmup should have values
        for (let i = warmup; i < 50; i++) {
            assert(!isNaN(rowData[i]), 
                `Unexpected NaN at index ${i} for period=${period}, offset=${offset}`);
        }
    }
});

test('EPMA batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array(20).fill(0).map((_, i) => i + 1);
    
    // Single value sweep
    const singleBatch = wasm.epma_batch_js(
        close,
        5, 5, 1,
        1, 1, 1
    );
    
    assert.strictEqual(singleBatch.length, 20);
    
    // Step = 0 should return single value when start=end
    const zeroStepBatch = wasm.epma_batch_js(
        close,
        7, 7, 0,
        2, 2, 0
    );
    
    assert.strictEqual(zeroStepBatch.length, 20); // Single combination
    
    // Empty data should throw
    assert.throws(() => {
        wasm.epma_batch_js(
            new Float64Array([]),
            11, 11, 0,
            4, 4, 0
        );
    }, /Input data slice is empty|All values are NaN/);
});

test('EPMA batch performance test', () => {
    // Test that batch is more efficient than multiple single calls
    const close = new Float64Array(testData.close.slice(0, 200));
    
    // Batch calculation
    const startBatch = Date.now();
    const batchResult = wasm.epma_batch_js(
        close,
        5, 15, 2,     // 6 period values
        1, 4, 1       // 4 offset values (max 4 to ensure offset < min period of 5)
    );
    const batchTime = Date.now() - startBatch;
    
    // Equivalent single calculations
    const startSingle = Date.now();
    const singleResults = [];
    for (let period = 5; period <= 15; period += 2) {
        for (let offset = 1; offset <= 4; offset += 1) {
            // All offsets are valid since max offset=4 < min period=5
            singleResults.push(...wasm.epma_js(close, period, offset));
        }
    }
    const singleTime = Date.now() - startSingle;
    
    // Batch should have same total length
    assert.strictEqual(batchResult.length, singleResults.length);
    
    // Log performance (batch should be faster)
    console.log(`  EPMA Batch time: ${batchTime}ms, Single calls time: ${singleTime}ms`);
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('EPMA WASM tests completed');
});