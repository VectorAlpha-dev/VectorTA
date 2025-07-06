/**
 * WASM binding tests for FRAMA indicator.
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

test('FRAMA partial params', () => {
    // Test with default parameters - mirrors check_frama_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Default parameters (window=10, sc=300, fc=1)
    const result = wasm.frama_js(high, low, close, undefined, undefined, undefined);
    assert.strictEqual(result.length, close.length);
    
    // Partial custom parameters
    const resultCustomWindow = wasm.frama_js(high, low, close, 14, undefined, undefined);
    assert.strictEqual(resultCustomWindow.length, close.length);
    
    const resultCustomSc = wasm.frama_js(high, low, close, undefined, 200, undefined);
    assert.strictEqual(resultCustomSc.length, close.length);
    
    const resultCustomFc = wasm.frama_js(high, low, close, undefined, undefined, 2);
    assert.strictEqual(resultCustomFc.length, close.length);
});

test('FRAMA accuracy', async () => {
    // Test accuracy matches expected values - mirrors check_frama_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const expectedLast5 = [
        59337.23056930512,
        59321.607512374605,
        59286.677929994796,
        59268.00202402624,
        59160.03888720062
    ];
    
    const result = wasm.frama_js(high, low, close, 10, 300, 1);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        0.1,  // 1e-1 tolerance as in Rust test
        "FRAMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('frama', result, 'high,low,close', {window: 10, sc: 300, fc: 1});
});

test('FRAMA empty input', () => {
    // Test error with empty input - mirrors check_frama_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.frama_js(empty, empty, empty, undefined, undefined, undefined);
    }, /Input data slice is empty/);
});

test('FRAMA zero window', () => {
    // Test error with zero window - mirrors check_frama_zero_window
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    const close = new Float64Array([7.0, 17.0, 27.0]);
    
    assert.throws(() => {
        wasm.frama_js(high, low, close, 0, undefined, undefined);
    }, /Invalid window/);
});

test('FRAMA window exceeds length', () => {
    // Test error when window exceeds data length - mirrors check_frama_window_exceeds_length
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    const close = new Float64Array([7.0, 17.0, 27.0]);
    
    assert.throws(() => {
        wasm.frama_js(high, low, close, 10, undefined, undefined);
    }, /Invalid window/);
});

test('FRAMA very small dataset', () => {
    // Test error with insufficient data - mirrors check_frama_very_small_dataset
    const high = new Float64Array([42.0]);
    const low = new Float64Array([40.0]);
    const close = new Float64Array([41.0]);
    
    assert.throws(() => {
        wasm.frama_js(high, low, close, 10, undefined, undefined);
    }, /Invalid window|Not enough valid data/);
});

test('FRAMA mismatched lengths', () => {
    // Test error with mismatched input lengths - mirrors check_frama_mismatched_len
    const high = new Float64Array([1.0, 2.0, 3.0]);
    const low = new Float64Array([0.5, 1.5]);
    const close = new Float64Array([1.0]);
    
    assert.throws(() => {
        wasm.frama_js(high, low, close, undefined, undefined, undefined);
    }, /Mismatched slice lengths/);
});

test('FRAMA all NaN input', () => {
    // Test error with all NaN values - mirrors check_frama_all_nan
    const allNaN = new Float64Array(10);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.frama_js(allNaN, allNaN, allNaN, undefined, undefined, undefined);
    }, /All values are NaN/);
});

test('FRAMA not enough valid data', () => {
    // Test error when there's not enough valid data after NaN prefix
    const high = new Float64Array([NaN, NaN, 10.0, 20.0, 30.0]);
    const low = new Float64Array([NaN, NaN, 5.0, 15.0, 25.0]);
    const close = new Float64Array([NaN, NaN, 7.0, 17.0, 27.0]);
    
    // With window=10 and data length=5, it will fail with "Invalid window"
    assert.throws(() => {
        wasm.frama_js(high, low, close, 10, undefined, undefined);
    }, /Invalid window/);
    
    // Test case where window is valid but not enough data after NaN
    const high2 = new Float64Array([NaN, NaN, NaN, NaN, NaN, 10.0, 20.0, 30.0, 40.0, 50.0]);
    const low2 = new Float64Array([NaN, NaN, NaN, NaN, NaN, 5.0, 15.0, 25.0, 35.0, 45.0]);
    const close2 = new Float64Array([NaN, NaN, NaN, NaN, NaN, 7.0, 17.0, 27.0, 37.0, 47.0]);
    
    // With window=10 and only 5 valid values after NaN
    assert.throws(() => {
        wasm.frama_js(high2, low2, close2, 10, undefined, undefined);
    }, /Not enough valid data/);
});

test('FRAMA reinput', () => {
    // Test applying indicator twice - mirrors check_frama_reinput
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.frama_js(high, low, close, 10, undefined, undefined);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply to output (using output for all three inputs)
    const secondResult = wasm.frama_js(firstResult, firstResult, firstResult, 5, undefined, undefined);
    assert.strictEqual(secondResult.length, firstResult.length);
});

test('FRAMA NaN handling', () => {
    // Test NaN handling - mirrors check_frama_nan_handling
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.frama_js(high, low, close, 10, 300, 1);
    assert.strictEqual(result.length, close.length);
    
    // First window-1 values should be NaN
    assertAllNaN(result.slice(0, 9), "Expected NaN in warmup period");
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('FRAMA batch single parameter set', () => {
    // Test batch with single parameter combination
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Single parameter set
    const batchResult = wasm.frama_batch_js(
        high, low, close,
        10, 10, 0,    // window range
        300, 300, 0,  // sc range
        1, 1, 0       // fc range
    );
    
    // Should match single calculation
    const singleResult = wasm.frama_js(high, low, close, 10, 300, 1);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('FRAMA batch multiple parameters', () => {
    // Test batch with multiple parameter combinations
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Multiple parameters: window 8,10,12 x sc 200,300 x fc 1,2
    const batchResult = wasm.frama_batch_js(
        high, low, close,
        8, 12, 2,      // window: 8, 10, 12
        200, 300, 100, // sc: 200, 300
        1, 2, 1        // fc: 1, 2
    );
    
    // Should have 3 x 2 x 2 = 12 rows * 100 cols = 1200 values
    assert.strictEqual(batchResult.length, 12 * 100);
    
    // Verify first combination matches individual calculation
    const firstRow = batchResult.slice(0, 100);
    const singleResult = wasm.frama_js(high, low, close, 8, 200, 1);
    assertArrayClose(
        firstRow, 
        singleResult, 
        1e-10, 
        "First batch row mismatch"
    );
});

test('FRAMA batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.frama_batch_metadata_js(
        8, 12, 2,      // window: 8, 10, 12
        200, 300, 100, // sc: 200, 300
        1, 2, 1        // fc: 1, 2
    );
    
    // Should have 3 x 2 x 2 = 12 combinations, each with 3 values
    assert.strictEqual(metadata.length, 12 * 3);
    
    // Check first combination
    assert.strictEqual(metadata[0], 8);    // window
    assert.strictEqual(metadata[1], 200);  // sc
    assert.strictEqual(metadata[2], 1);    // fc
    
    // Check second combination
    assert.strictEqual(metadata[3], 8);    // window
    assert.strictEqual(metadata[4], 200);  // sc
    assert.strictEqual(metadata[5], 2);    // fc
    
    // Check third combination
    assert.strictEqual(metadata[6], 8);    // window
    assert.strictEqual(metadata[7], 300);  // sc
    assert.strictEqual(metadata[8], 1);    // fc
});

test('FRAMA batch warmup validation', () => {
    // Test batch warmup period handling
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.frama_batch_js(
        high, low, close,
        6, 10, 4,     // window: 6, 10
        300, 300, 0,  // sc: 300
        1, 1, 0       // fc: 1
    );
    
    const metadata = wasm.frama_batch_metadata_js(6, 10, 4, 300, 300, 0, 1, 1, 0);
    const numCombos = metadata.length / 3;
    assert.strictEqual(numCombos, 2);  // 2 windows x 1 sc x 1 fc
    
    // Check warmup periods for each combination
    for (let combo = 0; combo < numCombos; combo++) {
        const window = metadata[combo * 3];
        const warmup = window - 1;
        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);
        
        // First warmup values should be NaN
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(rowData[i]), 
                `Expected NaN at index ${i} for window=${window}`);
        }
        
        // After warmup should have values
        for (let i = warmup; i < 50; i++) {
            assert(!isNaN(rowData[i]), 
                `Unexpected NaN at index ${i} for window=${window}`);
        }
    }
});

test('FRAMA batch edge cases', () => {
    // Test edge cases for batch processing
    const size = 20;
    const high = new Float64Array(size).fill(0).map((_, i) => i + 2);
    const low = new Float64Array(size).fill(0).map((_, i) => i);
    const close = new Float64Array(size).fill(0).map((_, i) => i + 1);
    
    // Single value sweep
    const singleBatch = wasm.frama_batch_js(
        high, low, close,
        10, 10, 1,
        300, 300, 1,
        1, 1, 1
    );
    
    assert.strictEqual(singleBatch.length, size);
    
    // Step = 0 should return single value when start=end
    const zeroStepBatch = wasm.frama_batch_js(
        high, low, close,
        10, 10, 0,
        300, 300, 0,
        1, 1, 0
    );
    
    assert.strictEqual(zeroStepBatch.length, size); // Single combination
    
    // Empty data should throw
    assert.throws(() => {
        wasm.frama_batch_js(
            new Float64Array([]),
            new Float64Array([]),
            new Float64Array([]),
            10, 10, 0,
            300, 300, 0,
            1, 1, 0
        );
    }, /Input data slice is empty|All values are NaN/);
});

test('FRAMA batch performance test', () => {
    // Test that batch is more efficient than multiple single calls
    const high = new Float64Array(testData.high.slice(0, 200));
    const low = new Float64Array(testData.low.slice(0, 200));
    const close = new Float64Array(testData.close.slice(0, 200));
    
    // Batch calculation
    const startBatch = Date.now();
    const batchResult = wasm.frama_batch_js(
        high, low, close,
        8, 12, 2,     // 3 window values
        200, 400, 100,// 3 sc values
        1, 2, 1       // 2 fc values = 18 total combinations
    );
    const batchTime = Date.now() - startBatch;
    
    // Equivalent single calculations
    const startSingle = Date.now();
    const singleResults = [];
    for (let window = 8; window <= 12; window += 2) {
        for (let sc = 200; sc <= 400; sc += 100) {
            for (let fc = 1; fc <= 2; fc += 1) {
                singleResults.push(...wasm.frama_js(high, low, close, window, sc, fc));
            }
        }
    }
    const singleTime = Date.now() - startSingle;
    
    // Batch should have same total length
    assert.strictEqual(batchResult.length, singleResults.length);
    
    // Log performance (batch should be faster)
    console.log(`  FRAMA Batch time: ${batchTime}ms, Single calls time: ${singleTime}ms`);
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('FRAMA WASM tests completed');
});