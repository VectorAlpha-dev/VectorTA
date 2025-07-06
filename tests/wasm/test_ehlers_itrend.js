/**
 * WASM binding tests for Ehlers ITrend indicator.
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

test('Ehlers ITrend partial params', () => {
    // Test with default parameters - mirrors check_itrend_partial_params
    const close = new Float64Array(testData.close);
    
    // Default parameters (warmup_bars=12, max_dc_period=50)
    const result = wasm.ehlers_itrend_js(close, undefined, undefined);
    assert.strictEqual(result.length, close.length);
    
    // Partial custom parameters
    const resultCustomWarmup = wasm.ehlers_itrend_js(close, 15, undefined);
    assert.strictEqual(resultCustomWarmup.length, close.length);
    
    const resultCustomMaxDc = wasm.ehlers_itrend_js(close, undefined, 40);
    assert.strictEqual(resultCustomMaxDc.length, close.length);
});

test('Ehlers ITrend accuracy', () => {
    // Test accuracy matches expected values - mirrors check_itrend_accuracy
    const close = new Float64Array(testData.close);
    const expectedLast5 = [59097.88, 59145.9, 59191.96, 59217.26, 59179.68];
    
    const result = wasm.ehlers_itrend_js(close, 12, 50);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        0.1,  // 1e-1 tolerance as in Rust test
        "Ehlers ITrend last 5 values mismatch"
    );
});

test('Ehlers ITrend empty input', () => {
    // Test error with empty input - mirrors check_itrend_no_data
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.ehlers_itrend_js(empty, undefined, undefined);
    }, /Input data is empty/);
});

test('Ehlers ITrend all NaN input', () => {
    // Test error with all NaN values - mirrors check_itrend_all_nan_data
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.ehlers_itrend_js(allNaN, undefined, undefined);
    }, /All values are NaN/);
});

test('Ehlers ITrend insufficient data', () => {
    // Test error with insufficient data - mirrors check_itrend_small_data_for_warmup
    const smallData = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    assert.throws(() => {
        wasm.ehlers_itrend_js(smallData, 10, undefined);
    }, /Not enough data for warmup/);
});

test('Ehlers ITrend zero warmup', () => {
    // Test error with zero warmup bars - mirrors check_itrend_zero_warmup
    const close = new Float64Array(testData.close);
    
    assert.throws(() => {
        wasm.ehlers_itrend_js(close, 0, undefined);
    }, /Invalid warmup_bars/);
});

test('Ehlers ITrend invalid max_dc', () => {
    // Test error with invalid max_dc_period - mirrors check_itrend_invalid_max_dc
    const close = new Float64Array(testData.close);
    
    assert.throws(() => {
        wasm.ehlers_itrend_js(close, undefined, 0);
    }, /Invalid max_dc_period/);
});

test('Ehlers ITrend reinput', () => {
    // Test applying indicator twice - mirrors check_itrend_reinput
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.ehlers_itrend_js(close, 12, 50);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply to output
    const secondResult = wasm.ehlers_itrend_js(firstResult, 10, 40);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Verify no NaN values after warmup
    if (secondResult.length > 20) {
        for (let i = 20; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('Ehlers ITrend NaN handling', () => {
    // Test NaN handling - mirrors check_itrend_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.ehlers_itrend_js(close, 12, 50);
    assert.strictEqual(result.length, close.length);
    
    // First 12 values should echo input (warmup period)
    for (let i = 0; i < 12; i++) {
        assertClose(
            result[i],
            close[i],
            1e-10,
            `Warmup echo failed at index ${i}`
        );
    }
    
    // After warmup, should have filtered values
    if (result.length > 12) {
        for (let i = 12; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('Ehlers ITrend batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Single parameter set
    const batchResult = wasm.ehlers_itrend_batch_js(
        close,
        12, 12, 0,    // warmup_bars range
        50, 50, 0     // max_dc_period range
    );
    
    // Should match single calculation
    const singleResult = wasm.ehlers_itrend_js(close, 12, 50);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('Ehlers ITrend batch multiple parameters', () => {
    // Test batch with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Multiple parameters: warmup 10,12,14 x max_dc 40,50
    const batchResult = wasm.ehlers_itrend_batch_js(
        close,
        10, 14, 2,    // warmup_bars: 10, 12, 14
        40, 50, 10    // max_dc_period: 40, 50
    );
    
    // Should have 3 x 2 = 6 rows * 100 cols = 600 values
    assert.strictEqual(batchResult.length, 6 * 100);
    
    // Verify first combination matches individual calculation
    const firstRow = batchResult.slice(0, 100);
    const singleResult = wasm.ehlers_itrend_js(close, 10, 40);
    assertArrayClose(
        firstRow, 
        singleResult, 
        1e-10, 
        "First batch row mismatch"
    );
});

test('Ehlers ITrend batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.ehlers_itrend_batch_metadata_js(
        10, 14, 2,    // warmup_bars: 10, 12, 14
        40, 50, 10    // max_dc_period: 40, 50
    );
    
    // Should have 3 x 2 = 6 combinations, each with 2 values
    assert.strictEqual(metadata.length, 6 * 2);
    
    // Check first combination
    assert.strictEqual(metadata[0], 10);  // warmup_bars
    assert.strictEqual(metadata[1], 40);  // max_dc_period
    
    // Check second combination
    assert.strictEqual(metadata[2], 10);  // warmup_bars
    assert.strictEqual(metadata[3], 50);  // max_dc_period
    
    // Check third combination  
    assert.strictEqual(metadata[4], 12);  // warmup_bars
    assert.strictEqual(metadata[5], 40);  // max_dc_period
});

test('Ehlers ITrend batch warmup validation', () => {
    // Test batch warmup period handling
    const close = new Float64Array(testData.close.slice(0, 30));
    
    const batchResult = wasm.ehlers_itrend_batch_js(
        close,
        10, 15, 5,    // warmup_bars: 10, 15
        50, 50, 0     // max_dc_period: 50
    );
    
    const metadata = wasm.ehlers_itrend_batch_metadata_js(10, 15, 5, 50, 50, 0);
    const numCombos = metadata.length / 2;
    assert.strictEqual(numCombos, 2);
    
    // Check warmup echo for each combination
    for (let combo = 0; combo < numCombos; combo++) {
        const warmupBars = metadata[combo * 2];
        const rowStart = combo * 30;
        const rowData = batchResult.slice(rowStart, rowStart + 30);
        
        // First warmup_bars values should echo input
        for (let i = 0; i < warmupBars; i++) {
            assertClose(
                rowData[i],
                close[i],
                1e-10,
                `Warmup echo failed at index ${i} for warmup_bars=${warmupBars}`
            );
        }
    }
});

test('Ehlers ITrend batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array(20).fill(0).map((_, i) => i + 1);
    
    // Single value sweep
    const singleBatch = wasm.ehlers_itrend_batch_js(
        close,
        10, 10, 1,
        30, 30, 1
    );
    
    assert.strictEqual(singleBatch.length, 20);
    
    // Step = 0 should return single value when start=end
    const zeroStepBatch = wasm.ehlers_itrend_batch_js(
        close,
        12, 12, 0,
        50, 50, 0
    );
    
    assert.strictEqual(zeroStepBatch.length, 20); // Single combination
    
    // Empty data should throw
    assert.throws(() => {
        wasm.ehlers_itrend_batch_js(
            new Float64Array([]),
            12, 12, 0,
            50, 50, 0
        );
    }, /Input data is empty|All values are NaN/);
});

test('Ehlers ITrend batch performance test', () => {
    // Test that batch is more efficient than multiple single calls
    const close = new Float64Array(testData.close.slice(0, 200));
    
    // Batch calculation
    const startBatch = Date.now();
    const batchResult = wasm.ehlers_itrend_batch_js(
        close,
        10, 20, 2,    // 6 warmup values
        40, 50, 5     // 3 max_dc values = 18 total combinations
    );
    const batchTime = Date.now() - startBatch;
    
    // Equivalent single calculations
    const startSingle = Date.now();
    const singleResults = [];
    for (let warmup = 10; warmup <= 20; warmup += 2) {
        for (let maxDc = 40; maxDc <= 50; maxDc += 5) {
            singleResults.push(...wasm.ehlers_itrend_js(close, warmup, maxDc));
        }
    }
    const singleTime = Date.now() - startSingle;
    
    // Batch should have same total length
    assert.strictEqual(batchResult.length, singleResults.length);
    
    // Log performance (batch should be faster)
    console.log(`  Ehlers ITrend Batch time: ${batchTime}ms, Single calls time: ${singleTime}ms`);
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('Ehlers ITrend WASM tests completed');
});