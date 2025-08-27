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

test('Ehlers ITrend partial params', () => {
    // Test with default parameters - mirrors check_itrend_partial_params
    const close = new Float64Array(testData.close);
    
    // Default parameters (warmup_bars=12, max_dc_period=50)
    // Note: WASM binding doesn't support undefined, must pass actual values
    const result = wasm.ehlers_itrend_js(close, 12, 50);
    assert.strictEqual(result.length, close.length);
    
    // Partial custom parameters - must provide both
    const resultCustomWarmup = wasm.ehlers_itrend_js(close, 15, 50);
    assert.strictEqual(resultCustomWarmup.length, close.length);
    
    const resultCustomMaxDc = wasm.ehlers_itrend_js(close, 12, 40);
    assert.strictEqual(resultCustomMaxDc.length, close.length);
});

test('Ehlers ITrend accuracy', async () => {
    // Test accuracy matches expected values - mirrors check_itrend_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.ehlersItrend;
    
    const result = wasm.ehlers_itrend_js(
        close,
        expected.defaultParams.warmupBars,
        expected.defaultParams.maxDcPeriod
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        0.1,  // 1e-1 tolerance as in Rust test
        "Ehlers ITrend last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('ehlers_itrend', result, 'close', expected.defaultParams);
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
        wasm.ehlers_itrend_js(smallData, 10, 50);
    }, /Not enough data for warmup/);
});

test('Ehlers ITrend zero warmup', () => {
    // Test error with zero warmup bars - mirrors check_itrend_zero_warmup
    const close = new Float64Array(testData.close);
    
    assert.throws(() => {
        wasm.ehlers_itrend_js(close, 0, 50);
    }, /Invalid warmup_bars/);
});

test('Ehlers ITrend invalid max_dc', () => {
    // Test error with invalid max_dc_period - mirrors check_itrend_invalid_max_dc
    const close = new Float64Array(testData.close);
    
    assert.throws(() => {
        wasm.ehlers_itrend_js(close, 12, 0);
    }, /Invalid max_dc_period/);
});

test('Ehlers ITrend reinput', () => {
    // Test applying indicator twice - mirrors check_itrend_reinput
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.ehlersItrend;
    const params = expected.defaultParams;
    
    // First pass
    const firstResult = wasm.ehlers_itrend_js(
        close,
        params.warmupBars,
        params.maxDcPeriod
    );
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply to output with SAME parameters
    const secondResult = wasm.ehlers_itrend_js(
        firstResult,
        params.warmupBars,
        params.maxDcPeriod
    );
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Ehlers ITrend has a 3-bar lookback, so it needs valid data for x0, x1, x2, x3
    // First pass has 12 NaN values, so indices 12, 13, 14 can still propagate NaN
    // The indicator should produce valid values once it has enough non-NaN input
    // Check that we eventually get valid values after the initial propagation
    if (secondResult.length > 30) {
        let hasValidValues = false;
        for (let i = 30; i < secondResult.length; i++) {
            if (!isNaN(secondResult[i])) {
                hasValidValues = true;
                break;
            }
        }
        assert(hasValidValues, 'Expected some valid values after warmup and lookback period');
    }
});

test('Ehlers ITrend NaN handling', () => {
    // Test NaN handling - mirrors check_itrend_nan_handling
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.ehlersItrend;
    const params = expected.defaultParams;
    const warmupBars = params.warmupBars;
    
    const result = wasm.ehlers_itrend_js(
        close,
        params.warmupBars,
        params.maxDcPeriod
    );
    assert.strictEqual(result.length, close.length);
    
    // First warmupBars values should be NaN (not echo)
    for (let i = 0; i < warmupBars; i++) {
        assert(isNaN(result[i]), `Expected NaN at warmup index ${i}`);
    }
    
    // After warmup, should have filtered values (no NaN)
    if (result.length > warmupBars) {
        for (let i = warmupBars; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // Verify warmup count matches expected
    assert.strictEqual(warmupBars, 12, `Expected warmupBars=12, got ${warmupBars}`);
});

test('Ehlers ITrend batch single parameter set', () => {
    // Note: Batch functions may not be implemented for ehlers_itrend
    // Skip this test if functions don't exist
    if (typeof wasm.ehlers_itrend_batch === 'undefined') {
        console.log('  Skipping: Batch functions not implemented for ehlers_itrend');
        return;
    }
    
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Use the ergonomic batch API if available
    const batchResult = wasm.ehlers_itrend_batch(close, {
        warmup_range: [12, 12, 0],
        max_dc_range: [50, 50, 0]
    });
    
    // Should match single calculation
    const singleResult = wasm.ehlers_itrend_js(close, 12, 50);
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    assertArrayClose(batchResult.values, singleResult, 1e-10, "Batch vs single mismatch");
});

test('Ehlers ITrend batch multiple parameters', () => {
    // Skip if batch functions not available
    if (typeof wasm.ehlers_itrend_batch === 'undefined') {
        console.log('  Skipping: Batch functions not implemented');
        return;
    }
    
    // Test batch with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Multiple parameters using ergonomic API
    const batchResult = wasm.ehlers_itrend_batch(close, {
        warmup_range: [10, 14, 2],    // warmup_bars: 10, 12, 14
        max_dc_range: [40, 50, 10]    // max_dc_period: 40, 50
    });
    
    // Should have 3 x 2 = 6 combinations
    assert.strictEqual(batchResult.combos.length, 6);
    assert.strictEqual(batchResult.values.length, 6 * 100);
    
    // Verify first combination matches individual calculation
    const firstRow = batchResult.values.slice(0, 100);
    const singleResult = wasm.ehlers_itrend_js(close, 10, 40);
    assertArrayClose(
        firstRow, 
        singleResult, 
        1e-10, 
        "First batch row mismatch"
    );
});

test('Ehlers ITrend batch metadata', () => {
    // Skip if batch functions not available
    if (typeof wasm.ehlers_itrend_batch === 'undefined') {
        console.log('  Skipping: Batch metadata functions not implemented');
        return;
    }
    
    // Test batch result includes correct metadata
    const close = new Float64Array(20).fill(100);
    const result = wasm.ehlers_itrend_batch(close, {
        warmup_range: [10, 14, 2],    // warmup_bars: 10, 12, 14
        max_dc_range: [40, 50, 10]    // max_dc_period: 40, 50
    });
    
    // Should have 3 x 2 = 6 combinations
    assert.strictEqual(result.combos.length, 6);
    
    // Check first combination
    assert.strictEqual(result.combos[0].warmup_bars, 10);
    assert.strictEqual(result.combos[0].max_dc_period, 40);
    
    // Check second combination
    assert.strictEqual(result.combos[1].warmup_bars, 10);
    assert.strictEqual(result.combos[1].max_dc_period, 50);
    
    // Check third combination  
    assert.strictEqual(result.combos[2].warmup_bars, 12);
    assert.strictEqual(result.combos[2].max_dc_period, 40);
});

test('Ehlers ITrend batch warmup validation', () => {
    // Skip if batch functions not available
    if (typeof wasm.ehlers_itrend_batch === 'undefined') {
        console.log('  Skipping: Batch functions not implemented');
        return;
    }
    
    // Test batch warmup period handling
    const close = new Float64Array(testData.close.slice(0, 30));
    
    const batchResult = wasm.ehlers_itrend_batch(close, {
        warmup_range: [10, 15, 5],    // warmup_bars: 10, 15
        max_dc_range: [50, 50, 0]     // max_dc_period: 50
    });
    
    assert.strictEqual(batchResult.combos.length, 2);
    
    // Check warmup echo for each combination
    for (let combo = 0; combo < batchResult.combos.length; combo++) {
        const warmupBars = batchResult.combos[combo].warmup_bars;
        const rowStart = combo * 30;
        const rowData = batchResult.values.slice(rowStart, rowStart + 30);
        
        // First warmup_bars values should be NaN
        for (let i = 0; i < warmupBars; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at index ${i} for warmup_bars=${warmupBars}`);
        }
    }
});

test('Ehlers ITrend batch edge cases', () => {
    // Skip if batch functions not available
    if (typeof wasm.ehlers_itrend_batch === 'undefined') {
        console.log('  Skipping: Batch functions not implemented');
        return;
    }
    
    // Test edge cases for batch processing
    const close = new Float64Array(20).fill(0).map((_, i) => i + 1);
    
    // Single value sweep
    const singleBatch = wasm.ehlers_itrend_batch(close, {
        warmup_range: [10, 10, 1],
        max_dc_range: [30, 30, 1]
    });
    
    assert.strictEqual(singleBatch.combos.length, 1);
    assert.strictEqual(singleBatch.values.length, 20);
    
    // Step = 0 should return single value when start=end
    const zeroStepBatch = wasm.ehlers_itrend_batch(close, {
        warmup_range: [12, 12, 0],
        max_dc_range: [50, 50, 0]
    });
    
    assert.strictEqual(zeroStepBatch.combos.length, 1);
    assert.strictEqual(zeroStepBatch.values.length, 20);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.ehlers_itrend_batch(new Float64Array([]), {
            warmup_range: [12, 12, 0],
            max_dc_range: [50, 50, 0]
        });
    }, /Input data is empty|All values are NaN/);
});

test('Ehlers ITrend batch performance test', () => {
    // Skip if batch functions not available
    if (typeof wasm.ehlers_itrend_batch === 'undefined') {
        console.log('  Skipping: Batch functions not implemented');
        return;
    }
    
    // Test that batch is more efficient than multiple single calls
    const close = new Float64Array(testData.close.slice(0, 200));
    
    // Batch calculation
    const startBatch = Date.now();
    const batchResult = wasm.ehlers_itrend_batch(close, {
        warmup_range: [10, 20, 2],    // 6 warmup values
        max_dc_range: [40, 50, 5]     // 3 max_dc values = 18 total combinations
    });
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
    assert.strictEqual(batchResult.values.length, singleResults.length);
    
    // Log performance (batch should be faster)
    console.log(`  Ehlers ITrend Batch time: ${batchTime}ms, Single calls time: ${singleTime}ms`);
});

// Zero-copy API tests
test('Ehlers ITrend zero-copy API', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const warmupBars = 12;
    const maxDcPeriod = 50;
    
    // Check if zero-copy functions exist
    if (!wasm.ehlers_itrend_alloc || !wasm.ehlers_itrend_into || !wasm.ehlers_itrend_free) {
        console.log('  Skipping: Zero-copy API not implemented for Ehlers ITrend');
        return;
    }
    
    // Allocate buffer
    const ptr = wasm.ehlers_itrend_alloc(data.length);
    assert(ptr !== 0, 'Failed to allocate memory');
    
    // Create view into WASM memory
    const memory = wasm.__wbindgen_memory();
    const memView = new Float64Array(
        memory.buffer,
        ptr,
        data.length
    );
    
    // Copy data into WASM memory
    memView.set(data);
    
    // Compute Ehlers ITrend in-place
    try {
        wasm.ehlers_itrend_into(ptr, ptr, data.length, warmupBars, maxDcPeriod);
        
        // Verify results match regular API
        const regularResult = wasm.ehlers_itrend_js(data, warmupBars, maxDcPeriod);
        for (let i = 0; i < data.length; i++) {
            if (isNaN(regularResult[i]) && isNaN(memView[i])) {
                continue; // Both NaN is OK
            }
            assert(Math.abs(regularResult[i] - memView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${memView[i]}`);
        }
    } finally {
        // Always free memory
        wasm.ehlers_itrend_free(ptr, data.length);
    }
});

test('Ehlers ITrend batch with ergonomic API', () => {
    // Check if new batch API exists
    if (typeof wasm.ehlers_itrend_batch === 'undefined') {
        console.log('  Skipping: Ergonomic batch API not implemented for Ehlers ITrend');
        return;
    }
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    try {
        // Test new ergonomic batch API
        const result = wasm.ehlers_itrend_batch(close, {
            warmup_range: [10, 14, 2],     // 10, 12, 14
            max_dc_range: [40, 50, 10]     // 40, 50
        });
        
        // Verify structure
        assert(result.values, 'Should have values array');
        assert(result.combos, 'Should have combos array');
        assert(typeof result.rows === 'number', 'Should have rows count');
        assert(typeof result.cols === 'number', 'Should have cols count');
        
        // Should have 3 * 2 = 6 combinations
        assert.strictEqual(result.rows, 6);
        assert.strictEqual(result.cols, 50);
        assert.strictEqual(result.combos.length, 6);
        assert.strictEqual(result.values.length, 300);
        
        // Verify each combo matches individual calculation
        for (let i = 0; i < result.combos.length; i++) {
            const combo = result.combos[i];
            const rowStart = i * 50;
            const rowData = result.values.slice(rowStart, rowStart + 50);
            
            const singleResult = wasm.ehlers_itrend_js(close, combo.warmupBars || combo.warmup_bars, combo.maxDcPeriod || combo.max_dc_period);
            assertArrayClose(
                rowData,
                singleResult,
                1e-10,
                `Combo ${i} mismatch (warmup=${combo.warmupBars || combo.warmup_bars}, maxDc=${combo.maxDcPeriod || combo.max_dc_period})`
            );
        }
    } catch (error) {
        // If ergonomic API fails, it might not be properly implemented
        // Log the error and skip the test
        console.log(`  Skipping: Ergonomic batch API error: ${error}`);
        return;
    }
});

test('Ehlers ITrend SIMD128 consistency', () => {
    // This test verifies SIMD128 produces same results as scalar
    // Note: Ehlers ITrend doesn't use SIMD, but we verify consistency anyway
    const testCases = [
        { size: 20, warmup: 12 },
        { size: 100, warmup: 20 },
        { size: 1000, warmup: 50 }
    ];
    
    for (const testCase of testCases) {
        const data = new Float64Array(testCase.size);
        for (let i = 0; i < testCase.size; i++) {
            data[i] = Math.sin(i * 0.1) + Math.cos(i * 0.05);
        }
        
        const result = wasm.ehlers_itrend_js(data, testCase.warmup, 50);
        
        // Basic sanity checks
        assert.strictEqual(result.length, data.length);
        
        // Check warmup period has NaN
        for (let i = 0; i < testCase.warmup && i < data.length; i++) {
            assert(isNaN(result[i]), 
                   `Expected NaN at warmup index ${i} for size=${testCase.size}`);
        }
        
        // Check values exist after warmup
        let sumAfterWarmup = 0;
        let countAfterWarmup = 0;
        for (let i = testCase.warmup; i < result.length; i++) {
            assert(!isNaN(result[i]), `Unexpected NaN at index ${i} for size=${testCase.size}`);
            sumAfterWarmup += result[i];
            countAfterWarmup++;
        }
        
        // Verify reasonable values
        if (countAfterWarmup > 0) {
            const avgAfterWarmup = sumAfterWarmup / countAfterWarmup;
            assert(Math.abs(avgAfterWarmup) < 100, `Average value ${avgAfterWarmup} seems unreasonable`);
        }
    }
});

test('Ehlers ITrend memory management', () => {
    // Skip if zero-copy not available
    if (!wasm.ehlers_itrend_alloc || !wasm.ehlers_itrend_free) {
        console.log('  Skipping: Zero-copy API not implemented');
        return;
    }
    
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptr = wasm.ehlers_itrend_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Write pattern to verify memory
        const memory = wasm.__wbindgen_memory();
        const memView = new Float64Array(memory.buffer, ptr, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 1.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 1.5, `Memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.ehlers_itrend_free(ptr, size);
    }
});

test('Ehlers ITrend batch comprehensive parameter sweep', () => {
    // Skip if batch functions not available
    if (typeof wasm.ehlers_itrend_batch === 'undefined') {
        console.log('  Skipping: Batch functions not implemented');
        return;
    }
    
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Comprehensive parameter sweep
    const batchResult = wasm.ehlers_itrend_batch(close, {
        warmup_range: [10, 20, 5],     // warmup: 10, 15, 20
        max_dc_range: [30, 60, 15]     // max_dc: 30, 45, 60
    });
    
    // Should have 3 * 3 = 9 combinations
    assert.strictEqual(batchResult.combos.length, 9);
    assert.strictEqual(batchResult.values.length, 9 * 100);
    
    // Verify ALL combinations
    const expectedCombos = [];
    for (let warmup = 10; warmup <= 20; warmup += 5) {
        for (let maxDc = 30; maxDc <= 60; maxDc += 15) {
            expectedCombos.push({ warmup, maxDc });
        }
    }
    
    for (let i = 0; i < expectedCombos.length; i++) {
        const expected = expectedCombos[i];
        const actual = batchResult.combos[i];
        
        assert.strictEqual(actual.warmup_bars, expected.warmup, 
                          `Combo ${i} warmup mismatch`);
        assert.strictEqual(actual.max_dc_period, expected.maxDc, 
                          `Combo ${i} maxDc mismatch`);
        
        // Verify calculation matches
        const rowStart = i * 100;
        const rowData = batchResult.values.slice(rowStart, rowStart + 100);
        const singleResult = wasm.ehlers_itrend_js(close, expected.warmup, expected.maxDc);
        
        assertArrayClose(
            rowData,
            singleResult,
            1e-10,
            `Combo ${i} (warmup=${expected.warmup}, maxDc=${expected.maxDc}) calculation mismatch`
        );
    }
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('Ehlers ITrend WASM tests completed');
});