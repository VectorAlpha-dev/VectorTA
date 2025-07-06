/**
 * WASM binding tests for EDCF indicator.
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

test('EDCF partial params', () => {
    // Test with default parameters - mirrors check_edcf_with_slice_data
    const close = new Float64Array(testData.close);
    
    const result = wasm.edcf_js(close, 15);
    assert.strictEqual(result.length, close.length);
    
    // Test with custom period
    const resultCustom = wasm.edcf_js(close, 20);
    assert.strictEqual(resultCustom.length, close.length);
});

test('EDCF accuracy', async () => {
    // Test EDCF matches expected values from Rust tests - mirrors check_edcf_accuracy_last_five
    // Note: Rust test uses "hl2" source, we need to calculate it
    const hl2 = new Float64Array(testData.high.length);
    for (let i = 0; i < testData.high.length; i++) {
        hl2[i] = (testData.high[i] + testData.low[i]) / 2;
    }
    
    const expectedLast5 = [
        59593.332275678375,
        59731.70263288801,
        59766.41512339413,
        59655.66162110993,
        59332.492883847
    ];
    
    const result = wasm.edcf_js(hl2, 15);
    
    assert.strictEqual(result.length, hl2.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-8,
        "EDCF last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('edcf', result, 'hl2', { period: 15 });
});

test('EDCF default candles', () => {
    // Test EDCF with default parameters - mirrors check_edcf_with_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.edcf_js(close, 15);
    assert.strictEqual(result.length, close.length);
});

test('EDCF zero period', () => {
    // Test EDCF fails with zero period - mirrors check_edcf_with_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.edcf_js(inputData, 0);
    }, /Invalid period/);
});

test('EDCF period exceeds length', () => {
    // Test EDCF fails when period exceeds data length - mirrors check_edcf_with_period_exceeding_data_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.edcf_js(dataSmall, 10);
    }, /Invalid period/);
});

test('EDCF very small dataset', () => {
    // Test EDCF fails with insufficient data - mirrors check_edcf_very_small_data_set
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.edcf_js(singlePoint, 15);
    }, /Invalid period/);
});

test('EDCF empty input', () => {
    // Test EDCF fails with empty input - mirrors check_edcf_with_no_data
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.edcf_js(empty, 15);
    }, /No data provided/);
});

test('EDCF reinput', () => {
    // Test EDCF applied twice (re-input) - mirrors check_edcf_with_slice_data_reinput
    const close = new Float64Array(testData.close);
    
    // First pass with period 15
    const firstResult = wasm.edcf_js(close, 15);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass with period 5 - apply EDCF to EDCF output
    const secondResult = wasm.edcf_js(firstResult, 5);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // After warmup period (240), no NaN values should exist
    if (secondResult.length > 240) {
        for (let i = 240; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('EDCF NaN handling', () => {
    // Test EDCF handles NaN values correctly - mirrors check_edcf_accuracy_nan_check
    const close = new Float64Array(testData.close);
    const period = 15;
    
    const result = wasm.edcf_js(close, period);
    assert.strictEqual(result.length, close.length);
    
    // After 2*period, no NaN values should exist
    const startIndex = 2 * period;
    if (result.length > startIndex) {
        for (let i = startIndex; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('EDCF all NaN input', () => {
    // Test EDCF with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.edcf_js(allNaN, 15);
    }, /All values are NaN/);
});

test('EDCF accuracy hl2 verification', () => {
    // Additional test to verify EDCF with hl2 data matches Rust exactly
    const hl2 = new Float64Array(testData.high.length);
    for (let i = 0; i < testData.high.length; i++) {
        hl2[i] = (testData.high[i] + testData.low[i]) / 2;
    }
    
    const result = wasm.edcf_js(hl2, 15);
    
    // Expected values from Rust test (check_edcf_accuracy_last_five)
    const expectedLast5 = [
        59593.332275678375,
        59731.70263288801,
        59766.41512339413,
        59655.66162110993,
        59332.492883847
    ];
    
    // Check last 5 values with the same tolerance as Rust (1e-8)
    const actualLast5 = result.slice(-5);
    for (let i = 0; i < 5; i++) {
        const diff = Math.abs(actualLast5[i] - expectedLast5[i]);
        assert(diff < 1e-8, `EDCF mismatch at index ${i}: got ${actualLast5[i]}, expected ${expectedLast5[i]}, diff ${diff}`);
    }
});

test('EDCF batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Single parameter set: period=15
    const batchResult = wasm.edcf_batch_js(
        close,
        15, 15, 0  // period range
    );
    
    // Should match single calculation
    const singleResult = wasm.edcf_js(close, 15);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('EDCF batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 10, 15, 20, 25
    const batchResult = wasm.edcf_batch_js(
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
        
        const singleResult = wasm.edcf_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('EDCF batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.edcf_batch_metadata_js(
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

test('EDCF batch warmup validation', () => {
    // Test that batch correctly handles warmup periods for EDCF
    const close = new Float64Array(testData.close.slice(0, 60));
    
    const batchResult = wasm.edcf_batch_js(
        close,
        10, 20, 10  // periods: 10, 20
    );
    
    const metadata = wasm.edcf_batch_metadata_js(10, 20, 10);
    const numCombos = metadata.length;
    assert.strictEqual(numCombos, 2);
    
    // EDCF has warmup period of 2*period
    for (let combo = 0; combo < numCombos; combo++) {
        const period = metadata[combo];
        const rowStart = combo * 60;
        const rowData = batchResult.slice(rowStart, rowStart + 60);
        
        // First 2*period values should be NaN
        const warmup = 2 * period;
        for (let i = 0; i < warmup && i < 60; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        // After warmup should have values
        for (let i = warmup; i < 60; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('EDCF batch with hl2 data', () => {
    // Test EDCF batch with hl2 data (matching Rust accuracy test)
    const hl2 = new Float64Array(testData.high.length);
    for (let i = 0; i < testData.high.length; i++) {
        hl2[i] = (testData.high[i] + testData.low[i]) / 2;
    }
    
    // Use smaller dataset for speed
    const hl2Small = hl2.slice(0, 100);
    
    const batchResult = wasm.edcf_batch_js(
        hl2Small,
        15, 15, 0  // Single period matching Rust test
    );
    
    // Should match single calculation
    const singleResult = wasm.edcf_js(hl2Small, 15);
    assertArrayClose(batchResult, singleResult, 1e-10, "HL2 batch vs single mismatch");
});

test('EDCF batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array(Array.from({length: 50}, (_, i) => i + 1));
    
    // Single value sweep
    const singleBatch = wasm.edcf_batch_js(
        close,
        10, 10, 1
    );
    
    assert.strictEqual(singleBatch.length, 50);
    
    // Step = 0 should return single value
    const zeroStepBatch = wasm.edcf_batch_js(
        close,
        15, 25, 0
    );
    
    assert.strictEqual(zeroStepBatch.length, 50); // Single period=15
    
    // Step larger than range
    const largeBatch = wasm.edcf_batch_js(
        close,
        10, 12, 10  // Step larger than range
    );
    
    // Should only have period=10
    assert.strictEqual(largeBatch.length, 50);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.edcf_batch_js(
            new Float64Array([]),
            15, 15, 0
        );
    }, /No data provided/);
});

test('EDCF batch performance test', () => {
    // Test that batch is more efficient than multiple single calls
    const close = new Float64Array(testData.close.slice(0, 200));
    
    // Batch calculation
    const startBatch = Date.now();
    const batchResult = wasm.edcf_batch_js(
        close,
        10, 30, 2  // 11 periods
    );
    const batchTime = Date.now() - startBatch;
    
    // Equivalent single calculations
    const startSingle = Date.now();
    const singleResults = [];
    for (let period = 10; period <= 30; period += 2) {
        singleResults.push(...wasm.edcf_js(close, period));
    }
    const singleTime = Date.now() - startSingle;
    
    // Batch should have same total length
    assert.strictEqual(batchResult.length, singleResults.length);
    
    // Log performance (batch should be faster)
    console.log(`  EDCF Batch time: ${batchTime}ms, Single calls time: ${singleTime}ms`);
});

test('EDCF batch volatility analysis scenario', () => {
    // Test realistic volatility analysis scenario using EDCF
    const close = new Float64Array(testData.close.slice(0, 300));
    
    // Different period ranges for volatility analysis
    // Short-term: 10-20, Medium-term: 25-40, Long-term: 45-60
    const shortTermBatch = wasm.edcf_batch_js(close, 10, 20, 2);
    const mediumTermBatch = wasm.edcf_batch_js(close, 25, 40, 3);
    const longTermBatch = wasm.edcf_batch_js(close, 45, 60, 5);
    
    // Verify sizes
    assert.strictEqual(shortTermBatch.length, 6 * 300);  // 6 periods
    assert.strictEqual(mediumTermBatch.length, 6 * 300); // 6 periods
    assert.strictEqual(longTermBatch.length, 4 * 300);   // 4 periods
    
    // Extract first series from each batch
    const short10 = shortTermBatch.slice(0, 300);
    const medium25 = mediumTermBatch.slice(0, 300);
    const long45 = longTermBatch.slice(0, 300);
    
    // Verify warmup periods (2*period for EDCF)
    for (let i = 0; i < 20; i++) {
        assert(isNaN(short10[i]), `Expected NaN at index ${i} for short-term EDCF`);
    }
    for (let i = 0; i < 50; i++) {
        assert(isNaN(medium25[i]), `Expected NaN at index ${i} for medium-term EDCF`);
    }
    for (let i = 0; i < 90; i++) {
        assert(isNaN(long45[i]), `Expected NaN at index ${i} for long-term EDCF`);
    }
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('EDCF WASM tests completed');
});