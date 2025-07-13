/**
 * WASM binding tests for DEMA indicator.
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
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('DEMA partial params', () => {
    // Test with default parameters - mirrors check_dema_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.dema_js(close, 30);
    assert.strictEqual(result.length, close.length);
    
    // Test with custom period
    const resultCustom = wasm.dema_js(close, 14);
    assert.strictEqual(resultCustom.length, close.length);
});

test('DEMA accuracy', async () => {
    // Test DEMA matches expected values from Rust tests - mirrors check_dema_accuracy
    const close = new Float64Array(testData.close);
    const expectedLast5 = [
        59189.73193987478,
        59129.24920772847,
        59058.80282420511,
        59011.5555611042,
        58908.370159946775
    ];
    
    const result = wasm.dema_js(close, 30);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-6,
        "DEMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('dema', result, 'close', {period: 30});
});

test('DEMA default candles', () => {
    // Test DEMA with default parameters - mirrors check_dema_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.dema_js(close, 30);
    assert.strictEqual(result.length, close.length);
});

test('DEMA zero period', () => {
    // Test DEMA fails with zero period - mirrors check_dema_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.dema_js(inputData, 0);
    }, /Invalid period/);
});

test('DEMA period exceeds length', () => {
    // Test DEMA fails when period exceeds data length - mirrors check_dema_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.dema_js(dataSmall, 10);
    }, /Invalid period|Not enough data/);
});

test('DEMA very small dataset', () => {
    // Test DEMA fails with insufficient data - mirrors check_dema_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.dema_js(singlePoint, 9);
    }, /Invalid period|Not enough data/);
});

test('DEMA empty input', () => {
    // Test DEMA fails with empty input - mirrors check_dema_empty_input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.dema_js(empty, 30);
    }, /Input data slice is empty/);
});

test('DEMA reinput', () => {
    // Test DEMA applied twice (re-input) - mirrors check_dema_reinput
    const close = new Float64Array(testData.close);
    
    // First pass with period 80
    const firstResult = wasm.dema_js(close, 80);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass with period 60 - apply DEMA to DEMA output
    const secondResult = wasm.dema_js(firstResult, 60);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // After warmup period (240), no NaN values should exist
    if (secondResult.length > 240) {
        for (let i = 240; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('DEMA NaN handling', () => {
    // Test DEMA handles NaN values correctly - mirrors check_dema_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.dema_js(close, 30);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('DEMA all NaN input', () => {
    // Test DEMA with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.dema_js(allNaN, 30);
    }, /All values are NaN/);
});

test('DEMA not enough valid data', () => {
    // Test DEMA with not enough valid data after NaN values
    const data = new Float64Array([NaN, NaN, 1.0, 2.0]);
    
    assert.throws(() => {
        wasm.dema_js(data, 3);
    }, /Not enough valid data/);
});

test('DEMA batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Single parameter set: period=30
    const batchResult = wasm.dema_batch_js(
        close,
        30, 30, 0  // period range
    );
    
    // Should match single calculation
    const singleResult = wasm.dema_js(close, 30);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('DEMA batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 10, 20, 30, 40
    const batchResult = wasm.dema_batch_js(
        close,
        10, 40, 10  // period range
    );
    
    // Should have 4 rows * 100 cols = 400 values
    assert.strictEqual(batchResult.length, 4 * 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 20, 30, 40];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.dema_js(close, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('DEMA batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.dema_batch_metadata_js(
        10, 50, 10  // period: 10, 20, 30, 40, 50
    );
    
    // Should have 5 periods
    assert.strictEqual(metadata.length, 5);
    
    // Check values
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 20);
    assert.strictEqual(metadata[2], 30);
    assert.strictEqual(metadata[3], 40);
    assert.strictEqual(metadata[4], 50);
});

test('DEMA batch warmup validation', () => {
    // Test that batch correctly handles warmup periods for DEMA
    const close = new Float64Array(testData.close.slice(0, 60));
    
    const batchResult = wasm.dema_batch_js(
        close,
        10, 20, 10  // periods: 10, 20
    );
    
    const metadata = wasm.dema_batch_metadata_js(10, 20, 10);
    const numCombos = metadata.length;
    assert.strictEqual(numCombos, 2);
    
    // DEMA has warmup period of period-1
    for (let combo = 0; combo < numCombos; combo++) {
        const period = metadata[combo];
        const rowStart = combo * 60;
        const rowData = batchResult.slice(rowStart, rowStart + 60);
        
        // DEMA doesn't have NaN warmup period - it starts calculating from the first value
        // All values should be non-NaN
        for (let i = 0; i < 60; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('DEMA batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
    
    // Single value sweep
    const singleBatch = wasm.dema_batch_js(
        close,
        10, 10, 1
    );
    
    assert.strictEqual(singleBatch.length, 20);
    
    // Step = 0 with period that requires more data than available should throw
    assert.throws(() => {
        wasm.dema_batch_js(
            close,
            15, 25, 0  // Period 15 needs 2*(15-1) = 28 values, but we only have 20
        );
    }, /Not enough data/);
    
    // Step larger than range
    const largeBatch = wasm.dema_batch_js(
        close,
        5, 7, 10  // Step larger than range
    );
    
    // Should only have period=5
    assert.strictEqual(largeBatch.length, 20);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.dema_batch_js(
            new Float64Array([]),
            30, 30, 0
        );
    }, /Input data slice is empty/);
});

test('DEMA batch performance test', () => {
    // Test that batch is more efficient than multiple single calls
    const close = new Float64Array(testData.close.slice(0, 300));
    
    // Batch calculation
    const startBatch = Date.now();
    const batchResult = wasm.dema_batch_js(
        close,
        10, 100, 5  // 19 periods
    );
    const batchTime = Date.now() - startBatch;
    
    // Equivalent single calculations
    const startSingle = Date.now();
    const singleResults = [];
    for (let period = 10; period <= 100; period += 5) {
        singleResults.push(...wasm.dema_js(close, period));
    }
    const singleTime = Date.now() - startSingle;
    
    // Batch should have same total length
    assert.strictEqual(batchResult.length, singleResults.length);
    
    // Log performance (batch should be faster)
    console.log(`  DEMA Batch time: ${batchTime}ms, Single calls time: ${singleTime}ms`);
});

test('DEMA batch MA crossover scenario', () => {
    // Test realistic MA crossover scenario
    const close = new Float64Array(testData.close.slice(0, 200));
    
    // Fast MA periods: 10, 15, 20
    // Slow MA periods: 30, 40, 50
    const fastBatch = wasm.dema_batch_js(close, 10, 20, 5);
    const slowBatch = wasm.dema_batch_js(close, 30, 50, 10);
    
    // Should have correct sizes
    assert.strictEqual(fastBatch.length, 3 * 200); // 3 fast periods
    assert.strictEqual(slowBatch.length, 3 * 200); // 3 slow periods
    
    // Test that we can extract individual MA series
    const fast10 = fastBatch.slice(0, 200);
    const slow30 = slowBatch.slice(0, 200);
    
    // DEMA doesn't have NaN warmup period - verify all values are valid
    for (let i = 0; i < 200; i++) {
        assert(!isNaN(fast10[i]), `Unexpected NaN at index ${i} for fast MA`);
        assert(!isNaN(slow30[i]), `Unexpected NaN at index ${i} for slow MA`);
    }
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('DEMA WASM tests completed');
});