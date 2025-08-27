/**
 * WASM binding tests for SMMA indicator.
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
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('SMMA partial params', () => {
    // Test with default parameters - mirrors check_smma_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.smma(close, 7);
    assert.strictEqual(result.length, close.length);
});

test('SMMA accuracy', async () => {
    // Test SMMA matches expected values from Rust tests - mirrors check_smma_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.smma(close, 7);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected values from Rust test
    const expectedLast5 = [59434.4, 59398.2, 59346.9, 59319.4, 59224.5];
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5,
        1e-1, // Using same tolerance as Rust test
        "SMMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('smma', result, 'close', { period: 7 });
});

test('SMMA default candles', () => {
    // Test SMMA with default parameters - mirrors check_smma_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.smma(close, 7);
    assert.strictEqual(result.length, close.length);
});

test('SMMA zero period', () => {
    // Test SMMA fails with zero period - mirrors check_smma_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.smma(inputData, 0);
    }, /Invalid period/);
});

test('SMMA empty input', () => {
    // Test SMMA fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.smma(empty, 7);
    }, /Input data slice is empty/);
});

test('SMMA period exceeds length', () => {
    // Test SMMA fails when period exceeds data length - mirrors check_smma_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.smma(dataSmall, 10);
    }, /Invalid period/);
});

test('SMMA very small dataset', () => {
    // Test SMMA fails with insufficient data - mirrors check_smma_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.smma(singlePoint, 9);
    }, /Invalid period|Not enough valid data/);
});

test('SMMA reinput', () => {
    // Test SMMA applied twice (re-input) - mirrors check_smma_reinput
    const close = new Float64Array(testData.close);
    
    // First pass with period 7
    const firstResult = wasm.smma(close, 7);
    assert.strictEqual(firstResult.length, close.length);
    
    // Verify first pass matches expected
    const expectedFirstPass = [59434.4, 59398.2, 59346.9, 59319.4, 59224.5];
    assertArrayClose(
        firstResult.slice(-5),
        expectedFirstPass,
        1e-1,
        "First pass SMMA values mismatch"
    );
    
    // Second pass with period 5 - apply SMMA to SMMA output
    const secondResult = wasm.smma(firstResult, 5);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Verify the re-input produces expected smoothing
    // Calculate standard deviation to verify smoothing
    const calcStdDev = (arr) => {
        const validValues = arr.filter(v => !isNaN(v));
        const mean = validValues.reduce((a, b) => a + b, 0) / validValues.length;
        const variance = validValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / validValues.length;
        return Math.sqrt(variance);
    };
    
    assert(calcStdDev(secondResult) < calcStdDev(firstResult),
           "Second pass should produce smoother results");
});

test('SMMA NaN handling', () => {
    // Test SMMA handles NaN values correctly - mirrors check_smma_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.smma(close, 7);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First period-1 values should be NaN (warmup = first + period - 1)
    // Since first valid index is 0, warmup = 0 + 7 - 1 = 6  
    assertAllNaN(result.slice(0, 6), "Expected NaN in warmup period (indices 0-5)");
    // First valid value should be at index 6 (period-1)
    assert(!isNaN(result[6]), "Expected valid value at index 6 (period-1)");
});

test('SMMA all NaN input', () => {
    // Test SMMA with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.smma(allNaN, 7);
    }, /All values are NaN/);
});

test('SMMA batch single period', () => {
    // Test batch with single period - mirrors check_batch_default_row
    const close = new Float64Array(testData.close);
    
    const batchValues = wasm.smma_batch_legacy(close, 7, 7, 0); // Default period only
    const metadata = wasm.smma_batch_metadata(7, 7, 0);
    const dims = wasm.smma_batch_rows_cols(7, 7, 0, close.length);
    
    assert(batchValues instanceof Float64Array, "Values should be Float64Array");
    assert(metadata instanceof Uint32Array, "Metadata should be Uint32Array");
    assert.strictEqual(dims[0], 1); // rows
    assert.strictEqual(dims[1], close.length); // cols
    
    const expected = [59434.4, 59398.2, 59346.9, 59319.4, 59224.5];
    
    // Check last 5 values match
    const last5 = batchValues.slice(-5);
    assertArrayClose(
        last5,
        expected,
        1e-1, // Using same tolerance as Rust test
        "SMMA batch default row mismatch"
    );
});

test('SMMA batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Smaller dataset for speed
    
    const batchValues = wasm.smma_batch_legacy(close, 5, 10, 1); // Periods 5, 6, 7, 8, 9, 10
    const metadata = wasm.smma_batch_metadata(5, 10, 1);
    const dims = wasm.smma_batch_rows_cols(5, 10, 1, 100);
    
    assert.strictEqual(dims[0], 6); // 6 periods
    assert.strictEqual(dims[1], 100); // cols
    assert.strictEqual(batchValues.length, 6 * 100);
    assert.strictEqual(metadata.length, 6);
    
    // Check periods are correct
    const expectedPeriods = [5, 6, 7, 8, 9, 10];
    for (let i = 0; i < expectedPeriods.length; i++) {
        assert.strictEqual(metadata[i], expectedPeriods[i]);
    }
    
    // Verify first row matches individual calculation
    const firstRow = batchValues.slice(0, 100);
    const singleResult = wasm.smma(close, 5);
    assertArrayClose(firstRow, singleResult, 1e-10, "First row mismatch");
});

test('SMMA batch large range', () => {
    // Test batch with large period range
    const close = new Float64Array(testData.close.slice(0, 200)); // Use subset for speed
    
    const batchValues = wasm.smma_batch_legacy(close, 7, 100, 1); // Default range from Rust
    const metadata = wasm.smma_batch_metadata(7, 100, 1);
    const dims = wasm.smma_batch_rows_cols(7, 100, 1, 200);
    
    const expectedPeriodCount = 94; // 7 to 100 inclusive
    assert.strictEqual(dims[0], expectedPeriodCount);
    assert.strictEqual(dims[1], 200);
    assert.strictEqual(metadata.length, expectedPeriodCount);
});

test('SMMA edge case period one', () => {
    // Test SMMA with period=1 (edge case)
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    const result = wasm.smma(data, 1);
    
    // With period=1, SMMA should equal the input data
    assertArrayClose(result, data, 1e-10, "SMMA with period=1 should equal input");
});

test('SMMA constant values', () => {
    // Test SMMA with constant values
    const constantValue = 42.0;
    const data = new Float64Array(50);
    data.fill(constantValue);
    
    const result = wasm.smma(data, 10);
    
    // After warmup, all values should equal the constant
    for (let i = 10; i < result.length; i++) {
        assertClose(result[i], constantValue, 1e-10, 
                   `SMMA of constant should be ${constantValue} at index ${i}`);
    }
});

test('SMMA formula verification', () => {
    // Verify SMMA formula implementation
    const data = new Float64Array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]);
    const period = 3;
    
    const result = wasm.smma(data, period);
    
    // First period-1 values should be NaN (warmup = first + period - 1)
    assertAllNaN(result.slice(0, period - 1), "First period-1 values should be NaN");
    
    // First SMMA value (at index period-1) should be mean of first period values
    const expectedFirst = (10 + 12 + 14) / 3; // 12.0
    assertClose(result[period - 1], expectedFirst, 1e-10, "First SMMA value incorrect");
    
    // Second SMMA value follows the formula: (prev * (period - 1) + new_value) / period
    const expectedSecond = (expectedFirst * (period - 1) + data[period]) / period;
    // (12.0 * 2 + 16) / 3 = 40 / 3 = 13.333...
    assertClose(result[period], expectedSecond, 1e-10, "Second SMMA value incorrect");
});


test('SMMA warmup period', () => {
    // Test SMMA warmup period is correct
    const close = new Float64Array(testData.close.slice(0, 50));
    const period = 10;
    
    const result = wasm.smma(close, period);
    
    // First period-1 values should be NaN (warmup = first + period - 1)
    // Since first valid index is 0, warmup = 0 + period - 1 = period - 1
    assertAllNaN(result.slice(0, period - 1), `Expected NaN in first ${period - 1} values`);
    
    // Value at period-1 index should not be NaN
    assert(!isNaN(result[period - 1]), `Expected valid value at index ${period - 1}`);
});

test('SMMA batch step parameter', () => {
    // Test batch with step parameter
    const close = new Float64Array(testData.close.slice(0, 50));
    
    // Test with step=2 (periods 5, 7, 9)
    const batchValues = wasm.smma_batch_legacy(close, 5, 9, 2);
    const metadata = wasm.smma_batch_metadata(5, 9, 2);
    const dims = wasm.smma_batch_rows_cols(5, 9, 2, 50);
    
    assert.strictEqual(dims[0], 3);
    assert.strictEqual(metadata.length, 3);
    assert.strictEqual(metadata[0], 5);
    assert.strictEqual(metadata[1], 7);
    assert.strictEqual(metadata[2], 9);
});

test('SMMA batch zero step', () => {
    // Test batch with step=0 (single value)
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchValues = wasm.smma_batch_legacy(close, 7, 7, 0);
    const metadata = wasm.smma_batch_metadata(7, 7, 0);
    const dims = wasm.smma_batch_rows_cols(7, 7, 0, 50);
    
    assert.strictEqual(dims[0], 1);
    assert.strictEqual(metadata.length, 1);
    assert.strictEqual(metadata[0], 7);
});

// ============ Fast API Tests ============

test('SMMA fast API basic', () => {
    // Test fast API matches safe API
    const close = new Float64Array(testData.close.slice(0, 100));
    const period = 7;
    
    // Safe API result
    const safeResult = wasm.smma(close, period);
    
    // Fast API
    const inPtr = wasm.smma_alloc(close.length);
    const outPtr = wasm.smma_alloc(close.length);
    
    try {
        // Copy data to WASM memory
        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        memory.set(close, inPtr / 8);
        
        // Compute
        wasm.smma_into(inPtr, outPtr, close.length, period);
        
        // Get results
        const fastResult = new Float64Array(wasm.__wasm.memory.buffer, outPtr, close.length);
        
        // Compare results
        assertArrayClose(fastResult, safeResult, 1e-10, "Fast API should match safe API");
    } finally {
        wasm.smma_free(inPtr, close.length);
        wasm.smma_free(outPtr, close.length);
    }
});

test('SMMA fast API aliasing', () => {
    // Test fast API with aliasing (in-place computation)
    const close = new Float64Array(testData.close.slice(0, 50));
    const period = 7;
    
    // Get expected result
    const expected = wasm.smma(close, period);
    
    // Fast API with aliasing
    const ptr = wasm.smma_alloc(close.length);
    
    try {
        // Copy data to WASM memory
        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        memory.set(close, ptr / 8);
        
        // Compute in-place (input and output are the same)
        wasm.smma_into(ptr, ptr, close.length, period);
        
        // Get results
        const result = new Float64Array(wasm.__wasm.memory.buffer, ptr, close.length);
        
        // Compare results
        assertArrayClose(result, expected, 1e-10, "In-place computation should work correctly");
    } finally {
        wasm.smma_free(ptr, close.length);
    }
});

test('SMMA fast API null pointer', () => {
    // Test fast API with null pointers
    assert.throws(() => {
        wasm.smma_into(0, 0, 100, 7);
    }, /null pointer/);
});

test('SMMA batch new API', () => {
    // Test new batch API with config object
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const config = {
        period_range: [5, 10, 1]  // 6 values: 5, 6, 7, 8, 9, 10
    };
    
    const result = wasm.smma_batch(close, config);
    
    assert.strictEqual(result.rows, 6);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.values.length, 6 * 50);
    assert.strictEqual(result.combos.length, 6);
    
    // Check combos have correct periods
    for (let i = 0; i < 6; i++) {
        assert.strictEqual(result.combos[i].period, 5 + i);
    }
});

test('SMMA batch fast API', () => {
    // Test fast batch API
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const inPtr = wasm.smma_alloc(close.length);
    const outPtr = wasm.smma_alloc(6 * close.length); // 6 periods
    
    try {
        // Copy data to WASM memory
        const memory = new Float64Array(wasm.__wasm.memory.buffer);
        memory.set(close, inPtr / 8);
        
        // Compute batch
        const rows = wasm.smma_batch_into(inPtr, outPtr, close.length, 5, 10, 1);
        
        assert.strictEqual(rows, 6);
        
        // Get results
        const batchResult = new Float64Array(wasm.__wasm.memory.buffer, outPtr, rows * close.length);
        
        // Verify first row matches individual calculation
        const firstRow = batchResult.slice(0, close.length);
        const singleResult = wasm.smma(close, 5);
        assertArrayClose(firstRow, singleResult, 1e-10, "First batch row should match single calculation");
    } finally {
        wasm.smma_free(inPtr, close.length);
        wasm.smma_free(outPtr, 6 * close.length);
    }
});

test('SMMA streaming', () => {
    // Test SMMA streaming matches batch calculation - mirrors Python streaming test
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset
    const period = 7;
    
    // Batch calculation
    const batchResult = wasm.smma(close, period);
    
    // Simulate streaming (no streaming API in WASM, so we'll test incremental)
    const streamValues = new Float64Array(close.length);
    streamValues.fill(NaN);
    
    // Process data incrementally
    for (let i = period; i <= close.length; i++) {
        const partialData = close.slice(0, i);
        const partialResult = wasm.smma(partialData, period);
        streamValues[i - 1] = partialResult[i - 1];
    }
    
    // Compare batch vs "streaming" for valid values
    for (let i = period - 1; i < close.length; i++) {
        if (!isNaN(batchResult[i]) && !isNaN(streamValues[i])) {
            assertClose(batchResult[i], streamValues[i], 1e-10, 
                       `Streaming mismatch at index ${i}`);
        }
    }
});

test('SMMA batch validation', () => {
    // Test SMMA batch parameter validation
    const close = new Float64Array(testData.close.slice(0, 50));
    
    // Test invalid config (missing required fields)
    assert.throws(() => {
        wasm.smma_batch(close, {});
    }, /Invalid config/);
    
    // Test invalid period range
    assert.throws(() => {
        wasm.smma_batch(close, {
            period_range: [10, 5, 1]  // start > end
        });
    }, /Invalid|Error/);
    
    // Test zero period
    assert.throws(() => {
        wasm.smma_batch(close, {
            period_range: [0, 5, 1]
        });
    }, /Invalid period|unreachable/);
});

test('SMMA batch matches individual', () => {
    // Test that batch results match individual calculations
    const close = new Float64Array(testData.close.slice(0, 100)); // Use subset for speed
    
    // Test multiple periods
    const periods = [5, 7, 10, 14];
    const batchResult = wasm.smma_batch(close, {
        period_range: [5, 14, 1]
    });
    
    // Verify each period matches individual calculation
    for (const period of periods) {
        const rowIdx = period - 5; // Since we start at 5
        const rowStart = rowIdx * close.length;
        const rowEnd = rowStart + close.length;
        const batchRow = batchResult.values.slice(rowStart, rowEnd);
        
        // Calculate individual result
        const individualResult = wasm.smma(close, period);
        
        // They should match exactly
        assertArrayClose(
            batchRow,
            individualResult,
            1e-10,
            `Batch row for period ${period} doesn't match individual calculation`
        );
    }
});

test('SMMA leading NaNs', () => {
    // Test SMMA with leading NaN values in data
    const data = new Float64Array([NaN, NaN, NaN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    const period = 3;
    
    const result = wasm.smma(data, period);
    
    // First valid data point is at index 3
    // Warmup = first + period - 1 = 3 + 3 - 1 = 5
    // So indices 0-4 should be NaN, index 5 should be first valid
    assertAllNaN(result.slice(0, 5), "Expected NaN through index 4");
    assert(!isNaN(result[5]), "Expected valid value at index 5");
    
    // The first valid SMMA value should be mean of [1.0, 2.0, 3.0]
    const expectedFirst = (1.0 + 2.0 + 3.0) / 3;
    assertClose(result[5], expectedFirst, 1e-10, "First valid value incorrect");
});

test('SMMA batch fast API error handling', () => {
    // Test error conditions in batch_into
    const close = new Float64Array([1, 2, 3, 4, 5]);
    
    // Test with null pointers
    assert.throws(() => {
        wasm.smma_batch_into(0, 0, close.length, 2, 3, 1);
    }, /null pointer/);
    
    // Test with invalid period range
    const inPtr = wasm.smma_alloc(close.length);
    const outPtr = wasm.smma_alloc(close.length);
    
    try {
        assert.throws(() => {
            wasm.smma_batch_into(inPtr, outPtr, close.length, 10, 5, 1); // start > end
        });
    } finally {
        wasm.smma_free(inPtr, close.length);
        wasm.smma_free(outPtr, close.length);
    }
});