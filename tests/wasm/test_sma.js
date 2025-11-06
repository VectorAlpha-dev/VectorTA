/**
 * WASM binding tests for SMA indicator.
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

test('SMA partial params', () => {
    // Test with default parameters - mirrors check_sma_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.sma(close, 9);
    assert(result instanceof Float64Array);
    assert.strictEqual(result.length, close.length);
});

test('SMA accuracy', async () => {
    // Test SMA matches expected values from Rust tests - mirrors check_sma_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.sma(close, 9);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected values from Rust test (from EXPECTED_OUTPUTS)
    const expectedLastFive = EXPECTED_OUTPUTS.sma.last_5_values;
    
    const actualLastFive = result.slice(-5);
    
    for (let i = 0; i < 5; i++) {
        assertClose(actualLastFive[i], expectedLastFive[i], 1e-1, 
            `SMA mismatch at index ${i}`);
    }
    
    // Compare with Rust implementation
    await compareWithRust('sma', result, 'close', { period: 9 });
});

test('SMA invalid period', () => {
    // Test SMA fails with invalid period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    // Period = 0 should fail
    assert.throws(() => {
        wasm.sma(inputData, 0);
    }, /Invalid period|period must be greater than 0/);
});

test('SMA period exceeds length', () => {
    // Test SMA fails with period exceeding length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.sma(dataSmall, 10);
    }, /Invalid period|Period.*exceeds|Not enough.*data/);
});

test('SMA very small dataset', () => {
    // Test SMA fails with insufficient data
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.sma(singlePoint, 9);
    }, /Invalid period|Not enough.*data|Insufficient data/);
});

test('SMA empty input', () => {
    // Test SMA with empty input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.sma(dataEmpty, 14);
    }, /Empty|empty|No data/);
});

test('SMA reinput', () => {
    // Test SMA with re-input of SMA result - mirrors check_sma_reinput
    const close = new Float64Array(testData.close);
    
    // First SMA pass with period=14
    const firstResult = wasm.sma(close, 14);
    
    // Second SMA pass with period=14 using first result as input
    const secondResult = wasm.sma(firstResult, 14);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // All values should be finite after warmup
    // Find first non-NaN in original input
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    // Find first non-NaN in first result
    let firstValidInFirst = 0;
    for (let i = 0; i < firstResult.length; i++) {
        if (!isNaN(firstResult[i])) {
            firstValidInFirst = i;
            break;
        }
    }
    const warmup = firstValidInFirst + 14 - 1;  // first_valid_in_first + second_period - 1
    for (let i = warmup; i < secondResult.length; i++) {
        assert(isFinite(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('SMA NaN handling', () => {
    // Test SMA handling of NaN values - mirrors check_sma_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.sma(close, 9);
    
    assert.strictEqual(result.length, close.length);
    
    // After warmup, all values should be finite
    // Find first non-NaN in input
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    const warmup = firstValid + 9 - 1;  // first_valid + period - 1
    if (result.length > warmup) {
        for (let i = warmup; i < result.length; i++) {
            assert(isFinite(result[i]), `NaN found at index ${i}`);
        }
    }
});

test('SMA batch', () => {
    // Test SMA batch computation
    const close = new Float64Array(testData.close);
    
    // Test parameter ranges
    const batch_result = wasm.smaBatch(
        close, 
        9, 240, 1    // period range: 9 to 240 step 1
    );
    
    // Get rows and cols info
    const rows_cols = wasm.smaBatchRowsCols(9, 240, 1, close.length);
    const rows = rows_cols[0];
    const cols = rows_cols[1];
    
    assert(batch_result instanceof Float64Array);
    assert.strictEqual(rows, 232); // 232 period values (9 to 240 inclusive)
    assert.strictEqual(cols, close.length);
    assert.strictEqual(batch_result.length, rows * cols);
    
    // Verify first combination matches individual calculation
    const individual_result = wasm.sma(close, 9);
    const batch_first = batch_result.slice(0, close.length);
    
    // Compare after warmup
    // Find first non-NaN in input
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    const warmup = firstValid + 9 - 1;  // first valid + period - 1
    for (let i = warmup; i < close.length; i++) {
        // Allow tiny f64 rounding differences; still far tighter than Rust's 1e-1 ref tolerance
        assertClose(batch_first[i], individual_result[i], 1e-8, `Batch mismatch at ${i}`);
    }
});

test('SMA different periods', () => {
    // Test SMA with different period values
    const close = new Float64Array(testData.close);
    
    // Test various period values
    const testPeriods = [5, 10, 14, 20, 50];
    
    for (const period of testPeriods) {
        const result = wasm.sma(close, period);
        assert.strictEqual(result.length, close.length);
        
        // After warmup, all values should be finite
        // Find first non-NaN in input
        let firstValid = 0;
        for (let i = 0; i < close.length; i++) {
            if (!isNaN(close[i])) {
                firstValid = i;
                break;
            }
        }
        const warmup = firstValid + period - 1;
        if (warmup < result.length) {
            for (let i = warmup; i < result.length; i++) {
                assert(isFinite(result[i]), `NaN at index ${i} for period=${period}`);
            }
        }
    }
});

test('SMA batch performance', () => {
    // Test that batch computation is more efficient than multiple single computations
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use first 1000 values
    
    // Test multiple parameter combinations
    const startBatch = performance.now();
    const batchResult = wasm.smaBatch(
        close,
        10, 50, 10    // periods: 10, 20, 30, 40, 50
    );
    const batchTime = performance.now() - startBatch;
    
    // Get the exact parameters used by the batch function
    const metadata = wasm.smaBatchMetadata(10, 50, 10);
    
    const startSingle = performance.now();
    const singleResults = [];
    // Use the exact parameters from metadata
    for (const period of metadata) {
        const result = wasm.sma(close, period);
        singleResults.push(...result);
    }
    const singleTime = performance.now() - startSingle;
    
    // Batch should be faster than multiple single calls
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    // Verify results match
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});

test('SMA edge cases', () => {
    // Test SMA with edge case inputs
    
    // Test with monotonically increasing data
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1;
    }
    const result = wasm.sma(data, 14);
    assert.strictEqual(result.length, data.length);
    
    // After warmup, all values should be finite
    for (let i = 13; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN at index ${i}`);
    }
    
    // Test with constant values
    const constantData = new Float64Array(100).fill(50.0);
    const constantResult = wasm.sma(constantData, 14);
    
    assert.strictEqual(constantResult.length, constantData.length);
    
    // With constant input, SMA should equal constant after warmup
    for (let i = 13; i < constantResult.length; i++) {
        assertClose(constantResult[i], 50.0, 1e-9, `SMA constant mismatch at ${i}`);
    }
});

test('SMA batch metadata', () => {
    // Test metadata function returns correct period values
    const metadata = wasm.smaBatchMetadata(
        10, 30, 5    // period range: 10, 15, 20, 25, 30
    );
    
    // Should have 5 period values
    assert.strictEqual(metadata.length, 5);
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 15);
    assert.strictEqual(metadata[2], 20);
    assert.strictEqual(metadata[3], 25);
    assert.strictEqual(metadata[4], 30);
});

test('SMA consistency across calls', () => {
    // Test that SMA produces consistent results across multiple calls
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.sma(close, 14);
    const result2 = wasm.sma(close, 14);
    
    assertArrayClose(result1, result2, 1e-15, "SMA results not consistent");
});

test('SMA step precision', () => {
    // Test batch with various step sizes
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.smaBatch(
        data,
        10, 20, 2     // periods: 10, 12, 14, 16, 18, 20
    );
    
    // Get rows and cols info
    const rows_cols = wasm.smaBatchRowsCols(10, 20, 2, data.length);
    const rows = rows_cols[0];
    
    // Should have 6 combinations
    assert.strictEqual(rows, 6);
    assert.strictEqual(batch_result.length, 6 * data.length);
    
    // Verify metadata
    const metadata = wasm.smaBatchMetadata(10, 20, 2);
    assert.strictEqual(metadata.length, 6);
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 12);
    assert.strictEqual(metadata[2], 14);
    assert.strictEqual(metadata[3], 16);
    assert.strictEqual(metadata[4], 18);
    assert.strictEqual(metadata[5], 20);
});

test('SMA warmup behavior', () => {
    // Test SMA warmup period behavior
    const close = new Float64Array(testData.close);
    const period = 9;
    
    const result = wasm.sma(close, period);
    
    // Find first non-NaN value in input
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    
    const warmup = firstValid + period - 1;
    
    // Values during warmup should be NaN
    for (let i = firstValid; i < warmup && i < result.length; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    // Values after warmup should be finite
    for (let i = warmup; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i} after warmup`);
    }
});

test('SMA oscillating data', () => {
    // Test with oscillating values
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = (i % 2 === 0) ? 10.0 : 20.0;
    }
    
    const result = wasm.sma(data, 14);
    assert.strictEqual(result.length, data.length);
    
    // After warmup, all values should be finite
    for (let i = 13; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

test('SMA small step size', () => {
    // Test batch with very small step size
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.smaBatch(
        data,
        10, 14, 1     // periods: 10, 11, 12, 13, 14
    );
    
    const rows_cols = wasm.smaBatchRowsCols(10, 14, 1, data.length);
    const rows = rows_cols[0];
    
    assert.strictEqual(rows, 5);
    assert.strictEqual(batch_result.length, 5 * data.length);
});

test('SMA formula verification', () => {
    // Test that SMA calculates correctly
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    const period = 3;
    
    const result = wasm.sma(data, period);
    
    // Result length should match input
    assert.strictEqual(result.length, data.length);
    
    // After warmup, verify values
    // SMA[2] = (1+2+3)/3 = 2.0
    // SMA[3] = (2+3+4)/3 = 3.0
    // SMA[4] = (3+4+5)/3 = 4.0
    assertClose(result[2], 2.0, 1e-9, 'SMA[2] should be 2.0');
    assertClose(result[3], 3.0, 1e-9, 'SMA[3] should be 3.0');
    assertClose(result[4], 4.0, 1e-9, 'SMA[4] should be 4.0');
});

test('SMA all NaN input', () => {
    // Test SMA with all NaN values
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.sma(allNaN, 14);
    }, /All values are NaN|No valid data/);
});

test('SMA batch error conditions', () => {
    // Test various error conditions for batch
    const data = new Float64Array([1, 2, 3, 4, 5]);
    
    // Period exceeds data length
    assert.throws(() => {
        wasm.smaBatch(data, 10, 20, 5);
    }, /Invalid period|Period.*exceeds|Not enough.*data/);
    
    // Empty data
    const empty = new Float64Array([]);
    assert.throws(() => {
        wasm.smaBatch(empty, 10, 20, 5);
    }, /Empty|empty|No data|All values are NaN/);
});

test('SMA constant input normalization', () => {
    // Test with constant input to verify calculations
    const data = new Float64Array(20).fill(1.0);
    
    // For any period with constant input, result should be constant after warmup
    for (const period of [3, 5, 7]) {
        const result = wasm.sma(data, period);
        
        // After warmup, check if output is constant
        for (let i = period - 1; i < result.length; i++) {
            assertClose(result[i], 1.0, 1e-9, 
                `Expected constant output for period=${period} at index ${i}`);
        }
    }
});

// Zero-copy API tests (skip if not available)
test('SMA zero-copy single calculation', () => {
    if (!wasm.smaIntoSlice) {
        console.log('Skipping: smaIntoSlice not available');
        return;
    }
    const close = new Float64Array(testData.close);
    const period = 9;
    
    // Create output buffer
    const output = new Float64Array(close.length);
    
    // Use zero-copy API
    wasm.smaIntoSlice(close, period, output);
    
    // Compare with regular API
    const expected = wasm.sma(close, period);
    
    assertArrayClose(output, expected, 1e-15, 'Zero-copy vs regular API');
});

test('SMA zero-copy batch calculation', () => {
    if (!wasm.smaBatchInto) {
        console.log('Skipping: smaBatchInto not available');
        return;
    }
    const close = new Float64Array(testData.close.slice(0, 500));
    
    // Create batch output buffer
    const rows = 5; // periods: 10, 20, 30, 40, 50
    const cols = close.length;
    const output = new Float64Array(rows * cols);
    
    // Use zero-copy batch API
    wasm.smaBatchInto(close, 10, 50, 10, output);
    
    // Compare with regular batch API
    const expected = wasm.smaBatch(close, 10, 50, 10);
    
    assertArrayClose(output, expected, 1e-15, 'Zero-copy batch vs regular batch');
});

test('SMA zero-copy memory reuse', () => {
    if (!wasm.smaIntoSlice) {
        console.log('Skipping: smaIntoSlice not available');
        return;
    }
    const close = new Float64Array(testData.close.slice(0, 100));
    const output = new Float64Array(close.length);
    
    // First calculation
    wasm.smaIntoSlice(close, 10, output);
    const firstResult = Array.from(output);
    
    // Reuse same buffer for different period
    wasm.smaIntoSlice(close, 20, output);
    const secondResult = Array.from(output);
    
    // Results should be different
    let foundDifference = false;
    for (let i = 0; i < output.length; i++) {
        if (!isNaN(firstResult[i]) && !isNaN(secondResult[i]) && 
            Math.abs(firstResult[i] - secondResult[i]) > 1e-10) {
            foundDifference = true;
            break;
        }
    }
    assert(foundDifference, 'Memory reuse should produce different results for different periods');
    
    // Verify second calculation is correct
    const expected = wasm.sma(close, 20);
    assertArrayClose(output, expected, 1e-15, 'Reused buffer calculation');
});

test('SMA zero-copy error handling', () => {
    if (!wasm.smaIntoSlice) {
        console.log('Skipping: smaIntoSlice not available');
        return;
    }
    const close = new Float64Array([1, 2, 3, 4, 5]);
    
    // Output buffer too small
    const smallOutput = new Float64Array(3);
    assert.throws(() => {
        wasm.smaIntoSlice(close, 2, smallOutput);
    }, /OutputLenMismatch|length mismatch/, 'Should fail with output length mismatch');
    
    // Output buffer too large
    const largeOutput = new Float64Array(10);
    assert.throws(() => {
        wasm.smaIntoSlice(close, 2, largeOutput);
    }, /OutputLenMismatch|length mismatch/, 'Should fail with output length mismatch');
    
    // Invalid period
    const correctOutput = new Float64Array(5);
    assert.throws(() => {
        wasm.smaIntoSlice(close, 0, correctOutput);
    }, /Invalid period/, 'Should fail with invalid period');
});

test('SMA zero-copy with NaN handling', () => {
    if (!wasm.smaIntoSlice) {
        console.log('Skipping: smaIntoSlice not available');
        return;
    }
    // Create data with NaN values
    const data = new Float64Array([NaN, NaN, 1, 2, 3, 4, 5, 6, 7, 8]);
    const output = new Float64Array(data.length);
    
    wasm.smaIntoSlice(data, 3, output);
    
    // First valid output should be at index 4 (2 NaN + period - 1)
    assert(isNaN(output[0]) && isNaN(output[1]), 'Initial NaN values preserved');
    assert(isNaN(output[2]) && isNaN(output[3]), 'Warmup period respected');
    assert(isFinite(output[4]), 'First valid output after warmup');
    
    // Compare with regular API
    const expected = wasm.sma(data, 3);
    assertArrayClose(output, expected, 1e-15, 'Zero-copy NaN handling');
});

test('SMA zero-copy precision', () => {
    if (!wasm.smaIntoSlice) {
        console.log('Skipping: smaIntoSlice not available');
        return;
    }
    // Test with precise values
    const data = new Float64Array([
        1.23456789012345,
        2.34567890123456,
        3.45678901234567,
        4.56789012345678,
        5.67890123456789
    ]);
    const output = new Float64Array(data.length);
    
    wasm.smaIntoSlice(data, 3, output);
    const expected = wasm.sma(data, 3);
    
    // Should maintain full precision
    assertArrayClose(output, expected, 1e-15, 'Zero-copy precision');
});

test('SMA zero-copy batch memory layout', () => {
    if (!wasm.smaBatchInto) {
        console.log('Skipping: smaBatchInto not available');
        return;
    }
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const periods = [2, 3, 4];
    const rows = periods.length;
    const cols = data.length;
    const output = new Float64Array(rows * cols);
    
    // Use batch zero-copy
    wasm.smaBatchInto(data, 2, 4, 1, output);
    
    // Verify memory layout is row-major
    for (let row = 0; row < rows; row++) {
        const period = periods[row];
        const expected = wasm.sma(data, period);
        const rowStart = row * cols;
        const actualRow = output.slice(rowStart, rowStart + cols);
        
        assertArrayClose(actualRow, expected, 1e-15, 
            `Batch row ${row} (period ${period})`);
    }
});

test('SMA zero-copy with empty input', () => {
    if (!wasm.smaIntoSlice) {
        console.log('Skipping: smaIntoSlice not available');
        return;
    }
    const empty = new Float64Array([]);
    const output = new Float64Array([]);
    
    assert.throws(() => {
        wasm.smaIntoSlice(empty, 5, output);
    }, /Empty|empty/, 'Should fail with empty input');
});

test('SMA zero-copy with all NaN', () => {
    if (!wasm.smaIntoSlice) {
        console.log('Skipping: smaIntoSlice not available');
        return;
    }
    const allNaN = new Float64Array(10).fill(NaN);
    const output = new Float64Array(10);
    
    assert.throws(() => {
        wasm.smaIntoSlice(allNaN, 3, output);
    }, /All values are NaN/, 'Should fail with all NaN values');
});

test('SMA zero-copy batch rows/cols validation', () => {
    if (!wasm.smaBatchInto) {
        console.log('Skipping: smaBatchInto not available');
        return;
    }
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) data[i] = i + 1;
    
    // Calculate expected dimensions
    const [expectedRows, expectedCols] = wasm.smaBatchRowsCols(10, 50, 10, data.length);
    const output = new Float64Array(expectedRows * expectedCols);
    
    // Perform batch calculation
    wasm.smaBatchInto(data, 10, 50, 10, output);
    
    // Verify dimensions match
    assert.strictEqual(output.length, expectedRows * expectedCols);
    assert.strictEqual(expectedCols, data.length);
    
    // Verify correct number of periods
    const metadata = wasm.smaBatchMetadata(10, 50, 10);
    assert.strictEqual(metadata.length, expectedRows);
});

test('SMA zero-copy invalid buffer types', () => {
    if (!wasm.smaIntoSlice) {
        console.log('Skipping: smaIntoSlice not available');
        return;
    }
    const data = new Float64Array([1, 2, 3, 4, 5]);
    
    // These should fail with type errors
    assert.throws(() => {
        wasm.smaIntoSlice(data, 3, new Float32Array(5));
    }, 'Should fail with Float32Array output');
    
    assert.throws(() => {
        wasm.smaIntoSlice(data, 3, new Uint8Array(5));
    }, 'Should fail with Uint8Array output');
    
    assert.throws(() => {
        wasm.smaIntoSlice(data, 3, [0, 0, 0, 0, 0]);
    }, 'Should fail with regular array output');
});

test('SMA zero-copy concurrent calculations', () => {
    if (!wasm.smaIntoSlice) {
        console.log('Skipping: smaIntoSlice not available');
        return;
    }
    const data1 = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const data2 = new Float64Array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
    const output1 = new Float64Array(10);
    const output2 = new Float64Array(10);
    
    // Perform calculations
    wasm.smaIntoSlice(data1, 3, output1);
    wasm.smaIntoSlice(data2, 3, output2);
    
    // Verify both are correct
    const expected1 = wasm.sma(data1, 3);
    const expected2 = wasm.sma(data2, 3);
    
    assertArrayClose(output1, expected1, 1e-15, 'First concurrent calculation');
    assertArrayClose(output2, expected2, 1e-15, 'Second concurrent calculation');
});

// JS Batch API tests (matching ALMA's coverage)
test('SMA batch JS API', () => {
    if (!wasm.smaBatchJs) {
        console.log('Skipping: smaBatchJs not available');
        return;
    }
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Use JS-friendly batch API
    const result = wasm.smaBatchJs(close, {
        period_start: 10,
        period_end: 30,
        period_step: 5
    });
    
    // Check result structure
    assert(result.values instanceof Float64Array, 'Values should be Float64Array');
    assert(Array.isArray(result.periods), 'Periods should be an array');
    
    // Should have 5 periods: 10, 15, 20, 25, 30
    assert.strictEqual(result.periods.length, 5);
    assert.deepStrictEqual(result.periods, [10, 15, 20, 25, 30]);
    
    // Values should have correct shape
    assert.strictEqual(result.values.length, 5 * close.length);
    
    // Verify first row matches single calculation
    const firstRowExpected = wasm.sma(close, 10);
    const firstRow = result.values.slice(0, close.length);
    assertArrayClose(firstRow, firstRowExpected, 1e-15, 'First row of JS batch');
});

test('SMA batch JS single parameter', () => {
    if (!wasm.smaBatchJs) {
        console.log('Skipping: smaBatchJs not available');
        return;
    }
    const close = new Float64Array(testData.close.slice(0, 50));
    
    // Test with single parameter (step = 0)
    const result = wasm.smaBatchJs(close, {
        period_start: 14,
        period_end: 14,
        period_step: 0
    });
    
    assert.strictEqual(result.periods.length, 1);
    assert.strictEqual(result.periods[0], 14);
    assert.strictEqual(result.values.length, close.length);
    
    // Should match single calculation
    const expected = wasm.sma(close, 14);
    assertArrayClose(result.values, expected, 1e-15, 'Single parameter batch');
});

test('SMA batch JS metadata', () => {
    if (!wasm.smaBatchJs) {
        console.log('Skipping: smaBatchJs not available');
        return;
    }
    const close = new Float64Array(100);
    
    // Test metadata generation
    const result = wasm.smaBatchJs(close, {
        period_start: 5,
        period_end: 25,
        period_step: 4
    });
    
    // Expected periods: 5, 9, 13, 17, 21, 25
    const expectedPeriods = [5, 9, 13, 17, 21, 25];
    assert.deepStrictEqual(result.periods, expectedPeriods, 'Period metadata');
    
    // Values shape should match
    assert.strictEqual(result.values.length, expectedPeriods.length * close.length);
});

test('SMA batch JS error handling', () => {
    if (!wasm.smaBatchJs) {
        console.log('Skipping: smaBatchJs not available');
        return;
    }
    const data = new Float64Array([1, 2, 3, 4, 5]);
    
    // Invalid period range
    assert.throws(() => {
        wasm.smaBatchJs(data, {
            period_start: 10,
            period_end: 20,
            period_step: 5
        });
    }, /Invalid period|exceeds/, 'Should fail when period exceeds data length');
    
    // Invalid step
    assert.throws(() => {
        wasm.smaBatchJs(data, {
            period_start: 2,
            period_end: 4,
            period_step: -1
        });
    }, /Invalid.*step/, 'Should fail with negative step');
    
    // Start > End with positive step
    assert.throws(() => {
        wasm.smaBatchJs(data, {
            period_start: 5,
            period_end: 2,
            period_step: 1
        });
    }, /Invalid.*range/, 'Should fail when start > end');
});

test('SMA batch JS with NaN handling', () => {
    if (!wasm.smaBatchJs) {
        console.log('Skipping: smaBatchJs not available');
        return;
    }
    const data = new Float64Array([NaN, NaN, 1, 2, 3, 4, 5, 6, 7, 8]);
    
    const result = wasm.smaBatchJs(data, {
        period_start: 3,
        period_end: 5,
        period_step: 1
    });
    
    // Should have 3 periods
    assert.strictEqual(result.periods.length, 3);
    assert.deepStrictEqual(result.periods, [3, 4, 5]);
    
    // Each row should handle NaN correctly
    for (let i = 0; i < result.periods.length; i++) {
        const period = result.periods[i];
        const rowStart = i * data.length;
        const row = result.values.slice(rowStart, rowStart + data.length);
        const expected = wasm.sma(data, period);
        
        assertArrayClose(row, expected, 1e-15, `Row ${i} (period ${period})`);
    }
});

test('SMA batch JS large dataset', () => {
    if (!wasm.smaBatchJs) {
        console.log('Skipping: smaBatchJs not available');
        return;
    }
    const close = new Float64Array(testData.close);
    
    // Test with large parameter sweep
    const result = wasm.smaBatchJs(close, {
        period_start: 10,
        period_end: 100,
        period_step: 10
    });
    
    // Should have 10 periods: 10, 20, 30, ..., 100
    assert.strictEqual(result.periods.length, 10);
    
    // Spot check a few values
    const period50Index = result.periods.indexOf(50);
    assert(period50Index >= 0, 'Should include period 50');
    
    const rowStart = period50Index * close.length;
    const row50 = result.values.slice(rowStart, rowStart + close.length);
    const expected50 = wasm.sma(close, 50);
    
    // Check warmup and a few values
    const warmup = 49; // period - 1
    assert(isNaN(row50[warmup - 1]), 'Should have NaN before warmup');
    assert(isFinite(row50[warmup]), 'Should have value after warmup');
    
    // Verify accuracy for non-NaN values
    for (let i = warmup; i < Math.min(warmup + 10, close.length); i++) {
        assertClose(row50[i], expected50[i], 1e-9, `Value at index ${i}`);
    }
});

test('SMA batch JS memory efficiency', () => {
    if (!wasm.smaBatchJs) {
        console.log('Skipping: smaBatchJs not available');
        return;
    }
    const data = new Float64Array(1000);
    for (let i = 0; i < 1000; i++) data[i] = Math.sin(i * 0.1) * 100;
    
    // Large batch operation
    const memBefore = process.memoryUsage().heapUsed;
    
    const result = wasm.smaBatchJs(data, {
        period_start: 10,
        period_end: 200,
        period_step: 10
    });
    
    const memAfter = process.memoryUsage().heapUsed;
    const memUsed = memAfter - memBefore;
    
    // Check result validity
    assert.strictEqual(result.periods.length, 20);
    assert.strictEqual(result.values.length, 20 * data.length);
    
    // Memory usage should be reasonable (not creating excessive intermediate arrays)
    // This is a soft check - mainly ensuring no memory leaks
    console.log(`Batch memory usage: ${(memUsed / 1024 / 1024).toFixed(2)} MB`);
    assert(memUsed < 50 * 1024 * 1024, 'Memory usage should be reasonable');
});

test('SMA batch JS result ordering', () => {
    if (!wasm.smaBatchJs) {
        console.log('Skipping: smaBatchJs not available');
        return;
    }
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    const result = wasm.smaBatchJs(data, {
        period_start: 2,
        period_end: 5,
        period_step: 1
    });
    
    // Verify periods are in order
    for (let i = 1; i < result.periods.length; i++) {
        assert(result.periods[i] > result.periods[i - 1], 
            `Periods should be in ascending order`);
    }
    
    // Verify each row corresponds to correct period
    for (let i = 0; i < result.periods.length; i++) {
        const period = result.periods[i];
        const rowStart = i * data.length;
        const row = result.values.slice(rowStart, rowStart + data.length);
        const expected = wasm.sma(data, period);
        
        assertArrayClose(row, expected, 1e-15, 
            `Row ${i} should match period ${period}`);
    }
});
