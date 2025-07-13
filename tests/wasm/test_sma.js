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
    
    // Expected values from Rust test
    const expectedLastFive = [59180.8, 59175.0, 59129.4, 59085.4, 59133.7];
    
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
    });
});

test('SMA period exceeds length', () => {
    // Test SMA fails with period exceeding length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.sma(dataSmall, 10);
    });
});

test('SMA very small dataset', () => {
    // Test SMA fails with insufficient data
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.sma(singlePoint, 9);
    });
});

test('SMA empty input', () => {
    // Test SMA with empty input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.sma(dataEmpty, 14);
    });
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
        assertClose(batch_first[i], individual_result[i], 1e-9, `Batch mismatch at ${i}`);
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
    }, /All values are NaN/);
});

test('SMA batch error conditions', () => {
    // Test various error conditions for batch
    const data = new Float64Array([1, 2, 3, 4, 5]);
    
    // Period exceeds data length
    assert.throws(() => {
        wasm.smaBatch(data, 10, 20, 5);
    });
    
    // Empty data
    const empty = new Float64Array([]);
    assert.throws(() => {
        wasm.smaBatch(empty, 10, 20, 5);
    });
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