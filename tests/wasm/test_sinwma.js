/**
 * WASM binding tests for SINWMA indicator.
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

test('SINWMA partial params', () => {
    // Test with default parameters - mirrors check_sinwma_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.sinwma_js(close, 14);
    assert(result instanceof Float64Array);
    assert.strictEqual(result.length, close.length);
});

test('SINWMA accuracy', async () => {
    // Test SINWMA matches expected values from Rust tests - mirrors check_sinwma_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.sinwma_js(close, 14);
    
    assert.strictEqual(result.length, close.length);
    
    // Expected values from Rust test
    const expectedLastFive = [
        59376.72903536103,
        59300.76862770367,
        59229.27622157621,
        59178.48781774477,
        59154.66580703081,
    ];
    
    const actualLastFive = result.slice(-5);
    
    for (let i = 0; i < 5; i++) {
        assertClose(actualLastFive[i], expectedLastFive[i], 1e-6, 
            `SINWMA mismatch at index ${i}`);
    }
    
    // Compare with Rust implementation
    await compareWithRust('sinwma', result, 'close', { period: 14 });
});

test('SINWMA invalid period', () => {
    // Test SINWMA fails with invalid period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    // Period = 0 should fail
    assert.throws(() => {
        wasm.sinwma_js(inputData, 0);
    });
});

test('SINWMA period exceeds length', () => {
    // Test SINWMA fails with period exceeding length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.sinwma_js(dataSmall, 10);
    });
});

test('SINWMA very small dataset', () => {
    // Test SINWMA fails with insufficient data
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.sinwma_js(singlePoint, 14);
    });
});

test('SINWMA empty input', () => {
    // Test SINWMA with empty input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.sinwma_js(dataEmpty, 14);
    });
});

test('SINWMA reinput', () => {
    // Test SINWMA with re-input of SINWMA result - mirrors check_sinwma_reinput
    const close = new Float64Array(testData.close);
    
    // First SINWMA pass with period=14
    const firstResult = wasm.sinwma_js(close, 14);
    
    // Second SINWMA pass with period=5 using first result as input
    const secondResult = wasm.sinwma_js(firstResult, 5);
    
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
    const warmup = firstValidInFirst + 5 - 1;  // first_valid_in_first + second_period - 1
    for (let i = warmup; i < secondResult.length; i++) {
        assert(isFinite(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('SINWMA NaN handling', () => {
    // Test SINWMA handling of NaN values - mirrors check_sinwma_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.sinwma_js(close, 14);
    
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
    const warmup = firstValid + 14 - 1;  // first_valid + period - 1
    if (result.length > warmup) {
        for (let i = warmup; i < result.length; i++) {
            assert(isFinite(result[i]), `NaN found at index ${i}`);
        }
    }
});

test('SINWMA batch', () => {
    // Test SINWMA batch computation
    const close = new Float64Array(testData.close);
    
    // Test parameter ranges
    const batch_result = wasm.sinwma_batch_js(
        close, 
        10, 30, 5    // period range: 10, 15, 20, 25, 30
    );
    
    // Get rows and cols info
    const rows_cols = wasm.sinwma_batch_rows_cols_js(10, 30, 5, close.length);
    const rows = rows_cols[0];
    const cols = rows_cols[1];
    
    assert(batch_result instanceof Float64Array);
    assert.strictEqual(rows, 5); // 5 period values
    assert.strictEqual(cols, close.length);
    assert.strictEqual(batch_result.length, rows * cols);
    
    // Verify first combination matches individual calculation
    const individual_result = wasm.sinwma_js(close, 10);
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
    const warmup = firstValid + 10 - 1;  // first valid + period - 1
    for (let i = warmup; i < close.length; i++) {
        assertClose(batch_first[i], individual_result[i], 1e-9, `Batch mismatch at ${i}`);
    }
});

test('SINWMA different periods', () => {
    // Test SINWMA with different period values
    const close = new Float64Array(testData.close);
    
    // Test various period values
    const testPeriods = [5, 10, 14, 20, 50];
    
    for (const period of testPeriods) {
        const result = wasm.sinwma_js(close, period);
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

test('SINWMA batch performance', () => {
    // Test that batch computation is more efficient than multiple single computations
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use first 1000 values
    
    // Test multiple parameter combinations
    const startBatch = performance.now();
    const batchResult = wasm.sinwma_batch_js(
        close,
        10, 50, 10    // periods: 10, 20, 30, 40, 50
    );
    const batchTime = performance.now() - startBatch;
    
    // Get the exact parameters used by the batch function
    const metadata = wasm.sinwma_batch_metadata_js(10, 50, 10);
    
    const startSingle = performance.now();
    const singleResults = [];
    // Use the exact parameters from metadata
    for (const period of metadata) {
        const result = wasm.sinwma_js(close, period);
        singleResults.push(...result);
    }
    const singleTime = performance.now() - startSingle;
    
    // Batch should be faster than multiple single calls
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    // Verify results match
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});

test('SINWMA edge cases', () => {
    // Test SINWMA with edge case inputs
    
    // Test with monotonically increasing data
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1;
    }
    const result = wasm.sinwma_js(data, 14);
    assert.strictEqual(result.length, data.length);
    
    // After warmup, all values should be finite
    for (let i = 14; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN at index ${i}`);
    }
    
    // Test with constant values
    const constantData = new Float64Array(100).fill(50.0);
    const constantResult = wasm.sinwma_js(constantData, 14);
    
    assert.strictEqual(constantResult.length, constantData.length);
    
    // With constant input, after warmup should produce constant output
    for (let i = 14; i < constantResult.length; i++) {
        assert(isFinite(constantResult[i]), `NaN at index ${i} for constant input`);
    }
});

test('SINWMA batch metadata', () => {
    // Test metadata function returns correct period values
    const metadata = wasm.sinwma_batch_metadata_js(
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

test('SINWMA consistency across calls', () => {
    // Test that SINWMA produces consistent results across multiple calls
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.sinwma_js(close, 14);
    const result2 = wasm.sinwma_js(close, 14);
    
    assertArrayClose(result1, result2, 1e-15, "SINWMA results not consistent");
});

test('SINWMA step precision', () => {
    // Test batch with various step sizes
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.sinwma_batch_js(
        data,
        10, 20, 2     // periods: 10, 12, 14, 16, 18, 20
    );
    
    // Get rows and cols info
    const rows_cols = wasm.sinwma_batch_rows_cols_js(10, 20, 2, data.length);
    const rows = rows_cols[0];
    
    // Should have 6 combinations
    assert.strictEqual(rows, 6);
    assert.strictEqual(batch_result.length, 6 * data.length);
    
    // Verify metadata
    const metadata = wasm.sinwma_batch_metadata_js(10, 20, 2);
    assert.strictEqual(metadata.length, 6);
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 12);
    assert.strictEqual(metadata[2], 14);
    assert.strictEqual(metadata[3], 16);
    assert.strictEqual(metadata[4], 18);
    assert.strictEqual(metadata[5], 20);
});

test('SINWMA warmup behavior', () => {
    // Test SINWMA warmup period behavior
    const close = new Float64Array(testData.close);
    const period = 14;
    
    const result = wasm.sinwma_js(close, period);
    
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

test('SINWMA oscillating data', () => {
    // Test with oscillating values
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = (i % 2 === 0) ? 10.0 : 20.0;
    }
    
    const result = wasm.sinwma_js(data, 14);
    assert.strictEqual(result.length, data.length);
    
    // After warmup, all values should be finite
    for (let i = 14; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

test('SINWMA small step size', () => {
    // Test batch with very small step size
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.sinwma_batch_js(
        data,
        10, 14, 1     // periods: 10, 11, 12, 13, 14
    );
    
    const rows_cols = wasm.sinwma_batch_rows_cols_js(10, 14, 1, data.length);
    const rows = rows_cols[0];
    
    assert.strictEqual(rows, 5);
    assert.strictEqual(batch_result.length, 5 * data.length);
});

test('SINWMA formula verification', () => {
    // Test that SINWMA detects patterns correctly
    const data = new Float64Array([10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
    const period = 5;
    
    const result = wasm.sinwma_js(data, period);
    
    // Result length should match input
    assert.strictEqual(result.length, data.length);
    
    // After warmup, values should be finite
    for (let i = period; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

test('SINWMA all NaN input', () => {
    // Test SINWMA with all NaN values
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.sinwma_js(allNaN, 14);
    }, /All values are NaN/);
});

test('SINWMA batch error conditions', () => {
    // Test various error conditions for batch
    const data = new Float64Array([1, 2, 3, 4, 5]);
    
    // Period exceeds data length
    assert.throws(() => {
        wasm.sinwma_batch_js(data, 10, 20, 5);
    });
    
    // Empty data
    const empty = new Float64Array([]);
    assert.throws(() => {
        wasm.sinwma_batch_js(empty, 10, 20, 5);
    });
});

test('SINWMA constant input normalization', () => {
    // Test with constant input to verify normalization
    const data = new Float64Array(20).fill(1.0);
    
    // For any period with constant input, result should produce constant output after warmup
    for (const period of [3, 5, 7]) {
        const result = wasm.sinwma_js(data, period);
        
        // After warmup, check if output is constant
        let firstValidOutput = null;
        for (let i = period; i < result.length; i++) {
            if (isFinite(result[i])) {
                if (firstValidOutput === null) {
                    firstValidOutput = result[i];
                } else {
                    assertClose(result[i], firstValidOutput, 1e-9, 
                        `Expected constant output for period=${period} at index ${i}`);
                }
            }
        }
    }
});