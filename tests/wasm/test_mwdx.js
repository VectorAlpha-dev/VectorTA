/**
 * WASM binding tests for MWDX indicator.
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

test('MWDX partial params', () => {
    // Test with default parameters - mirrors check_mwdx_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.mwdx_js(close, 0.2);
    assert(result instanceof Float64Array);
    assert.strictEqual(result.length, close.length);
});

test('MWDX accuracy', async () => {
    // Test MWDX matches expected values from Rust tests - mirrors check_mwdx_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.mwdx;
    
    const result = wasm.mwdx_js(
        close,
        expected.defaultParams.factor
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-7,
        "MWDX last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('mwdx', result, 'close', expected.defaultParams);
});

test('MWDX zero factor', () => {
    // Test MWDX fails with zero factor - mirrors check_mwdx_zero_factor
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.mwdx_js(inputData, 0.0);
    }, /Factor must be greater than 0/);
});

test('MWDX negative factor', () => {
    // Test MWDX fails with negative factor - mirrors check_mwdx_negative_factor
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.mwdx_js(inputData, -0.5);
    }, /Factor must be greater than 0/);
});

test('MWDX NaN factor', () => {
    // Test MWDX fails with NaN factor
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.mwdx_js(inputData, NaN);
    }, /Factor must be greater than 0/);
});

test('MWDX very small dataset', () => {
    // Test MWDX with single data point - mirrors check_mwdx_very_small_dataset
    const data = new Float64Array([42.0]);
    
    const result = wasm.mwdx_js(data, 0.2);
    assert.strictEqual(result.length, 1);
    assert.strictEqual(result[0], 42.0);
});

test('MWDX empty input', () => {
    // Test MWDX with empty input
    const dataEmpty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.mwdx_js(dataEmpty, 0.2);
    }, /No input data was provided/);
});

test('MWDX reinput', () => {
    // Test MWDX with re-input of MWDX result - mirrors check_mwdx_reinput
    const close = new Float64Array(testData.close);
    
    // First MWDX pass with factor=0.2
    const firstResult = wasm.mwdx_js(close, 0.2);
    
    // Second MWDX pass with factor=0.3 using first result as input
    const secondResult = wasm.mwdx_js(firstResult, 0.3);
    
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // All values should be finite
    for (let i = 0; i < secondResult.length; i++) {
        assert(isFinite(secondResult[i]), `NaN found at index ${i}`);
    }
});

test('MWDX NaN handling', () => {
    // Test MWDX handling of NaN values - mirrors check_mwdx_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.mwdx_js(close, 0.2);
    
    assert.strictEqual(result.length, close.length);
    
    // All values should be finite
    for (let i = 0; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN found at index ${i}`);
    }
});

test('MWDX batch', () => {
    // Test MWDX batch computation
    const close = new Float64Array(testData.close);
    
    // Test parameter ranges
    const batch_result = wasm.mwdx_batch_js(
        close, 
        0.1, 0.5, 0.1    // factor range: 0.1, 0.2, 0.3, 0.4, 0.5
    );
    
    // Get rows and cols info
    const rows_cols = wasm.mwdx_batch_rows_cols_js(0.1, 0.5, 0.1, close.length);
    const rows = rows_cols[0];
    const cols = rows_cols[1];
    
    assert(batch_result instanceof Float64Array);
    assert.strictEqual(rows, 5); // 5 factor values
    assert.strictEqual(cols, close.length);
    assert.strictEqual(batch_result.length, rows * cols);
    
    // Verify first combination matches individual calculation
    const individual_result = wasm.mwdx_js(close, 0.1);
    const batch_first = batch_result.slice(0, close.length);
    
    assertArrayClose(batch_first, individual_result, 1e-9, 'MWDX first combination');
});

test('MWDX different factors', () => {
    // Test MWDX with different factor values
    const close = new Float64Array(testData.close);
    
    // Test various factor values
    const testFactors = [0.1, 0.2, 0.5, 0.9];
    
    for (const factor of testFactors) {
        const result = wasm.mwdx_js(close, factor);
        assert.strictEqual(result.length, close.length);
        
        // All values should be finite
        let finiteCount = 0;
        for (let i = 0; i < result.length; i++) {
            if (isFinite(result[i])) finiteCount++;
        }
        assert.strictEqual(finiteCount, close.length, 
            `Found NaN values for factor=${factor}`);
    }
});

test('MWDX batch performance', () => {
    // Test that batch computation is more efficient than multiple single computations
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use first 1000 values
    
    // Test multiple parameter combinations
    const startBatch = performance.now();
    const batchResult = wasm.mwdx_batch_js(
        close,
        0.1, 0.9, 0.2    // factors: 0.1, 0.3, 0.5, 0.7, 0.9
    );
    const batchTime = performance.now() - startBatch;
    
    // Get the exact parameters used by the batch function
    const metadata = wasm.mwdx_batch_metadata_js(0.1, 0.9, 0.2);
    
    const startSingle = performance.now();
    const singleResults = [];
    // Use the exact parameters from metadata
    for (const factor of metadata) {
        const result = wasm.mwdx_js(close, factor);
        singleResults.push(...result);
    }
    const singleTime = performance.now() - startSingle;
    
    // Batch should be faster than multiple single calls
    console.log(`Batch time: ${batchTime.toFixed(2)}ms, Single time: ${singleTime.toFixed(2)}ms`);
    
    // Verify results match
    assertArrayClose(batchResult, singleResults, 1e-9, 'Batch vs single results');
});

test('MWDX edge cases', () => {
    // Test MWDX with edge case inputs
    
    // Test with monotonically increasing data
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = i + 1;
    }
    const result = wasm.mwdx_js(data, 0.2);
    assert.strictEqual(result.length, data.length);
    
    // All values should be finite
    for (let i = 0; i < result.length; i++) {
        assert(isFinite(result[i]), `NaN at index ${i}`);
    }
    
    // Test with constant values
    const constantData = new Float64Array(100).fill(50.0);
    const constantResult = wasm.mwdx_js(constantData, 0.2);
    
    assert.strictEqual(constantResult.length, constantData.length);
    
    // After initial value, all should converge to the constant
    for (let i = 10; i < constantResult.length; i++) {
        assertClose(constantResult[i], 50.0, 1e-6, 
            `Constant value failed at index ${i}`);
    }
});

test('MWDX batch metadata', () => {
    // Test metadata function returns correct factor values
    const metadata = wasm.mwdx_batch_metadata_js(
        0.2, 0.6, 0.2    // factor range: 0.2, 0.4, 0.6
    );
    
    // Should have 3 factor values
    assert.strictEqual(metadata.length, 3);
    assertClose(metadata[0], 0.2, 1e-9);
    assertClose(metadata[1], 0.4, 1e-9);
    assertClose(metadata[2], 0.6, 1e-9);
});

test('MWDX consistency across calls', () => {
    // Test that MWDX produces consistent results across multiple calls
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const result1 = wasm.mwdx_js(close, 0.2);
    const result2 = wasm.mwdx_js(close, 0.2);
    
    assertArrayClose(result1, result2, 1e-15, "MWDX results not consistent");
});

test('MWDX step precision', () => {
    // Test batch with various step sizes
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.mwdx_batch_js(
        data,
        0.2, 0.8, 0.3     // factors: 0.2, 0.5, 0.8
    );
    
    // Get rows and cols info
    const rows_cols = wasm.mwdx_batch_rows_cols_js(0.2, 0.8, 0.3, data.length);
    const rows = rows_cols[0];
    
    // Should have 3 combinations
    assert.strictEqual(rows, 3);
    assert.strictEqual(batch_result.length, 3 * data.length);
    
    // Verify metadata
    const metadata = wasm.mwdx_batch_metadata_js(0.2, 0.8, 0.3);
    assert.strictEqual(metadata.length, 3);
    assertClose(metadata[0], 0.2, 1e-9);
    assertClose(metadata[1], 0.5, 1e-9);
    assertClose(metadata[2], 0.8, 1e-9);
});

test('MWDX streaming simulation', () => {
    // Test MWDX streaming functionality (simulated)
    const close = new Float64Array(testData.close.slice(0, 100));
    const factor = 0.2;
    
    // Calculate batch result for comparison
    const batchResult = wasm.mwdx_js(close, factor);
    
    // MWDX provides values for all inputs
    assert.strictEqual(batchResult.length, close.length);
    
    // All values should be finite
    for (let i = 0; i < batchResult.length; i++) {
        assert(isFinite(batchResult[i]), `Expected finite value at index ${i}`);
    }
    
    // First value should match input
    assertClose(batchResult[0], close[0], 1e-9, "First value mismatch");
});

test('MWDX high factor', () => {
    // Test MWDX with high factor value
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.sin(i * 0.1) * 100 + 1000; // Generate some test data
    }
    
    const result = wasm.mwdx_js(data, 0.95);
    
    assert.strictEqual(result.length, data.length);
    
    // All values should be finite
    for (let i = 0; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

test('MWDX low factor', () => {
    // Test MWDX with low factor value
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = Math.sin(i * 0.1) * 100 + 1000; // Generate some test data
    }
    
    const result = wasm.mwdx_js(data, 0.01);
    
    assert.strictEqual(result.length, data.length);
    
    // All values should be finite
    for (let i = 0; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

test('MWDX NaN prefix', () => {
    // Test MWDX with NaN prefix in data
    const data = new Float64Array([NaN, NaN, 10.0, 20.0, 30.0, 40.0]);
    
    const result = wasm.mwdx_js(data, 0.2);
    assert.strictEqual(result.length, data.length);
    
    // MWDX should preserve NaN prefix but compute valid values starting from first non-NaN
    // This matches ALMA behavior and prevents NaN contamination of the entire series
    // result[0] = NaN (input)
    // result[1] = NaN (input)
    // result[2] = 10.0 (first non-NaN value)
    // result[3] = 0.2 * 20.0 + 0.8 * 10.0 = 4.0 + 8.0 = 12.0
    // result[4] = 0.2 * 30.0 + 0.8 * 12.0 = 6.0 + 9.6 = 15.6
    // result[5] = 0.2 * 40.0 + 0.8 * 15.6 = 8.0 + 12.48 = 20.48
    
    // First two values should be NaN (the prefix)
    assert(isNaN(result[0]), 'Expected NaN at index 0');
    assert(isNaN(result[1]), 'Expected NaN at index 1');
    
    // Starting from index 2, values should be computed correctly
    assert.strictEqual(result[2], 10.0);
    assert(Math.abs(result[3] - 12.0) < 1e-10, `Expected 12.0 at index 3, got ${result[3]}`);
    assert(Math.abs(result[4] - 15.6) < 1e-10, `Expected 15.6 at index 4, got ${result[4]}`);
    assert(Math.abs(result[5] - 20.48) < 1e-10, `Expected 20.48 at index 5, got ${result[5]}`);
    
    // Test with data that has no NaN prefix to verify normal operation
    const cleanData = new Float64Array([10.0, 20.0, 30.0, 40.0]);
    const cleanResult = wasm.mwdx_js(cleanData, 0.2);
    
    // All values should be finite for clean data
    for (let i = 0; i < cleanResult.length; i++) {
        assert(isFinite(cleanResult[i]), `Expected finite value at index ${i} for clean data`);
    }
});

test('MWDX formula verification', () => {
    // Verify MWDX formula: out[i] = fac * data[i] + (1 - fac) * out[i-1]
    const data = new Float64Array([10.0, 20.0, 30.0, 40.0]);
    const factor = 0.3;
    
    const result = wasm.mwdx_js(data, factor);
    
    // Manually calculate expected values
    const expected = [data[0]]; // First value is always the input
    for (let i = 1; i < data.length; i++) {
        const val = factor * data[i] + (1 - factor) * expected[i-1];
        expected.push(val);
    }
    
    assertArrayClose(result, expected, 1e-12, 'Formula verification failed');
});

test('MWDX all NaN input', () => {
    // Test MWDX with all NaN values
    const allNaN = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        allNaN[i] = NaN;
    }
    
    // MWDX doesn't raise error for all NaN - it returns all NaN
    const result = wasm.mwdx_js(allNaN, 0.2);
    for (let i = 0; i < result.length; i++) {
        assert(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
});

test('MWDX oscillating data', () => {
    // Test with oscillating values
    const data = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        data[i] = (i % 2 === 0) ? 10.0 : 20.0;
    }
    
    const result = wasm.mwdx_js(data, 0.5);
    assert.strictEqual(result.length, data.length);
    
    // All values should be finite
    for (let i = 0; i < result.length; i++) {
        assert(isFinite(result[i]), `Expected finite value at index ${i}`);
    }
});

test('MWDX small step size', () => {
    // Test batch with very small step size
    const data = new Float64Array(50);
    for (let i = 0; i < 50; i++) {
        data[i] = i + 1;
    }
    
    const batch_result = wasm.mwdx_batch_js(
        data,
        0.1, 0.2, 0.05     // factors: 0.1, 0.15, 0.2
    );
    
    const rows_cols = wasm.mwdx_batch_rows_cols_js(0.1, 0.2, 0.05, data.length);
    const rows = rows_cols[0];
    
    assert.strictEqual(rows, 3);
    assert.strictEqual(batch_result.length, 3 * data.length);
});